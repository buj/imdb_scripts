from fastai.text.learner import *
from fastai.text import *
from torch_util import *


def get_prs(c, nt):
    """
    Returns the proportions (probabilities) for each word in <c>.
    <c>: a numpy array of word ids
    <nt>: size of the vocabulary == number of distinct words in <c>
    """
    uni_counter = Counter(c)
    uni_counts = np.array([uni_counter[o] for o in range(nt)])
    return uni_counts/uni_counts.sum()

class LinearDecoder(nn.Module):
    initrange=0.1
    def __init__(self, n_out, nhid, dropout, tie_encoder=None, decode_train=True):
        """
        <n_out>: size of the output layer (e.g. number of distinct classes)
        <nhid>: size of the hidden layer, which is to be projected into
            the output layer.
        <droupout>: probability of drouput. At each time step and for
            each node, it is the same.
        <tie_encoder>: if True, ties the decoder weights to the encoder
            (embedding) weights (?)
        <decode_train>: whether the decoder layer should be trained
            or not.
        """
        super().__init__()
        self.decode_train = decode_train
        self.decoder = nn.Linear(nhid, n_out, bias=False)
        self.decoder.weight.data.uniform_(-self.initrange, self.initrange)
        self.dropout = LockedDropout(dropout)
        if tie_encoder: self.decoder.weight = tie_encoder.weight

    def forward(self, input):
        raw_outputs, outputs = input
        output = self.dropout(outputs[-1])
        output = output.view(output.size(0)*output.size(1), output.size(2))
        if self.decode_train or not self.training:
            decoded = self.decoder(output)
            output = decoded.view(-1, decoded.size(1))
        return output, raw_outputs, outputs


def get_language_model(n_tok, em_sz, nhid, nlayers, pad_token, decode_train=True, dropouts=None):
    """
    Constructs and returns a language model with the given parameters.
    The model consists of a recurrent part (RNNEncoder), which takes as
    input the current state and the next word and returns the next state;
    and a linear decoder part that makes predictions based on the current
    state.
    
    <n_tok>: number of different words (vocabulary size)
    <em_sz>: embedding size (length of vectors corresponding to words)
    <nhid>: size of the hidden representation layer
    <n_layers>: number of layers of RNN
    <pad_token>: 'the int value used for padding text'
    
    <dropouts[0]>: dropout to apply to the activations going from one LSTM layer to another
    <dropouts[1]>: dropout used by linear decoder
    <dropouts[2]>: dropout to apply to the input layer.
    <dropouts[3]>: dropout to apply to the embedding layer.
    <dropouts[4]>: dropout used for a LSTM's internal (or hidden) recurrent weights.
    """
    if dropouts is None: dropouts = [0.5,0.4,0.5,0.05,0.3]
    rnn_enc = RNN_Encoder(n_tok, em_sz, n_hid=nhid, n_layers=nlayers, pad_token=pad_token,
                dropouti=dropouts[0], wdrop=dropouts[2], dropoute=dropouts[3], dropouth=dropouts[4])
    rnn_dec = LinearDecoder(n_tok, em_sz, dropouts[1], decode_train=decode_train, tie_encoder=rnn_enc.encoder)
    return SequentialRNN(rnn_enc, rnn_dec)


def pt_sample(pr, ns):
    """"""
    w = -torch.log(make_cuda(FloatTensor(len(pr))).uniform_())/(pr+1e-10)
    return torch.topk(w, ns, largest=False)[1]


class CrossEntDecoder(nn.Module):
    initrange=0.1
    def __init__(self, prs, decoder, n_neg=4000, sampled=True):
        super().__init__()
        self.prs,self.decoder,self.sampled = make_cuda(T(prs)),decoder,sampled
        self.set_n_neg(n_neg)

    def set_n_neg(self, n_neg): self.n_neg = n_neg

    def get_rand_idxs(self): return pt_sample(self.prs, self.n_neg)

    def sampled_softmax(self, input, target):
        idxs = V(self.get_rand_idxs())
        dw = self.decoder.weight
        #db = self.decoder.bias
        output = input @ dw[idxs].t() #+ db[idxs]
        max_output = output.max()
        output = output - max_output
        num = (dw[target] * input).sum(1) - max_output
        negs = torch.exp(num) + (torch.exp(output)*2).sum(1)
        return (torch.log(negs) - num).mean()

    def forward(self, input, target):
        if self.decoder.training:
            if self.sampled: return self.sampled_softmax(input, target)
            else: input = self.decoder(input)
        return F.cross_entropy(input, target)

def get_learner(drops, n_neg, sampled, md, em_sz, nh, nl, opt_fn, prs):
    m = to_gpu(get_language_model(md.n_tok, em_sz, nh, nl, md.pad_idx, decode_train=False, dropouts=drops))
    crit = make_cuda(CrossEntDecoder(prs, m[1].decoder, n_neg=n_neg, sampled=sampled))
    learner = RNN_Learner(md, LanguageModel(m), opt_fn=opt_fn)
    crit.dw = learner.model[0].encoder.weight
    learner.crit = crit
    learner.reg_fn = partial(seq2seq_reg, alpha=2, beta=1)
    learner.clip=0.3
    return learner,crit

