import torch as T
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
# from data_util import config_word2vec
from data_util import config_word2vec, data_word2vec
from data_util.data_word2vec import Vocab
import torch.nn.functional as F
from train_util import get_cuda
from gensim.models import KeyedVectors
import numpy as np
# from gensim.scripts.glove2word2vec import glove2word2vec
# # 将glove转化为word2vec形式
# def transfer(gloveFile, word2vecFile):
#     glove2word2vec(gloveFile, word2vecFile)


def init_lstm_wt(lstm):
    for name, _ in lstm.named_parameters():
        if 'weight' in name:
            wt = getattr(lstm, name)
            wt.data.uniform_(-config_word2vec.rand_unif_init_mag,
                             config_word2vec.rand_unif_init_mag)
        elif 'bias' in name:
            # set forget bias to 1
            bias = getattr(lstm, name)
            n = bias.size(0)
            start, end = n // 4, n // 2
            bias.data.fill_(0.)
            bias.data[start:end].fill_(1.)


def init_linear_wt(linear):
    linear.weight.data.normal_(std=config_word2vec.trunc_norm_init_std)
    if linear.bias is not None:
        linear.bias.data.normal_(std=config_word2vec.trunc_norm_init_std)


def init_wt_normal(wt):
    wt.data.normal_(std=config_word2vec.trunc_norm_init_std)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.lstm = nn.LSTM(config_word2vec.emb_dim,
                            config_word2vec.hidden_dim,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=True)
        init_lstm_wt(self.lstm)

        self.reduce_h = nn.Linear(config_word2vec.hidden_dim * 2, config_word2vec.hidden_dim)
        init_linear_wt(self.reduce_h)
        self.reduce_c = nn.Linear(config_word2vec.hidden_dim * 2, config_word2vec.hidden_dim)
        init_linear_wt(self.reduce_c)

    def forward(self, x, seq_lens):
        packed = pack_padded_sequence(x, seq_lens, batch_first=True)
        enc_out, enc_hid = self.lstm(packed)
        enc_out, _ = pad_packed_sequence(enc_out, batch_first=True)
        enc_out = enc_out.contiguous()  #bs, n_seq, 2*n_hid
        h, c = enc_hid  #shape of h: 2, bs, n_hid
        h = T.cat(list(h), dim=1)  #bs, 2*n_hid
        c = T.cat(list(c), dim=1)
        h_reduced = F.relu(self.reduce_h(h))  #bs,n_hid
        c_reduced = F.relu(self.reduce_c(c))
        return enc_out, (h_reduced, c_reduced)

class encoder_attention(nn.Module):
    def __init__(self):
        super(encoder_attention, self).__init__()
        self.W_h = nn.Linear(config_word2vec.hidden_dim * 2,
                             config_word2vec.hidden_dim * 2,
                             bias=False)
        self.W_s = nn.Linear(config_word2vec.hidden_dim * 2, config_word2vec.hidden_dim * 2)
        self.v = nn.Linear(config_word2vec.hidden_dim * 2, 1, bias=False)

    def forward(self, st_hat, h, enc_padding_mask, sum_temporal_srcs):
        ''' Perform attention over encoder hidden states
        :param st_hat: decoder hidden state at current time step
        :param h: encoder hidden states
        :param enc_padding_mask:
        :param sum_temporal_srcs: if using intra-temporal attention, contains summation of attention weights from previous decoder time steps
        '''

        # Standard attention technique (eq 1 in https://arxiv.org/pdf/1704.04368.pdf)
        et = self.W_h(h)  # bs,n_seq,2*n_hid
        dec_fea = self.W_s(st_hat).unsqueeze(1)  # bs,1,2*n_hid
        et = et + dec_fea
        et = T.tanh(et)  # bs,n_seq,2*n_hid
        et = self.v(et).squeeze(2)  # bs,n_seq


        # intra-temporal attention     (eq 3 in https://arxiv.org/pdf/1705.04304.pdf)
        if config_word2vec.intra_encoder:
            exp_et = T.exp(et)
            if sum_temporal_srcs is None:
                et1 = exp_et
                sum_temporal_srcs = get_cuda(
                    T.FloatTensor(et.size()).fill_(1e-10)) + exp_et
            else:
                et1 = exp_et / sum_temporal_srcs  #bs, n_seq
                sum_temporal_srcs = sum_temporal_srcs + exp_et
        else:
            et1 = F.softmax(et, dim=1)

        # assign 0 probability for padded elements
        at = et1 * enc_padding_mask
        normalization_factor = at.sum(1, keepdim=True)
        at = at / normalization_factor

        at = at.unsqueeze(1)  #bs,1,n_seq
        # Compute encoder context vector
        ct_e = T.bmm(at, h)  #bs, 1, 2*n_hid
        ct_e = ct_e.squeeze(1)
        at = at.squeeze(1)

        return ct_e, at, sum_temporal_srcs


class decoder_attention(nn.Module):
    def __init__(self):
        super(decoder_attention, self).__init__()
        if config_word2vec.intra_decoder:
            self.W_prev = nn.Linear(config_word2vec.hidden_dim,
                                    config_word2vec.hidden_dim,
                                    bias=False)
            self.W_s = nn.Linear(config_word2vec.hidden_dim, config_word2vec.hidden_dim)
            self.v = nn.Linear(config_word2vec.hidden_dim, 1, bias=False)

    def forward(self, s_t, prev_s):
        '''Perform intra_decoder attention
        Args
        :param s_t: hidden state of decoder at current time step
        :param prev_s: If intra_decoder attention, contains list of previous decoder hidden states
        '''
        if config_word2vec.intra_decoder is False:
            ct_d = get_cuda(T.zeros(s_t.size()))
        elif prev_s is None:
            ct_d = get_cuda(T.zeros(s_t.size()))
            prev_s = s_t.unsqueeze(1)  #bs, 1, n_hid
        else:
            # Standard attention technique (eq 1 in https://arxiv.org/pdf/1704.04368.pdf)
            et = self.W_prev(prev_s)  # bs,t-1,n_hid
            dec_fea = self.W_s(s_t).unsqueeze(1)  # bs,1,n_hid
            et = et + dec_fea
            et = T.tanh(et)  # bs,t-1,n_hid
            et = self.v(et).squeeze(2)  # bs,t-1
            # intra-decoder attention     (eq 7 & 8 in https://arxiv.org/pdf/1705.04304.pdf)
            at = F.softmax(et, dim=1).unsqueeze(1)  #bs, 1, t-1
            ct_d = T.bmm(at, prev_s).squeeze(1)  #bs, n_hid,bmm计算乘法
            prev_s = T.cat([prev_s, s_t.unsqueeze(1)], dim=1)  #bs, t, n_hid

        return ct_d, prev_s


class sentiment_attention(nn.Module):
    def __init__(self):
        super(sentiment_attention, self).__init__()
        self.W_h = nn.Linear(config_word2vec.hidden_dim * 2,
                             config_word2vec.hidden_dim * 2)
        self.v1 = nn.Linear(config_word2vec.hidden_dim * 2, 1, bias=False)
        self.v2 = nn.Linear(config_word2vec.max_enc_steps, 1, bias=False)

    def forward(self, h, enc_padding_mask):
        ''' Perform attention over encoder hidden states
        :param h: encoder hidden states
        :param enc_padding_mask:
        '''

        # Standard attention technique (eq 3 in https://arxiv.org/pdf/1906.00318.pdf)
        et = self.W_h(h)  # bs,n_seq,2*n_hid
        et = T.tanh(et)  # bs,n_seq,2*n_hid
        et0 = et
        et1 = self.v1(et0).squeeze(2)  # bs,n_seq
        et1 = T.sigmoid(et1)

        # assign 0 probability for padded elements
        au = et1 * enc_padding_mask
        normalization_factor = au.sum(1, keepdim=True)
        au = au / normalization_factor
        et = et.permute(0,2,1)
        if h.size()[1] < config_word2vec.max_enc_steps:
            diff = config_word2vec.max_enc_steps - h.size()[1]
            # t2 = get_cuda(T.zeros([4*config_word2vec.batch_size, 1200, diff]))
            t2 = get_cuda(T.zeros([et.size()[0], 1200, diff]))
            et = T.cat((et, t2), 2)
        et = self.v2(et).squeeze(2)
        et2 = T.sigmoid(et)
        return au, et2


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.enc_attention = encoder_attention()
        self.dec_attention = decoder_attention()
        self.senti_attention = sentiment_attention()
        self.x_context = nn.Linear(config_word2vec.hidden_dim * 2 + config_word2vec.emb_dim,
                                   config_word2vec.emb_dim)

        self.lstm = nn.LSTMCell(config_word2vec.emb_dim, config_word2vec.hidden_dim)
        init_lstm_wt(self.lstm)

        # self.p_gen_linear = nn.Linear(config.hidden_dim * 5 + config.emb_dim,1)
        #修改权重添加的
        self.p_gen_emotion = nn.Linear(config_word2vec.hidden_dim*2, 1, bias=False)
        self.p_gen_w_ct_e = nn.Linear(config_word2vec.hidden_dim * 2, 1, bias=False)
        # self.p_gen_w_h = T.transpose(p_gen_w_h, 0, 1)
        self.p_gen_w_ct_d = nn.Linear(config_word2vec.hidden_dim, 1, bias=False)
        # self.p_gen_w_s = T.transpose(p_gen_w_s, 0, 1)
        self.p_gen_w_st_hat = nn.Linear(config_word2vec.hidden_dim * 2, 1)
        # self.p_gen_x_t = T.transpose(p_gen_x_t, 0, 1)
        # p_vocab
        self.V = nn.Linear(config_word2vec.hidden_dim * 6, config_word2vec.hidden_dim)    # 情感注意力这里也要改
        self.V1 = nn.Linear(config_word2vec.hidden_dim, config_word2vec.vocab_size)    # Linear(input_size, output_size)
        init_linear_wt(self.V1)

    def forward(self, x_t, s_t, enc_out, enc_padding_mask, ct_e, extra_zeros,
                enc_batch_extend_vocab, sum_temporal_srcs, prev_s):
        x = self.x_context(T.cat([x_t, ct_e], dim=1))   # emb_dim,即256  bs,256
        s_t = self.lstm(x, s_t)

        dec_h, dec_c = s_t   # dec_h,next_hidden_state; dec_c,next_cell_state
        st_hat = T.cat([dec_h, dec_c], dim=1)  # st_hat表示当前时刻编码层的隐藏状态
        ct_e, attn_dist, sum_temporal_srcs = self.enc_attention(
            st_hat, enc_out, enc_padding_mask, sum_temporal_srcs)   # attn_dist对应at，size为bs,n_seq
        au, et2 = self.senti_attention(enc_out, enc_padding_mask)

        ct_d, prev_s = self.dec_attention(dec_h,prev_s)  #intra-decoder attention

        # p_gen = T.cat([ct_e, ct_d, st_hat, x], 1)    # 按列拼，p_gen: hidden_dim * 5 + config.emb
        # p_gen = self.p_gen_linear(p_gen)  # bs,1
        # p_gen = T.sigmoid(p_gen)  # bs,1
        p_gen_emotion = self.p_gen_emotion(et2)
        p_gen_ct_e = self.p_gen_w_ct_e(ct_e)  #  ct_e: bs, 2*n_hid
        p_gen_ct_d = self.p_gen_w_ct_d(ct_d)  # ct_d: bs,n_hid
        p_gen_st_hat = self.p_gen_w_st_hat(st_hat)   # st_hat: bs,2*n_hid
        p_gen = p_gen_ct_e + p_gen_ct_d + p_gen_st_hat + p_gen_emotion
        p_gen = T.sigmoid(p_gen)
        # print(dec_h.size(), ct_e.size(), ct_d.size(), et2.size())
        out = T.cat([dec_h, ct_e, ct_d, et2], dim=1)  # bs, 4*n_hid
        out = self.V(out)  # bs,n_hid
        out = self.V1(out)  # bs, n_vocab
        vocab_dist = F.softmax(out, dim=1)  # bs, n_vocab
        vocab_dist = p_gen * vocab_dist
        attn_dist_ = (1 - p_gen) * attn_dist

        # pointer mechanism (as suggested in eq 9 https://arxiv.org/pdf/1704.04368.pdf), extra_zeros is not None即意味着有oov
        if extra_zeros is not None:
            # print("这里扩充词典")
            vocab_dist = T.cat([vocab_dist, extra_zeros], dim=1)    # 如果有oov就扩充词典大小
        final_dist = vocab_dist.scatter_add(1, enc_batch_extend_vocab, attn_dist_)
        # 最后生成的final_dist与vocab_dist维度是一致的,vocab_dist的维度见config中的vocab_size,把attn_dist发散到vocab_dist上生成final_dist
        # enc_batch_extnd_vocab, (bs,max_enc_seq_len) attn_dist:(bs, max_enc_seq_len)
        return final_dist, s_t, ct_e, au, et2, sum_temporal_srcs, prev_s


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.vocab = Vocab(config_word2vec.vocab_path, config_word2vec.vocab_size)    #初始化vocab
        self.encoder = Encoder()
        self.decoder = Decoder()
        # 导入预训练的词向量
        # transfer('./glove.840B.300d.txt', './glove_vectors.txt')
        tmp_file = 'sgns.renmin.bigram-char'
        wvmodel = KeyedVectors.load_word2vec_format(tmp_file)
        # print('*********model_loaded**********')
        self.embeds = nn.Embedding(config_word2vec.vocab_size, config_word2vec.emb_dim)
        weight = T.zeros(config_word2vec.vocab_size, config_word2vec.emb_dim)


        # 读取unk向量表示
        # with open('vectors.txt', 'r') as f:
        #     line = f.readlines()[-1]
        #     unk_vec = np.array([float(n) for n in line.split(' ')[1:]], dtype=np.float32)
        for i in range(len(wvmodel.index2word)):
            # print(i)
            # print(wvmodel.index2word[i])
            # print(self.vocab.word2id('一潭死水'))
            # print(self.vocab.word2id(wvmodel.index2word[i]))
            index = self.vocab.word2id(wvmodel.index2word[i])
            if index != 0:
                weight[index, :] = T.from_numpy(wvmodel.get_vector(self.vocab.id2word(index)))

        self.embeds = nn.Embedding.from_pretrained(weight)
        self.embeds.weight.requires_grad = True
        # print(self.embeds.weight)

        init_wt_normal(self.embeds.weight)

        self.encoder = get_cuda(self.encoder)
        self.decoder = get_cuda(self.decoder)
        self.embeds = get_cuda(self.embeds)


# if __name__ == '__main__':
#     from torchstat import stat
#     model = Model()
#     print("model loaded")
    
