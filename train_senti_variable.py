import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"    #指定gpu
from data_util.log import logger
import time
import torch as T
import torch.nn.functional as F
from model_pretrained_senti import Model
from data_util import config_word2vec, data_word2vec
from data_util.batcher_word2vec import Batcher
from data_util.data_word2vec import Vocab
from train_util import get_enc_data, get_cuda, get_dec_data
from torch.distributions import Categorical
from rouge import Rouge
# from bert_score import score
from numpy import random
import argparse
import numpy as np


random.seed(123)
T.manual_seed(123)
if T.cuda.is_available():
    T.cuda.manual_seed_all(123)


class Train(object):
    def __init__(self, opt):
        self.vocab = Vocab(config_word2vec.vocab_path, config_word2vec.vocab_size)    #初始化vocab
        print(self.vocab.size())
        self.batcher = Batcher(config_word2vec.train_data_path,
                               self.vocab,
                               mode='train',
                               batch_size=config_word2vec.batch_size,
                               single_pass=False)
        self.opt = opt
        self.start_id = self.vocab.word2id(data_word2vec.START_DECODING)
        self.end_id = self.vocab.word2id(data_word2vec.STOP_DECODING)
        self.pad_id = self.vocab.word2id(data_word2vec.PAD_TOKEN)
        self.unk_id = self.vocab.word2id(data_word2vec.UNKNOWN_TOKEN)
        time.sleep(5)

    def save_model(self, iter):
        save_path = config_word2vec.save_model_path + "/%07d.tar" % iter
        T.save(
            {
                "iter": iter + 1,
                "model_dict": self.model.state_dict(),
                "trainer_dict": self.trainer.state_dict()
            }, save_path)

    def setup_train(self):
        self.model = Model()
        # print('get model')
        # if T.cuda.device_count() > 1:
        # self.model = T.nn.DataParallel(self.model, device_ids=[0,1])
        self.model = get_cuda(self.model)

        self.trainer = T.optim.Adam(self.model.parameters(), lr=config_word2vec.lr)
        # 学习率衰减
        # self.scheduler = T.optim.lr_scheduler.StepLR(self.trainer, step_size=50, gamma=0.1)
        start_iter = 0
        if self.opt.load_model is not None:
            load_model_path = os.path.join(config_word2vec.save_model_path,
                                           self.opt.load_model)
            checkpoint = T.load(load_model_path)
            start_iter = checkpoint["iter"]
            self.model.load_state_dict(checkpoint["model_dict"])
            self.trainer.load_state_dict(checkpoint["trainer_dict"])
            print("Loaded model at " + load_model_path)
            # print(self.vocab.size())
        if self.opt.new_lr is not None:
            self.trainer = T.optim.Adam(self.model.parameters(),
                                        lr=self.opt.new_lr)
        return start_iter

    def train_batch_MLE(self, s_e, enc_out, enc_hidden, enc_padding_mask, ct_e,
                        extra_zeros, enc_batch_extend_vocab, batch):
        ''' Calculate Negative Log Likelihood Loss for the given batch. In order to reduce exposure bias,
                pass the previous generated token as input with a probability of 0.25 instead of ground truth label
        Args:
        :param enc_out: Outputs of the encoder for all time steps (batch_size, length_input_sequence, 2*hidden_size)
        :param enc_hidden: Tuple containing final hidden state & cell state of encoder. Shape of h & c: (batch_size, hidden_size)
        :param enc_padding_mask: Mask for encoder input; Tensor of size (batch_size, length_input_sequence) with values of 0 for pad tokens & 1 for others
        :param ct_e: encoder context vector for time_step=0 (eq 5 in https://arxiv.org/pdf/1705.04304.pdf)
        :param extra_zeros: Tensor used to extend vocab distribution for pointer mechanism
        :param enc_batch_extend_vocab: Input batch that stores OOV ids
        :param batch: batch object
        '''
        dec_batch, max_dec_len, dec_lens, target_batch = get_dec_data(
            batch)  #Get input and target batchs for training decoder
        step_losses = []
        s_t = (enc_hidden[0], enc_hidden[1])  #Decoder hidden states
        x_t = get_cuda(T.LongTensor(len(enc_out)).fill_(
            self.start_id))  #Input to the decoder
        prev_s = None  #Used for intra-decoder attention (section 2.2 in https://arxiv.org/pdf/1705.04304.pdf)
        sum_temporal_srcs = None  #Used for intra-temporal attention (section 2.1 in https://arxiv.org/pdf/1705.04304.pdf)
        # print("*****len_enc*****************")
        # print(len(enc_out))
        for t in range(min(max_dec_len, config_word2vec.max_dec_steps)):
            use_gound_truth = get_cuda((T.rand(len(enc_out)) > 0.25)).long()  #Probabilities indicating whether to use ground truth labels instead of previous decoded tokens
            x_t = use_gound_truth * dec_batch[:, t] + (
                1 - use_gound_truth
            ) * x_t  #Select decoder input based on use_ground_truth probabilities
            x_t = self.model.embeds(x_t)
            final_dist, s_t, ct_e, au, et2, sum_temporal_srcs, prev_s = self.model.decoder(
                x_t, s_t, enc_out, enc_padding_mask, ct_e, extra_zeros,
                enc_batch_extend_vocab, sum_temporal_srcs, prev_s)
            # print(au.size())
            # print(s_e.size())
            # print("********情感注意力的大小*********")
            # print(au.size())
            target = target_batch[:, t]
            log_probs = T.log(final_dist + config_word2vec.eps)
            step_loss = F.cross_entropy(log_probs,
                                   target,
                                   reduction="none",
                                   ignore_index=self.pad_id)
            # print(au.size(), s_e.size())
            # 情感损失
            senti_loss = F.binary_cross_entropy(get_cuda(au), get_cuda(s_e))
            # print("********senti loss*********")
            # print(senti_loss)
            final_loss = self.opt.normal_weight * step_loss + self.opt.sentiment_weight * senti_loss
            # print("************final_loss***************")
            # print(final_loss)
            step_losses.append(final_loss)
            # print("*******final_dist***********")
            # print(final_dist)
            x_t = T.multinomial(final_dist, 1).squeeze()  #Sample words from final distribution which can be used as input in next time step，从finaldistribution中进行词采样，并作为下一步的输入
            is_oov = (x_t >= config_word2vec.vocab_size).long()  #Mask indicating whether sampled word is OOV
            x_t = (1 - is_oov) * x_t.detach() + (is_oov) * self.unk_id  #Replace OOVs with [UNK] token

        losses = T.sum(
            T.stack(step_losses, 1), 1
        )  #unnormalized losses for each example in the batch; (batch_size)
        batch_avg_loss = losses / dec_lens  #Normalized losses; (batch_size)
        mle_loss = T.mean(batch_avg_loss)  #Average batch loss
        return mle_loss

    def train_batch_RL(self, enc_out, enc_hidden, enc_padding_mask, ct_e,
                       extra_zeros, enc_batch_extend_vocab, article_oovs,
                       greedy):
        '''Generate sentences from decoder entirely using sampled tokens as input. These sentences are used for ROUGE evaluation
        Args
        :param enc_out: Outputs of the encoder for all time steps (batch_size, length_input_sequence, 2*hidden_size)
        :param enc_hidden: Tuple containing final hidden state & cell state of encoder. Shape of h & c: (batch_size, hidden_size)
        :param enc_padding_mask: Mask for encoder input; Tensor of size (batch_size, length_input_sequence) with values of 0 for pad tokens & 1 for others
        :param ct_e: encoder context vector for time_step=0 (eq 5 in https://arxiv.org/pdf/1705.04304.pdf)
        :param extra_zeros: Tensor used to extend vocab distribution for pointer mechanism
        :param enc_batch_extend_vocab: Input batch that stores OOV ids
        :param article_oovs: Batch containing list of OOVs in each example
        :param greedy: If true, performs greedy based sampling, else performs multinomial sampling
        Returns:
        :decoded_strs: List of decoded sentences
        :log_probs: Log probabilities of sampled words
        '''
        s_t = enc_hidden  #Decoder hidden states
        x_t = get_cuda(T.LongTensor(len(enc_out)).fill_(self.start_id))  #Input to the decoder,enc_out为encoder的输出
        prev_s = None  #Used for intra-decoder attention (section 2.2 in https://arxiv.org/pdf/1705.04304.pdf)    解码的注意力机制
        sum_temporal_srcs = None  #Used for intra-temporal attention (section 2.1 in https://arxiv.org/pdf/1705.04304.pdf)    输入序列的注意力机制
        inds = []  #Stores sampled indices for each time step
        decoder_padding_mask = []  #Stores padding masks of generated samples
        log_probs = []  #Stores log probabilites of generated samples
        mask = get_cuda(
            T.LongTensor(len(enc_out)).fill_(1)
        )  #Values that indicate whether [STOP] token has already been encountered; 1 => Not encountered, 0 otherwise；    mask用1填充

        for t in range(config_word2vec.max_dec_steps):
            x_t = self.model.embeds(x_t)
            probs, s_t, ct_e, au, et2, sum_temporal_srcs, prev_s = self.model.decoder(
                x_t, s_t, enc_out, enc_padding_mask, ct_e, extra_zeros,
                enc_batch_extend_vocab, sum_temporal_srcs, prev_s)
            if greedy is False:
                multi_dist = Categorical(probs)
                x_t = multi_dist.sample()  #perform multinomial sampling，根据概率分布来产生样本
                log_prob = multi_dist.log_prob(x_t)
                log_probs.append(log_prob)
            else:
                _, x_t = T.max(probs, dim=1)  #perform greedy sampling
            x_t = x_t.detach()
            inds.append(x_t)
            mask_t = get_cuda(T.zeros(
                len(enc_out)))  #Padding mask of batch for current time step
            mask_t[
                mask == 1] = 1  #If [STOP] is not encountered till previous time step, mask_t = 1 else mask_t = 0
            mask[
                (mask == 1) + (x_t == self.end_id) == 2] = 0  #If [STOP] is not encountered till previous time step and current word is [STOP], make mask = 0
            decoder_padding_mask.append(mask_t)
            # print(x_t)
            # print(self.vocab.size())
            is_oov = (x_t >= config_word2vec.vocab_size ).long()  #Mask indicating whether sampled word is OOV
            x_t = (1 - is_oov) * x_t + (is_oov) * self.unk_id  #Replace OOVs with [UNK] token


        inds = T.stack(inds, dim=1)
        decoder_padding_mask = T.stack(decoder_padding_mask, dim=1)
        if greedy is False:  #If multinomial based sampling, compute log probabilites of sampled words
            log_probs = T.stack(log_probs, dim=1)
            log_probs = log_probs * decoder_padding_mask  #Not considering sampled words with padding mask = 0
            lens = T.sum(decoder_padding_mask, dim=1)  #Length of sampled sentence
            log_probs = T.sum( log_probs, dim=1 ) / lens  # (bs,)                                     #compute normalizied log probability of a sentence
        decoded_strs = []
        # print(inds)
        for i in range(len(enc_out)):    #enc_out所有时刻的编码输出
            id_list = inds[i].cpu().numpy()
            oovs = article_oovs[i]    #每个步长的oov，输出发现都是[]  #article_oovs=batch.art_oovs
            # print('here')
            # print(oovs)
            S = data_word2vec.outputids2words(id_list, self.vocab,oovs)  #Generate sentence corresponding to sampled words，根据给定的词来生成句子
            try:
                end_idx = S.index(data_word2vec.STOP_DECODING)
                S = S[:end_idx]
            except ValueError:
                S = S
            if len(S) < 2:  #If length of sentence is less than 2 words, replace it with "xxx"; Avoids setences like "." which throws error while calculating ROUGE
                S = ["xxx"]
            S = " ".join(S)
            decoded_strs.append(S)

        return decoded_strs, log_probs

    # 原始的使用rouge作为reward
    def reward_function(self, decoded_sents, original_sents):
        rouge = Rouge()
        try:
            scores = rouge.get_scores(decoded_sents, original_sents)
        except Exception:
            print(
                "Rouge failed for multi sentence evaluation.. Finding exact pair"
            )
            scores = []
            for i in range(len(decoded_sents)):
                try:
                    score = rouge.get_scores(decoded_sents[i],
                                             original_sents[i])
                except Exception:
                    print("Error occured at:")
                    print("decoded_sents:", decoded_sents[i])
                    print("original_sents:", original_sents[i])
                    score = [{"rouge-l": {"f": 0.0}}]
                scores.append(score[0])
        rouge_l_f1 = [score["rouge-l"]["f"] for score in scores]
        rouge_l_f1 = get_cuda(T.FloatTensor(rouge_l_f1))
        return rouge_l_f1

    # def reward_function(self, decoded_sents, original_sents):
    #     # try:
    #     # print(decoded_sents, original_sents)
    #     (P, R, F), hashname = score(decoded_sents, original_sents, lang="zh", return_hash=True, rescale_with_baseline=True, device='cuda:3')
    #     # print(f"{hashname}: P={P.mean().item():.6f} R={R.mean().item():.6f} F={F.mean().item():.6f}")
    #     # except Exception:
    #     #     print(
    #     #         "Bertscoer failed"
    #     #     )
    #         # scores = []
    #         # for i in range(len(decoded_sents)):
    #         #     try:
    #         #         score = score([decoded_sents[i]],
    #         #                                  [original_sents[i]])
    #         #     except Exception:
    #         #         print("Error occured at:")
    #         #         print("decoded_sents:", decoded_sents[i])
    #         #         print("original_sents:", original_sents[i])
    #         #         score = [{"rouge-l": {"f": 0.0}}]
    #         #     scores.append(score[0])
    #     bert_f = F.mean().item()
    #     # print("111111111111111111")
    #     # print(bert_f,type(bert_f))
    #     bert_f = get_cuda(T.FloatTensor([bert_f]))
    #     # print("22222222222222222222")
    #     # print(bert_f)
    #     return bert_f

    # def write_to_file(self, decoded, max, original, sample_r, baseline_r, iter):
    #     with open("temp.txt", "w") as f:
    #         f.write("iter:"+str(iter)+"\n")
    #         for i in range(len(original)):
    #             f.write("dec: "+decoded[i]+"\n")
    #             f.write("max: "+max[i]+"\n")
    #             f.write("org: "+original[i]+"\n")
    #             f.write("Sample_R: %.4f, Baseline_R: %.4f\n\n"%(sample_r[i].item(), baseline_r[i].item()))

    def train_one_batch(self, batch, iter):
        enc_batch, enc_lens, enc_padding_mask, enc_batch_extend_vocab, extra_zeros, context = get_enc_data(batch)
        # enc_batch存了每篇新闻的id表示(其中oov的id表示为0)， enc_batch_extend_vocab存了每篇新闻的id表示(其中oov的id是计算后的表示)，enc_padding_mask用1填充
        # print("***********************")
        # print(enc_batch)
        # print(enc_batch_extend_vocab)    # 如果没有oov生成的enc_batch和enc_batch_extend_vocab是一样的
        # print("***********输入文章的大小************")
        # print(enc_batch.size(), enc_batch)


        # 加载情感词典
        senti_dict = open('union.txt', 'r', encoding='utf-8-sig')
        senti_list = []
        for line in senti_dict.readlines():
            line = line.strip('\n')
            senti_list.append(line)
        # senti_len = len(senti_list)
        enc_strs = []
        for i in enc_batch:    #enc_out所有时刻的编码输出
            id_list = i.cpu().numpy()
            oovs = []    #每个步长的oov，输出发现都是[]  #article_oovs=batch.art_oovs
            S = data_word2vec.outputids2words(id_list, self.vocab,oovs)  #Generate sentence corresponding to sampled words，根据给定的词来生成句子
            try:
                end_idx = S.index(data_word2vec.STOP_DECODING)
                S = S[:end_idx]
            except ValueError:
                S = S
            if len(S) < 2:  #If length of sentence is less than 2 words, replace it with "xxx"; Avoids setences like "." which throws error while calculating ROUGE
                S = ["xxx"]
            S = " ".join(S)
            enc_strs.append(S)
        # print("*************输入的文章***************")
        # print(enc_strs)
        # 这里获取情感向量
        s_e = []

        # words = enc_strs[0].split(' ')
        # for word in words:
        #     # print(word)
        #     if word in senti_list:
        #         s_e.append(1)
        #     else:
        #         s_e.append(0)

        # batch_size
        for i in range(0, config_word2vec.batch_size):
            s_e_tmp = []
            words = enc_strs[i].split(' ')
            for word in words:
                if word in senti_list:
                    s_e_tmp.append(1)
                else:
                    s_e_tmp.append(0)
            s_e.append(s_e_tmp)
        s_e = np.array(s_e)
        s_e = T.Tensor(s_e)
        # print("情感向量的大小")
        # print(s_e.size())
        # print(enc_batch.size())    #

        enc_batch = self.model.embeds(
            enc_batch)  #Get embeddings for encoder input
        enc_out, enc_hidden = self.model.encoder(enc_batch, enc_lens)
        # print("*******enc***********")
        # print(enc_out)
        # print(len(enc_out))
        # -------------------------------Summarization-----------------------
        if self.opt.train_mle == "yes":  #perform MLE training
            # print("extra_zeros:" + extra_zeros)
            mle_loss = self.train_batch_MLE(s_e, enc_out, enc_hidden,
                                            enc_padding_mask, context,
                                            extra_zeros,
                                            enc_batch_extend_vocab, batch)
            # print("******mle_Loss finished********")
            # print(mle_loss)    # train_mle没有问题
        else:
            mle_loss = get_cuda(T.FloatTensor([0]))
        # --------------RL training-----------------------------------------------------
        if self.opt.train_rl == "yes":  #perform reinforcement learning training
            # multinomial sampling
            # print('*********oovs*************')
            # print(batch.art_oovs)
            sample_sents, RL_log_probs = self.train_batch_RL(
                enc_out,
                enc_hidden,
                enc_padding_mask,
                context,
                extra_zeros,
                enc_batch_extend_vocab,
                batch.art_oovs,
                greedy=False)
            with T.autograd.no_grad():
                # greedy sampling
                greedy_sents, _ = self.train_batch_RL(enc_out,
                                                      enc_hidden,
                                                      enc_padding_mask,
                                                      context,
                                                      extra_zeros,
                                                      enc_batch_extend_vocab,
                                                      batch.art_oovs,
                                                      greedy=True)

            sample_reward = self.reward_function(sample_sents,
                                                 batch.original_abstracts)
            baseline_reward = self.reward_function(greedy_sents,
                                                   batch.original_abstracts)
            # if iter%200 == 0:
            #     self.write_to_file(sample_sents, greedy_sents, batch.original_abstracts, sample_reward, baseline_reward, iter)
            rl_loss = -(
                sample_reward - baseline_reward
            ) * RL_log_probs  #Self-critic policy gradient training (eq 15 in https://arxiv.org/pdf/1705.04304.pdf)
            # print("rl_loss")
            # print(rl_loss)
            rl_loss = T.mean(rl_loss)

            batch_reward = T.mean(sample_reward).item()
        else:
            rl_loss = get_cuda(T.FloatTensor([0]))
            batch_reward = 0

    # ------------------------------------------------------------------------------------
        self.trainer.zero_grad()
        rl_weight = (iter - int(opt.load_model.split('.')[0])) / config_word2vec.iteration_add
        mle_weight = 1 - rl_weight
        # print(mle_weight, rl_weight)z
        (mle_weight * mle_loss + rl_weight * rl_loss).backward()
        self.trainer.step()
        # 学习率衰减
        # self.scheduler.step()
        return mle_loss.item(), rl_loss.item(), batch_reward, mle_weight, rl_weight

    def trainIters(self):
        iter = self.setup_train()
        count = mle_total = r_total = rl_total = 0
        while iter <= config_word2vec.max_iterations:
            batch = self.batcher.next_batch()
            try:
                mle_loss, rl_loss, r, mle_weight, rl_weight = self.train_one_batch(batch, iter)
            except KeyboardInterrupt:
                print(
                    "-------------------Keyboard Interrupt------------------")
                exit(0)

            mle_total += mle_loss
            rl_total += rl_loss
            r_total += r
            count += 1
            iter += 1

            if iter % 50 == 0:    #if iter % 50 == 0
                mle_avg = mle_total / count
                rl_avg = rl_total / count
                r_avg = r_total / count
                logger.info("iter:" + str(iter) + "  mle_loss:" + "%.3f" % mle_avg + "  rl_loss:" + "%.3f" % rl_avg + "  reward:" + "%.4f" % r_avg)
                logger.info("mle_weight:" + "%.3f" % mle_weight + " rl_weight:"+ "%.3f" % rl_weight)
                count = mle_total = r_total = rl_total = 0

            if iter % 100== 0:    #if iter % 5000 == 0
                self.save_model(iter)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_mle', type=str, default="yes")
    parser.add_argument('--train_rl', type=str, default="no")
    parser.add_argument('--mle_weight', type=float, default=1.0)
    parser.add_argument('--sentiment_weight', type=float, default=0.0)
    parser.add_argument('--load_model', type=str, default=None)
    parser.add_argument('--new_lr', type=float, default=None)
    opt = parser.parse_args()
    opt.rl_weight = 1 - opt.mle_weight
    opt.normal_weight = 1 - opt.sentiment_weight
    print(
        "Training mle: %s, Training rl: %s, mle weight: %.2f, rl weight: %.2f"
        % (opt.train_mle, opt.train_rl, opt.mle_weight, opt.rl_weight))
    print("intra_encoder:", config_word2vec.intra_encoder, "intra_decoder:",
          config_word2vec.intra_decoder)

    train_processor = Train(opt)
    train_processor.trainIters()
