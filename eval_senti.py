import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import time
from data_util.log import logger
import torch as T
import rouge
from model_pretrained_senti import Model
from data_util import config_word2vec, data_word2vec
from data_util.batcher_word2vec import Batcher, Example, Batch
from data_util.data_word2vec import Vocab
from beam_search_senti import beam_search
from train_util import get_enc_data
from rouge import Rouge
import argparse
import jieba
import numpy as np

# jieba.load_userdict("userdict.txt")


def get_cuda(tensor):
    if T.cuda.is_available():
        tensor = tensor.cuda()
    return tensor


class Evaluate(object):
    def __init__(self, data_path, opt, batch_size=config_word2vec.batch_size):
        self.vocab = Vocab(config_word2vec.vocab_path, config_word2vec.vocab_size)
        self.batcher = Batcher(data_path,
                               self.vocab,
                               mode='eval',
                               batch_size=batch_size,
                               single_pass=True)
        self.opt = opt
        time.sleep(5)

    def setup_valid(self):
        self.model = Model()
        self.model = get_cuda(self.model)
        if config_word2vec.cuda:
            checkpoint = T.load(os.path.join(config_word2vec.save_model_path, self.opt.load_model))
        else:
            checkpoint = T.load(os.path.join(config_word2vec.save_model_path, self.opt.load_model), map_location='cpu')
        self.model.load_state_dict(checkpoint["model_dict"])

    def print_original_predicted(self, decoded_sents, ref_sents, article_sents,
                                 loadfile):
        filename = "test_" + loadfile.split(".")[0] + ".txt"

        with open(os.path.join("CLTS/eval/table9/SCE_0.0001_8+MRL0.9984_0.0001_2/", filename), "w") as f:
            for i in range(len(decoded_sents)):
                f.write("article: " + article_sents[i] + "\n")
                f.write("ref: " + ref_sents[i] + "\n")
                f.write("dec: " + decoded_sents[i] + "\n\n")

    def evaluate_batch(self, article):

        self.setup_valid()
        batch = self.batcher.next_batch()
        start_id = self.vocab.word2id(data_word2vec.START_DECODING)
        end_id = self.vocab.word2id(data_word2vec.STOP_DECODING)
        unk_id = self.vocab.word2id(data_word2vec.UNKNOWN_TOKEN)
        decoded_sents = []
        ref_sents = []
        article_sents = []
        rouge = Rouge()
        while batch is not None:
            enc_batch, enc_lens, enc_padding_mask, enc_batch_extend_vocab, extra_zeros, ct_e = get_enc_data(
                batch)
            with T.autograd.no_grad():
                enc_batch = self.model.embeds(enc_batch)
                enc_out, enc_hidden = self.model.encoder(enc_batch, enc_lens)

            # 修改sentiment后的部分

            # 这里获取情感向量
            article = batch.original_articles
            artile = article[0].strip("'").split(' ')[0:config_word2vec.max_enc_steps]
            # print("********")
            # print(article)
            # 处理情感词
            s_tensor = []
            s_e_pos = np.random.rand(1200)
            s_e_neg = np.random.rand(1200)
            # 读取情感词典，构建情感列表senti_list
            senti_dict = open('union.txt', 'r', encoding='utf-8-sig')
            senti_list = []
            for line in senti_dict.readlines():
                line = line.strip('\n')
                senti_list.append(line)

            for token in artile:
                if token in senti_list:
                    s_tensor.append(s_e_pos)
                else:
                    s_tensor.append(s_e_neg)
            s_tensor = T.from_numpy(np.array(s_tensor))


            #-----------------------Summarization----------------------------------------------------
            with T.autograd.no_grad():
                pred_ids = beam_search(s_tensor, enc_hidden, enc_out, enc_padding_mask,
                                       ct_e, extra_zeros,
                                       enc_batch_extend_vocab, self.model,
                                       start_id, end_id, unk_id)

            for i in range(len(pred_ids)):
                decoded_words = data_word2vec.outputids2words(pred_ids[i], self.vocab,
                                                     batch.art_oovs[i])
                if len(decoded_words) < 2:
                    decoded_words = "xxx"
                else:
                    decoded_words = " ".join(decoded_words)
                decoded_sents.append(decoded_words)
                abstract = batch.original_abstracts[i]
                article = batch.original_articles[i]
                ref_sents.append(abstract)
                article_sents.append(article)

            batch = self.batcher.next_batch()

        load_file = self.opt.load_model

        if article:
            self.print_original_predicted(decoded_sents, ref_sents,
                                          article_sents, load_file)

        scores = rouge.get_scores(decoded_sents, ref_sents)
        rouge_1 = sum([x["rouge-1"]["f"] for x in scores]) / len(scores)
        rouge_2 = sum([x["rouge-2"]["f"] for x in scores]) / len(scores)
        rouge_l = sum([x["rouge-l"]["f"] for x in scores]) / len(scores)
        logger.info(load_file + " rouge_1:" + "%.4f" % rouge_1 + " rouge_2:" + "%.4f" % rouge_2 + " rouge_l:" + "%.4f" % rouge_l)


class Demo(Evaluate):
    def __init__(self, opt):
        self.vocab = Vocab(config_word2vec.demo_vocab_path, config_word2vec.demo_vocab_size)
        self.opt = opt
        self.setup_valid()

    def evaluate(self, article, ref):
        dec = self.abstract(article)
        scores = rouge.get_scores(dec, ref)
        rouge_1 = sum([x["rouge-1"]["f"] for x in scores]) / len(scores)
        rouge_2 = sum([x["rouge-2"]["f"] for x in scores]) / len(scores)
        rouge_l = sum([x["rouge-l"]["f"] for x in scores]) / len(scores)
        return {
            'dec': dec,
            'rouge_1': rouge_1,
            'rouge_2': rouge_2,
            'rouge_l': rouge_l
        }

    def abstract(self, article):
        start_id = self.vocab.word2id(data_word2vec.START_DECODING)
        end_id = self.vocab.word2id(data_word2vec.STOP_DECODING)
        unk_id = self.vocab.word2id(data_word2vec.UNKNOWN_TOKEN)
        example = Example(' '.join(jieba.cut(article)), '', self.vocab)
        batch = Batch([example], self.vocab, 1)
        enc_batch, enc_lens, enc_padding_mask, enc_batch_extend_vocab, extra_zeros, ct_e = get_enc_data(
            batch)
        with T.autograd.no_grad():
            enc_batch = self.model.embeds(enc_batch)
            enc_out, enc_hidden = self.model.encoder(enc_batch, enc_lens)
            pred_ids = beam_search(enc_hidden, enc_out, enc_padding_mask, ct_e,
                                   extra_zeros, enc_batch_extend_vocab,
                                   self.model, start_id, end_id, unk_id)

        for i in range(len(pred_ids)):
            decoded_words = data_word2vec.outputids2words(pred_ids[i], self.vocab,
                                                 batch.art_oovs[i])
            decoded_words = " ".join(decoded_words)
        # logger.info('get opinion here')
        return decoded_words


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--task",
                        type=str,
                        default="validate",
                        choices=["validate", "test", "demo"])
    parser.add_argument("--start_from", type=str, default="0005000.tar")
    # parser.add_argument("--load_model", type=str, default='0060000.tar')
    opt = parser.parse_args()

    if opt.task == "validate":
        saved_models = os.listdir(config_word2vec.save_model_path)
        saved_models.sort()
        file_idx = saved_models.index(opt.start_from)
        saved_models = saved_models[file_idx:]
        for f in saved_models:
            opt.load_model = f
            eval_processor = Evaluate(config_word2vec.valid_data_path, opt)
            eval_processor.evaluate_batch(False)
    elif opt.task == "test":
        eval_processor = Evaluate(config_word2vec.test_data_path, opt)
        eval_processor.evaluate_batch(True)
    else:
        demo_processor = Demo(opt)
        logger.info('get opinion here')
        logger.info(
            demo_processor.abstract(
                '随着美国解除对中兴通讯的禁令，中兴事件终于解决了，虽然付出了不小的代价，这个教训也很惨痛，但是中兴也迎来了重生的机会。可以说，这对很多人来说是一个好消息。但是，接下来国内的另外一家通信巨头——华为可能将会遭遇一个坏消息，因为澳大利亚或将正式发布禁令，封杀华为。中兴通讯正式解禁，今晨已全面恢复全球业务！ 凌晨，美国商务部发布公告称，中兴通讯已缴纳亿美元罚款，同时亿美元的保证金也已存入美国银行的托管账户，正式解除对其的禁售令。中兴将可重新向美国公司购买关键零部件，恢复运营。据媒体报道称，北京时间 点分，中兴通讯已正式启动了之前早已预备的业务重建计划，开始全面恢复全球业务。实际上，从昨晚开始，中兴员工就陆续来到工作岗位上。并且接到了恢复运营后的首个订单，来自中兴墨西哥的终端业务线。日早上，中兴通讯内部多处荧屏均打上了“解禁了！痛定思痛！再踏征程！”的标语。与此同时，中兴通讯今早在其官方微博也发布了一条新的微博称：满怀信心再出发！值得一提的是，日晚间，中兴通讯发布了年半年度业绩预告，预计归属于上市公司普通股股东的净亏损人民币亿元-亿元；上年同期盈利人民币约.亿元。而报告期业绩与上年同期相比下降幅度较大，主要原因为（被美国制裁期间）公司主要经营活动无法进行导致的经营损失、预提损失以及缴纳亿美元罚款所致。总的来说，中兴通讯此次美国禁运事件带来的教训不可谓不深，而且即便是恢复运营之后，中兴通讯头上仍悬着一把“达摩克利斯之剑”，即来自美国的合规“监管体系”，以及暂缓执行的新的年禁制令。希望中兴通讯能够痛定思痛，知耻而后勇！以国家安全为名，澳大利亚或将发出华为禁令。早在今年期间，华为手机与美国第二大移动运营商&;的合作，因为美国政府的叫停而失败。华为副总裁余承东在接受专访时也公开表示，”美国人利用政治禁止华为进入美国市场”。随后，美国又开始进一步限制限制华为、中兴公司的美国的通信业务，并以停发网络基建补贴，迫使美国小型、农村地区运营商弃用华为、中兴的电信设备。而随着美国老大针对华为的频频发难，作为美国小弟的澳大利亚政府也开始积极响应。澳大利亚频频向华为发难。今年月底，澳大利亚国防部发言人就表示，国防部以前采购过中兴和华为的产品，但现在决定将用其他生产商的产品取代它们。这位发言人说，以前购买的中国公司产品暂时还在使用，但过一段时间将被取消。随后在今年月，澳大利亚政府方面表示，它将调拨逾亿澳元的对外援助、用于建设一条通往所罗门群岛的电信光缆，该项目将实现所罗门群岛和巴布亚新几内亚与澳大利亚大陆的联通。而澳大利亚政府此举实际上可能也是出于对于华为的担忧。此前华为针对太平洋国家巴布亚新几内亚和所罗门群岛的电信光缆项目，提出的自行铺设光缆计划和修建光缆的合同已经得到了巴布亚新几内亚和所罗门群岛认可。但是作为澳大利亚的“后院”，澳大利亚处于对于华为的“安全”顾虑，所以决定拨款亿澳元对外援助，支付其中三分之二的资金，而该项目光缆建设项目将会交由澳大利亚电信企业沃卡斯集团。而随着澳大利亚准备宣布大规模部署移动通信网络，有关华为存在安全问题的担忧开始增多。澳大利亚有媒体报道称，该国情报机构建议不要把华为纳入设备供应商名单。与美国一样，澳大利亚国内同样有人担忧采用华为的电信设备可能危及到其国内安全。澳反对党工党议员迈克尔•丹比()就曾要求自由党-国家党(-)联合政府禁止华为和中兴供应用于网络的设备。一位知情人士也称，澳大利亚总理在今年月接到了来自美国国家安全局和国土安全部关于华为威胁的简报。月中旬，《澳大利亚金融评论》报道说，澳大利亚政府很快会宣布华为将被禁止参与网络竞标。澳大利亚司法部长随后拒绝确认或否认该报道，但表示正在全力评估网络竞标参与者，称任何涉及关键基础设施的合同都会考虑国家安全问题。 ，澳大利亚战略政策研究所 还发布研究报告称，根据政界人士披露的信息，年至年，华为为澳大利亚政界人士赴深圳华为总部的次行程买单，费用涉及商务舱机票、当地旅行、住宿与用餐。接受这些安排的就包括：一直对中国“指手画脚”、渲染中国威胁的“急先锋”——澳外长毕晓普 ，以及现任贸易部长乔博 与前贸易部长罗布 等。随后，众多澳大利亚媒体开始渲染“华为的赞助是在故意讨好政界人士。”华为在澳大利亚的发言人米切尔 当日回应称，公司没有任何不当行为。“我们公开邀请媒体、企业、智库以及政界人士来访问，以更好地了解我们。”华为仍在积极努力争取。针对澳大利亚政府声称的华为“构成安全风险”说法， ，华为向澳大利亚政府发出公开信，称这种观点“过于片面，没有以事实为基础”。并表示“我们已经进入了全球个国家，为大电信公司中的个供应设备。我们会遵守每一个国家的法律和准则。否则，我们的生意一夜之间就会结束。”华为澳大利亚高管约翰·洛德在信中还强调了该公司在英国、加拿大和新西兰的投资，称上述国家的政府已接受评估华为技术的提议，以确保该公司遵守网络安全协议。他还表示，华为已提议在澳大利亚修建一座评估和测试中心，“确保对华为的设备进行独立认证。”实际上，目前华为已经是澳大利亚网络设备的主要供应商，满足着澳大利亚全国超过%的需求。如果没有奥大利亚政府的阻挠，华为有望在澳大利亚网络建设当中取得竞争优势。目前，澳洲的电信公司巨头和都在与华为合作推出，只有澳洲电信使用爱立信 作为的供应商。为了减轻产品带来的国家安全担忧，华为近几个月来一直在卖力游说澳洲政府。华为还准备向澳政府提供对其网络设备的全面监管权。这种监管模式已经被其他国家尤其是英国接受。包括新西兰、加拿大和德国在内的其他西方国家也表示，它们有足够的保障措施。但是澳大利亚情报机构告诉立法者，这种监督并不会减轻他们的担忧。尽管面临着贸易压力，但是由于澳大利亚政府内部对华鹰派的崛起，很难推翻该国的安全部门的看法。北京时间 ，路透社报道，据两位消息人士称，澳大利亚正准备禁止华为公司为该国的网络提供设备，因为其情报机构担心中国政府可能会强迫这家中国通信设备巨头交出敏感数据。从目前的信息来看，澳大利亚政府仍有很大可能禁止华为参与其网络建设，而这或将对华为海外业务的开展造成重创。而且，早在年，华为就曾因安全担忧而被禁止向澳大利亚投资额达亿澳元的国家宽带网络()供应设备。不管结果如何，相信华为应该已经做好了应对的准备。在月份的华为全球分析师大会上，华为轮值董事长徐直军（对于进军美国市场受挫）就曾表示，“有些事情不是以我们的意志为转移的，与其无法左右，不如不理会。有些事情，放下了反而轻松。”在大环境难以左右的情况下，华为也只能如此。'                ))


