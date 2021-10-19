import torch

train_data_path = "CLTS/chunked/train/train_*"     # 读取训练数据，每次换数据集训练时需要更改
valid_data_path = "CLTS/chunked/valid/valid_*"    # 读取验证集数据，每次换数据集训练时需要更改
test_data_path = "CLTS/chunked/test/test_*"    # 读取测试集数据，每次换数据集训练时需要更改
vocab_path = "vocab_CLTS"
# demo_vocab_path = "fire_zx_124/vocab"
# demo_vocab_size = 40000

# Hyperparameterss
hidden_dim = 600   #512
emb_dim = 300    #256
batch_size = 8   #batch_size = 10
max_enc_steps = 800  #99% of the articles are within length 55 zh-800；symantec=600
max_dec_steps = 100  #99% of the titles are within length 1s5 zh-100；symantec=80
beam_size = 4
min_dec_steps = 3    #生成句子的最短长度
vocab_size = 50000

lr = 0.0001    #学习率 出现NaN时意味着出现了梯度爆炸问题，尝试减小学习率 初始lr = 0.001
rand_unif_init_mag = 0.02
trunc_norm_init_std = 1e-4

eps = 1e-12
max_iterations = 56700  #max_iterations = 5000000，最大迭代次数，每次训练时视情况修改
#iteration_add = 10000
save_model_path = "CLTS/table9/0.75/SCE_0.0001_8+MRL0.9984_0.0001_2/"    #训练好的模型存放的路径，每次训练时需要修改，建议不要跟之前的混到一起
# demo_model_path = "fire_zx_124/demo_models"

intra_encoder = True
intra_decoder = True

# cuda = False
cuda = torch.device('cuda')    #使用gpu
