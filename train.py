# -- coding: utf-8 --
# author : TangQiang
# time   : 2023/8/8
# email  : tangqiang.0701@gmail.com
# file   : train.py

import os
import sys
import numpy as np
import pandas as pd
import torch

from model.deepb3p import DeepB3P
from utils.config_transformer import Config
from utils.amino_acid import fasta_to_pandas, fasta_to_numpy, pandas_to_numpy, AMINO_ACID_SIZE
from utils.utils import SeqDataset, set_seed
from validation import predict

os.environ["CUDA_VISIBLE_DEVICES"] = ""

# ======================================================================
# 1. 路径 & 读原始序列（和你原来 deep 逻辑完全一致）
# ======================================================================

train_pos_org = '/home/bio/old_deepb3p/deepB3P-main/il-17/pos_train.fasta'  # 定义 TNF-alpha 阳性训练集的文件路径。
train_pos_pseudo = '/home/bio/old_deepb3p/deepB3P-main/fbgan/out/df_all_final_300.csv'  # 将 FBGAN 生成的伪阳性数据路径设为 None，暂时不使用。
train_neg_org = '/home/bio/old_deepb3p/deepB3P-main/il-17/neg_train.fasta'  # 定义 TNF-alpha 阴性训练集的文件路径。
pos_test = '/home/bio/old_deepb3p/deepB3P-main/il-17/test/pos_test.fasta' # 定义 TNF-alpha 阳性测试集的文件路径。
neg_test = '/home/bio/old_deepb3p/deepB3P-main/il-17/test/neg_test.fasta'# 定义 TNF-alpha 阴性测试集的文件路径。

# 阳性：原始 + 伪阳性
train_pos_pseudo = pd.read_csv(train_pos_pseudo)
train_pos_org= fasta_to_pandas(train_pos_org)
pos_all = pd.concat([train_pos_org, train_pos_pseudo], ignore_index=True)

# 编码为 ordinal（ID 序列）
pos_train_fea, pos_train_label = pandas_to_numpy(pos_all, label=1, max_length=30)
neg_train_fea, neg_train_label = fasta_to_numpy(train_neg_org, label=0, max_length=30)

# 测试集（暂时用原来逻辑）
pos_test_fea, pos_test_label = fasta_to_numpy(pos_test)
neg_test_fea, neg_test_label = fasta_to_numpy(neg_test, label=0)

# 训练集 = 阳性(原始+伪) + 阴性
train_feas = np.concatenate((pos_train_fea, neg_train_fea), axis=0)   # (N, 30)
train_labels = np.concatenate((pos_train_label, neg_train_label), axis=0)

seq_len = 30  # 和 max_length 一致


# 读 AAindex 特征
aa_train = pd.read_csv(
    "/home/bio/old_deepb3p/deepB3P-main/feature/norm/AAindex_train3774_norm.csv",
    header=None
)   # (N, 30, 531)
#
# # 读 BLOSUM62 特征
blo_train = pd.read_csv(
    "/home/bio/old_deepb3p/deepB3P-main/feature/norm/BLOSUM62_train3774_norm.csv",
    header=None
)   # (N, 30, 23)
# esm2_train = pd.read_csv(
#     "/home/bio/old_deepb3p/deepB3P-main/feature/norm/esm2_train3774_norm.csv",
#     header=None
# )   # (N, 30, 23)
print("aa_train.shape =", aa_train.shape)
print("blo_train.shape =", blo_train.shape)
# print("esm2_train.shape =", esm2_train.shape)
# print("train_feas.shape =", train_feas.shape)
#
aa_train = aa_train.values
blo_train = blo_train.values
# esm2_train = esm2_train.values
#
aa_train_tensor = torch.tensor(aa_train, dtype=torch.float32)
blo_train_tensor = torch.tensor(blo_train, dtype=torch.float32)
# esm2_train_tensor = torch.tensor(esm2_train, dtype=torch.float32)
#
# # 调整为 (N, seq_len, feat_dim) 形状
aa_train_tensor = aa_train_tensor.reshape(-1, 30, 531) # (N, seq_len, feat_dim)
blo_train_tensor = blo_train_tensor.reshape(-1, 30, 23)
# esm2_train_tensor = esm2_train_tensor.reshape(3774, 30, 320)
#
# # feat_pseudo_aa = torch.zeros((1457, 30, 531))  # For pseudo AAindex features
# # feat_pseudo_blo = torch.zeros((1457, 30, 23))  # For pseudo BLOSUM62 features
# # feat_pseudo_esm2 = torch.zeros((1457, 30, 320))# For pseudo esm2 features
#
# # # Concatenate real and pseudo features for each type
# # aa_train_final = torch.cat([aa_train_tensor, feat_pseudo_aa], dim=0)  # Shape: (3774, 30, 531)
# # blo_train_final = torch.cat([blo_train_tensor, feat_pseudo_blo], dim=0)  # Shape: (3774, 30, 23)
# # esm2_train_final = torch.cat([esm2_train_tensor, feat_pseudo_esm2], dim=0)  # Shape: (3774, 30, 320)
# #
# # # Concatenate all the feature types (AAindex, BLOSUM62, esm2) together
# # extra_train = torch.cat([aa_train_final, blo_train_final, esm2_train_final], dim=2)  # Shape: (3774, 30, 531+23+320)
#
# # # Ensure alignment with the original 'train_feas' data
# # assert extra_train.shape[0] == train_feas.shape[0], "Number of samples in final_train doesn't match train_feas."
#
# # Verify the shape of the final feature tensor
#,esm2_train_tensor
# # 在最后一维拼接特征： (N, 30, 531+23)
print(aa_train_tensor.shape)  # 应该是 (N, 30, 531)
print(blo_train_tensor.shape)  # 应该是 (N, 30, 23)
# print(esm2_train_tensor.shape)  # 应该是 (N, 30, 23)
#
extra_train = np.concatenate([aa_train_tensor, blo_train_tensor], axis=2)
F_total = extra_train.shape[1]
print(extra_train.shape)  # Should be (3774, 30, 874)
#构建带有额外特征的 Dataset
train_dataset = SeqDataset(
    train_feas,     # (N, 30) ordinal 编码
    train_labels,   # (N,)
    extra_feats=extra_train#.numpy()
)
# train_dataset = SeqDataset(train_feas, train_labels)
# ======================================================================
# 3. 超参数 & Config
# ======================================================================

d_model = 64
d_ff = 64
d_k = 16
n_layers = 1
n_heads = 2
lr = 0.0001
drop = 0.3
bs = 16

results = open('res.txt', 'w')
results.write('lr\tdrop\tn_head\tn_layers\td_k\td_ff\td_model\tauc\tsn\tsp\tacc\tmcc\n')
results.close()

params = Config(
    d_model=d_model,
    d_ff=d_ff,
    d_k=d_k,
    n_layers=n_layers,
    n_heads=n_heads,
    lr=lr,
    drop=drop,
    seq_len=seq_len,
    bs=bs
)

#这里手动补充 deepB3P 需要的额外字段
params.feat_dim = extra_train.shape[2]
# params.feat_dim =           # 额外特征维度
params.vocab_size = AMINO_ACID_SIZE + 1 # 20 aa + PAD，一般是 21
params.d_v = d_k                   # 通常 d_v = d_k
params.max_length = seq_len        # 方便内部模块使用
#
# print(">>> vocab_size =", params.vocab_size)
# print(">>> feat_dim   =", params.feat_dim)
# print(">>> d_v        =", params.d_v)

params.make_dir()
logfile = params.model_file / 'deepb3p.log'
logger = params.set_logging(logfile)
set_seed(2025, logger)

logger.info('read pos train seqs: {0}'.format(len(pos_train_fea)))
logger.info('read neg train seqs: {0}'.format(len(neg_train_fea)))

# ======================================================================
# 4. 训练 + 验证
# ======================================================================

model = DeepB3P(params, logger)
model.cv_train(train_dataset, kFlod=params.kFold, earlyStop=params.earlyStop)

auc, sn, sp, acc, mcc = predict(params, logger)
with open('res.txt', 'a') as f:
    f.write(str(lr)+'\t'+str(drop)+'\t'+str(n_heads)+'\t'+
            str(n_layers)+'\t'+str(d_k)+'\t'+str(d_ff)+'\t'+
            str(d_model)+'\t')
    f.write(str(auc)+'\t'+str(sn)+'\t'+str(sp)+'\t'+str(acc)+'\t'+str(mcc)+'\n')

    # 假设你已经记录了训练和测试的损失
# import matplotlib.pyplot as plt
#
# plot_loss_curve(losses_train, losses_valid)
# 假设你已经记录了训练和测试的损失
# epochs = range(1, len(train_losses) + 1)
# plt.plot(epochs, train_losses, label="Training Loss")
# plt.plot(epochs, val_losses, label="Validation Loss")
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.title('Training vs Validation Loss')
# plt.show()


# #
# -- coding: utf-8 --
# author : TangQiang
# # time   : 2023/8/8
# # email  : tangqiang.0701@gmail.com
# # file   : train.py
# import pandas as pd
# from isort import Config
# import logging
#
#
# from model.deepb3p import DeepB3P
# from utils.config_transformer import *
# from utils.amino_acid import *
# from validation import predict
# """"
# train_pos_org = './data/train_pos_org.fasta'
# train_pos_pseudo = './data/df_all_final_1000.csv'
# train_neg_org = './data/neg_train.fasta'
# pos_test = './data/pos_test.fasta'
# neg_test = './data/neg_test.fasta'"""
# train_pos_org = '/home/bio/old_deepb3p/deepB3P-main/il-17/pos_train.fasta'  # 定义 TNF-alpha 阳性训练集的文件路径。
# train_pos_pseudo = '/home/bio/old_deepb3p/deepB3P-main/seq.csv'  # 将 FBGAN 生成的伪阳性数据路径设为 None，暂时不使用。
# train_neg_org = '/home/bio/old_deepb3p/deepB3P-main/il-17/neg_train.fasta'  # 定义 TNF-alpha 阴性训练集的文件路径。
# pos_test = '/home/bio/old_deepb3p/deepB3P-main/il-17/test/pos_test.fasta' # 定义 TNF-alpha 阳性测试集的文件路径。
# neg_test = '/home/bio/old_deepb3p/deepB3P-main/il-17/test/neg_test.fasta'# 定义 TNF-alpha 阴性测试集的文件路径。
#
# train_pos_pseudo = pd.read_csv(train_pos_pseudo)
# train_pos_org = fasta_to_pandas(train_pos_org)
# pos_all = pd.concat([train_pos_org, train_pos_pseudo])
#
#
# pos_train_fea, pos_train_label = pandas_to_numpy(pos_all, label=1,max_length=30)
# neg_train_fea, neg_train_label = fasta_to_numpy(train_neg_org, label=0,max_length=30)
# logging.info('read pos train seqs: {0}'.format(len(pos_train_fea)))
# logging.info('read neg train seqs: {0}'.format(len(neg_train_fea)))
#
# pos_test_fea, pos_test_label = fasta_to_numpy(pos_test)
# neg_test_fea, neg_test_label = fasta_to_numpy(neg_test, label=0)
# logging.info('read pos test seqs: {0}'.format(len(pos_test_fea)))
# logging.info('read neg test seqs: {0}'.format(len(neg_test_fea)))
#
# train_feas = np.concatenate((pos_train_fea, neg_train_fea), axis=0)
# train_labels = np.concatenate((pos_train_label, neg_train_label), axis=0)
#
#
# train_dataset = SeqDataset(train_feas, train_labels)
#
# # d_model=64, d_ff=16, d_k=32, n_layers=1, n_heads=2, lr=0.0001, drop=0.3
# d_model = 64
# d_ff = 64
# d_k = 16
# n_layers = 1
# n_heads = 2
# lr = 0.0001
# drop = 0.1
# seq_len=30
# bs=16
# results = open('res.txt', 'w')
# results.write('lr\tdrop\tn_head\tn_layers\td_k\td_ff\td_model\tauc\tsn\tsp\tacc\tmcc\n')
# results.close()
# params = Config(
#     d_model=d_model,
#     d_ff=d_ff,
#     d_k=d_k,
#     n_layers=n_layers,
#     n_heads=n_heads,
#     lr=lr,
#     drop=drop,
#     seq_len=seq_len,
#     bs=bs
# )
# params.make_dir()
# logfile = params.model_file / 'deepb3p.log'
# logger = params.set_logging(logfile)
# set_seed(2025, logger)
# logger.info('read pos train seqs: {0}'.format(len(pos_train_fea)))
# logger.info('read neg train seqs: {0}'.format(len(neg_train_fea)))
# model = DeepB3P(params, logger)
# model.cv_train(train_dataset, kFlod=params.kFold, earlyStop=params.earlyStop)
# auc, sn, sp, acc, mcc = predict(params, logger)
# with open('res.txt','a') as results:
#     results.write('\tauc\tsn\tsp\tacc\tmcc\tpre\tf1\tauprc\n')
#     results.write(
#         f"{auc}\t{sn}\t{sp}\t{acc}\t{mcc}\n"
#     )
#
# import sys
# # import os
# sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
