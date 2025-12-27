# -- coding: utf-8 --
# author : TangQiang
# time   : 2023/8/9
# email  : tangqiang.0701@gmail.com
# file   : validation.py

from model.deepb3p import DeepB3P
from utils.utils import *
from utils.amino_acid import *
from sklearn.metrics import roc_curve, auc

def get_models(params, logger):
    model_list = []
    for i in range(1, params.kFold+1):
        model_file = params.model_file / f'deepb3p_{i}.pth'
        model = DeepB3P(params, logger)
        model.model.reset_parameters()
        model.load_model(directory=model_file)
        model_list.append(model)
    return model_list


def cv_predict(params, dataloader, logger):
    model_list = get_models(params, logger)
    y_prob_df = pd.DataFrame(columns=[1, 2, 3, 4, 5])
    y_pred_df = pd.DataFrame(columns=[1, 2, 3, 4, 5])
    auc_list, sn_list, sp_list, acc_list, mcc_list = [], [], [], [], []
    for idx, model in enumerate(model_list):
        ys_train, loss_list, metrics_train, time_epoch = model.valid_epoch(dataloader)
        enc_att, dec_att, dec_enc_att = model.model.get_atts()
        y_true_list, y_pred_list, y_prob_list = ys_train
        y_prob_df[idx+1] = y_prob_list
        y_pred_df[idx+1] = y_pred_list
        auc, sn, sp, acc, mcc = metrics_train
        auc_list.append(auc)
        sn_list.append(sn)
        sp_list.append(sp)
        acc_list.append(acc)
        mcc_list.append(mcc)
    return auc_list, sn_list, sp_list, acc_list, mcc_list, y_prob_df, y_pred_df, y_true_list

#create--计算不同阈值下的AUC或MCC
def optimize_threshold(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    best_threshold = thresholds[np.argmax(tpr - fpr)]  # 选择最佳阈值，最大化Youden指数（tpr-fpr）
    return best_threshold


def predict(params, logger):
    pos_test = '/home/bio/old_deepb3p/deepB3P-main/il-17/test/pos_test.fasta'  # 定义 TNF-alpha 阳性测试集的文件路径。
    neg_test = '/home/bio/old_deepb3p/deepB3P-main/il-17/test/neg_test.fasta'  # 定义 TNF-alpha 阴性测试集的文件路径。
    pos_test_fea, pos_test_label = fasta_to_numpy(pos_test)
    neg_test_fea, neg_test_label = fasta_to_numpy(neg_test, label=0)
    logger.info('read pos test seqs: {0}'.format(len(pos_test_fea)))
    logger.info('read neg test seqs: {0}'.format(len(neg_test_fea)))

    test_feas = np.concatenate((pos_test_fea, neg_test_fea), axis=0)
    test_labels = np.concatenate((pos_test_label, neg_test_label), axis=0)



    seq_len = params.seq_len  # 一般是 30

   #  #读测试集外部特征
    aa_test = pd.read_csv(
        "/home/bio/old_deepb3p/deepB3P-main/feature/AAindex_test.csv",
        header=None
    )  # (N, 30, 531)

   #  # 读 BLOSUM62 特征
    blo_test = pd.read_csv(
        "/home/bio/old_deepb3p/deepB3P-main/feature/norm/BLOSUM62_test_norm.csv",
        header=None
    )  # (N, 30, 23)
   #  esm2_test = pd.read_csv(
   #      "/home/bio/old_deepb3p/deepB3P-main/feature/norm/esm2_test_norm.csv",
   #      header=None
   #  )  # (N, 30, 320)
    print("aa_test.shape =", aa_test.shape)
    print("blo_test.shape =", blo_test.shape)
   #  print("esm2_test.shape =", esm2_test.shape)
   #  print("test_feas.shape =", test_feas.shape)
   #
    aa_test = aa_test.values
    blo_test = blo_test.values
   #  esm2_test = esm2_test.values
   #
    aa_test_tensor = torch.tensor(aa_test, dtype=torch.float32)
    blo_test_tensor = torch.tensor(blo_test, dtype=torch.float32)
   #  esm2_test_tensor = torch.tensor(esm2_test, dtype=torch.float32)
   #
   #  # 调整为 (N, seq_len, feat_dim) 形状
    aa_test_tensor = aa_test_tensor.reshape(580, 30, 531)  # (N, seq_len, feat_dim)
    blo_test_tensor = blo_test_tensor.reshape(580, 30, 23)
   #  esm2_test_tensor = esm2_test_tensor.reshape(580, 30, 320)
   #
    assert aa_test.shape[0] == test_feas.shape[0]
    assert blo_test.shape[0] == test_feas.shape[0]
   #  assert esm2_test.shape[0] == test_feas.shape[0]
   #  # 在最后一维拼接特征： (N, 30, 531+23)
    print(aa_test_tensor.shape)  # 应该是 (N, 30, 531)
    print(blo_test_tensor.shape)  # 应该是 (N, 30, 23)
   #  print(esm2_test_tensor.shape)  # 应该是 (N, 30, 23)
   # #, esm2_test_tensor
    extra_test = np.concatenate([aa_test_tensor,blo_test_tensor], axis=2)
    F_total = extra_test.shape[0]

    test_dataset = SeqDataset(test_feas, test_labels, extra_feats=extra_test)#
    dataloader = DataLoader(test_dataset, batch_size=len(test_labels))
    (
        auc_list, sn_list, sp_list, acc_list, mcc_list,
        # pre_list, f1_list, auprc_list,
        prob_df, pred_df, y_true) = cv_predict(params, dataloader, logger)

    # ---------------------- 4. CV 平均五折指标 -------------------------
    auc = np.mean(auc_list)
    sn = np.mean(sn_list)
    sp = np.mean(sp_list)
    acc = np.mean(acc_list)
    mcc = np.mean(mcc_list)
    # precision = np.mean(pre_list)
    # f1 = np.mean(f1_list)
    # auprc = np.mean(auprc_list)

    logger.info('Average result for {}-fold'.format(params.kFold))
    row_first = ['Average', 'auc', 'sn', 'sp', 'acc', 'mcc']#, 'pre', 'f1', 'auprc'
    logger.info(''.join(f'{item:<12}' for item in row_first))
    logger.info(
        f"{'Average1':<12}"
        f"{auc:<12.3f}{sn:<12.3f}{sp:<12.3f}{acc:<12.3f}{mcc:<12.3f}"
        # f"{precision:<12.3f}{f1:<12.3f}{auprc:<12.3f}"
    )

    # ---------------------- 5. 平均概率方式  -------------------------
    avg_prob = prob_df.mean(axis=1)
    avg_prob_pred = transfer(avg_prob, 0.5)

    metrics_train_avg = cal_performance(y_true, avg_prob_pred, avg_prob, logger, logging_=True)

    # ---------------------- 6. Vote 模式 -------------------------
    vote_pred = pred_df.mean(axis=1)
    vote_pred = transfer(vote_pred, 0.5)

    tn, fp, fn, tp = metrics.confusion_matrix(y_true, vote_pred, labels=[0, 1]).ravel().tolist()
    acc_v = metrics.accuracy_score(y_true, vote_pred)
    mcc_v = metrics.matthews_corrcoef(y_true, vote_pred)
    sn_v = tp / (tp + fn)
    sp_v = tn / (tn + fp)

    vote_row = ['vote', 'sn', 'sp', 'acc', 'mcc']
    metrics_vote = [sn_v, sp_v, acc_v, mcc_v]

    logger.info(''.join(f'{item:<12}' for item in vote_row))
    logger.info(f"{'vote':<12}{sn_v:<12.3f}{sp_v:<12.3f}{acc_v:<12.3f}{mcc_v:<12.3f}")

    # ---------------------- 7. 提取 cal_performance 的 8 指标 -------------------------
    auc_f, sn_f, sp_f, acc_f, mcc_f = metrics_train_avg
  #  , pre_f, f1_f, auprc_f
    # ---------------------- 8. 返回八指标 -------------------------
    return auc_f, sn_f, sp_f, acc_f, mcc_f


    test_dataset = SeqDataset(test_feas, test_labels)
    dataloader = DataLoader(test_dataset, batch_size=len(test_labels))
    auc_list, sn_list, sp_list, acc_list, mcc_list, prob_df, pred_df, y_true = cv_predict(params, dataloader, logger)
    auc = sum(auc_list)/len(auc_list)
    sn = sum(sn_list) / len(sn_list)
    sp = sum(sp_list) / len(sp_list)
    acc = sum(acc_list) / len(acc_list)
    mcc = sum(mcc_list) / len(mcc_list)


    th_file = params.model_file / 'val_thresholds.json'
    if th_file.exists():
        with open(th_file, 'r') as f:
            th_info = json.load(f)
        best_threshold = float(th_info.get('avg_th', 0.5))
        logger.info(f'Use avg_th from val: {best_threshold:.4f}')
    else:
        # 找不到的话就退回 0.5，或者你自己指定一个
        best_threshold = 0.5
        logger.warning('val_thresholds.json not found, fallback to 0.5')

    logger.info('Average result for {}-fold'.format(params.kFold))
    row_first = ['Average', 'auc', 'sn', 'sp', 'acc', 'mcc']
    metrics_list = [auc, sn, sp, acc, mcc]
    logger.info(''.join(f'{item:<12}' for item in row_first))
    logger.info(f'%-12s' % 'Average1' + ''.join(f'{key:<12.3f}' for key in metrics_list))

    avg_prob = prob_df.mean(axis=1)
    avg_prob_pred = transfer(avg_prob, 0.5)
    metrics_train_avg = cal_performance(y_true, avg_prob_pred, avg_prob, logger, logging_=True)
    logger.info(''.join(f'{item:<12}' for item in row_first))
    logger.info(f'%-12s' % 'Average' + ''.join(f'{key:<12.3f}' for key in metrics_train_avg))

    vote_pred = pred_df.mean(axis=1)
    vote_pred = transfer(vote_pred, 0.5)

    tn, fp, fn, tp = metrics.confusion_matrix(y_true, vote_pred, labels=[0, 1]).ravel().tolist()
    acc = metrics.accuracy_score(y_true, vote_pred)
    mcc = metrics.matthews_corrcoef(y_true, vote_pred)
    sn = tp / (tp + fn)
    sp = tn / (tn + fp)
    vote_row = ['vote', 'sn', 'sp', 'acc', 'mcc']
    metrics_train = [sn, sp, acc, mcc]
    logger.info(''.join(f'{item:<12}' for item in vote_row))
    logger.info(f'%-12s' % 'vote' + ''.join(f'{key:<12.3f}' for key in metrics_train))
    auc, sn, sp, acc, mcc = metrics_train_avg
    return auc, sn, sp, acc, mcc




# # -- coding: utf-8 --
# # author : TangQiang
# # file   : validation.py
#
# from model.deepb3p import DeepB3P
# from utils.utils import *
# from utils.amino_acid import *
# from sklearn.metrics import precision_score, f1_score, average_precision_score
#
#
# def get_models(params, logger):
#     model_list = []
#     for i in range(1, params.kFold + 1):
#         model_file = params.model_file / f'deepb3p_{i}.pth'
#         model = DeepB3P(params, logger)
#         model.model.reset_parameters()
#         model.load_model(directory=model_file)
#         model_list.append(model)
#     return model_list
#
#
# def cv_predict(params, dataloader, logger):
#     model_list = get_models(params, logger)
#     y_prob_df = pd.DataFrame(columns=[1, 2, 3, 4, 5])
#     y_pred_df = pd.DataFrame(columns=[1, 2, 3, 4, 5])
#
#     auc_list, sn_list, sp_list, acc_list, mcc_list = [], [], [], [], []
#     pre_list, f1_list, auprc_list = [], [], []
#
#     for idx, model in enumerate(model_list):
#         ys_train, loss_list, metrics_train, time_epoch = model.valid_epoch(dataloader)
#         y_true_list, y_pred_list, y_prob_list = ys_train
#
#         y_prob_df[idx + 1] = y_prob_list
#         y_pred_df[idx + 1] = y_pred_list
#
#         # 8 指标解析
#         auc, sn, sp, acc, mcc = metrics_train
#
#         auc_list.append(auc)
#         sn_list.append(sn)
#         sp_list.append(sp)
#         acc_list.append(acc)
#         mcc_list.append(mcc)
#         # pre_list.append(precision)
#         # f1_list.append(f1)
#         # auprc_list.append(auprc)
#
#     return (
#         auc_list, sn_list, sp_list, acc_list, mcc_list,
#         # pre_list, f1_list, auprc_list,
#         y_prob_df, y_pred_df, y_true_list
#     )
#
#
# def predict(params, logger):
#
#     # ---------------------- 1. 读取测试集 -------------------------
#     pos_test = '/home/bio/old_deepb3p/deepB3P-main/il-17/test/pos_test.fasta'  # 定义 TNF-alpha 阳性测试集的文件路径。
#     neg_test = '/home/bio/old_deepb3p/deepB3P-main/il-17/test/neg_test.fasta'  # 定义 TNF-alpha 阴性测试集的文件路径。
#
#     pos_test_fea, pos_test_label = fasta_to_numpy(pos_test)
#     neg_test_fea, neg_test_label = fasta_to_numpy(neg_test, label=0)
#
#     logger.info(f"read pos test seqs: {len(pos_test_fea)}")
#     logger.info(f"read neg test seqs: {len(neg_test_fea)}")
#
#     test_feas = np.concatenate((pos_test_fea, neg_test_fea), axis=0)
#     test_labels = np.concatenate((pos_test_label, neg_test_label),axis=0)
#      # ---------------------- 2. 不加特征 -------------------------
#     test_dataset = SeqDataset(test_feas, test_labels)
#     dataloader = DataLoader(test_dataset, batch_size=len(test_labels))
#
#     # ---------------------- 3. CV 预测 -------------------------
#     (
#         auc_list, sn_list, sp_list, acc_list, mcc_list,
#         # pre_list, f1_list, auprc_list,
#         prob_df, pred_df, y_true
#     ) = cv_predict(params, dataloader, logger)
#
#     # ---------------------- 4. CV 平均五折指标 -------------------------
#     auc = np.mean(auc_list)
#     sn = np.mean(sn_list)
#     sp = np.mean(sp_list)
#     acc = np.mean(acc_list)
#     mcc = np.mean(mcc_list)
#     # precision = np.mean(pre_list)
#     # f1 = np.mean(f1_list)
#     # auprc = np.mean(auprc_list)
#
#     logger.info('Average result for {}-fold'.format(params.kFold))
#     row_first = ['Average', 'auc', 'sn', 'sp', 'acc', 'mcc']
#     logger.info(''.join(f'{item:<12}' for item in row_first))
#     logger.info(
#         f"{'Average1':<12}"
#         f"{auc:<12.3f}{sn:<12.3f}{sp:<12.3f}{acc:<12.3f}{mcc:<12.3f}"
#         # f"{precision:<12.3f}{f1:<12.3f}{auprc:<12.3f}"
#     )
#
#     # ---------------------- 5. 平均概率方式  -------------------------
#     avg_prob = prob_df.mean(axis=1)
#     avg_prob_pred = transfer(avg_prob, 0.5)
#
#     metrics_train_avg = cal_performance(y_true, avg_prob_pred, avg_prob, logger, logging_=True)
#
#     # ---------------------- 6. Vote 模式 -------------------------
#     vote_pred = pred_df.mean(axis=1)
#     vote_pred = transfer(vote_pred, 0.5)
#
#     tn, fp, fn, tp = metrics.confusion_matrix(y_true, vote_pred, labels=[0, 1]).ravel().tolist()
#     acc_v = metrics.accuracy_score(y_true, vote_pred)
#     mcc_v = metrics.matthews_corrcoef(y_true, vote_pred)
#     sn_v = tp / (tp + fn)
#     sp_v = tn / (tn + fp)
#
#     vote_row = ['vote', 'sn', 'sp', 'acc', 'mcc']
#     metrics_vote = [sn_v, sp_v, acc_v, mcc_v]
#
#     logger.info(''.join(f'{item:<12}' for item in vote_row))
#     logger.info(f"{'vote':<12}{sn_v:<12.3f}{sp_v:<12.3f}{acc_v:<12.3f}{mcc_v:<12.3f}")
#
#     # ---------------------- 7. 提取 cal_performance 的 8 指标 -------------------------
#     auc_f, sn_f, sp_f, acc_f, mcc_f = metrics_train_avg
#
#     # ---------------------- 8. 返回八指标 -------------------------
#     return auc_f, sn_f, sp_f, acc_f, mcc_f
