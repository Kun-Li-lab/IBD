# _*_coding:utf_8_
# @author:李坤
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os
from utils import read_csv_table, read_csv_table_raw
import argparse
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from lime import lime_tabular
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from scipy import stats
from boruta import BorutaPy
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from IPython.display import (display, display_html, display_png, display_svg)
def main(args):
    # 读取特征和分类结果数据
    features_df = os.path.join(args.data_root, "ibd-x1-met-zComp-clr.csv")
    subject_ids, ids, met_fea = read_csv_table(features_df)
    print(ids)
    labels_df = os.path.join(args.data_root, "ibd-x2-mic-zComp-clr.csv")
    subject_ids, label, mic_fea = read_csv_table(labels_df)
    train_idx = []
    val_idx = []
    for i, row_name in enumerate(subject_ids):
        if "Validation" in subject_ids[i]:
            val_idx.append(i)
        else:
            train_idx.append(i)

    metfeatures = met_fea.astype(np.float64)
    metfeatures = torch.tensor(metfeatures, dtype=torch.float64)
    print(metfeatures.shape)
    micfeatures = mic_fea.astype(np.float64)
    micfeatures = torch.tensor(micfeatures, dtype=torch.float64)
    print(micfeatures.shape)

    y_train = []
    fea_train = []
    fea_test = []
    metfea_train = metfeatures[train_idx]
    metfea_test = metfeatures[val_idx]
    print(metfea_train)
    micfea_train =  micfeatures[train_idx]
    micfea_test =  micfeatures[val_idx]
    print( micfea_train)
    print(len(micfea_train))
    print(len(metfea_train))
    # 转换数据为PyTorch张量
    gbm = lgb.LGBMRegressor()
    gbm.fit(micfeatures.T.reshape(-1), metfeatures.T.reshape(-1))
    lr = LogisticRegression()
    print("开始")
    accuracies = []
    #获取到建立的树
    model = gbm.booster_
    # 每个样本落在每个树的位置 ， 下面两个是矩阵  (样本个数, 树的棵树)  ， 每一个数字代表某个样本落在了某个数的哪个叶子节点
    gbdt_feats_train = model.predict(micfea_train, pred_leaf=True)
    print(gbdt_feats_train.shape)
    gbdt_feats_test = model.predict(micfea_test, pred_leaf=True)
    # 把上面的矩阵转成新的样本-特征的形式， 与原有的数据集合并
    gbdt_feats_name = ['gbdt_leaf_' + str(i) for i in range(gbdt_feats_train.shape[1])]
    df_train_gbdt_feats = pd.DataFrame(gbdt_feats_train, columns=gbdt_feats_name)
    df_test_gbdt_feats = pd.DataFrame(gbdt_feats_test, columns=gbdt_feats_name)

    micfea_train = pd.DataFrame(micfea_train)
    micfea_test = pd.DataFrame(micfea_test)
    # 构造新数据集
    mictrain = pd.concat([micfea_train, df_train_gbdt_feats], axis=1)
    mictest = pd.concat([micfea_test, df_test_gbdt_feats], axis=1)
    mictrain_len = mictrain.shape[0]
    print("train的长度", mictrain_len)
    data = pd.concat([mictrain, mictest])
    print('GBDT.data: ', data)

    x_train, x_val, y_train, y_val = train_test_split(mictrain, metfea_train, test_size=0.2, random_state=2023)
    lr.fit(x_train, y_train)

    y_pred = lr.predict(mictest)
    XGboost_measure_result = classification_report(metfea_test, y_pred)
    print('XGboost:measure_result = \n', XGboost_measure_result)
    accuracy = accuracy_score(metfea_test, y_pred)
    print("准确率:", accuracy)
    #auc = roc_auc_score(y_test, y_pred)
    #print("AUC:", auc)
    print(gbm.predict_proba)
    micfeatures = micfea.astype(np.float64)
    micfeatures = torch.tensor(micfeatures, dtype=torch.float64)
    micfea_train = micfeatures[train_idx]
    micfea_test = micfeatures[val_idx]
    class_names = label
    explainer = lime_tabular.LimeTabularExplainer(training_data=np.array(micfea_train), # 训练集特征，必须是 numpy 的 Array
    feature_names=ids, # 特征列名
    class_names=label, # 预测类别名称
    mode='classification' )
    exp = explainer.explain_instance(np.array(micfea_train[5]),predict_fn=gbm.predict_proba,num_features=10)

    plt = exp.as_pyplot_figure()
    exp.show_in_notebook(show_table=True, show_all=False)
    plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default='.\data\BioHealth\IBD\datas')
    parser.add_argument("--dataset_name", type=str, default='ibd')
    parser.add_argument("--data_type", type=str, default='clr', help="clr: log(x) - mean(log(x)), 0-1: log (x/sum(x)))")
    args = parser.parse_args()

    main(args)
