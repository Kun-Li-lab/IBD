# _*_coding:utf_8_
# @author:李坤
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os
import seaborn as sns
from utils import read_csv_table, read_csv_table_raw
import argparse
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.metrics import confusion_matrix,classification_report, roc_auc_score, accuracy_score
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
    features_df = os.path.join(args.data_root, "mic-met-grouped.csv")
    subject_ids, ids, fea = read_csv_table(features_df)
    print(ids)
    labels_df = os.path.join(args.data_root, "y.csv")
    subject_ids, label, label_fea = read_csv_table(labels_df)
    train_idx = []
    val_idx = []
    for i, row_name in enumerate(subject_ids):
        if "Validation" in subject_ids[i]:
            val_idx.append(i)
        else:
            train_idx.append(i)

    features = fea.astype(np.float64)
    features = torch.tensor(features, dtype=torch.float64)
    print(features.shape)
    labels = label_fea.astype(np.str_)
    labels = labels.tolist()
    label_mapping = {'CD': 2, 'UC': 1, 'Control': 0}
    numeric_labels = [label_mapping[str(label[0])] for label in labels]
    print("标签为%s" % numeric_labels)
    print("特征为%s" % features)

    y_train = []
    fea_train = []
    fea_test = []
    fea_train = features[train_idx]
    fea_test = features[val_idx]
    print(fea_train)
    y_train = [numeric_labels[i] for i in train_idx]
    print(y_train)
    y_test = [numeric_labels[i] for i in val_idx]
    print(len(fea_train))
    print(len(y_train))
    print(len(y_test))
    # 转换数据为PyTorch张量
    gbm = lgb.LGBMClassifier()
    gbm.fit(features, numeric_labels)
    lr = LogisticRegression()
    print("开始")
    accuracies = []
    #获取到建立的树
    model = gbm.booster_
    # 每个样本落在每个树的位置 ， 下面两个是矩阵  (样本个数, 树的棵树)  ， 每一个数字代表某个样本落在了某个数的哪个叶子节点
    gbdt_feats_train = model.predict(fea_train, pred_leaf=True)
    print(gbdt_feats_train.shape)
    gbdt_feats_test = model.predict(fea_test, pred_leaf=True)
    # 把上面的矩阵转成新的样本-特征的形式，与原有的数据集合并
    gbdt_feats_name = ['gbdt_leaf_' + str(i) for i in range(gbdt_feats_train.shape[1])]
    df_train_gbdt_feats = pd.DataFrame(gbdt_feats_train, columns=gbdt_feats_name)
    df_test_gbdt_feats = pd.DataFrame(gbdt_feats_test, columns=gbdt_feats_name)

    fea_train = pd.DataFrame(fea_train)
    fea_test = pd.DataFrame(fea_test)
    # 构造新数据集
    train = pd.concat([fea_train, df_train_gbdt_feats], axis=1)
    test = pd.concat([fea_test, df_test_gbdt_feats], axis=1)
    train_len = train.shape[0]
    print("train的长度", train_len)
    data = pd.concat([train, test])
    print('GBDT.data: ', data)
    x_train, x_val, y_train, y_val = train_test_split(train, y_train, test_size=0.2, random_state=2023)
    # lr.fit(df_train_gbdt_feats, y_train)
    #y_pred = lr.predict(df_test_gbdt_feats)
    #accuracy = accuracy_score(y_test, y_pred)
    #print("准确率:", accuracy)

    lr.fit(x_train, y_train)
    y_pred = lr.predict(test)
    print(y_pred)
    XGboost_measure_result = classification_report(y_test, y_pred)
    print('XGboost:measure_result = \n', XGboost_measure_result)
    accuracy = accuracy_score(y_test, y_pred)
    print("准确率:", accuracy)
    # 计算混淆矩阵
    cm = confusion_matrix(y_test, y_pred)

    # 打印混淆矩阵
    print("Confusion Matrix:")
    print(cm)

    # 利用seaborn绘制热力图
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1, 2], yticklabels=[0, 1, 2])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()
    print(gbm.predict_proba)
    features = fea.astype(np.float64)
    features = torch.tensor(features, dtype=torch.float64)
    fea_train = features[train_idx]
    fea_test = features[val_idx]
    class_names = ['CD', 'UC', 'Control']
    explainer = lime_tabular.LimeTabularExplainer(training_data=np.array(fea_train), # 训练集特征，必须是 numpy 的 Array
    feature_names=ids, # 特征列名
    class_names=['CD','UC','Control'], # 预测类别名称
    mode='classification' )
    exp = explainer.explain_instance(np.array(fea_train[5]),predict_fn=gbm.predict_proba,num_features=10)
    feature_importances = exp.as_list()
    print(feature_importances)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default='.\data\BioHealth\IBD\datas')
    parser.add_argument("--dataset_name", type=str, default='ibd')
    parser.add_argument("--data_type", type=str, default='clr', help="clr: log(x) - mean(log(x)), 0-1: log (x/sum(x)))")
    args = parser.parse_args()

    main(args)
