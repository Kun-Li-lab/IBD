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
from sklearn.metrics import confusion_matrix,classification_report, roc_auc_score, accuracy_score,roc_curve,auc
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from lime import lime_tabular
import seaborn as sns
from lime import explanation
from lime.lime_tabular import LimeTabularExplainer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from scipy import stats
from boruta import BorutaPy
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import matplotlib.pyplot as plt

#  交叉验证、特征提取、集成

def main(args):
    # 读取特征和分类结果数据
    features_df = os.path.join(args.data_root, "mic.csv")
    subject_ids, ids, fea = read_csv_table(features_df)
    labels_df = os.path.join(args.data_root, "y.csv")
    subject_ids, label, label_fea = read_csv_table(labels_df)
    kfold = KFold(n_splits=5, shuffle=True)
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

    auc_accuracies = []
    accuracies = []

    y_train = [numeric_labels[i] for i in train_idx]
    print(y_train)
    y_test = [numeric_labels[i] for i in val_idx]
    print(len(fea_train))
    print(len(y_train))
    print(len(y_test))
    # 转换数据为PyTorch张量
    gbm = lgb.LGBMClassifier()
    lr = LogisticRegression()
    print("开始")

    gbm.fit(features, numeric_labels)
    #获取到建立的树
    model = gbm.booster_

    # 每个样本落在每个树的位置 ， 下面两个是矩阵  (样本个数, 树的棵树)  ， 每一个数字代表某个样本落在了某个数的哪个叶子节点
    gbdt_feats_train = model.predict(fea_train, pred_leaf=True)
    gbdt_feats_test = model.predict(fea_test, pred_leaf=True)
    # 把上面的矩阵转成新的样本-特征的形式， 与原有的数据集合并
    gbdt_feats_name = ['gbdt_leaf_' + str(i) for i in range(gbdt_feats_train.shape[1])]
    df_train_gbdt_feats = pd.DataFrame(gbdt_feats_train, columns=gbdt_feats_name)
    df_test_gbdt_feats = pd.DataFrame(gbdt_feats_test, columns=gbdt_feats_name)

    fea_train = pd.DataFrame(fea_train)
    print(df_test_gbdt_feats)
    fea_test = pd.DataFrame(fea_test)
    # 构造新数据集
    train = pd.concat([fea_train, df_train_gbdt_feats], axis=1)
    test = pd.concat([fea_test,df_test_gbdt_feats], axis=1)
    train_len = train.shape[1]
    print("train的长度",train_len)
    data = pd.concat([df_train_gbdt_feats, df_test_gbdt_feats])
    #data = pd.concat([train, test])
    print('GBDT.data: ', data)
    data = np.array(data)
    data_list = data.tolist()
    #print(data_list)
    #data = data.astype(np.float64)
    data_list = torch.tensor(data_list, dtype=torch.float64)
    num_classes = len(np.unique(numeric_labels))
    plt.figure(figsize=(16, 12))

    for i, (train_index, test_index) in enumerate(kfold.split(features)):
        # 将数据集分成训练集和测试集
        X_train, X_test = data_list[train_index], data_list[test_index]
        Y_train, Y_test = np.array(numeric_labels)[train_index], np.array(numeric_labels)[test_index]
        # 在训练集上拟合模型
        lr.fit(X_train, Y_train)
        # 在测试集上进行预测
        Y_pred = lr.predict(X_test)
        Y_prob = lr.predict_proba(X_test)[:, 1]
        print(Y_pred)
        print(Y_test)
        XGboost_measure_result = classification_report(Y_test, Y_pred)
        print('XGboost:measure_result = \n', XGboost_measure_result)
        #auc = roc_auc_score(Y_test, Y_pred)
        accuracy = accuracy_score(Y_test, Y_pred)
        #auc_accuracies.append(auc)
        accuracies.append(accuracy)
        print("准确率:", accuracy)
        print("AUC:", auc)
        # 计算混淆矩阵
        cm = confusion_matrix(Y_test, Y_pred)

        # 打印混淆矩阵
        print("Confusion Matrix:")
        print(cm)
        plt.subplot(2, 3, i + 1)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        # 利用seaborn绘制热力图
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Control', 'UC', 'CD'], yticklabels=['Control', 'UC', 'CD'])
    #plt.title('Confusion Matrices for Each Fold', fontsize=16)
    plt.suptitle('Confusion Matrix of GBDT-LR for predicting UC, CD and Control in Five-fold', fontsize=12)
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    save_path = os.path.join(os.getcwd(), 'results', 'Confusion Matrix.png')
    plt.savefig(save_path, dpi=300)
    plt.show()

    average_AUC = np.mean(auc_accuracies)
    average_accuracy = np.mean(accuracies)
    # 打印平均准确率
    #print("平均准确率:", average_AUC)
    print("平均准确率:", average_accuracy)
    predict_fn = lr.predict
    predict = gbm.predict_proba
    print(predict_fn)
    print(predict)
    features = fea.astype(np.float64)
    features = torch.tensor(features, dtype=torch.float64)
    fea_train = features[train_idx]
    fea_test = features[val_idx]
    class_names = ['IBD', '非IBD', 'Control']
    explainer = lime_tabular.LimeTabularExplainer(training_data=np.array(fea_train), # 训练集特征，必须是 numpy 的 Array
    feature_names=ids, # 特征列名
    class_names=['CD','UC', 'Control'], # 预测类别名称
    mode='classification' # 分类模式、
    )
    for i in range(155):
        print("第%s个样本的可解释性结果:"%(1+i))
        exp = explainer.explain_instance(np.array(fea_test[i]), predict_fn=gbm.predict_proba)
        ans=explanation.Explanation.as_list(exp)
        print("前10特征：%s"%ans)
        #plt = exp.as_pyplot_figure()
        #plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default='.\data\BioHealth\IBD\datas')
    parser.add_argument("--dataset_name", type=str, default='ibd')
    parser.add_argument("--data_type", type=str, default='clr', help="clr: log(x) - mean(log(x)), 0-1: log (x/sum(x)))")
    args = parser.parse_args()

    main(args)
