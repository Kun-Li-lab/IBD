import os  
import sys
import numpy as np
import pickle      #Python对象结构的二进制序列化和反序列化的核心库
import argparse     #给Python脚本传入参数的库
import matplotlib.pyplot as plt
#from pyrcca import rcca    正则化内核规范
#import logg ing
import torch
from sklearn import metrics
import tensorboard_logger as tl
import array
import lime
import seaborn as sns
import csv
from lime import lime_tabular
from lime.lime_tabular import LimeTabularExplainer
from biome_ae import BiomeAE
from utils import write_matrix_to_csv, get_subgraph, consistency_index,prune_subgraph
'''
CENTER_AND_SCALE===数据预处理方法,将数据集中的每个特征（列）进行中心化和标准化处理
中心化（centering）：将数据集中的每个特征的均值移动到 0，即将每个特征的值减去该特征的均值。
中心化后的数据集每个特征的均值为 0。

标准化（scaling）：将数据集中的每个特征缩放到相同的尺度，即将每个特征的值除以该特征的标准差。
标准化后的数据集每个特征的标准差为 1
列表list[有序]、集合set[无序]、字典dict[键值对]
set 是一种无序、不重复元素的集合数据类型
epoch是所有训练样本完成一次训练、batch是批次 一次训练的数据、iteration迭代 训练一个batch多少次
'''

RANDOM_SEED =1  #随机种子
parser = argparse.ArgumentParser()   #创建对象，ArgumentParser对象包含将命令行解析成Python数据类型所需的全部信息
FORMAT = '[%(asctime)s: %(filename)s: %(lineno)4d]: %(message)s'   #日志记录格式字符串
markers = {0: '.', 1: '+', 2: 'o'}
colors = {0: 'g', 1: 'r', 2: 'b', 3: 'm', 4: 'y', -1: 'black'}
def get_parts(x_full, parts, num_parts):
    this_part = []  #划分x_full列表 分批提取数据到this_part列表
    part_length = int(len(x_full) / num_parts)
    for part in parts:
        part_start = part_length * part
        part_end = part_length * (part + 1) if part < num_parts - 1 else len(x_full)
        this_part += x_full[part_start:part_end]
    return this_part


def visualize_vectors(x1_val, x2_pred, fold, val_ind, vis_dir):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.title(f'Input Vector Fold {fold} Val Ind {val_ind}')
    plt.hist(x1_val.flatten(), bins=50, color='blue', alpha=0.7)
    plt.xlabel('Feature Value')
    plt.ylabel('Frequency')

    plt.subplot(1, 2, 2)
    plt.title(f'Output Vector Fold {fold} Val Ind {val_ind}')
    plt.hist(x2_pred.flatten(), bins=50, color='red', alpha=0.7)
    plt.xlabel('Feature Value')
    plt.ylabel('Frequency')
    print("图片图片图片图片图片图片图片图片图片图片图片图片图片图片图片图片")
    plt.tight_layout()
    plt.close()
def init_model(args):
    # Train X1-->X2
    translator = BiomeAE(args)
    return translator
def main(args):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    DATA_ROOT=args.data_root
    data_type = args.data_type
    data_path = os.path.join(DATA_ROOT, "ibd_{}.pkl".format(data_type))
    with open(data_path,"rb") as df: #使用pickle.load(f)从f中读取一个字符串，并将它重构为原来的python对象，反序列化出对象过程
        content = pickle.load(df)
    extra = ""
    extra += "+bs_%d+ac_%s+lr_%s" %(
            args.batch_size, #批次大小
            args.activation,
            args.learning_rate
    )

    model_alias = 'Translator+%s_%s+%s-%s+cv_%d+%s+ls_%d+%s' % (
        args.dataset_name, #数据集名称
        args.data_type, #数据类型
        args.fea1, args.fea2, #特征1,2
        args.cross_val_folds,
        args.model,
        args.latent_size,  #潜在表示大小
        extra + args.extra)  #其他可选信息

    args.model_alias = model_alias
    print(model_alias)
    vis_dir = os.path.join(DATA_ROOT, 'vis', model_alias)  #用于存储训练过程中可视化结果的文件，例如损失曲线、重构结果等
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    args.vis_dir = vis_dir
    print("准备-------------------------------------------------------")
    X1 = content[args.fea1] # n x m1  读入特征  n样本数量、m特征数量
    X2 = content[args.fea2] # n x m2
    Name1 = content[args.fea1.replace("fea","ids")]
    Name2 = content[args.fea2.replace("fea","ids")]
    Y = content["diagnosis"]
    print(Y)
    #分为两类，疾病/健康
    for i in range(len(Y)):
        if Y[i] !=0:
            Y[i] =1
    print(Y)
    test_y_keys = [-1]+list(set(Y)) #-1==ALL所有样本
    y_ind  = {}  #该字典存放键为-1、0、1的数据
    #y_ind[-1] = range(len(Y))
    y_ind = {-1: Y}
    print(y_ind[-1])
    for y_val in set(Y):
        y_ind[y_val] = [i for i in range(len(Y)) if Y[i] == y_val]
    print(y_ind)
    ind_pos = []
    ind_neg = []
    for i, y in enumerate(Y):
        if y > 0:
            ind_pos.append(i)
        else:
            ind_neg.append(i)
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(ind_pos)
    np.random.shuffle(ind_neg)

    #TODO: Cross validation  交叉验证
    num_folds =args.cross_val_folds
    if not num_folds:
        num_folds =1
    CENTER_AND_SCALE = 0

    print("y 数据准备好-------------------------------------------------------")
    CV_results = [{} for i in range(num_folds)] #CV_results是空列表，里面元素是字典

    def get_parts(x_full, parts, num_parts):
        this_part = []  # 划分x_full列表 分批提取数据到this_part列表
        part_length = int(len(x_full) / num_parts)
        for part in parts:
            part_start = part_length * part
            part_end = part_length * (part + 1) if part < num_parts - 1 else len(x_full)
            this_part += x_full[part_start:part_end]
        return this_part
    for fold in range(num_folds):
        test_parts = [fold]
        train_parts = list(set(range(num_folds)) - set(test_parts))  #得到训练集的折号
        x1_train_pos = get_parts([X1[i] for i in ind_pos], train_parts, num_folds)  # 微生物训练正向作用
        x1_train_neg = get_parts([X1[i] for i in ind_neg], train_parts, num_folds)  # 微生物训练负向作用
        x1_train_all = np.vstack(x1_train_pos + x1_train_neg)  # 微生物训练集
        ind1_train_pos = [ind_pos[i] for i in train_parts]
        ind1_train_neg = [ind_neg[i] for i in train_parts]
        ind1_train = np.vstack(ind1_train_pos + ind1_train_pos)
        x2_train_pos = get_parts([X2[i] for i in ind_pos], train_parts, num_folds)  # 代谢物训练正向作用
        x2_train_neg = get_parts([X2[i] for i in ind_neg], train_parts, num_folds)  # 代谢物训练负向作用
        x2_train_all = np.vstack(x2_train_pos + x2_train_neg)  # 代谢物训练集
        ind2_train_pos = [ind_pos[i] for i in train_parts]
        ind2_train_neg = [ind_neg[i] for i in train_parts]
        ind2_train = np.vstack(ind2_train_pos + ind2_train_pos)
        x1_val_pos = get_parts([X1[i] for i in ind_pos], test_parts, num_folds)  # 微生物验证正向作用
        x1_val_neg = get_parts([X1[i] for i in ind_neg], test_parts, num_folds)  # 微生物验证负向作用
        x1_val_all = x1_val_pos + x1_val_neg  # 微生物验证集
        x1_vals = [np.vstack(x1_val_all), np.vstack(x1_val_neg), np.vstack(x1_val_pos)]

        x2_val_pos = get_parts([X2[i] for i in ind_pos], test_parts, num_folds)  # 代谢物验证正向作用
        x2_val_neg = get_parts([X2[i] for i in ind_neg], test_parts, num_folds)  # 代谢物验证负向作用
        x2_val_all = x2_val_pos + x2_val_neg  ##代谢物验证集
        x2_vals = [np.vstack(x2_val_all), np.vstack(x2_val_neg), np.vstack(x2_val_pos)]

        vals_meanings = ["all", "negatives", "positives"]

        ind_test_pos = get_parts(ind_pos, test_parts, num_folds)
        ind_test_neg = get_parts(ind_neg, test_parts, num_folds)
        ind_train_pos = get_parts(ind_pos, train_parts, num_folds)
        ind_train_neg = get_parts(ind_neg, train_parts, num_folds)

        y_train = np.concatenate((np.ones((len(ind_train_pos), 1)), np.zeros((len(ind_train_neg), 1))), axis=0)  #合并数组
        y_test = np.concatenate((np.ones((len(ind_test_pos), 1)), np.zeros((len(ind_test_neg), 1))), axis=0)
        y_vals = [y_test,  np.ones((len(ind_test_pos), 1)), np.zeros((len(ind_test_neg), 1))]
        print("交叉验证 数据准备好-------------------------------------------------------")

        x1_train = x1_train_all
        x2_train = x2_train_all
        if (CENTER_AND_SCALE):
            mu1 = (x1_train.max(axis=0) + x1_train.min(axis=0))/2.0
            #print("x1_train%s"%x1_train)
            x1_train = x1_train - mu1
            #print("x1_train%s"%x1_train)
            std1 = (x1_train.max(axis=0) - x1_train.min(axis=0))/2.0
            x1_train /= std1
            x1_vals = [(x1_val - mu1) / std1 for x1_val in x1_vals]
            mu2 = (x2_train.max(axis=0) + x2_train.min(axis=0))/2.0
            x2_train = x2_train - mu2
            std2 = (x2_train.max(axis=0) - x2_train.min(axis=0))/2.0
            x2_train /= std2
            x2_vals = [(x2_val - mu2) / std2 for x2_val in x2_vals]

        #Train
        args.contr = fold
        print("接下来开始第%s次模型-------------------------------------------------------"%fold)
        translator = init_model(args)
        print("初始化第%s次模型完成，模型开始训练-------------------------------------------------------"%fold)
        translator.fit(x1_train,x2_train,y_train, x1_vals[0], x2_vals[0], y_test, args)
        nodes, weights = translator.get_graph()
        #translator.visualize_heatmap(weights)
        print("训练第%s次模型完成完成-------------------------------------------------"%fold)
        print(content["bac_group_ids"])

        latent_corrs_train_cv = [0 for i in range(x1_train.shape[1])]
        print("latent_corrs_train_cv%s-----------------------"%latent_corrs_train_cv)
        print("x1_train.shape=%s"%x1_train.shape[1])

        #Test
        print("测试第%s次开始------------------------------------------------------"%fold)
        test_corrs1_cv = [None for _ in range(len(vals_meanings))]#存储一组交叉验证实验中第一种测试数据与对应预测结果之间的相关系数
        test_corrs2_cv = [None for _ in range(len(vals_meanings))]#存储一组交叉验证实验中第二种测试数据与对应预测结果之间的相关系数
        latent_corrs_val_cv = [None for _ in range(len(vals_meanings))]#存储一组交叉验证实验中预测结果与实际结果之间的相关系数
        avg_acc1_cv = [None for _ in range(len(vals_meanings))]#存储一组交叉验证实验中第一种测试数据的平均准确率
        x2_val_hat = [None for _ in range(len(vals_meanings))]
        for val_ind, (x1_val, x2_val, y_val, meaning) in enumerate(zip(x1_vals, x2_vals, y_vals, vals_meanings)):
            # Test correlation: Correlation between org signal and the predicted one
            print("测试meaning的%s开始------------------------------------------------------" % meaning)
            x2_val_hat[val_ind] = translator.predict(x1_val, x2_val, y_val, args)
            print(len(x2_val_hat[val_ind]))
            print(x1_val.shape[0])
            print(x1_val.shape[1])
            print(x2_val.shape[0])
            print(x2_val.shape[1])
            example_index = 0
            input_vector_1 = x1_val[example_index]
            input_vector_2 = x2_val[example_index]
            output_vector = x2_val_hat[val_ind][example_index]

            print("\n展示第{}个折叠下的第{}个样本".format(fold, example_index))
            print("Input Vector 1 (Microbiome Features):", input_vector_1)
            print("Input Vector 2 (Metabolomic Features):", input_vector_2)
            print("Output Vector (Predicted Metabolites):", output_vector)
            test_corrs1_cv[val_ind] = np.nan_to_num([np.corrcoef(x2_val_hat[val_ind][:, i], x2_val[:, i])[0, 1] for i in range(x2_val.shape[1])])
            avg_acc1_cv[val_ind] = np.sqrt(metrics.mean_squared_error(x2_val, x2_val_hat[val_ind]))
            nodes12, weights12 = translator.get_graph()
            latent_corrs_val_cv = [0 for i in range(x1_val.shape[1])]

            argsort = np.argsort(-test_corrs1_cv[val_ind])
            argsort1_filtered = argsort[test_corrs1_cv[val_ind]  > 0.7]
            # 输出满足条件的 argsort1
            print("相关性>0.7的索引%s" % argsort1_filtered)

            met_predict = x2_val_hat[val_ind][:, argsort1_filtered]
            # print("预测出来的代谢物%d"%met_predict.shape)
            visualize_vectors(x1_val, x2_val_hat[val_ind], fold, val_ind, vis_dir)

            if val_ind == 0:
                with open(os.path.join(vis_dir, 'new----met0.7.csv'), mode='a', newline='') as file:
                    writer = csv.writer(file)
                    print("yixie")
                    # 写入二维数组的每一行
                    for row in met_predict:
                        writer.writerow(row)

        CV_results[fold] = {'test_corrs1':test_corrs1_cv,  #相关性
                            'avg_acc1':avg_acc1_cv,   #均方误差准确性
                            'latent_corrs_train': latent_corrs_train_cv,
                            'latent_corrs': latent_corrs_val_cv,
                            'trans12':translator.get_transformation(),
                            'nodes12':nodes12,
                            'weights12':weights12,
                            }
        print("[CV_results[fold]['latent_corrs_train']%s-----------------------" %CV_results[fold]['latent_corrs_train'])
    #Join the  folds
    latent_corrs_train_allfolds = [CV_results[fold]['latent_corrs_train'] for fold in range(num_folds)]
    latent_corrs_train = np.array(latent_corrs_train_allfolds).mean(axis=0)
    latent_corrs = [None for _ in range(len(vals_meanings))]
    for v in range(len(vals_meanings)):
        latent_corrs_allfolds = [CV_results[fold]['latent_corrs'][v] for fold in range(num_folds)]
        latent_corrs[v] = np.array(latent_corrs_allfolds).mean(axis=0)
    print("latent_corrs[v]%s-----------------------" % latent_corrs)
    NUM_PLOT_COL = len(vals_meanings)  # cca fit + y_test types
    NUM_PLOT_ROW = 1
    '''
    绘制了训练集和验证集中每个“潜在变量”（即模型中的“Canonical component”）与目标变量之间的“Canonical correlation”
    “latent_corrs_train”表示训练集的“潜在变量”相关性
    “latent_corrs[v]”表示验证集中第v个数据集的“潜在变量”相关性。
    Canonical correlation量化两组变量之间线性关系的统计量度,
    通过找出每组变量之间相关性最高的线性组合来衡量两组变量之间的关联程度
    Canonical component
    在CCA中，对于两个不同的数据集合，它们各自有自己的特征向量和特征值，
    通过将它们的特征向量按照特征值从大到小排序，选择其中的前k个，就可以得到k个最大的Canonical component，
    这些Canonical component可以用来表示数据集合之间的相关性或协方差。
    '''

    print("-------------------------------------------------------")
    print("计算所有CV的平均相关性和准确性")
   #计算交叉验证实验中所有折叠的每个验证选择方法的平均相关性和准确性
    #values_greater_than_0_7 = [list() for _ in range(len(vals_meanings))]
    # first row 1->2
    if args.visualize == "subplot":
        fig, axes = plt.subplots(nrows=NUM_PLOT_ROW, ncols=NUM_PLOT_COL, figsize=(np.sqrt(2) * 12, 1 * 12))
        #plt.subplot(NUM_PLOT_ROW, NUM_PLOT_COL, 1 + v)
        fig.suptitle("%s" % model_alias, fontsize=16)
        row_names = ["X2->X1", "X1->X2"]

    if args.visualize == "subplot":
        plt.subplot(NUM_PLOT_ROW, NUM_PLOT_COL, 1)
    else:
        plt.figure()

    for v, meaning in enumerate(vals_meanings):  # for each choice of val selection, we take the mean of the folds
        test_corrs1_allfolds = [CV_results[fold]['test_corrs1'][v] for fold in range(num_folds)]
        test_corrs1 = np.array(test_corrs1_allfolds).mean(axis=0)

        meancc1 = test_corrs1.mean()
        argsort1 = np.argsort(-test_corrs1)  # descending

        sort1 = test_corrs1[argsort1]
        meantopk1 = sort1[:args.topk].mean()

        avg_acc1_allfolds = [CV_results[fold]['avg_acc1'][v] for fold in range(num_folds)]
        avg_acc1 = np.array(avg_acc1_allfolds).mean(axis=0)
        # first row 1->2
        if args.visualize == "subplot":
            plt.subplot(NUM_PLOT_ROW, NUM_PLOT_COL, 1 + v)
            print("执行了")
        else:
            plt.figure()

        for fold in range(num_folds):
            plt.plot(np.arange(len(test_corrs1_allfolds[fold])) + 1, test_corrs1_allfolds[fold], marker='x',
                        c=colors[fold], label="fold %s" % (fold))
        plt.plot(np.arange(len(test_corrs1)) + 1, test_corrs1, marker=markers[val_ind],
                        c=colors[-1], label="mean cv")
        plt.title(
                    'Y %s, x1->x2 top %d:%3.2f, best %d:%3.2f' % (meaning, args.topk, meantopk1, argsort1[0], sort1[0]))
        plt.legend()
        plt.ylim(-1.0, 1.0)
        plt.savefig(os.path.join(vis_dir, 'cv_val%s_corcoeff.png' % (meaning)))
        plt.show()



        # 在指定位置创建子图

        # FINAL MODEL
        # Train a single model on all data
        if (CENTER_AND_SCALE):
            # Center and Scale
            mu1 = (X1.max(axis=0) + X1.min(axis=0)) / 2.0
            X1 = X1 - mu1
            std1 = (X1.max(axis=0) - X1.min(axis=0)) / 2.0
            X1 /= std1
            mu2 = (X2.max(axis=0) + X2.min(axis=0)) / 2.0
            X2 = X2 - mu2
            std2 = (X2.max(axis=0) - X2.min(axis=0)) / 2.0
            X2 /= std2
        translator.fit(X1, X2, Y, X1, X2, Y, args)

        # 权重写入文件
        nodes, weights = translator.get_graph()
        if args.model in ["BiomeAE"]:
            z = translator.transform(X1, X2, Y, args)
            write_matrix_to_csv(z, os.path.join(vis_dir, 'latent_z.txt'))
            write_matrix_to_csv(weights[0], os.path.join(vis_dir, 'x1_to_z_weight.txt'))
            write_matrix_to_csv(weights[1], os.path.join(vis_dir, 'z_to_x2_weight.txt'))

        X2_hat = translator.predict(X1, X2, Y, args)
        # nan_to_num计算两个数组之间的相关系数；corrcoef计算两个数组之间的相关系数;然后对相关性进行排序
        test_corrs = np.nan_to_num(
            [np.corrcoef(X2_hat[:, i], X2[:, i])[0, 1] for i in range(X2.shape[1])])
        argsort = np.argsort(-test_corrs)  # descending
        sort = test_corrs[argsort]
        avg_acc = np.sqrt(metrics.mean_squared_error(X2, X2_hat))
        meancc = test_corrs.mean()
        meantopk = sort[:args.topk].mean()

        print(
            "CV  %s-->%s,        RMSE %4.2f,    mean corrcoef:%4.2f,      met best：%s/%4.2f,     top %d average:%4.2f,      \n top10:%s " % (
                args.fea1,
                args.fea2,

                avg_acc1,  # 均方误差
                meancc1,  #
                argsort1[0], sort1[0],
                 args.topk, meantopk1,
                 ", ".join(["%d:%4.2f"%(k,a) for k,a in zip(argsort1[:args.topk],sort1[:args.topk])]))
            )

        print(
            "All %s-->%s,     RMSE %4.2f,     mean corrcoef:%4.2f,     met best：%s/%4.2f,     top average%d:%4.2f,     \n top10:%s" % (
                args.fea1,
                args.fea2,
                avg_acc,
                meancc,
                argsort[0], sort[0],
                 args.topk, meantopk, ", ".join(["%d:%4.2f"%(k,a) for k,a in zip(argsort[:args.topk],sort[:args.topk])]))
            )
        pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()   #parse对象相当于一个总容器，存放着全部的信息
    parser.add_argument("--data_root", type=str, default='./data/BioHealth/IBD')   #type读取参数类型，default默认值
    parser.add_argument("--model", type=str, default='BiomeAE')
    parser.add_argument("--fea1", type=str, default='bac_group_fea')
    parser.add_argument("--fea2", type=str, default='met_group_fea')
    parser.add_argument("--dataset_name", type=str, default='ibd')
    parser.add_argument("--data_type", type=str, default='clr')
    parser.add_argument("--cross_val_folds", type=int, default=5)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--sparse", type=str, default=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=20)   #单次训练的样本
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--activation", type=str, default="tanh_true")
    parser.add_argument('--dropout', default=0, type=float)
    parser.add_argument('--batch_norm', default=1, type=bool)   #加速神经网络训练
    parser.add_argument("--latent_size", type=int, default=20)
    parser.add_argument("--print_every", type=int, default=10)
    parser.add_argument("--fig_root", type=str, default='figs')
    parser.add_argument("--conditional", action='store_true')
    parser.add_argument("--nonneg_weight", action='store_true')
    parser.add_argument("--normalize_input", action='store_true')
    parser.add_argument('--gpu_num', default="0", type=str)
    parser.add_argument("--extra", type=str, default='')
    parser.add_argument("--visualize", type=str, default='subplot')
    parser.add_argument("--draw_graph", type=str, default=True)
    args = parser.parse_args()
    main(args)
    """
    创建一个解析器---创建ArgumentParser()对象
    添加参数---调用add_argument()方法添加参数
    解析参数---使用parse_args()解析添加的参数
    action：命令行遇到参数时的动作，默认值为store
    store_true：表示赋值为true
    default:不指定参数时的默认值
    type：命令行参数应该被转换成的类型
    在文中使用参数时，调用使用的是arg.
    Python类中__init__函数-----创建该类实例时立即调用该函数，为该类初始化
        init(self,参数）和__init__(self)区别：
            __init__()方法中只有self，但在方法的类部包含三个属性。
            __init__(self,参数）在定义方法的时候，直接给定三个参数（实例化类时传过来的）
    下划线开头的函数是声明该属性为私有，不能在类的外部被使用或访问；
    nn.functional.x直接传入参数调用，nn.x需要先实例化再传参调用
    python创建张量  torch.Tensor()    torch.tensor()
    python重构张量  .reshape(a,b)     .resize(a,b)        .transpose(a,b)
    nn.moudle模块中有许多神经网络层：卷积层、池化层、
    常见深度学习模型：卷积神经网络CNN
                    循环神经网络RNN
                    长短时记忆网络LSTM
                    生成对抗网络GAN
                    自编码器AE
                    残差网络ResNet
                    注意力机制Attention
                    seq2seq transtormer bert
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'      # 只允许使用0号显卡
    os.environ["CUDA_VISIBLE_DEVICES"] = '1,2,3,4'  # 允许使用多块显卡
    """

