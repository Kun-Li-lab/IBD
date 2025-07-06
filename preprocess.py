import os
import time
import argparse
import csv
import numpy as np
import pickle
from utils import read_csv_table, read_csv_table_raw

def count_zeros(arr):
    count0= 0
    count1= 0
    count2= 0# 用于统计0的数量
    for num in arr:
        if num == 0:  # 检查数组中的元素是否为0
            count0 += 1  # 如果是0，计数器加1
        elif num==1:
            count1 +=1
        elif num == 2:
            count2 +=1

    return count0,count1,count2

def main(args):
    metabolite_file = os.path.join(args.data_root, "ibd-x1-met-zComp-{}.csv".format(args.data_type))  #代谢物文件
    bacteria_file   = os.path.join(args.data_root, "ibd-x2-mic-zComp-{}.csv".format(args.data_type))  #微生物文件
    metabolite_group_file = os.path.join(args.data_root, "ibd-x1-met-zComp-grouped-{}.csv".format(args.data_type))
    bacteria_group_file = os.path.join(args.data_root, "ibd-x2-mic-zComp-grouped-{}.csv".format(args.data_type))
    bacteria_taxa_file = os.path.join(args.data_root, "mic-taxa-itis-v2.csv")
    label_file      = os.path.join(args.data_root, "ibd_y.csv")
    output_file =os.path.join(args.data_root, "ibd_{}.pkl".format(args.data_type))
    met_group_DA_file =os.path.join(args.data_root, "6-DA-x1-Met-Group.csv")
    bac_group_DA_file = os.path.join(args.data_root, "6-DA-x2-Mic-Group.csv")
    z_DA_file = os.path.join(args.data_root, "7-latent-z-DA.csv")

    #Process metabolite:
    subject_ids, met_ids, met_fea = read_csv_table(metabolite_file)
    subject_ids, met_group_ids, met_group_fea = read_csv_table(metabolite_group_file)
    met_c_name, met_group_DA = read_csv_table_raw(met_group_DA_file)
    bac_c_name, bac_group_DA = read_csv_table_raw(bac_group_DA_file)
    z_c_name, z_DA = read_csv_table_raw(z_DA_file)
    met_fea = met_fea.astype(np.float64)  #astype转换类型
    met_group_fea = met_group_fea.astype(np.float64)

    if args.data_type == "0-1":
        met_fea = np.log(met_fea)  #求对数，数据平滑处理
        met_group_fea = np.log(met_group_fea)


    # Process bacteria
    row_name_bac, bac_ids, bac_fea = read_csv_table(bacteria_file)
    row_name_bac_group, bac_group_ids, bac_group_fea = read_csv_table(bacteria_group_file)
    assert row_name_bac == subject_ids
    bac_fea = bac_fea.astype(np.float64)
    bac_group_fea = bac_group_fea.astype(np.float64)
    if args.data_type == "0-1":
        bac_fea = np.log(bac_fea)
        bac_group_fea = np.log(bac_group_fea)

    # Grouping bac myself
    bac_taxa = read_csv_table(bacteria_taxa_file,output_dict=True)
    print(bac_taxa)
    #Get a list of unique genuses  获取独特属的列表
    genuses = set()
    for bac_id, this_bac_taxa in bac_taxa.items():  #items() 函数以列表返回可遍历的(键, 值) 元组
        genuses.add(this_bac_taxa["genus属"])
    genuses = list(genuses)
    print("属：%s"%genuses)
    # assign a list of col ind to each genus ind  为每个属 ind 分配一个 col ind 列表
    gen_to_bac = [[] for i in range(len(genuses))]
    #print(gen_to_bac)  #一个属长度的空数组    52个属
    for bac_col, bac_id in enumerate(bac_ids):
        gen_to_bac[genuses.index(bac_taxa[bac_id]["genus属"])].append(bac_col)
    #print(len(gen_to_bac))  #属对应的细菌在一个数组中
    genus_fea = np.zeros((len(subject_ids),len(genuses)))
    for i in range(len(subject_ids)):
        for g in range(len(genuses)):
            if args.data_type =="clr":
                genus_fea[i,g] = np.mean(bac_fea[i, gen_to_bac[g]])
            elif args.data_type =="0-1":
                genus_fea[i, g] = np.sum(bac_fea[i, gen_to_bac[g]])
    print("属的结构：{}".format("genus_fea.shape"))   #属的含量
    # Process y
    row_name_lab, col_name_lab, labels = read_csv_table(label_file)
    assert row_name_bac == subject_ids
    diagnosis = labels[:, 1]
    factors = np.delete(labels, 1, axis=1)
    print("临床因素有：%s"%factors)
    diagnosis = labels[:,1]
    diag_encode = {
        "Control":0,
        "CD":1,
        "UC":2
    }
    train_idx = []
    val_idx = []
    for i, row_name in enumerate(row_name_lab):
        if "Validation" in row_name_lab[i]:
            val_idx.append(i)
        else:
            train_idx.append(i)

    diagnosis = [diag_encode[d] for d in diagnosis]
    diagnosis = np.array(diagnosis)
    print(diagnosis)
    # TODO:use tree structure or binning to preprocess
    count0,count1,count2=count_zeros(diagnosis)

    print(count2)
    print("对照样本人数：{},CD人数：{},UC人数：{}".format("count0","count1","count2"))
    out_data = {
            "subject_ids":subject_ids,   #样本
            "met_ids": met_ids,  #代谢物ID
            "met_group_ids": met_group_ids,  #代谢物组ID
            "met_fea":met_fea,  #代谢物含量
            "met_group_fea": met_group_fea,  #代谢物组含量
            "bac_ids":bac_ids,   #微生物ID
            "bac_group_ids": bac_group_ids,  #微生物组ID
            "bac_fea":bac_fea,  #微生物含量
            "bac_group_fea": bac_group_fea,  #微生物组含量
            "train_idx":train_idx,  #训练集
            "val_idx":val_idx,   #验证集
            "diagnosis":diagnosis,   #预测结果
            "met_group_DA": met_group_DA,   #代谢物信息
            "bac_group_DA": bac_group_DA,   #微生物信息
            "z_DA": z_DA,       #z_DA信息，隐藏层信息---
            "factors":factors
             }
    print("met_ids:%s"%met_ids)
    print("met_group_ids:%s" % met_group_ids)
    print("met_ids:%s" % len(met_ids))
    print("met_group_ids:%s" % len(met_group_ids))
    print("bac_ids:%s" % len(bac_ids))
    print("bac_group_ids:%s" % len(bac_group_ids))
    with open(output_file,'wb') as of:
        pickle.dump(out_data, of, protocol=2)
    print("written %s"%output_file)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default='.\data\BioHealth\IBD\datas')
    parser.add_argument("--dataset_name", type=str, default='ibd')
    parser.add_argument("--data_type", type=str, default='clr', help="clr: log(x) - mean(log(x)), 0-1: log (x/sum(x)))")
    args = parser.parse_args()

    main(args)
