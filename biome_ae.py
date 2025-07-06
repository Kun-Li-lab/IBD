import os
import copy
import time
import torch
import torch.nn as nn
import pickle
import argparse
import types
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from collections import defaultdict
from sklearn.cross_decomposition import CCA
from sklearn import linear_model
#from models import  AE,VAE, MLPerceptron, Decoder,Encoder
from models import  VAE,Decoder,Encoder
from snip.snip import snip_forward_linear, snip_forward_conv2d
import seaborn as sns

"""
Dataset提供一种方式去获取每个数据及其对应的label，告诉我们总共有多少个数据，位置在哪里。
Dataloader为后面的网络提供不同的数据形式，它将一批一批数据进行一个打包，从dataset里面取。
L1正则化会使某些参数权重为0，达到特征选择、稀疏化
L2正则化会将参数值压缩到较小的范围，不依赖于某个特征，使权重在不同特征之间平衡
"""
#获得数据
def get_dataloader(X1, X2, y, batch_size, shuffle=True,drop_last=True):
    X1_tensor = torch.FloatTensor(X1)
    X2_tensor = torch.FloatTensor(X2)
    y_tensor = torch.LongTensor(y)
    ds = TensorDataset(X1_tensor, X2_tensor, y_tensor)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
    return dl
class BiomeAE():

    def __init__(self, args):
        self.mlp_type = None
        self.model_alias = args.model_alias
        self.model= args.model
        self.snap_loc = os.path.join(args.vis_dir, "snap.pt")

        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
        self.predictor = None

        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def get_transformation(self):
        return None
    def loss_fn(self, recon_x, x, mean, log_var):#计算损失函数   重建输出、输入数据、均值、方差
        mseloss = torch.nn.MSELoss()
        return torch.sqrt(mseloss(recon_x, x))

        """
          BCE = torch.nn.functional.binary_cross_entropy(
            recon_x.view(-1), x.view(-1), reduction='sum')
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        return (BCE + KLD)
      
        """
    def param_l0(self): #返回稀疏自动编码器的 L0 正则化参数
        return self.predictor.param_l0()
    def init_fit(self, X1_train, X2_train, y_train, X1_val, X2_val, y_val, args, ):  #初始化训练和测试数据加载器，创建自动编码器模型和优化器
        self.train_loader = get_dataloader (X1_train, X2_train, y_train, args.batch_size)
        self.test_loader = get_dataloader(X1_val, X2_val, y_val, args.batch_size)
        self.predictor = VAE(
            encoder_layer_sizes=X1_train.shape[1],
            latent_size=args.latent_size,
            decoder_layer_sizes=X2_train.shape[1],
            activation=args.activation,
            batch_norm= args.batch_norm,
            dropout=args.dropout,
            mlp_type=self.mlp_type,
            conditional=args.conditional,
            num_labels=10 if args.conditional else 0).to(self.device)
        self.optimizer = torch.optim. Adam(self.predictor.parameters(), lr=args.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.8)
    def train(self, args):
        if args.contr:
            print("Loading from ", self.snap_loc)  #从指定的文件中加载已保存的模型，self.snap_loc并将模型的状态设置self.predictor为加载模型的状态
            loaded_model_para = torch.load(self.snap_loc)
            self.predictor.load_state_dict(loaded_model_para)
        t = 0
        logs = defaultdict(list)  #初始化一个字典
        iterations_per_epoch = len(self.train_loader.dataset) / args.batch_size  #计算每个时期迭代数量
        num_iterations = int(iterations_per_epoch * args.epochs)  #整个迭代总数
        for epoch in range(args.epochs):  #两层循环

            tracker_epoch = defaultdict(lambda: defaultdict(dict))

            for iteration, (x1, x2, y) in enumerate(self.train_loader):
                t+=1

                x1, x2, y = x1.to(self.device), x2.to(self.device), y.to(self.device)

                if args.conditional:
                    x2_hat, z, mean, log_var = self.predictor(x1, y)
                else:
                    x2_hat, z, mean, log_var = self.predictor(x1)
                for i, yi in enumerate(y):
                    id = len(tracker_epoch)
                    tracker_epoch[id]['x'] = z[i, 0].item()
                    tracker_epoch[id]['y'] = z[i, 1].item()
                    tracker_epoch[id]['label'] = yi.item()
                loss = self.loss_fn(x2_hat, x2, mean, log_var)
                self.optimizer.zero_grad()
                loss.backward()
                if (t + 1) % int(num_iterations / 10) == 0:
                    self.scheduler.step()
                self.optimizer.step()

                logs['loss'].append(loss.item())
                #if iteration % args.print_every == 0 or iteration == len(self.train_loader) - 1:
                    #print("Epoch {:02d}/{:02d} Batch {:04d}/{:d}, Loss {:9.4f}".format(
                        #epoch, args.epochs, iteration, len(self.train_loader) - 1, loss.item()))
                #enforce non-negative
                if args.nonneg_weight:
                    for layer in self.predictor.modules():
                        #if isinstance(layer, nn.Linear):
                            layer.weight.data.clamp_(0.0)
                            print("将神经网络层的权重张量限制为非负")#将神经网络层的权重张量限制为非负
        if not args.contr:
            print("Saving to ", self.snap_loc)
            torch.save(self.predictor.state_dict(), self.snap_loc)
    def fit(self,X1_train, X2_train, y_train, X1_val, X2_val, y_val, args,):
        self.init_fit(X1_train, X2_train, y_train, X1_val, X2_val, y_val, args)
        self.train(args)

    def visualize_heatmap(weights):

        sns.heatmap(weights, cmap="YlGnBu")
        plt.xlabel("Input Nodes")
        plt.ylabel("Output Nodes")
        plt.show()

    def get_graph(self):
        """
        return nodes and weights
        :return:
        """
        nodes = []
        weights = []
        for l, layer in enumerate(self.predictor.modules()):
            if isinstance(layer, nn.Linear):
                lin_layer =layer
                nodes.append(["%s"%(x) for x in list(range(lin_layer.in_features))])
                weights.append(lin_layer.weight.detach().cpu().numpy().T)
        nodes.append(["%s"%(x) for x in list(range(lin_layer.out_features))]) #last linear layer
        return (nodes, weights)


    def predict(self,X1_val, X2_val, y_val, args):  #对验证数据进行预测
        #Batch test
        x1, x2, y = torch.FloatTensor(X1_val).to(self.device), torch.FloatTensor(X2_val).to(self.device), torch.FloatTensor(y_val).to(self.device)
        if args.conditional:
            x2_hat, z, mean, log_var = self.predictor(x1, y)
        else:
            x2_hat, z, mean, log_var = self.predictor(x1)

        val_loss = self.loss_fn( x2_hat, x2, mean, log_var)
        print("val_loss: {:9.4f}", val_loss.item())
        return x2_hat.detach().cpu().numpy()
    def transform(self,X1_val, X2_val, y_val, args):  #返回验证数据的潜在表示
        x1, x2, y = torch.FloatTensor(X1_val).to(self.device), torch.FloatTensor(X2_val).to(
            self.device), torch.FloatTensor(y_val).to(self.device)
        if args.conditional: #检查是否有条件
            x2_hat, z, mean, log_var = self.predictor(x1, y)
        else:
            x2_hat, z, mean, log_var = self.predictor(x1)

        return z.detach().cpu().numpy()
    def get_influence_matrix(self):  #返回模型的影响矩阵
        return self.predictor.get_influence_matrix()
