# generate adversary examples using a assortment of algorithm,
# save them


# 获取MNIST数据集
from torch.utils.data import DataLoader
from torchvision.datasets import cifar
import torchvision.transforms as torchtrans

import torch
from advertorch.utils import predict_from_logits
from advertorch.test_utils import LeNet5
from advertorch_examples.utils import TRAINED_MODEL_PATH
from advertorch_examples.utils import get_mnist_test_loader

from attacker import attacker
import shutil
import os

# logger
import logger
import sys
logPath = "./adv_sample/logout.txt"
sys.stdout = logger.Logger(logPath)
print("log saved to " + logPath)


def createDict(filepath):
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    else:
        shutil.rmtree(filepath)
        os.makedirs(filepath)
    return


torch.manual_seed(0)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# 加载网络状态
filename = "./model/RESNET18.pth"
model = torch.load(filename)
model.to(device)
model.eval()

# 加载攻击器，设定攻击目标数量
method_num = attacker.items().__len__()
batchsize = 256
batchnum = 0
start_batch = 80
cln_leastNum = 25000  # 最小干净样本数

TrainSet = cifar.CIFAR10('./', train=True, download=False, transform=torchtrans.ToTensor())
loader = DataLoader(TrainSet, batch_size=batchsize, shuffle=True)

print("batch size: {}, ATK method num: {}".format(batchsize, method_num ))
print("start from {}th batch (begin with 0), sample SER num: {}".format(start_batch, batchsize * start_batch))
print(" =====================================")

totalSuccessSample_num = 0    # 全部攻击成功数
methodlist = list(attacker.keys())  # 攻击列表
clean_sample = torch.tensor([]).to(device)
curSample_serialNum = - batchsize    # 统计样本位置
currmethodnum = 0  # 当前攻击算法
currbatchnum = -1   # 当前 batch数
for ii, (cln_data, true_label) in enumerate(loader):
    curSample_serialNum += batchsize
    if ii < start_batch:  # 跳过最开始的 batch
        continue

    if ii - start_batch >= method_num*batchnum:
        # 将没有使用的batch存储下来
        print(" - save remain clean from SER num: {}".format(curSample_serialNum))
        cln_data = cln_data.to(device)
        clean_sample = torch.cat((clean_sample, cln_data), 0)
        if ii - method_num*batchnum - start_batch >= int(cln_leastNum/batchsize):
            print(" - (remain clean) end, last batch at SER num:{} ~ {}".format(curSample_serialNum,
                                                                              curSample_serialNum + batchsize))
            break

    else:
        currmethodnum = int((ii - start_batch) / batchnum)
        print("currmethodnum: {}".format(currmethodnum))
        currbatchnum = (currbatchnum+1) % batchnum

        cln_data, true_label = cln_data.to(device), true_label.to(device)
        print(" - loading {} clean samples, from SER num: {}".format(batchsize, curSample_serialNum))

        algor_name = methodlist[currmethodnum]
        method_handle = attacker[algor_name]
        print(" - using attack algorithm:", algor_name)

        # use successAtk_sample to contain ADVERSARY examples
        # use successAtk_orig to contain corresponding CLEAN examples
        # use to contain corresponding CLEAN examples
        if (currbatchnum == 0):
            successAtk_num = 0
            successAtk_sample = torch.tensor([]).to(device)
            successAtk_orig = torch.tensor([]).to(device)
            #failedAtk_sample = torch.tensor([]).to(device)
            #failedAtk_orig = torch.tensor([]).to(device)

        adv_data = method_handle.perturb(cln_data, true_label)
        pred_cln = predict_from_logits(model(cln_data))
        pred_adv = predict_from_logits(model(adv_data))
        for j, clnlabel in enumerate(true_label):
            if not pred_adv[j] == clnlabel:  # 攻击成功
                successAtk_num += 1
                successAtk_sample = torch.cat((successAtk_sample, adv_data[j].unsqueeze(0)), 0)
                successAtk_orig = torch.cat((successAtk_orig, cln_data[j].unsqueeze(0)), 0)
                # unsqueeze: insert an dim in position,  unsqueeze(0): most outside dim
                # torch.cat: concatenate the tensor, 0: most outside dim
            else:    # 攻击不成功
                pass
                #failedAtk_sample = torch.cat((failedAtk_sample, adv_data[j].unsqueeze(0)), 0)
                #failedAtk_orig = torch.cat((failedAtk_orig, cln_data[j].unsqueeze(0)), 0)
        print(" - success generating {} adv samples, success ATK rate:{}".format(successAtk_num,successAtk_num/batchsize/(currbatchnum+1)))
        totalSuccessSample_num += successAtk_num

        if (currbatchnum == batchnum-1):
            savepath = os.path.join(".", "adv_sample", algor_name)
            createDict(savepath)
            torch.save(successAtk_sample, os.path.join(savepath, algor_name + "_adv" + ".pt"))    # 成功攻击样本
            torch.save(successAtk_orig, os.path.join(savepath, algor_name + "_advOrig" + ".pt"))    # 成功攻击原型
            # torch.save(failedAtk_sample, os.path.join(savepath, algor_name + "_fail" + ".pt"))    # 失败攻击样本
            # torch.save(failedAtk_orig, os.path.join(savepath, algor_name + "_failOrig" + ".pt"))    # 失败攻击原型
            print(" - saved")
            print(" -------------------------------------")

# 保存未使用的样本
print(" =====================================")
savepath = "./adv_sample"
torch.save(clean_sample, os.path.join(savepath, "cln_remained_example.pt"))
print(" - save {} (remain) clean samples to".format(clean_sample.shape[0]) + os.path.join(savepath, "cln_remained_example.pt"))
print(" - generate {} ADV in all".format(totalSuccessSample_num))

print(" generate ADVER SERIES num: {} ~ {}".format(batchsize * start_batch, batchsize * (start_batch+method_num) ) )
print(" generate CLEAN SERIES num: {} ~ {}".format(batchsize * (start_batch+method_num), curSample_serialNum + batchsize) )

print("end of process")



