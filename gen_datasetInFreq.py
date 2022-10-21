import torch
import dct_trans
import myiterater
import numpy as np
import displayer

# 展示block 用的
def trans_Debug(dataset):
    dataloader = myiterater.MyIterator(dataset, 1)
    dataset_freq = []
    for img in dataloader:
        pic = []
        for i in range(3):
            channel = dct_trans.imageBlockTrans_Debug(img[0][i].cpu().numpy())
            pic = pic + [channel]
        dataset_freq = dataset_freq + [pic]

    return torch.tensor(dataset_freq)

# 将图像转换为多通道的, 分块训练
def trans(dataset):
    dataloader = myiterater.MyIterator(dataset, 1)
    dataset_freq = []
    for channel_RGB in dataloader:
        channel_1 = dct_trans.imageBlockTrans(channel_RGB[0][0][:,:].cpu().numpy())
        channel_2 = dct_trans.imageBlockTrans(channel_RGB[0][1][:,:].cpu().numpy())
        channel_3 = dct_trans.imageBlockTrans(channel_RGB[0][2][:,:].cpu().numpy())
        out = np.concatenate((channel_1, channel_2, channel_3), axis=0)
        dataset_freq = dataset_freq + [out]
    return torch.tensor(dataset_freq)

# 全图变换
def trans_all(dataset):
    dataloader = myiterater.MyIterator(dataset, 1)
    dataset_freq = []
    for img in dataloader:
        pic = []
        for i in range(3):
            channel = dct_trans.imageTrans(img[0][i].cpu().numpy(), DCBias=0)
            pic = pic + [channel]
        dataset_freq = dataset_freq + [pic]

    return torch.tensor(dataset_freq)



if __name__ == "__main__":
    # RGB
    datapath = "./adv_sample/"

    adv_dataset = torch.load(datapath + "adv_mixed_example.pt")    # 成功对抗样
    cln_dataset = torch.load(datapath + "cln_mixed_example.pt")    # 失败对抗样本
    remain_dataset = torch.load(datapath + "cln_remained_example.pt")    # 失败对抗样本

    (temp1,temp2,temp3) = ( trans_Debug(adv_dataset), trans_Debug(cln_dataset), trans_Debug(remain_dataset))

    torch.save(temp1, datapath + "adv_mixed_example_freq_block.pt")  # N*48*8*8   48 = 16*3
    torch.save(temp2, datapath + "cln_mixed_example_freq_block.pt")
    torch.save(temp3, datapath + "cln_remained_example_freq_block.pt")

# if __name__=="__main__":
#     datapath = "./singletest/medium/"
#     name = "L1PGD_adv"
#     adv_dataset = torch.load(datapath + name + ".pt")
#
#     torch.save(trans(adv_dataset), datapath + name + "_freq" + ".pt")
#     torch.save(trans_all(adv_dataset), datapath + name + "_freq_all" + ".pt")
#     torch.save(trans_Debug(adv_dataset), datapath + name + "_freq_block" + ".pt")



