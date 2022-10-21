# 统计高频区域分量大小
import torch

def getMetricValue(datatensor_freq):
    # datatensor_freq: tensor shape [sample num, 1, 28, 28]
    sample_num = datatensor_freq.shape[0]
    value = 0
    for i in range(sample_num):
        matrix_freq = datatensor_freq[i][0]
        value += zigzagHighFreq(matrix_freq)

    value /= sample_num
    return value


def blockHighFreq(coe_freq):
    # coe_freq: tensor shape [28, 28]
    blocksize = 3
    high_freq = coe_freq[28-blocksize:28,28-blocksize:28]
    value = abs(high_freq).sum() / (blocksize ** 2)
    return value


def zigzagHighFreq(coe_freq):
    blocksize = 8
    value = 0

    for i in range(blocksize):
        for j in range(i+1):
            value += torch.abs(coe_freq[28-blocksize+i, 27-i+j])

    value /= (blocksize*(blocksize-1)/2)
    return value


if __name__ == "__main__":


    dataPath_base_adv = "./adv_sample/adv_mixed_example" + "_freq_total" + ".pt"
    dataPath_base_cln = "./adv_sample/cln_mixed_example" + "_freq_total" + ".pt"

    dataset_adv = torch.load(dataPath_base_adv).to("cuda")
    dataset_cln = torch.load(dataPath_base_cln).to("cuda")

    for i in range(10):
        print("-------------------------------")
        print(blockHighFreq(dataset_adv[i][0]))
        print(blockHighFreq(dataset_cln[i][0]))


        # print("adv:", getMetricValue(dataset_adv[i].unsqueeze(0)).tolist() )
        # print("cln:", getMetricValue(dataset_cln[i].unsqueeze(0)).tolist() )

