import matplotlib.pyplot as plt
from advertorch_examples.utils import _imshow
import torch
# from attacker import attacker

filename = "./model/RESNET18.pth"
model = torch.load(filename)
model.to("cuda")
model.eval()

# 空域展示
def displayer(dataset, display_row=3, display_col=5, batch_serialnum=6):
    batch_size = display_row * display_col
    plt.figure(figsize=(10, 8))
    for i in range(batch_size):
        plt.subplot(display_row, display_col, i + 1)
        _imshow(dataset[(batch_serialnum - 1) * batch_size + i])
        # target = torch.max(model(dataset[(batch_serialnum - 1) * batch_size + i].unsqueeze(0)), dim=1)[1]
        # plt.title("pred: {}".format(int(target)))

    print("display sequence number from {}".format(display_row * display_col * batch_serialnum + 1),
          "to {}".format(display_row * display_col * (batch_serialnum + 1) + 1))
    plt.tight_layout()
    return

# 选择一个通道展示
def displayer_Debug(dataset, display_row=3, display_col=5, batch_serialnum=6):
    batch_size = display_row * display_col
    plt.figure(figsize=(10, 8))
    for i in range(batch_size):
        plt.subplot(display_row, display_col, i + 1)
        R=dataset[(batch_serialnum - 1) * batch_size + i][0]
        G=dataset[(batch_serialnum - 1) * batch_size + i][1]
        B=dataset[(batch_serialnum - 1) * batch_size + i][2]
        _imshow(R.unsqueeze(0))
        # _imshow(G.unsqueeze(0))
        # _imshow(B.unsqueeze(0))
        # target = torch.max(model(dataset[(batch_serialnum - 1) * batch_size + i].unsqueeze(0)), dim=1)[1]
        # plt.title("pred: {}".format(int(target)))

    print("display sequence number from {}".format(display_row * display_col * batch_serialnum + 1),
          "to {}".format(display_row * display_col * (batch_serialnum + 1) + 1))
    plt.tight_layout()
    return

if __name__=="__main__":
    # freq : 分块的用于训练的 48*8*8
    # all: 分块用于展示的
    #
    # total: 分块用于展示的
    import torch
    mode = 1

    if mode ==1: append = "_freq_all"
    else: append = ""

    dataset_adv_part = torch.load("./adv_sample/adv_mixed_example"+append+".pt")
    dataset_cln_part = torch.load("./adv_sample/cln_mixed_example"+append+".pt")

    (row, col, bat) = (3,5,2)

    if mode ==1:
        displayer_Debug(1.2*torch.abs(dataset_adv_part), display_row=row, display_col=col, batch_serialnum=bat)
        displayer_Debug(1.2*torch.abs(dataset_cln_part), display_row=row, display_col=col, batch_serialnum=bat)
        displayer_Debug(10.0*torch.abs(dataset_adv_part-dataset_cln_part), display_row=row, display_col=col, batch_serialnum=bat)
        plt.show()
    else:
        displayer(torch.abs(dataset_adv_part), display_row=row, display_col=col, batch_serialnum=bat)
        displayer(torch.abs(dataset_cln_part), display_row=row, display_col=col, batch_serialnum=bat)
        displayer(0.5 + 3 *(dataset_adv_part - dataset_cln_part), display_row=row, display_col=col, batch_serialnum=bat)
        plt.show()







