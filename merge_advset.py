from attacker import attacker
import torch
import os
import random
import displayer


if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    algorithm_name = list(attacker)
    rootpath_name = "./adv_sample"

    dataset_adv = torch.tensor([]).to(device)
    dataset_cln = torch.tensor([]).to(device)
    dataset_fail = torch.tensor([]).to(device)
    for sub_path in algorithm_name:
        full_path = os.path.join(rootpath_name, sub_path)
        try:
            dataset_adv_part = torch.load(full_path + "/" + sub_path + "_adv" + ".pt")
            dataset_cln_part = torch.load(full_path + "/" + sub_path + "_advOrig" +".pt")
            #dataset_fail_part = torch.load(full_path + "/" + sub_path + "_failOrig" +".pt")
            print(" - from:" + full_path + "/" + sub_path + ".pt")
            print("   load success adv item: {},".format(dataset_adv_part.shape[0]) )
            print("   load corr cln item: {},".format(dataset_cln_part.shape[0]))
            #print("   load failed adv item: {},".format(dataset_fail_part.shape[0]))

            dataset_adv = torch.cat((dataset_adv, dataset_adv_part), 0)
            dataset_cln = torch.cat((dataset_cln, dataset_cln_part), 0)
            #dataset_fail = torch.cat((dataset_fail, dataset_fail_part), 0)

        except:
            print(" - path not found! {}".format(full_path + "/" + sub_path + ".pt"))

    # shuffle the data
    sample_num = dataset_adv.shape[0]
    label = [i for i in range(sample_num)]
    random.shuffle(label)
    dataset_adv_shuffled = dataset_adv[label]
    dataset_cln_shuffled = dataset_cln[label]


    # save the data
    torch.save(dataset_adv_shuffled, rootpath_name + "/adv_mixed_example.pt")
    torch.save(dataset_cln_shuffled, rootpath_name + "/cln_mixed_example.pt")
    # torch.save(dataset_fail_shuffled, rootpath_name + "/fail_mixed_example.pt")

    print("save {} adversary items,".format(dataset_adv.shape[0]), "to: " + rootpath_name + "/adv_mixed_example.pt")
    print("save {} corresponding clean items,".format(dataset_adv.shape[0]), "to: " + rootpath_name + "/cln_mixed_example.pt")
