import numpy as np
import torch
from model.LSTM import TwoStreamRNN
from dataset import TestDataset
from torch.utils.data import DataLoader
# 用于测试集推理


def test(model, batch_size, device, dataset, dataloader):
    cs = [24, 25, 12, 11, 10, 9, 21, 5, 6, 7, 8, 23, 22, 4, 3, 21, 2, 1, 20, 19, 18, 17, 1, 13, 14, 15, 16]
    test_loader = dataloader
    test_acc = 0
    model.eval()
    with torch.no_grad():
        for inputs in test_loader:
            skeleton0 = inputs[-1][0].to(device).float()
            skeleton1 = inputs[-1][1].to(device).float()
            label = inputs[-1][2].to(device).long()

            sp_skeleton0 = torch.rand([batch_size, len(cs), 300])
            sp_skeleton1 = torch.rand([batch_size, len(cs), 300])
            for b in range(batch_size):
                sequence_idx = 0
                for j in cs:
                    sp_skeleton0[b][sequence_idx] = skeleton0[b][:, 3 * (j - 1):3 * j].reshape(1, 300)
                    sp_skeleton1[b][sequence_idx] = skeleton1[b][:, 3 * (j - 1):3 * j].reshape(1, 300)
                    sequence_idx += 1
            sp_skeleton0 = sp_skeleton0.to(device).float()
            sp_skeleton1 = sp_skeleton1.to(device).float()

            pre = model(skeleton0, skeleton1, sp_skeleton0, sp_skeleton1)
            test_acc += np.sum(np.argmax(pre.cpu().detach().numpy(), axis=1) == label.cpu().detach().numpy())

    return test_acc / dataset.__len__()


if __name__ == "__main__":
    model_path = ""
    # model_path为模型的地址
    test_dataset_path = "h5/NTURGB-D_test.h5"
    batch_size = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TwoStreamRNN()
    model.load_state_dict(torch.load(model_path))
    test_data = TestDataset(test_dataset_path)
    test_loader = DataLoader(test_data, batch_size=batch_size,
                             num_workers=0, shuffle=False, drop_last=True)
    print("test_acc: ", str(test(model, batch_size, device, test_data, test_loader)))
