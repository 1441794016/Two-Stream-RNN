import h5py
import torch.utils.data as data
import torchvision.transforms as transforms
import torch
import numpy as np


class TrainDataset(data.Dataset):
    def __init__(self, path):
        # path 为训练集的.h5文件的路径
        super(TrainDataset, self).__init__()

        h5f = h5py.File(path, 'r')

        self.label = [v[()] for v in h5f["label"].values()]
        # 此时label为python int类型列表,如[1,1,2,5,6,....]
        self.nbodys = [v[()] for v in h5f["nbodys"].values()]
        # nbodys:动作的执行人数 [1,1,1,1,1,2,1,1,...]
        self.skeleton0 = [v[:] for v in h5f["skel_body0"].values()]
        # skeleton0是一个list，每个元素是(100, 75)大小的ndarray
        self.skeleton1 = [v[:] for v in h5f["skel_body1"].values()]
        # skeleton1是一个list，每个元素是(100, 75)大小的ndarray
        h5f.close()
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __getitem__(self, item):
        index = [(self.skeleton0[item], self.skeleton1[item], self.label[item])]
        # 注意需要加()把index的元素变成一个元组，否则下一步中for skeleton0, skeleton1, label in index 将报错
        return [(torch.from_numpy(skeleton0), torch.from_numpy(skeleton1), torch.from_numpy(np.array(label)))
                for skeleton0, skeleton1, label in index]

    def __len__(self):
        return len(self.label)


class TestDataset(data.Dataset):
    def __init__(self, path):
        # path 为测试集的.h5文件地址
        super(TestDataset, self).__init__()

        h5f = h5py.File(path, 'r')

        self.label = [v[()] for v in h5f["label"].values()]
        self.skeleton0 = [v[:] for v in h5f["skel_body0"].values()]
        # skeleton0是一个list，每个元素是(100, 75)大小的ndarray
        self.skeleton1 = [v[:] for v in h5f["skel_body1"].values()]
        # skeleton1是一个list，每个元素是(100, 75)大小的ndarray
        h5f.close()

    def __getitem__(self, item):
        index = [(self.skeleton0[item], self.skeleton1[item], self.label[item])]
        # 注意需要加()把index的元素变成一个元组，否则下一步中for skeleton0, skeleton1, label in index 将报错
        return [(torch.from_numpy(skeleton0), torch.from_numpy(skeleton1), torch.from_numpy(np.array(label)))
                for skeleton0, skeleton1, label in index]

    def __len__(self):
        return len(self.label)
