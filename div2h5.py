import glob
import os
import h5py
import numpy as np

dataset_dir = "data/raw_npy/"

train_id = ['01', '02', '04', '05', '08', '09', '13', '14', '15', '16', '17', '18', '19', '25', '27', '28', '31',
            '34', '35', '38']
# 执行者id在train_id列表中的数据为训练数据

f_train = h5py.File("NTURGB-D_train.h5", 'w')
f_test = h5py.File("NTURGB-D_test.h5", 'w')

grp_file_name_train = f_train.create_group("file_name")  # 文件的名称
grp_nbodys_train = f_train.create_group("nbodys")  # 动作的执行人数
grp_skel_body0_train = f_train.create_group("skel_body0")
# dimension of skel_body0: (frame, njoint, 3) ,njoint = 25
grp_skel_body1_train = f_train.create_group("skel_body1")
grp_label_train = f_train.create_group("label")

grp_file_name_test = f_test.create_group("file_name")
grp_nbodys_test = f_test.create_group("nbodys")  # 动作的执行人数
grp_skel_body0_test = f_test.create_group("skel_body0")
# dimension of skel_body0: (frame, njoint, 3) ,njoint = 25
grp_skel_body1_test = f_test.create_group("skel_body1")
grp_label_test = f_test.create_group("label")

im_path = glob.glob(os.path.join(dataset_dir, "*.skeleton.npy"))
im_path.sort()

for i in range(len(im_path)):
    person_id = im_path[i].split('/')[1][18:20]
    label = int(im_path[i].split('/')[1][26:28]) - 1
    # label: 1~60 -> 0~59
    skeleton_data = np.load(im_path[i], allow_pickle=True).item()
    if person_id in train_id:
        # 训练集
        grp_file_name_train.create_dataset(str(i), data=skeleton_data["file_name"])
        grp_label_train.create_dataset(str(i), data=label)
        if skeleton_data["nbodys"][0] == 1:
            # 动作执行者为1人时
            grp_nbodys_train.create_dataset(str(i), data=1)
            if skeleton_data["skel_body0"].shape[0] < 100:
                # 如果frame少于100，将frame补零至100
                temp_data_body0 = np.concatenate((skeleton_data["skel_body0"],
                                       np.zeros((100 - skeleton_data["skel_body0"].shape[0], 25, 3))), axis=0)
                temp_data_body0 = np.reshape(temp_data_body0, (100, 75))
                # (100, 25, 3) -> (100, 75)
                grp_skel_body0_train.create_dataset(str(i), data=temp_data_body0)
                grp_skel_body1_train.create_dataset(str(i), data=temp_data_body0)
            else:
                # 如果frame大于100，需要裁剪至100
                temp_data_body0 = skeleton_data["skel_body0"][0:100, :, :]
                temp_data_body0 = np.reshape(temp_data_body0, (100, 75))
                # (100, 25, 3) -> (100, 75)
                grp_skel_body0_train.create_dataset(str(i), data=temp_data_body0)
                grp_skel_body1_train.create_dataset(str(i), data=temp_data_body0)
        else:
            # 动作的执行者为2人时
            grp_nbodys_train.create_dataset(str(i), data=2)
            if skeleton_data["skel_body0"].shape[0] < 100:
                # 如果frame少于100，将frame补零至100
                temp_data_body0 = np.concatenate((skeleton_data["skel_body0"],
                                       np.zeros((100 - skeleton_data["skel_body0"].shape[0], 25, 3))), axis=0)
                temp_data_body1 = np.concatenate((skeleton_data["skel_body1"],
                                       np.zeros((100 - skeleton_data["skel_body1"].shape[0], 25, 3))), axis=0)
                temp_data_body0 = np.reshape(temp_data_body0, (100, 75))
                temp_data_body1 = np.reshape(temp_data_body1, (100, 75))
                grp_skel_body0_train.create_dataset(str(i), data=temp_data_body0)
                grp_skel_body1_train.create_dataset(str(i), data=temp_data_body1)
            else:
                # 如果frame大于100，需要裁剪至100
                temp_data_body0 = skeleton_data["skel_body0"][0:100, :, :]
                temp_data_body1 = skeleton_data["skel_body1"][0:100, :, :]
                temp_data_body0 = np.reshape(temp_data_body0, (100, 75))
                temp_data_body1 = np.reshape(temp_data_body1, (100, 75))
                grp_skel_body0_train.create_dataset(str(i), data=temp_data_body0)
                grp_skel_body1_train.create_dataset(str(i), data=temp_data_body1)
    else:
        # 测试集
        grp_label_test.create_dataset(str(i), data=label)
        grp_file_name_test.create_dataset(str(i), data=skeleton_data["file_name"])
        if skeleton_data["nbodys"][0] == 1:
            # 动作执行者为1人时
            grp_nbodys_test.create_dataset(str(i), data=1)
            if skeleton_data["skel_body0"].shape[0] < 100:
                # 如果frame少于100，将frame补零至100
                temp_data_body0 = np.concatenate((skeleton_data["skel_body0"],
                                       np.zeros((100 - skeleton_data["skel_body0"].shape[0], 25, 3))), axis=0)
                temp_data_body0 = np.reshape(temp_data_body0, (100, 75))
                grp_skel_body0_test.create_dataset(str(i), data=temp_data_body0)
                grp_skel_body1_test.create_dataset(str(i), data=temp_data_body0)
            else:
                # 如果frame大于100，需要裁剪至100
                temp_data_body0 = skeleton_data["skel_body0"][0:100, :, :]
                temp_data_body0 = np.reshape(temp_data_body0, (100, 75))
                grp_skel_body0_test.create_dataset(str(i), data=temp_data_body0)
                grp_skel_body1_test.create_dataset(str(i), data=temp_data_body0)
        else:
            # 动作的执行者为2人时
            grp_nbodys_test.create_dataset(str(i), data=2)
            if skeleton_data["skel_body0"].shape[0] < 100:
                # 如果frame少于100，将frame补零至100
                temp_data_body0 = np.concatenate((skeleton_data["skel_body0"],
                                       np.zeros((100 - skeleton_data["skel_body0"].shape[0], 25, 3))), axis=0)
                temp_data_body1 = np.concatenate((skeleton_data["skel_body1"],
                                       np.zeros((100 - skeleton_data["skel_body1"].shape[0], 25, 3))), axis=0)
                temp_data_body0 = np.reshape(temp_data_body0, (100, 75))
                temp_data_body1 = np.reshape(temp_data_body1, (100, 75))
                grp_skel_body0_test.create_dataset(str(i), data=temp_data_body0)
                grp_skel_body1_test.create_dataset(str(i), data=temp_data_body1)
            else:
                # 如果frame大于100，需要裁剪至100
                temp_data_body0 = skeleton_data["skel_body0"][0:100, :, :]
                temp_data_body1 = skeleton_data["skel_body1"][0:100, :, :]
                temp_data_body0 = np.reshape(temp_data_body0, (100, 75))
                temp_data_body1 = np.reshape(temp_data_body1, (100, 75))
                grp_skel_body0_test.create_dataset(str(i), data=temp_data_body0)
                grp_skel_body1_test.create_dataset(str(i), data=temp_data_body1)
    print(str(i + 1), '/', len(im_path))
# data = np.load(im_path[i], allow_pickle=True).item()

