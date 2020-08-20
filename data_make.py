import os
import numpy as np
import sys
import h5py
import random
from train import *
import matplotlib.pyplot as plt
from pathlib import Path
from torchvision.utils import save_image

root_path = 'D:\\dataset\\multiview_hand_pose_dataset_uploaded_v2'
proj_2d_path = root_path+'\\projections_2d'
proj_3d_path = root_path+'\\projections_3d'
img_path = root_path+'\\augmented_samples'


def heat_map_create(conf):
    def joint_scale(joint_path, scale):
        joint = read_points(joint_path)
        joint = joint / scale
        return joint

    file_list = conf['list']

    all_len = len(file_list)
    for iter in range(all_len):
        joint2 = joint_scale(os.path.join(proj_2d_path,file_list[iter][1]), np.asarray([480,640]))
        joint2 = np.asarray([joint_2_heat_map(joint, (48, 64)) for joint in joint2])

        for idx, heat_map in enumerate(joint2):
            if conf['is_train'] == 0:
                path = '{0}\\{1}'.format('D:\\dataset\\multiview_hand_pose_dataset_uploaded_v2\\train\\heat_map', idx)
            elif conf['is_train'] == 1:
                path = '{0}\\{1}'.format('D:\\dataset\\multiview_hand_pose_dataset_uploaded_v2\\validate\\heat_map', idx)
            else:
                path = '{0}\\{1}'.format('D:\\dataset\\multiview_hand_pose_dataset_uploaded_v2\\test\\heat_map', idx)

            heat_map = torch.from_numpy(heat_map)
            path = os.path.join(path, '{0}.png'.format(file_list[iter][1].split('.')[0]))
            save_image(heat_map, path)


def get_full_path(root_path):
    file_path = []
    path_dir = os.listdir(root_path)
    for d_p in path_dir:
        middle_path = os.path.join(root_path, d_p)
        file_list = os.listdir(middle_path)
        for file in file_list:
            last_path = os.path.join(middle_path, file)
            idx = last_path.rfind('data')
            file_path.append(last_path[idx:])
    return file_path


def file_save(save_path, img_path, index, beg, end):
    fp = open(save_path, 'w')
    for j in index[beg:end]:
        img = img_path[j]
        split = img.split('\\')
        folder_name = split[0]
        file_order = split[1].split('_')
        proj_2d = '{0}\\{1}_jointsCam_{2}.txt'.format(folder_name,file_order[0], file_order[2].split('.')[0])
        fp.write('{0},{1},{2}\n'.format(img, proj_2d, proj_2d))
    fp.close()


def create_data_pair():
    img_file_path = get_full_path(img_path)
    sorted(img_file_path)

    print(len(img_file_path))

    idx = np.arange(len(img_file_path))
    length = len(idx)
    train_len = int(length/10*7)
    validate_len = int(length/10*2)
    test_len = int(length/10*1)

    random.shuffle(idx)

    beg = 0
    end = train_len
    file_save('train_file_pair.csv', img_file_path, idx, beg, end)

    beg = end
    end = beg + validate_len
    file_save('validate_file_pair.csv', img_file_path, idx, beg, end)

    beg = end
    end = beg + test_len
    file_save('test_file_pair.csv', img_file_path, idx, beg, end)


def create_heat_map():
    train_file_list = file_list_load('train_file_pair.csv')
    validate_file_list = file_list_load('validate_file_pair.csv')
    test_file_list = file_list_load('test_file_pair.csv')

    train_root_folder = os.path.join(root_path, 'train')
    validate_root_folder = os.path.join(root_path, 'validate')
    test_root_folder = os.path.join(root_path, 'test')

    os.makedirs(train_root_folder, exist_ok=True)
    os.makedirs(validate_root_folder, exist_ok=True)
    os.makedirs(test_root_folder, exist_ok=True)

    os.makedirs(os.path.join(train_root_folder, 'heat_map'), exist_ok=True)
    os.makedirs(os.path.join(validate_root_folder, 'heat_map'), exist_ok=True)
    os.makedirs(os.path.join(test_root_folder, 'heat_map'), exist_ok=True)

    for j in range(21):
        os.makedirs(os.path.join(train_root_folder, 'heat_map')+'\\{0}'.format(j), exist_ok=True)
        os.makedirs(os.path.join(validate_root_folder, 'heat_map')+'\\{0}'.format(j), exist_ok=True)
        os.makedirs(os.path.join(test_root_folder, 'heat_map')+'\\{0}'.format(j), exist_ok=True)
        for i in range(1,22):
            train_heat_map_path = os.path.join(train_root_folder, 'heat_map')+'\\{0}\\data_{1}'.format(j, i)
            os.makedirs(train_heat_map_path, exist_ok=True)

            validate_heat_map_path = os.path.join(validate_root_folder, 'heat_map') + '\\{0}\\data_{1}'.format(j, i)
            os.makedirs(validate_heat_map_path, exist_ok=True)

            test_heat_map_path = os.path.join(test_root_folder, 'heat_map') + '\\{0}\\data_{1}'.format(j, i)
            os.makedirs(test_heat_map_path, exist_ok=True)

    heat_map_create({
        'list': train_file_list,
        'is_train': 0
    })
    heat_map_create({
        'list': validate_file_list,
        'is_train': 1
    })

    heat_map_create({
        'list': test_file_list,
        'is_train': 2
    })


if __name__ == "__main__1":
    with h5py.File('train_data.h5', 'r') as hf:
        print(hf.keys())
        print(hf['joint_3']['Joint_3_0'][:,])
        plt.imshow(hf['joint_2']['Joint_2_0'][:,:,0])
        plt.show()


def h5py_table_create(name, generator, conf):
    with h5py.File(name, 'w') as hf:
        x_g = hf.create_group('x')
        j_2_g = hf.create_group('joint_2')
        j_3_g = hf.create_group('joint_3')
        for i, (x, y) in enumerate(generator(conf)):
            print(i)

            x = np.squeeze(x)
            x = np.moveaxis(x, 0, -1)

            joint_2 = np.squeeze(y[0])
            joint_2 = np.moveaxis(joint_2, 0, -1)

            joint_3 = np.squeeze(y[1])
            joint_3 = np.moveaxis(joint_3, 0, -1)

            Xset = x_g.create_dataset(
                name='X_'+str(i),
                data=x,
                shape=(192, 256, 3),
                maxshape=(192, 256, 3),
                compression='gzip',
                compression_opts=9
            )

            joint_2_set = j_2_g.create_dataset(
                name='Joint_2_'+str(i),
                data=joint_2,
                shape=(48,64,21),
                maxshape=(48,64,21),
                compression="gzip",
                compression_opts=9
            )

            joint_3_set = j_3_g.create_dataset(
                name='Joint_3_'+str(i),
                data=joint_3,
                shape=(63,),
                maxshape=(None, ),
                compression="gzip",
                compression_opts=9
            )


def create_h5py():

    train_file_list = file_list_load('train_file_pair.csv')
    validate_file_list = file_list_load('validate_file_pair.csv')
    test_file_list = file_list_load('test_file_pair.csv')
    batch_size=1

    h5py_table_create('train_data.hdf5', data_generator, {
         'root': 'D:\\dataset\\multiview_hand_pose_dataset_uploaded_v2',
         'is_train': 0,
         'batch': batch_size,
         'list': train_file_list})

    h5py_table_create('validate_data.hdf5', data_generator, {
        'root': 'D:\\dataset\\multiview_hand_pose_dataset_uploaded_v2',
        'is_train': 1,
        'batch': batch_size,
        'list': validate_file_list})

    h5py_table_create('test_data.hdf5', data_generator, {
        'root': 'D:\\dataset\\multiview_hand_pose_dataset_uploaded_v2',
        'is_train': 2,
        'batch': batch_size,
        'list': test_file_list})


if __name__ == "__main___":

    hf = h5py.File('train_data.hdf5', 'r')

    for x, y in data_load_from_hdf5({'hdf5': hf, 'batch_size': 32, 'is_train': True}):
        print(x.shape)
        print(y[0].shape, y[1].shape)

if __name__ == "__main__":
    make_pair = 0
    make_heat_map = 1
    make_h5 = 2

    data_make_type = make_h5
    if data_make_type == make_pair:
        create_data_pair()
    elif data_make_type == make_heat_map:
        create_heat_map()
    elif data_make_type == make_h5:
        create_h5py()


