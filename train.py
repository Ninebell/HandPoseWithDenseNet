import os

import datetime

from multiprocessing import Process, Lock

import torch
from torch_module.losses import pixel_logistic_focal_loss
import random

import matplotlib.pyplot as plt


from torch_module.utils import train_model
import numpy as np
from PIL import Image
from model import DenseUNet, DenseUNetFilter

import h5py

root_path = 'result'
# root_path = 'D:\\dataset\\multiview_hand_pose_dataset_uploaded_v2'


def read_points(path):
    files = open(path, 'r')
    joints = [joint.strip().split(' ')[1:] for joint in files.readlines()]
    for j_idx in range(len(joints)):
        for i in range(len(joints[j_idx])):
            joints[j_idx][i] = float(joints[j_idx][i])
    return np.asarray(joints)


def image_encode(image):
    image = image.resize((256, 192))
    image = np.asarray(image)/255
    image = np.moveaxis(image, -1, 0)
    return image


def joint_2_heat_map(joint, shape):
    joint[0] = int(joint[0] * shape[0])
    joint[1] = int(joint[1] * shape[1])
    base = np.zeros(shape)
    for r in range(shape[0]):
        for c in range(shape[1]):
            base[r, c] = np.exp(-((joint[0]-c)**2 + (joint[1]-r)**2)/5)
    return base


def Save_IMG(target, predict, is_train, x):
    with torch.no_grad():
        t_c = target[0].detach().cpu().numpy()[0,:,:,:]
        p_c = predict[0].detach().cpu().numpy()[0,:,:,:]
        x = x.detach().cpu().numpy()[0,:,:,:]
        x = np.moveaxis(x,0,-1)
        base_t = t_c[0]
        base_p = p_c[0]
        for t, p in zip(t_c, p_c):
            base_t = np.maximum(base_t, t)
            base_p = np.maximum(base_p, p)
        plt.subplot(3, 1, 1)
        plt.imshow(base_t)
        plt.subplot(3, 1, 2)
        plt.imshow(base_p)
        plt.subplot(3, 1, 3)
        plt.imshow(x)
        img_path = 'D:\\dataset\\multiview_hand_pose_dataset_uploaded_v2\\result\\test_image'
        plt.savefig('{0}\\{2}_result_{1}.png'.format(img_path, datetime.datetime.now().strftime('%H_%M_%S_%f'), is_train))
        plt.close()
    return 0


def data_generator(conf):
    def joint_scale(joint_path, scale):
        joint = read_points(joint_path)
        joint = joint / scale
        return joint

    batch = conf['batch']
    xs = conf['x']
    ys = conf['y']

    all_len = len(xs)
    iter_len = all_len // batch
    for iter in range(iter_len):
        x = []
        y = []
        for b in range(batch):
            b_idx = iter*batch + b
            x.append(xs[b_idx])
            y.append(ys[b_idx])

        x = np.asarray(x)
        x = torch.from_numpy(x).type(torch.FloatTensor).cuda()

        y = np.asarray(y)
        y = torch.from_numpy(y).type(torch.FloatTensor).cuda()

        yield x, y


def data_generator(conf):
    def joint_scale(joint_path, scale):
        joint = read_points(joint_path)
        joint = joint / scale
        return joint

    batch = conf['batch']
    file_list = conf['list']
    root_path = conf['root']

    all_len = len(file_list)
    iter_len = all_len // batch
    for iter in range(iter_len):
        x = []
        point_2d = []
        point_3d = []
        for b in range(batch):
            b_idx = iter*batch + b
            input_img = image_encode(Image.open(os.path.join(root_path+'\\augmented_samples', file_list[b_idx][0])))
            x.append(input_img)

            joint2 = []

            for j in range(21):

                if conf['is_train'] == 0:
                    path = '{0}\\{1}\\{2}\\'.format('{0}'.format(root_path), 'train\\heat_map', j)
                elif conf['is_train'] == 1:
                    path = '{0}\\{1}\\{2}\\'.format('{0}'.format(root_path), 'validate\\heat_map', j)
                elif conf['is_train'] == 2:
                    path = '{0}\\{1}\\{2}\\'.format('{0}'.format(root_path), 'test\\heat_map', j)

                img = Image.open(path+file_list[b_idx][1].split('.')[0]+'.png').convert('L')
                img = np.asarray(img)
                img = img/255
                joint2.append(img)

            joint2 = np.asarray(joint2)

            joint3 = joint_scale(os.path.join(root_path+'\\projections_3d', file_list[b_idx][2]), np.asarray([500,500,1000]))
            joint3 = joint3 - joint3[-1, :]
            joint3 = np.reshape(joint3, (63,))
            point_2d.append(joint2)
            point_3d.append(joint3)

        x = np.asarray(x)
        point_2d = np.asarray(point_2d)
        point_3d = np.asarray(point_3d)

        yield x, (point_2d, point_3d)


def custom_loss(target, predict):
    shape = target[0][0,0].shape
    pixel_loss = pixel_logistic_focal_loss(target[0], predict[0], shape=shape)
    mse = torch.mean(torch.pow(target[1] - predict[1], 2))
    return pixel_loss + 100*mse


def calc_mae_px(target, result):
    with torch.no_grad():
        target = target.cpu().numpy() * np.asarray((640,480))
        result = result.cpu().numpy() * np.asarray((640,480))
        return np.mean(np.abs(target - result))


def check_point(model, train, validate):
    if validate > check_point.validate:
        check_point.validate = validate
        Path = root_path+'\\{0}'.format(train)
        os.makedirs(Path, exist_ok=True)
        torch.save(model.state_dict(), Path+'\\model.dict')


check_point.validate = 0


def file_list_load(path):
    file_path_list = open(path, 'r')
    file_list = [lines.strip().split(',') for lines in file_path_list.readlines()]
    return file_list


def PCK_2D(target, predict, t, x):
    with torch.no_grad():
        t_np = target[0].cpu().numpy()
        p_np = predict[0].cpu().numpy()
        t_flat = np.reshape(t_np, (target[0].shape[0], 21, -1))
        p_flat = np.reshape(p_np, (target[0].shape[0], 21, -1))
        t_idx = np.argmax(t_flat, axis=2)
        p_idx = np.argmax(p_flat, axis=2)
        t_idx = np.array([t_idx//64, t_idx%64])
        p_idx = np.array([p_idx//64, p_idx%64])
        dist = np.power(t_idx - p_idx, 2)
        dist = np.sqrt(dist[0]+dist[1])
        dist = np.mean(dist < 3)

    return dist


def PCK_3D(target, predict, t, x):
    with torch.no_grad():
        joint_3_p = predict[1][0].cpu().detach().numpy()
        joint_3_p = np.reshape(joint_3_p, (20, 3))

        joint_3_t = target[1][0].cpu().detach().numpy()
        joint_3_t = np.reshape(joint_3_t, (20, 3))

        x_3_p = joint_3_p[:-1, 0]
        y_3_p = joint_3_p[:-1, 1]
        z_3_p = joint_3_p[:-1, 2]

        x_3_t = joint_3_t[:-1, 0]
        y_3_t = joint_3_t[:-1, 1]
        z_3_t = joint_3_t[:-1, 2]

        dist_p = np.sqrt(np.square(x_3_p) + np.square(y_3_p) + np.square(z_3_p))
        dist_t = np.sqrt(np.square(x_3_t) + np.square(y_3_t) + np.square(z_3_t))

        dist_percent = dist_p / dist_t
        dist_percent = np.where(dist_percent > 0.9, 1, 0)
        return np.mean(dist_percent)


def multi_process_file_load(x, point_2d, point_3d, file_list, conf, idxes, b_idx, lock):
    def joint_scale(joint_path, scale):
        joint = read_points(joint_path)
        joint = joint / scale
        return joint
    input_img = image_encode(Image.open(os.path.join(root_path + '\\augmented_samples', file_list[idxes[b_idx]][0])))

    joint2 = []
    for j in range(21):
        if conf['is_train'] == 0:
            path = '{0}\\{1}\\{2}\\'.format('{0}'.format(root_path), 'train\\heat_map', j)
        elif conf['is_train'] == 1:
            path = '{0}\\{1}\\{2}\\'.format('{0}'.format(root_path), 'validate\\heat_map', j)
        elif conf['is_train'] == 2:
            path = '{0}\\{1}\\{2}\\'.format('{0}'.format(root_path), 'test\\heat_map', j)

        img = Image.open(path + file_list[idxes[b_idx]][1].split('.')[0] + '.png').convert('L')
        img = np.asarray(img)
        img = img / 255
        joint2.append(img)

    joint2 = np.asarray(joint2)
    joint2 = np.moveaxis(joint2, -1, 0)

    joint3 = joint_scale(os.path.join(root_path + '\\projections_3d', file_list[idxes[b_idx]][2]),
                         np.asarray([500, 500, 1000]))
    joint3 = joint3 - joint3[-1, :]
    joint3 = np.reshape(joint3, (63,))

    lock.acquire()
    x.append(input_img)
    point_2d.append(joint2)
    point_3d.append(joint3)
    lock.release()


def multi_process_data_load(conf):
    batch = conf['batch']
    file_list = conf['list']

    all_len = len(file_list)
    iter_len = all_len // batch
    idxes = np.arange(iter_len)
    lock = Lock()

    if conf['is_train']:
        random.shuffle(idxes)

    for iter in range(iter_len):
        x = []
        point_2d = []
        point_3d = []
        process_list = []
        for b in range(batch):
            b_idx = iter*batch + b

            p = Process(target=multi_process_file_load, args=((x, point_2d, point_3d, file_list, conf, idxes, b_idx, lock)))
            process_list.append(p)

        for p in process_list:
            p.start()

        for p in process_list:
            p.join()

        x = np.asarray(x)
        x = torch.from_numpy(x).type(torch.FloatTensor).cuda()

        point_2d = np.asarray(point_2d)
        point_3d = np.asarray(point_3d)

        point_2d = torch.from_numpy(point_2d).type(torch.FloatTensor).cuda()
        point_3d = torch.from_numpy(point_3d).type(torch.FloatTensor).cuda()

        yield x, (point_2d, point_3d)


def data_load_from_hdf5(conf):
    batch = conf['batch_size']
    hdf = conf['hdf5']
    Xs = hdf['x']
    joint_2 = hdf['joint_2']
    joint_3 = hdf['joint_3']
    iter_len = len(list(Xs.keys()))
    idxes = np.arange(iter_len)
    if conf['is_train']:
        random.shuffle(idxes)

    iter_len = len(list(Xs.keys()))//batch
    for iter in range(iter_len):
        x = []
        point_2d = []
        point_3d = []
        for b in range(batch):
            b_idx = iter*batch + b
            input_img = Xs['X_{}'.format(idxes[b_idx])][:,:,:]
            input_img = np.moveaxis(input_img, -1, 0)
            x.append(input_img)

            j2 = joint_2['Joint_2_{}'.format(idxes[b_idx])][:,:,:]
            j3 = joint_3['Joint_3_{}'.format(idxes[b_idx])][:-3,]
            j2 = np.moveaxis(j2, -1, 0)
            point_2d.append(j2)
            point_3d.append(j3)

        x = np.asarray(x)
        x = torch.from_numpy(x).type(torch.FloatTensor).cuda()

        point_2d = np.asarray(point_2d)
        point_3d = np.asarray(point_3d)

        point_2d = torch.from_numpy(point_2d).type(torch.FloatTensor).cuda()
        point_3d = torch.from_numpy(point_3d).type(torch.FloatTensor).cuda()

        yield x, (point_2d, point_3d)


def test_model(net, pretrain_path, data_loader):

    if torch.cuda.is_available():
        net.cuda()

    net.load_state_dict(torch.load(pretrain_path))
    net.eval()

    for iter, (x, y) in enumerate(data_loader['loader'](data_loader['conf'])):
        print(x.shape)
        result = net(x)

        PCK_2D(y, result)
        PCK_3D(y, result)

        del result


if __name__ == "__main__":

    net = DenseUNetFilter(using_down=True, using_up=False)

    optim = torch.optim.Adam(net.parameters(), lr=1e-3)

    batch_size = 32

    # train_file_list = file_list_load('train_file_pair.csv')
    # validate_file_list = file_list_load('validate_file_pair.csv')
    #
    # print(len(train_file_list), len(train_file_list)//64)

    save_path = 'result'
    os.makedirs(save_path, exist_ok=True)

    criterion = custom_loss

    train_hf = h5py.File('train_data.hdf5', 'r')
    test_hf = h5py.File('test_data.hdf5', 'r')
    validate_hf = h5py.File('validate_data.hdf5', 'r')

    is_train = True

    if is_train:
        train_model(50, net, custom_loss, optim,
                    {
                        'loader': data_load_from_hdf5,
                        'conf': {'hdf5': train_hf, 'batch_size': batch_size, 'is_train': True}
                     },
                    {
                        'loader': data_load_from_hdf5,
                        'conf': {'hdf5': validate_hf, 'batch_size': batch_size, 'is_train': False}
                     },
                    save_path, 'test', check_point,
                    accuracy=[
                        {'metrics': PCK_2D, 'name': 'PCK_2D'},
                        {'metrics': PCK_3D, 'name': 'PCK_3D'}
                    ]
                )

    if not is_train:
        pretrain_path = 'result\\0.09887924045324326\\model.dict'
        test_model(net, pretrain_path,
                   {
                       'loader': data_load_from_hdf5,
                       'conf': {'hdf5': test_hf, 'batch_size': 1, 'is_train': True}
                   })


