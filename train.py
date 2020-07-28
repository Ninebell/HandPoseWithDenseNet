import os

import torch
import torch.nn as nn

from torch_module.utils import train_model
import numpy as np
from PIL import Image
from model import NDenseNet

root_path = 'C:\\Users\\rnwhd\\Desktop\\git\\HandPoseWithDenseNet\\result'


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


def data_generator(conf):
    batch = conf['batch']
    file_list = conf['list']
    def joint_scale(joint_path, scale):
        joint = read_points(joint_path)
        joint = joint / scale
        return joint

    all_len = len(file_list)
    iter_len = all_len // batch
    for iter in range(iter_len):
        x = []
        point_2d = []
        point_3d = []
        for b in range(batch):
            b_idx = iter*batch + b
            input_img = image_encode(Image.open(file_list[b_idx][0]))
            x.append(input_img)
            joint2 = joint_scale(file_list[b_idx][1], np.asarray([640,480]))
            joint3 = joint_scale(file_list[b_idx][2], np.asarray([500,500,1000]))
            joint3 = joint3 - joint3[-1, :]
            point_2d.append(joint2)
            point_3d.append(joint3)

        x = np.asarray(x)
        x = torch.from_numpy(x).type(torch.FloatTensor).cuda()

        point_2d = np.asarray(point_2d)
        point_2d = torch.from_numpy(point_2d).type(torch.FloatTensor).cuda()

        point_3d = np.asarray(point_3d)
        point_3d = torch.from_numpy(point_3d).type(torch.FloatTensor).cuda()
        yield x, (point_2d, point_3d)


def custom_loss(target, predict):
    return torch.mean(torch.abs(target[0] - predict[0])) + torch.mean(torch.abs(target[1] - predict[1]))


def calc_mae_px(target, result):
    with torch.no_grad():
        target = target.cpu().numpy() * np.asarray((640,480))
        result = result.cpu().numpy() * np.asarray((640,480))
        return np.mean(np.abs(target - result))


def check_point(model, train, validate):
    if validate < check_point.validate:
        check_point.validate = validate
        Path = root_path+'\\{0}'.format(train)
        os.makedirs(Path, exist_ok=True)
        torch.save(model.state_dict(), Path+'\\model.dict')


check_point.validate = 100


def file_list_load(path):
    file_path_list = open(path, 'r')
    file_list = [lines.strip().split(',') for lines in file_path_list.readlines()]
    return file_list


if __name__ == "__main__":
    net = NDenseNet()

    is_cuda = torch.cuda.is_available()

    if is_cuda:
        net = net.cuda()

    optim = torch.optim.Adam(net.parameters(), lr=1e-3)

    batch_size = 64

    train_file_list = file_list_load('train_file_pair.csv')
    validate_file_list = file_list_load('validate_file_pair.csv')

    print(len(train_file_list), len(train_file_list)//64)


    save_path = '.\\result'
    os.makedirs(save_path, exist_ok=True)

    train_model(100, net, custom_loss, optim,
                {'loader': data_generator,
                 'conf': {
                     'batch': batch_size,
                     'list': train_file_list}},
                {'loader': data_generator,
                 'conf': {
                     'batch': batch_size,
                     'list': validate_file_list}},
                save_path, 'test', check_point)
