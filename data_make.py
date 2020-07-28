import os
import numpy as np
import random

root_path = 'D:\\dataset\\multiview_hand_pose_dataset_uploaded_v2'
proj_2d_path = root_path+'\\projections_2d'
proj_3d_path = root_path+'\\projections_3d'
img_path = root_path+'\\augmented_samples'


def get_fool_path(root_path):
    file_path = []
    path_dir = os.listdir(root_path)
    for d_p in path_dir:
        middle_path = os.path.join(root_path, d_p)
        file_list = os.listdir(middle_path)
        for file in file_list:
            last_path = os.path.join(middle_path, file)
            file_path.append(last_path)
    return file_path


def file_save(save_path, img_path, proj_2d_path, proj_3d_path, index, beg, end):
    fp = open(save_path, 'w')
    for j in index[beg:end]:
        img = img_path[j]
        proj_2d = proj_2d_path[j]
        proj_3d = proj_3d_path[j]
        fp.write('{0},{1},{2}\n'.format(img, proj_2d, proj_3d))
    fp.close()


if __name__ == "__main__":
    img_file_path = get_fool_path(img_path)
    proj_2d_file_path = get_fool_path(proj_2d_path)
    proj_3d_file_path = get_fool_path(proj_3d_path)

    idx = np.arange(len(img_file_path))
    length = len(idx)
    train_len = int(length/10*7)
    validate_len = int(length/10*2)
    test_len = int(length/10*1)

    random.shuffle(idx)

    beg = 0
    end = train_len
    file_save('train_file_pair.csv', img_file_path, proj_2d_file_path, proj_3d_file_path, idx, beg, end)

    beg = end
    end = beg + validate_len
    file_save('validate_file_pair.csv', img_file_path, proj_2d_file_path, proj_3d_file_path, idx, beg, end)

    beg = end
    end = beg + test_len
    file_save('test_file_pair.csv', img_file_path, proj_2d_file_path, proj_3d_file_path, idx, beg, end)

