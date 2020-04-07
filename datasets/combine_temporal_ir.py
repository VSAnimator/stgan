import os
import numpy as np
import cv2
import argparse

parser = argparse.ArgumentParser('create image pairs')
parser.add_argument('--fold_A', dest='fold_A', help='input directory for image A', type=str, default='../dataset/50kshoes_edges')
parser.add_argument('--fold_B', dest='fold_B', help='input directory for image B', type=str, default='../dataset/50kshoes_jpg')
parser.add_argument('--fold_AB', dest='fold_AB', help='output directory', type=str, default='../dataset/test_AB')
parser.add_argument('--num_imgs', dest='num_imgs', help='number of images', type=int, default=1000000)
args = parser.parse_args()

for arg in vars(args):
    print('[%s] = ' % arg, getattr(args, arg))

splits = os.listdir(args.fold_A)

for sp in splits:
    img_fold_A = os.path.join(args.fold_A, sp)
    img_fold_B = os.path.join(args.fold_B, sp)
    img_list = os.listdir(img_fold_B)

    num_imgs = min(args.num_imgs, len(img_list))
    print('split = %s, use %d/%d images' % (sp, num_imgs, len(img_list)))
    img_fold_AB = os.path.join(args.fold_AB, sp)
    if not os.path.isdir(img_fold_AB):
        os.makedirs(img_fold_AB)
    print('split = %s, number of images = %d' % (sp, num_imgs))
    for n in range(num_imgs):
        name_B = img_list[n]
        path_B = os.path.join(img_fold_B, name_B)
        name_A0 = name_B[:-4] + '_0.jpg' # Remove .jpg, add rest
        name_A1 = name_B[:-4] + '_1.jpg' # Remove .jpg, add rest
        name_A2 = name_B[:-4] + '_2.jpg' # Remove .jpg, add rest
        name_A0_ir = name_B[:-4] + '_0_ir.jpg' # Remove .jpg, add rest
        name_A1_ir = name_B[:-4] + '_1_ir.jpg' # Remove .jpg, add rest
        name_A2_ir = name_B[:-4] + '_2_ir.jpg' # Remove .jpg, add rest
        path_A0 = os.path.join(img_fold_A, name_A0)
        path_A1 = os.path.join(img_fold_A, name_A1)
        path_A2 = os.path.join(img_fold_A, name_A2)
        path_A0_ir = os.path.join(img_fold_A, name_A0_ir)
        path_A1_ir = os.path.join(img_fold_A, name_A1_ir)
        path_A2_ir = os.path.join(img_fold_A, name_A2_ir)
        # print(path_A0)
        if os.path.isfile(path_A0) and os.path.isfile(path_A1) and os.path.isfile(path_A2) and os.path.isfile(path_B):
            name_AB = name_B
            path_AB = os.path.join(img_fold_AB, name_AB)
            im_A0 = cv2.imread(path_A0, cv2.IMREAD_COLOR)
            im_A1 = cv2.imread(path_A1, cv2.IMREAD_COLOR)
            im_A2 = cv2.imread(path_A2, cv2.IMREAD_COLOR)
            im_A0_ir = cv2.imread(path_A0, cv2.IMREAD_GRAYSCALE)[:,:,np.newaxis]
            im_A1_ir = cv2.imread(path_A1, cv2.IMREAD_GRAYSCALE)[:,:,np.newaxis]
            im_A2_ir = cv2.imread(path_A2, cv2.IMREAD_GRAYSCALE)[:,:,np.newaxis]
            im_ir = np.concatenate([im_A0_ir, im_A1_ir, im_A2_ir], 2)
            im_B = cv2.imread(path_B, cv2.IMREAD_COLOR)
            im_AB = np.concatenate([im_A0, im_A1, im_A2, im_ir, im_B], 1)
            cv2.imwrite(path_AB, im_AB)
