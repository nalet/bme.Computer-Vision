import os

import numpy as np
from PIL import Image


def load_merton_college_data(folder):
    im1 = np.array(Image.open(os.path.join(folder, 'images/001.jpg')))
    im2 = np.array(Image.open(os.path.join(folder, 'images/002.jpg')))

    # load 2D points for each view to a list
    points_2d = [np.loadtxt(os.path.join(folder, f'2D/00{i + 1:d}.corners')).T for i in range(3)]

    # load 3D points
    points_3d = np.loadtxt(os.path.join(folder, '3D/p3d')).T

    # load correspondences
    corr = np.genfromtxt(os.path.join(folder, '2D/nview-corners'), dtype=int, missing_values='*')

    # create cameras
    P = [np.loadtxt(os.path.join(folder, f'2D/00{i + 1:d}.P')) for i in range(3)]

    return im1, im2, points_2d, points_3d, corr, P


def load_homogeneous_coordinates(data_dir):
    im1, im2, points_2d, _, corr, _ = load_merton_college_data(data_dir)

    # index for visible points in first two views
    ndx = (corr[:, 0] >= 0) & (corr[:, 1] >= 0)

    # get coordinates and make homogeneous
    x1 = points_2d[0][:, corr[ndx, 0]]
    x1 = np.vstack((x1, np.ones(x1.shape[1])))
    x2 = points_2d[1][:, corr[ndx, 1]]
    x2 = np.vstack((x2, np.ones(x2.shape[1])))

    return im1, im2, x1, x2


def _plot_points(xy, img):
    import cv2

    for i, (x, y) in enumerate(zip(xy[0, :], xy[1, :])):
        if i % 7 == 0:
            cv2.circle(img, (int(x), int(y)), 5, (255, 0, 255), -1)


def plot_matched_features():
    import cv2

    data_dir = './merton_college/'
    _, _, x1, x2 = load_homogeneous_coordinates(data_dir)
    img1 = cv2.imread(os.path.join(data_dir, 'images/001.jpg'))
    img2 = cv2.imread(os.path.join(data_dir, 'images/002.jpg'))

    _plot_points(x1, img1)
    _plot_points(x2, img2)
    vis = np.concatenate((img1, img2), axis=1)
    for i, (x, y, xx, yy) in enumerate(zip(x1[0, :], x1[1, :], x2[0, :], x2[1, :])):
        if i % 7 == 0:
            cv2.line(vis, (int(x), int(y)), (int(xx + 1024), int(yy)), (255 - 2 * (i // 7), 5 * (i // 7), 0))
    cv2.imwrite('matched_points.png', vis)
