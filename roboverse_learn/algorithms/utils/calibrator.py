import numpy as np
import cv2, PIL, os
from cv2 import aruco
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
%matplotlib nbagg
from roboverse_learn.algorithms.utils.multi_realsense import MultiRealsenseWrapper
import os
import time
from PIL import Image

def calibrateKd(im_fpath, aruco_len=60.0):
    (w, h) = (6,4)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
    objp = np.zeros((w*h,3), np.float32)
    objp[:,:2] = np.mgrid[0:w,0:h].T.reshape(-1,2)
    objp = objp*aruco_len
    objpoints,imgpoints = [],[]
    images = glob.glob(f'{im_fpath}/*')
    for fname in images:
        img = cv2.imread(fname)
        print('r %s'%(fname),end='')
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (w,h),None)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)
            img = cv2.drawChessboardCorners(img, (6,4), corners2, ret)

    ret,mtx,dist,rvecs,tvecs = cv2.calibrateCamera(objpoints,imgpoints,gray.shape[::-1],None,None)
    tot_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        tot_error += error
    print ('total error: ', tot_error/len(objpoints))
    return mtx, dist


def take_picture_and_save(real_sense_cam, save_dir):
    cam_dict = real_sense_cam()
    rgb = cam_dict["camera0"]["rgb"]
    if not rgb.shape[-1] == 3 or not len(rgb.shape) == 3 :
        raise ValueError(f"RGB images must have the correct shape, got {rgb.shape}")
    # ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # generate a timestamped filename
    timestamp = int(time.time() * 1000)
    filename = f"rgb_{timestamp}.png"
    filepath = os.path.join(save_dir, filename)

    # save the RGB image
    Image.fromarray(rgb).save(filepath)
    print(f"Saved image to {filepath}")

def read_chessboards(images):
    """
    Charuco base pose estimation.
    """
    print("POSE ESTIMATION STARTS:")
    allCorners = []
    allIds = []
    decimator = 0
    # SUB PIXEL CORNER DETECTION CRITERION
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)

    for im in images:
        print("=> Processing image {0}".format(im))
        frame = cv2.imread(im)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict)

        if len(corners)>0:
            # SUB PIXEL DETECTION
            for corner in corners:
                cv2.cornerSubPix(gray, corner,
                                 winSize = (3,3),
                                 zeroZone = (-1,-1),
                                 criteria = criteria)
            res2 = cv2.aruco.interpolateCornersCharuco(corners,ids,gray,board)
            if res2[1] is not None and res2[2] is not None and len(res2[1])>3 and decimator%1==0:
                allCorners.append(res2[1])
                allIds.append(res2[2])

        decimator+=1

    imsize = gray.shape
    return allCorners,allIds,imsize


def main():
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    board = aruco.CharucoBoard_create(9, 14, 2, 1.5, aruco_dict)
    real_sense_cam = MultiRealsenseWrapper()
    save_dir = "/home/balen/Projects/yktang/RoboVerse/tmp/PickCube/diffusion_policy/franka/L0/Calibration"
    os.makedirs(save_dir, exist_ok=True)
    take_picture_and_save(real_sense_cam, save_dir)
    images = np.array([save_dir + f for f in os.listdir(save_dir) if f.endswith(".png") ])
    order = np.argsort([int(p.split(".")[-2].split("_")[-1]) for p in images])
    images = images[order]
