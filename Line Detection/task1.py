###############
##Design the function "findRotMat" to  return 
# 1) rotMat1: a 2D numpy array which indicates the rotation matrix from xyz to XYZ 
# 2) rotMat2: a 2D numpy array which indicates the rotation matrix from XYZ to xyz 
###############

import numpy as np
import cv2

def findRotMat(alpha, beta, gamma):
    #......
    Rotate_z1 = np.array([[np.cos(np.deg2rad(alpha)),-np.sin(np.deg2rad(alpha)),0],
                        [(np.sin(np.deg2rad(alpha))),np.cos(np.deg2rad(alpha)),0],
                        [0,0,1]])
    Rotate_x2 = np.array([[1,0,0],
                        [0,np.cos(np.deg2rad(beta)),-np.sin(np.deg2rad(beta))],
                        [0,(np.sin(np.deg2rad(beta))),np.cos(np.deg2rad(beta))]])

    Rotate_z3 = np.array([[np.cos(np.deg2rad(gamma)),-np.sin(np.deg2rad(gamma)),0],
                        [(np.sin(np.deg2rad(gamma))),np.cos(np.deg2rad(gamma)),0],
                        [0,0,1,]])

    Rotate_Z1 = np.array ([[np.cos(np.deg2rad(360-alpha)),-np.sin(np.deg2rad(360-alpha)),0],
                           [np.sin(np.deg2rad(360-alpha)),np.cos(np.deg2rad(360-alpha)),0],
                           [0, 0, 1]])

    Rotate_X2 = np.array ([[1, 0, 0,],
                           [0, np.cos(np.deg2rad(360-beta)),-np.sin(np.deg2rad (360-beta))],
                           [0, np.sin(np.deg2rad(360-beta)),np.cos(np.deg2rad (360-beta))]])

    Rotate_Z3 = np.array ([[np.cos (np.deg2rad(360-gamma)),-np.sin(np.deg2rad(360-gamma)),0],
                           [(np.sin(np.deg2rad(360-gamma))),np.cos(np.deg2rad(360-gamma)),0],
                           [0, 0, 1, ]])

    xyz_to_XYZ = np.dot(np.dot(Rotate_z3,Rotate_x2),Rotate_z1)
    XYZ_to_xyz = np.dot(np.dot(Rotate_Z1,Rotate_X2),Rotate_Z3)

    # print(np.cos(45*np.pi/180))
    # print(np.cos(np.deg2rad(45)))

    return xyz_to_XYZ, XYZ_to_xyz


if __name__ == "__main__":
    alpha = 45
    beta = 30
    gamma = 50
    rotMat1, rotMat2 = findRotMat(alpha, beta, gamma)
    print("xyz_to_XYZ: \n",rotMat1)
    print("XYZ_to_xyz: \n",rotMat2)