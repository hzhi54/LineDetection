###############
##Design the function "calibrate" to  return 
# (1) intrinsic_params: should be a list with four elements: [f_x, f_y, o_x, o_y], where f_x and f_y is focal length, o_x and o_y is offset;
# (2) is_constant: shoulpd be bool data type. False if the intrinsic parameters differed from world coordinates.
#                                            True if the intrinsic parameters are invariable.
###############
import numpy as np
from cv2 import imread, cvtColor, COLOR_BGR2GRAY, TERM_CRITERIA_EPS, TERM_CRITERIA_MAX_ITER, \
    findChessboardCorners, cornerSubPix, drawChessboardCorners, imshow, waitKey, circle

def calibrate(imgname):
    #......
    criteria = (TERM_CRITERIA_EPS + TERM_CRITERIA_MAX_ITER, 30, 0.001)

    img = imread(imgname)
    img_left = img[:,0:1001]
    img_left_gray = cvtColor(img_left, COLOR_BGR2GRAY)
    img_left_points = []
    ret, corners = findChessboardCorners(img_left_gray,(4,4),None)
    if ret == True:
        corners2 = cornerSubPix(img_left_gray,corners,(5,5),(-1,-1),criteria)
        img_left_points.append(corners2)
        temp = []
        for p in img_left_points[0]:
            for t in p:
                temp.append(t)
        img_left_points = temp

    world_left = [[0,20,20,1],[0,15,20,1],[0,10,20,1],[0,5,20,1],
                  [0,20,15,1],[0,15,15,1],[0,10,15,1],[0,5,15,1],
                  [0,20,10,1],[0,15,10,1],[0,10,10,1],[0,5,10,1],
                  [0,20,5,1],[0,15,5,1],[0,10,5,1],[0,5,5,1]]

    img_right = img[:,1001:]
    img_right_gray = cvtColor(img_right, COLOR_BGR2GRAY)
    img_right_points = []
    ret, corners = findChessboardCorners(img_right_gray,(4,4),None)
    if ret == True:
        corners2 = cornerSubPix(img_right_gray,corners,(5,5),(-1,-1),criteria)
        img_right_points.append(corners2)
        temp = []
        for p in img_right_points[0]:
            for t in p:
                t[0] += len(img[0])/2
                temp.append(t)

        img_right_points = temp

    world_right = [[20,0,20,1],[20,0,15,1],[20,0,10,1],[20,0,5,1],
                   [15,0,20,1],[15,0,15,1],[15,0,10,1],[15,0,5,1],
                   [10,0,20,1],[10,0,15,1],[10,0,10,1],[10,0,5,1],
                   [5,0,20,1],[5,0,15,1],[5,0,10,1],[5,0,5,1]]

    world_cord = np.asarray(world_left + world_right)
    camera_cord = np.asarray(img_left_points + img_right_points)
    camera_cord = np.insert(camera_cord,int(2),[1],axis=1)
    A_matrix = []
    for idx in range(0,len(world_cord)):
        X,Y,Z,xi,yi = world_cord[idx][0], world_cord[idx][1], world_cord[idx][2], camera_cord[idx][0], camera_cord[idx][1]
        temp1 = [X,Y,Z,1,0,0,0,0,-xi*X,-xi*Y,-xi*Z,-xi]
        temp2 = [0,0,0,0,X,Y,Z,1,-yi*X,-yi*Y,-yi*Z,-yi]
        A_matrix.append(temp1)
        A_matrix.append(temp2)
    u, x, v = np.linalg.svd(A_matrix)
    x = v[len(v)-1]
    lamb = np.sqrt(1/(x[8]**2+x[9]**2+x[10]**2))
    m = lamb * x
    # print(m)
    m1 = np.array([m[0],m[1],m[2]])
    m2 = np.array([m[4],m[5],m[6]])
    m3 = np.array([m[8],m[9],m[10]])
    ox = np.dot(np.transpose(m1),m3)
    oy = np.dot(np.transpose(m2),m3)
    fx = np.sqrt(np.dot(np.transpose(m1),m1) - ox**2)
    fy = np.sqrt(np.dot(np.transpose(m2),m2) - oy**2)
    retVal = [fx,fy,ox,oy]

    # test
    # m = np.reshape(m,(3,4))
    # test = np.dot([0,0,10,1],np.transpose(m))
    # a = int(test[0]/test[2])
    # b = int(test[1]/test[2])
    # img1 = circle(img,(a,b),4,(255,0,0),-1)
    # imshow('',img1)
    # waitKey(100000000)

    return retVal, False

if __name__ == "__main__":
    intrinsic_params, is_constant = calibrate('checkboard.png')
    print(intrinsic_params, is_constant)