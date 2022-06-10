import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pandas as pd
import cv2
import time
from PIL import Image
import sys
sys.dont_write_bytecode = True

num = 3
WINDOW_WIDTH = 420
WINDOW_HEIGHT = 330
left = -100
right = 320
up = 160
bottom = -170
high = 1150
low = 800

def pointcloud_process(pcd_matrix, flat):
    pcd_matrix2 = pcd_matrix[(pcd_matrix[:,0]>left)&(pcd_matrix[:,0]< right) & (pcd_matrix[:, 1]>bottom)&(pcd_matrix[:, 1]<up)&(pcd_matrix[:, 2]>low)&(pcd_matrix[:, 2]<high)]
    pcd_matrix3 = pcd_matrix2.copy()
    pcd_matrix2[:, 0] += -left
    pcd_matrix2[:, 1] += -bottom

    depth, matrix_image = xyz2matrix(pcd_matrix2)
    depth = depth - flat
    depth[depth<3] = 0
    depth[0:10] = 0
    depth[:, 400:420] = 0
    depth[:, 0:10] = 0
    depth[depth>200] = 0

    return depth, matrix_image, pcd_matrix3

def PtoD(depth):
    MAX_DEPTH = depth.max()
    if MAX_DEPTH > 100:
        MAX_DEPTH = 100
    MIN_DEPTH = 0
    # MAX_DEPTH = 100

    for i, row in enumerate(depth):
        for j, z in enumerate(row):
            if z <= MAX_DEPTH:
                depth[i][j] = 255*((z-MIN_DEPTH)/(MAX_DEPTH-MIN_DEPTH))
            elif z > MAX_DEPTH:
                depth[i][j] = 255
    
    return depth, MAX_DEPTH

def xyz2matrix(matrix):
    depth = np.zeros((WINDOW_HEIGHT*num, WINDOW_WIDTH*num))
    matrix_image = np.zeros((WINDOW_HEIGHT*num, WINDOW_WIDTH*num))
    for i, xyz in enumerate(matrix):
        x = int(xyz[1]*num)
        y = int(xyz[0]*num)
        depth[x][y] = high - xyz[2]
        matrix_image[x][y] = i
    depth, matrix_image = max_pooling(depth, matrix_image)

    return depth, matrix_image

def max_pooling(depth, matrix_image):
    small_depth = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH))
    matrix_small_image = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH))
    for x in range(WINDOW_HEIGHT):
        for y in range(WINDOW_WIDTH):
            kernel = depth[x*3:x*3+3, y*3:y*3+3]
            max_num = np.max(kernel)
            index = np.argwhere(kernel == max_num)[0]
            small_depth[x][y] = max_num
            matrix_small_image[x][y] = matrix_image[x*3+index[0]][y*3+index[1]]
    
    return small_depth, matrix_small_image
 
def main(filepath):
    start = time.time()
    flatpath = "/home/takasu/ダウンロード/wire-test/result/flat.csv"
    flat_list = pd.read_csv(flatpath, header=None).values
    pcd = o3d.io.read_point_cloud(filepath)
    pcd_matrix = np.asanyarray(pcd.points)

    depth, matrix_image, matrix = pointcloud_process(pcd_matrix, flat_list)

    depth, MAX_DEPTH = PtoD(depth)
    depth[depth < 0] = 0

    # from PIL import Image
    # pil_depth = Image.fromarray(depth)
    # pil_depth = pil_depth.rotate(2)
    # depth = np.array(pil_depth)
    # depth[0:50] = 0
    # depth[:, 0:50] = 0

    elapsed_time = time.time() - start#処理の終了時間を取得
    print("Converting ply to depth image is succeeded!")
    print("Run time costs is {}".format(elapsed_time))

    return depth, matrix_image, matrix, MAX_DEPTH

if __name__ == "__main__":
    start = time.time()

    # filepath = "./ply/current_used/u-cylinder1.ply"
    filepath = "/home/takasu/ダウンロード/wire-test/ply/out.ply"
    
    # ptCloud = o3d.io.read_point_cloud(filepath)
    # o3d.visualization.draw_geometries([ptCloud])
  
    depth, _, _, _ = main(filepath)

    # df  = pd.DataFrame(depth)
    # df.to_csv('./result/depth_result.csv')
    # cv2.imwrite('./result/depthimg.png', depth)

    # elapsed_time = time.time() - start#処理の終了時間を取得
    # print("実行時間は{}秒でした．".format(elapsed_time))

    # print(depth.shape)
    # for y in range(WINDOW_HEIGHT):
    #     for x in range(WINDOW_WIDTH):
    #         print(x, y)
    #         print(depth[x][y])

    plt.imshow(depth)
    plt.show()
