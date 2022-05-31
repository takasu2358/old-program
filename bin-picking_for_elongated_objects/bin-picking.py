import enum
from multiprocessing.sharedctypes import Value
from cv2 import circle, normalize
from matplotlib import markers
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import random
from skimage.morphology import thin, skeletonize
import time
import math
from PIL import Image
from scipy.signal import argrelmax
from itertools import chain
from collections import Counter
import itertools
import sys
import ply2depth_csv
import warnings
import csv
from datetime import datetime as dt
import os
warnings.simplefilter('ignore')
sys.setrecursionlimit(10000)
tdatetime = dt.now()
tstr = tdatetime.strftime("%Y-%m-%d-%H-%M-%S")
foldername = "trash/" + "Experiment-" + tstr
os.mkdir("/home/takasu/ダウンロード/wire-test/result/Experiment/{}".format(foldername))
args = sys.argv
full_length = int(args[1])

#閾値集
extend_condition = 3     #compare_luminosity関数内
large_area_threshold = 30  #search_seed関数内
small_area_threshold = 100   #search_seed関数内
skel_length_threshold = 10  #detect_line関数内
small_area_threshold2 = 150 #small_region関数内
blocksize = 25              #valley_enhance関数内
line_length_threshold = 40  #valley_enhance関数内
cut_threshold = 50          #cutting, ad_cutting関数内 

filepath = "/home/takasu/ダウンロード/wire-test/ply/out1.ply"
img = ply2depth_csv.main(filepath)
cv2.imwrite('/home/takasu/ダウンロード/wire-test/result/Experiment/{}/depthimg.png'.format(foldername), img)
height, width = img.shape #画像サイズの取得
img_copy = img.copy() #画像のコピー

def get_neighbor(poi, gray_img):
    neighbor = []
    x0, y0 = poi[0], poi[1] #注目点の座標

    #近傍点取得
    for i in range(-1, 2):
        for j in range(-1, 2):
            #poi(注目点)の座標は格納しない
            if (i, j) == (0, 0):
                continue
            x, y = x0 + i, y0 + j #近傍点の座標
            #近傍点の座標が画像サイズ内かつ画素値が0より大きい
            if 0 < x and x < height and 0 < y and y < width:
                if gray_img[x][y] > 0: 
                    neighbor.append([x, y])#neighborに格納

    return neighbor

def gray2color(gray_img):
    height, width = gray_img.shape
    color_img = np.zeros((height, width, 3)) #色情報を3次元にして作成
    for i in range(0, height):
        for j in range(0, width):
            luminosity = gray_img[i][j]
            color_img[i][j] = [luminosity, luminosity, luminosity]

    return color_img

def get_unique_list(seq):
    seen = []
    return [x for x in seq if not x in seen and not seen.append(x)]

def line_circle_delete(line_img):
    contours, _ = cv2.findContours(line_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE) #輪郭を検出(輪郭の内部に輪郭があれば、その細線には周回する部分がある)

    #周回部分がなければそのまま返す
    if len(contours) == 1:
        return line_img

    #周回部分削除(contours[0]は最外部の輪郭)
    for i in range(1, len(contours)):
            line_img = cv2.drawContours(line_img, contours, i, 1, -1) #周回部分の内側を塗りつぶす
    line_img = skeletonize(line_img, method="lee") #再度細線化

    return line_img

class Visualize():
    
    def visualize_region(self, region_list):
        color_img = np.zeros((height, width, 3))
        for region in region_list:
            blue = random.random() #青色を0〜1の中でランダムに設定
            green = random.random() #緑色を0〜1の中でランダムに設定
            red = random.random() #赤色を0〜1の中でランダムに設定
            for xy in region:
                color_img[xy[0]][xy[1]] = [blue, green, red] #各領域ごとに異なる色を指定

        cv2.imshow('image', color_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return color_img

    def visualize_1img(self, img1):
        plt.imshow(img1)
        plt.show()

    def visualize_2img(self, img1, img2):
        fig, axes = plt.subplots(1, 2, figsize = (20, 6))#画像を横に3つ表示
        ax = axes.ravel()
        ax[0].imshow(img1, cmap=plt.cm.gray)
        ax[0].axis('off')#軸無し
        ax[0].set_title('original')

        ax[1].imshow(img2, cmap=plt.cm.gray)
        ax[1].axis('off')#軸無し
        ax[1].set_title('closed_img')

        fig.tight_layout()#余白が少なくなるように表示
        plt.show()

    def visualize_3img(self, img1, img2, img3):
        fig, axes = plt.subplots(1, 3, figsize = (20, 6))#画像を横に3つ表示
        ax = axes.ravel()
        ax[0].imshow(img1, cmap=plt.cm.gray)
        ax[0].axis('off')#軸無し
        ax[0].set_title('original')

        ax[1].imshow(img2, cmap=plt.cm.gray)
        ax[1].axis('off')#軸無し
        ax[1].set_title('closed_img')

        ax[2].imshow(img3, cmap=plt.cm.gray)
        ax[2].axis('off')#軸無し
        ax[2].set_title('skeleton')

        fig.tight_layout()#余白が少なくなるように表示
        plt.show()

    def visualize_branch_point(self, region_skel, branch_point, ad_branch_point):
        ske = region_skel.copy()
        ske = gray2color(ske)
        for xy in branch_point:
            ske[xy[0]][xy[1]] = (1, 0, 0)
        for xys in ad_branch_point:
            for xy in xys:
                ske[xy[0]][xy[1]] = (0,0,1)

        self.visualize_1img(ske)

class SaveImage():

    def save_region(self, region_list, img_name):
        color_img = np.zeros((height, width, 3))
        for region in region_list:
            blue = random.random()*255 #青色を0〜1の中でランダムに設定
            green = random.random()*255 #緑色を0〜1の中でランダムに設定
            red = random.random()*255 #赤色を0〜1の中でランダムに設定
            for xy in region:
                color_img[xy[0]][xy[1]] = [blue, green, red] #各領域ごとに異なる色を指定

        if not img_name == []:
            cv2.imwrite("/home/takasu/ダウンロード/wire-test/result/Experiment/{}/{}.png".format(foldername, img_name), color_img)

        return color_img

    def save_centerpoints(self, center_points_list):
        depth_centerpoint = img_copy.copy()
        depth_centerpoint = gray2color(depth_centerpoint)
        points = self.save_region(center_points_list, [])
        depth_centerpoint = depth_centerpoint + points
        cv2.imwrite("/home/takasu/ダウンロード/wire-test/result/Experiment/{}/depth_centerpoint.png".format(foldername), depth_centerpoint)

    def save_grasp_position(self, LC_skel_list, obj_index, optimal_grasp):
        depth = img_copy.copy()
        depth = gray2color(depth)
        alpha = 30
        skel = LC_skel_list[obj_index]
        for i in range(0, len(skel), 15):
            depth = cv2.circle(depth, (int(skel[i][1]), int(skel[i][0])), 4, (0, 255, 0), -1)
        grasp = optimal_grasp[0]
        grasp_point = grasp[3]
        theta = -(grasp[1] + 90)
        grasp_line_vector = np.array([int(alpha*math.cos(math.radians(theta))), int(-alpha*math.sin(math.radians(theta)))])
        circle_point1 = grasp_point + grasp_line_vector
        circle_point2 = grasp_point - grasp_line_vector
        cv2.line(depth, (int(circle_point1[1]), int(circle_point1[0])), (int(circle_point2[1]), int(circle_point2[0])), (0, 0, 255), 2)
        cv2.circle(depth, (int(circle_point1[1]), int(circle_point1[0])), 4, (0, 0, 255), -1)
        cv2.circle(depth, (int(circle_point2[1]), int(circle_point2[0])), 4, (0, 0, 255), -1)
        cv2.circle(depth, (int(grasp_point[1]), int(grasp_point[0])), 4, (0, 255, 0), -1)
        depth[depth < 1] *= 255
        cv2.imwrite('/home/takasu/ダウンロード/wire-test/result/grasp_position.png'.format(foldername), depth)
        cv2.imwrite('/home/takasu/ダウンロード/wire-test/result/Experiment/{}/grasp_position.png'.format(foldername), depth)
        # cv2.imwrite('/home/takasu/ダウンロード/wire-test/result/grasp_image/{}.png'.format(filename)), depth)

class MakeImage():

    def make_image(self, point_list, value):
        image = np.zeros((height, width), dtype = np.uint8)
        
        if value <= 0:
            for xy in point_list:
                image[xy[0]][xy[1]] = img_copy[xy[0]][xy[1]]
        else:
            for xy in point_list:
                image[xy[0]][xy[1]] = value

        return image
    
    def make_mini_image(self, point_list, value):
        margin = 30
        point_array = np.array(point_list)
        x_list = point_array[:, 0]
        y_list = point_array[:, 1]
        minX, maxX = x_list.min(), x_list.max()
        minY, maxY = y_list.min(), y_list.max()
        miniHeight = maxX - minX + margin*2
        miniWidth = maxY - minY + margin*2
        miniImg = np.zeros((miniHeight, miniWidth))

        for i in range(0, len(point_list)):
            x = x_list[i] - minX + margin
            y = y_list[i] - minY + margin
            miniImg[x][y] = value

        return miniImg, minX-margin, minY-margin, miniHeight, miniWidth, margin

class Detect():

    def detect_region(self, image, area_threshold):
        large_area = []
        _, labels, stats, _ = cv2.connectedComponentsWithStats(np.uint8(image)) #ラベリング
        large_area_labels = np.where(stats[:, 4] > area_threshold)[0] #閾値以上のラベルを取得

        #取得したラベルの領域を取り出す
        for i in range(1, len(large_area_labels)):
            large_area.append(list(zip(*np.where(labels == large_area_labels[i]))))
        
        return large_area

    def detect_line(self, image):
        line_list = []
        image = image.astype(np.uint8)
        retval, labels = cv2.connectedComponents(image)
        for i in range(1, retval):
            line = list(map(list, (zip(*np.where(labels == i)))))
            line_list.append(line)
        
        return line_list

    def detect_singularity(self, skeleton):
        branch_point, end_point, ad_branch_point = [], [], [] #1点連結分岐点、終点、3点連結分岐点を格納する配列
        branch_img = np.zeros((height, width), dtype = np.uint8)
        skeleton[skeleton>0] = 1 #値を1に統一
        nozeros = list(zip(*np.where(skeleton > 0))) #値が1の座標を探索
        #値が1となる座標の近傍9点の値を足し合わせる
        for xy in nozeros:
            point = np.sum(skeleton[xy[0]-1:xy[0]+2, xy[1]-1:xy[1]+2]) #近傍3×3画素の範囲で画素値の合計を求める
            #値が4以上なら分岐点
            if point >= 4:
                branch_point.append(xy) #branch_pointに中心座標xyを格納
                branch_img[xy[0]][xy[1]] = 1 #分岐点のみで構成された画像を作成
            #値が2なら終点
            if point == 2:
                end_point.append(xy) #end_pointに終点座標xyを格納
        
        #1点連結分岐点か3点連結分岐点かを判断する
        count = 0
        branch_copy = branch_point.copy() #branch_pointをコピー

        #branch_copyに含まれる各座標に対してループ
        for xy in branch_copy:
            points = []
            #xyがbranch_pointに含まれていなければ飛ばす
            if not xy in branch_point:
                continue
            point = branch_img[xy[0]-1][xy[1]] + branch_img[xy[0]+1][xy[1]] + branch_img[xy[0]][xy[1]-1] + branch_img[xy[0]][xy[1]+1] #注目座標の上下左右の点を足し合わせる
            points.append(branch_img[xy[0]+1][xy[1]] + branch_img[xy[0]][xy[1]+1]) #注目座標の右・上の点を足し合わせる
            points.append(branch_img[xy[0]][xy[1]+1] + branch_img[xy[0]-1][xy[1]]) #注目座標の上・左の点を足し合わせる
            points.append(branch_img[xy[0]-1][xy[1]] + branch_img[xy[0]][xy[1]-1]) #注目座標の左・下の点を足し合わせる
            points.append(branch_img[xy[0]][xy[1]-1] + branch_img[xy[0]+1][xy[1]]) #注目座標の下・右の点を足し合わせる
            #pointsに2が含まれるかpointが4の場合
            if 2 in points or point == 4:
                ad_branch_point.append([]) #ad_branch_pointに3点格納するための空の配列を用意
                ad_branch_point[count].append(xy) #まず注目座標を格納
                branch_point.remove(xy) #branch_pointから座標xyを削除
                branch_img[xy[0]][xy[1]] = 0 #branch_imgから座標xyを削除
                #注目点の近傍9点を探索
                for i in range(xy[0]-1, xy[0]+2):
                    for j in range(xy[1]-1, xy[1]+2):
                        #x座標またはy座標が注目座標と一致する場合
                        if i == xy[0] or j == xy[1]: 
                            #branch_imgにおける該当座標の値が1の場合
                            if branch_img[i][j] == 1:
                                ad_branch_point[count].append((i, j)) #ad_branch_pointに格納
                                branch_point.remove((i, j)) #branch_pointから削除
                                branch_img[i][j] = 0 #branch_imgから削除
                count += 1
        
        return ad_branch_point, branch_point, end_point 

class RegionGrowing():

    def __init__(self, image):
        self.img = image
        self.img_copy = img_copy
        self.height = height
        self.width = width
        self.ec = extend_condition
        self.lat = large_area_threshold
        self.sat = small_area_threshold

    def search_seed(self):
        V = Visualize()
        MI = MakeImage()
        D = Detect()

        if not self.img.ndim == 2:
            raise ValueError("入力画像は2次元(グレースケール)にしてください")

        region_list = []
        RG_edge_list = []
        nozeros = np.nonzero(self.img)
        while len(nozeros[0]) > 0:
            seed = [[nozeros[0][0], nozeros[1][0]]]
            self.img[seed[0][0]][seed[0][1]] = 0
            region, RG_edge = self.region_growing(seed, [], [])
            nozeros = np.nonzero(self.img)
            region.insert(0, seed[0])
            if len(region) > self.lat:
                region_list.append(region)
                RG_edge = get_unique_list(RG_edge)
                RG_edge_list.append(RG_edge)

        return region_list, RG_edge_list

    def region_growing(self, prepartial_region, region, RG_edge):
        if len(prepartial_region) == 0:
            return region, RG_edge

        for poi in prepartial_region:
            neighbor = get_neighbor(poi, self.img)
            if len(neighbor) == 0:
                continue
            partial_region, RG_edge = self.compare_luminosity(neighbor, poi, RG_edge)
            region.extend(partial_region)
            region, RG_edge = self.region_growing(partial_region, region, RG_edge)

        return region, RG_edge

    def compare_luminosity(self, neighbor, poi, RG_edge):
        partial_region = []

        poi_luminosity = self.img_copy[poi[0]][poi[1]]
        for xy in neighbor:
            neighbor_luminosity = self.img_copy[xy[0]][xy[1]]
            if np.abs(poi_luminosity - neighbor_luminosity) < self.ec:
                partial_region.append(xy)
                self.img[xy[0]][xy[1]] = 0
            else:
                RG_edge.append(xy)
        
        return partial_region, RG_edge

    def watershed_canny(self, region_list, RG_edge_list):
        all = np.zeros((height, width)) 
        for RG_edge, region in zip(RG_edge_list, region_list):
            if len(RG_edge) < 10:
                continue 
            RG_edge_img = MI.make_image(RG_edge, 1)
            region_img = MI.make_image(region, 1)
            dif_img2 = RG_edge_img - region_img
            ero_dif2 = cv2.erode(dif_img2, np.ones((3, 3), np.uint8))
            _, labels, stats, _ = cv2.connectedComponentsWithStats(np.uint8(ero_dif2))
            for i in range(1, len(stats)):
                if stats[i][4] < 10:
                    continue
                lab = labels.copy()
                lab[lab != i] = 0
                lab = lab.astype("uint8")
                close = cv2.morphologyEx(lab, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=3)
                all += close

        gray = img_copy.copy()
        gray[gray > 0] = 1
        big = cv2.dilate(gray, np.ones((3, 3), np.uint8))
        big = big.astype("uint8")
        all[all>1] = 1
        all = all.astype("uint8")
        unknown = big - all
        unknown[unknown<0] = 0
        _, markers = cv2.connectedComponents(all)
        markers = markers+1
        markers[unknown==1] = 0
        water = img_copy.copy()
        canny = cv2.Canny(water.astype("uint8"), 0, 30)
        water = gray2color(water)
        water = water.astype("uint8")
        markers = cv2.watershed(water, markers)
        water[markers==-1] = (0, 0, 255)
        water[markers>0] = (0, 0, 0)
        gray_water = cv2.cvtColor(water, cv2.COLOR_BGR2GRAY)
        gray_water[gray_water>0] = 1
        canny[canny>0] = 1
        canny = cv2.dilate(canny, np.ones((3, 3), dtype = np.uint8))
        check = gray_water * canny
        check = cv2.dilate(check, np.ones((3, 3), np.uint8))
        region_img = img_copy.copy()
        region_img[check > 0] = 0

        return region_img
        
class ConnectRegion():

    def __init__(self, region_list):
        self.region_list = region_list
        self.Size = 51
        self.Step = 16
        self.half_S = self.Size//2
        self.nextPdist = 6 #Don't change(loop process will go wrong)
        self.correct_length = 150
        self.error_length = 30
        self.filename = ["{}.png".format(i) for i in range(0, 10)]

    def search_center_points(self):
        line_templates = [[] for i in range(0, self.Step)]
        center_points_list, end_point_list = [], []
        count = 0
        MI = MakeImage()
        Visual = Visualize()

        ##########################テンプレートの作成############################################
        ground_img = np.zeros((self.Size, self.Size)) #テンプレートの下地を作成
        for i in range(0, self.Step):
            line_angle = ground_img.copy()
            radian = math.radians(180/self.Step*(i+1)) #thetaは各方位の角度
            y = int(self.Size*math.sin(radian)) #方位からy座標を取得
            x = int(self.Size*math.cos(radian)) #方位からx座標を取得
            line_angle = cv2.line(line_angle, (-x+self.half_S, -y+self.half_S), (x+self.half_S, y+self.half_S), 1) #直線を引く
            line_templates[i] = line_angle #画像を格納
        #######################################################################################

        #region_listに格納された各領域に対して中央線を求める
        for region in self.region_list:
            center_points, fore_points, back_points = [], [], []
            if count > -1:
                reg_img, minx, miny, Height, Width, _ = MI.make_mini_image(region, 1)  #領域の画像を作成

                #######################最初の中心点を求める#########################################
                reg_img = cv2.morphologyEx(reg_img, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))  #クロージング処理
                blur = cv2.GaussianBlur(reg_img, (45, 45), 0)  #ガウシアンブラーをかける
                bm = np.argmax(blur)  #輝度値が最大のインデックスを求め得る
                center_x = (bm+1)//Width  #中心のx座標を求める
                center_y = bm-center_x*Width  #中心のy座標を求める
                ####################################################################################

                pil_img = Image.fromarray(reg_img) #cropを使うためpillow型に変更
                pre_index = -1 #はじめはpre_indexは無いため、-1を代入
                reg_color = gray2color(reg_img) #確認用のカラー画像
                center1, center2, nextPdist1, nextPdist2, pre_index, first_dist, pmsign1, pmsign2 = self.center_orientaion([center_x, center_y], pre_index, 1, 0, pil_img, line_templates, reg_color)
                pre_index1 = pre_index
                pre_index2 = pre_index
                shift1, shift2 = 0, 0

                #はじめの中心点からは２方向に伸びるのでwhileも２つ
                #前方向の探索
                fore_count, fore_loop_flag = 0, 0
                while 1:
                    #終了条件：次の候補点の画素値が０の場合
                    if reg_img[center1[0]][center1[1]] == 0:
                        fore_ext = [center1[0]+minx, center1[1]+miny]
                        break
                    #点数が100点以上の場合は、ループのチェックが入る
                    if fore_count > 100:
                        fore_loop_flag, _ = self.check_loop(fore_points,  -1) #最後の点に対して近傍点を確認する
                        break
                    center1, current_center, current_nextPdist, pre_index1, pmsign1, dist, current_shift, reg_color = self.center_orientaion(center1, pre_index1, pmsign1, shift1, pil_img, line_templates, reg_color)
                    if pre_index1 == -100:
                        break
                    fore_points.append([current_center[0]+minx, current_center[1]+miny, nextPdist1, dist, shift1])
                    nextPdist1 = current_nextPdist
                    shift1 = current_shift
                    fore_count += 1

                #ループが存在すると判定された場合、最初の点の近傍を確認する(最初の点の超近傍に点が存在すればその点をループの終点とする)
                if fore_loop_flag == 1:
                    print("Loop exists in fore center points process")
                    start_flag, min_dist_index = self.check_loop(fore_points, 0)
                    if start_flag == 0:
                        # raise ValueError("ループ処理に場合分けが必要です")
                        print("ループ処理に場合分けが必要です")
                    else:
                        fore_points = np.delete(fore_points, np.s_[min_dist_index:-1], 0).astype(int)

                #後ろ方向の探索
                back_count, back_loop_flag = 0, 0
                while 1:
                    #終了条件：次の候補点の画素値が０の場合
                    if reg_img[center2[0]][center2[1]] == 0:
                        back_ext = [center2[0]+minx, center2[1]+miny]
                        break
                    #点数が100点以上の場合は、ループのチェックが入る
                    if back_count > 100:
                        back_loop_flag, _ = self.check_loop(back_points,  -1) #最後の点に対して近傍点を確認する
                        break
                    center2, current_center, current_nextPdist, pre_index2, pmsign2, dist, current_shift, reg_color = self.center_orientaion(center2, pre_index2, pmsign2, shift2, pil_img, line_templates, reg_color)
                    if pre_index2 == -100:
                        break
                    back_points.append([current_center[0]+minx, current_center[1]+miny, nextPdist2, dist, shift2])
                    nextPdist2 = current_nextPdist
                    shift2 = current_shift
                    back_count += 1

                if back_loop_flag == 1:
                    print("Loop exists in back center points process")
                    start_flag, min_dist_index = self.check_loop(back_points, 0)
                    if start_flag == 0:
                        # raise ValueError("ループ処理に場合分けが必要です")
                        print("ループ処理に場合分けが必要です")
                    else:
                        back_points = np.delete(back_points, np.s_[min_dist_index:-1], 0).astype(int)

                fore_points = list(reversed(fore_points))
                center_points.extend(fore_points)
                center_points.append([center_x+minx, center_y+miny, 30, first_dist, 0])
                center_points.extend(back_points)  
                center_points_copy = center_points.copy()

                if len(center_points) > 6:
                    if center_points[0][4] == 1 and center_points[2][4] == 0:
                        del center_points[0]
                    if center_points[-1][4] == 1 and center_points[-3][4] == 0:
                        del center_points[-1]

                if len(center_points) > 3:
                    if center_points[0][2] < 10:
                        del center_points[0]
                    if center_points[-1][2] < 10:
                        del center_points[-1]

                if len(center_points) >= 2:
                    dist1 = center_points[0][3]
                    dist2 = center_points[1][3]
                    if dist1 < dist2*0.75 and len(center_points) >= 3:
                        xy1 = [center_points[0][0], center_points[0][1]]
                        xy2 = [center_points[1][0], center_points[1][1]]
                        xy3 = [center_points[2][0], center_points[2][1]]
                        vectorx21 = xy1[0] - xy2[0]
                        vectory21 = xy1[1] - xy2[1]
                        vectorx32 = xy2[0] - xy3[0]
                        vectory32 = xy2[1] - xy3[1]
                        theta21 = int(math.degrees(math.atan2(vectory21, vectorx21)))
                        theta32 = int(math.degrees(math.atan2(vectory32, vectorx32)))
                        dif_theta321 = np.abs(theta21 - theta32)
                        if dif_theta321 > 180:
                            dif_theta321 = 360 - dif_theta321
                        if dif_theta321 > 30:
                            del center_points[0]           
                    dist1 = center_points[-1][3]
                    dist2 = center_points[-2][3]
                    if dist1 < dist2*0.75 and len(center_points) >= 3:
                        xy1 = [center_points[-1][0], center_points[-1][1]]
                        xy2 = [center_points[-2][0], center_points[-2][1]]
                        xy3 = [center_points[-3][0], center_points[-3][1]]
                        vectorx21 = xy1[0] - xy2[0]
                        vectory21 = xy1[1] - xy2[1]
                        vectorx32 = xy2[0] - xy3[0]
                        vectory32 = xy2[1] - xy3[1]
                        theta21 = int(math.degrees(math.atan2(vectory21, vectorx21)))
                        theta32 = int(math.degrees(math.atan2(vectory32, vectorx32)))
                        dif_theta321 = np.abs(theta21 - theta32)
                        if dif_theta321 > 180:
                            dif_theta321 = 360 - dif_theta321
                        if dif_theta321 > 30:
                            del center_points[-1]

                center_points_list.append(center_points)
                end_point_list.append([[center_points[0][0], center_points[0][1]], [center_points[-1][0], center_points[-1][1]]])

                ####################################################
                # print(center_points)    
                # rc = gray2color(reg_img)
                # rc2 = rc.copy()
                # for xy in center_points:
                #     rc = cv2.circle(rc, (xy[1]-miny, xy[0]-minx), 1, (255, 0, 0), -1)
                #     # rc[xy[0]-minx][xy[1]-miny] = (255, 0, 0)
                # for xy in center_points_copy:
                #     rc2[xy[0]-minx][xy[1]-miny] = (255, 0, 0)
                # Visual.visualize_3img(rc, rc2, rc)
                ####################################################

            count += 1

        return center_points_list, end_point_list

    def check_loop(self, points, index):
        if not len(points[0]) == 2:
            points = np.array([row[0:2] for row in points])
        
        poi = points[index]
        points = np.delete(points, index, 0)
        dif_points = np.abs(points - poi)
        sum_points = [np.sum(point) for point in dif_points]
        sort_points = np.argsort(sum_points)
        for min_index in sort_points:
            if np.abs(min_index - index) > 5:
                break
        min_dist = sum_points[min_index]
        if  min_dist < 7:
            min_dist_index = sum_points.index(min_dist)
            if index < min_dist_index:
                min_dist_index += 1
            return 1, min_dist_index

        return 0, []
        
    def search_next_center_point(self, mini_center_xy, xsin, ycos, cut, min_xy1, min_xy2):
        mini_center_x = mini_center_xy[0]
        mini_center_y = mini_center_xy[1]
        shift = 0

        nextPdist2 = self.nextPdist*(2/3)
        nextPdist3 = nextPdist2*(1/2)
        long_x = int(self.nextPdist*xsin)
        long_y = int(self.nextPdist*ycos)
        medium_x = int(nextPdist2*xsin)
        medium_y = int(nextPdist2*ycos)
        short_x = int(nextPdist3*xsin)
        short_y = int(nextPdist3*ycos)
        if short_x == 0 and short_y == 0:
            if nextPdist3*xsin >= nextPdist3*ycos:
                short_x = 1
            else:
                short_y = 1

        next_center_x = mini_center_x+long_x                                              #切り取られた画像上で、次の候補点のx座標を取得
        next_center_y = mini_center_y+long_y                                              #切り取られた画像上で、次の候補点のy座標を取得

        if cut[next_center_x][next_center_y] == 0:                                                          #次の候補点が黒い領域内なら、距離を短くしてもう一度探索                                                            #はじめの距離の３分の２に縮小
            next_center_x = mini_center_x+medium_x                                              #切り取られた画像上で、次の候補点のx座標を取得
            next_center_y = mini_center_y+medium_y                                              #切り取られた画像上で、次の候補点のy座標を取得
            if cut[next_center_x][next_center_y] == 0:                                                      #まだ黒い領域内なら、もう一度だけ距離を短くして探索                                                              #はじめの距離の３分の１に縮小
                next_center_x = mini_center_x+short_x                                          #切り取られた画像上で、次の候補点のx座標を取得
                next_center_y = mini_center_y+short_y                                          #切り取られた画像上で、次の候補点のy座標を取得
                if cut[next_center_x][next_center_y] == 0:                                                  #それでも黒い領域なら、切り取られた線上で中心点をずらして候補点探索

                    ####################################中心点をずらす############################################################################
                    shift = 1                                                                               #中心点をずらしたかのフラグ
                    cut_line = np.zeros((self.Size, self.Size))                                            
                    cut_line = cv2.line(cut_line, (min_xy1[1], min_xy1[0]), (mini_center_y, mini_center_x), 1)      #min_indexの方向の切り取られた線を画像上で作成
                    cut_line = cv2.line(cut_line, (min_xy2[1], min_xy2[0]), (mini_center_y, mini_center_x), 1)      #min_indexの方向の切り取られた線を画像上で作成
                    poi = [mini_center_x, mini_center_y]                                                    #現在の中心点を注目点に
                    next_pois = get_neighbor(poi, cut_line)                                                #近傍点探索
                    cut_line[mini_center_x][mini_center_y] = 0                                              #注目点を画像上から削除
                    if len(next_pois) == 2:                                                                 #近傍点が２点のとき
                        next_poi1 = [[next_pois[0][0], next_pois[0][1]]]                                                            #中心点の左の点を取得
                        next_poi2 = [[next_pois[1][0], next_pois[1][1]]]                                                            #中心点の右の点を取得
                        while 1:                                                                            #候補点が見つかるか、探索できる点がなくなるまで続ける
                            if len(next_poi1) > 0: 
                                next_poi1 = next_poi1[0]                          
                                mini_center_x1 = next_poi1[0]                                                   #中心点のx座標を更新
                                mini_center_y1 = next_poi1[1]                                                   #中心点のy座標を更新
                                next_center_x = mini_center_x1+long_x                         #nextPdistの距離で候補点のx座標を探索
                                next_center_y = mini_center_y1+long_y                         #nextPdistの距離で候補点のy座標を探索
                                if cut[next_center_x][next_center_y] == 0:
                                    next_center_x = mini_center_x1+medium_x                         #nextPdist2の距離で候補点のx座標を探索
                                    next_center_y = mini_center_y1+medium_y                         #nextPdist2の距離で候補点のy座標を探索
                                    if cut[next_center_x][next_center_y] == 0:
                                        next_center_x = mini_center_x1+short_x                     #nextPdist3の距離で候補点のx座標を探索
                                        next_center_y = mini_center_y1+short_y                     #nextPdist3の距離で候補点のy座標を探索
                                        if cut[next_center_x][next_center_y] == 0:                              #候補点の探索失敗
                                            next_poi1 = get_neighbor(next_poi1, cut_line)                       #現在の中心点の近傍点を探索
                                            cut_line[mini_center_x1][mini_center_y1] = 0                        #現在の中心点を画像上から削除
                                        else:                                       
                                            return_nextPdist = nextPdist3
                                            break
                                    else:
                                        return_nextPdist = nextPdist2
                                        break
                                else:
                                    return_nextPdist = self.nextPdist
                                    break
                            
                            if len(next_poi2) > 0:
                                next_poi2 = next_poi2[0]
                                mini_center_x2 = next_poi2[0]
                                mini_center_y2 = next_poi2[1]
                                next_center_x = mini_center_x2+long_x
                                next_center_y = mini_center_y2+long_y
                                if cut[next_center_x][next_center_y] == 0:
                                    next_center_x = mini_center_x2+medium_x 
                                    next_center_y = mini_center_y2+medium_y 
                                    if cut[next_center_x][next_center_y] == 0:
                                        next_center_x = mini_center_x2+short_x
                                        next_center_y = mini_center_y2+short_y
                                        if cut[next_center_x][next_center_y] == 0:
                                            next_poi2 = get_neighbor(next_poi2, cut_line)
                                            cut_line[mini_center_x2][mini_center_y2] = 0
                                        else:
                                            return_nextPdist = nextPdist3
                                            break
                                    else:
                                        return_nextPdist = nextPdist2
                                        break
                                else:
                                    return_nextPdist = self.nextPdist
                                    break
                            
                            if len(next_poi1) == 0 and len(next_poi2) == 0:
                                return_nextPdist = 0
                                break
                    else:
                        return_nextPdist = 0
                    ###############################################################################################################################

                else:
                    return_nextPdist = nextPdist3
            else:
                return_nextPdist = nextPdist2
        else:
            return_nextPdist = self.nextPdist

        # if [next_center_x, next_center_y] == [self.half_S, self.half_S]:
        #     [next_center_x, next_center_y] = [-100, -100]

        return [next_center_x, next_center_y], return_nextPdist, shift

    #中心線の方向を求め、次の候補点を探索する(現在の中心点、１つ前の方向の添字、テンプレート画像リスト、方位数、テンプレート画像サイズの半分、次の候補点までの距離、前方向か後方向か、確認用カラー画像)
    def center_orientaion(self, center_point, pre_index, pre_pmsign, shift, pil_img, line_angles, reg_color):
        center_x, center_y = center_point[0], center_point[1]
        cut = pil_img.crop((center_y-self.half_S, center_x-self.half_S, center_y+self.half_S+1, center_x+self.half_S+1)) #中心点を基準にテンプレート画像と同サイズに切り取る
        left, upper = center_x-self.half_S, center_y-self.half_S #切り取った画像の左端の座標を取得
        cut = np.asarray(cut) #pillow型からarray型に変更
        dists = [[] for n in range(0, self.Step)] #距離を格納する配列を作成 
        dist_ps = [[] for n in range(0, self.Step)] #直線の端の点を格納する配列を作成
        Visual = Visualize()
        
        ########################テンプレートの適用##############################################################
        for i in range(0, self.Step):            
            cut_line = cut*line_angles[i]                                                       #切り出した領域にテンプレートをかけ合わせる
            line_pixel = np.nonzero(cut_line)                                                   #値を持つ座標を取得する
            if len(line_pixel[0]) == 0:                                                         #かけ合わせた画像に値がない場合次に移る
                continue

            ########################切り取られた線の端点を求める#########################################################
            p1_x = np.min(line_pixel[0])                                                        #片方の端点のｘ座標を、x座標リストの最小値として取得する
            p1_ys = line_pixel[1][np.where(line_pixel[0] == p1_x)]                              #x座標がp1_xとなるy座標を取得
            p1_y = np.min(p1_ys)                                                                #p1_ysの中から最小のものを取り出す
            cutter = np.sum(cut_line[p1_x-1:p1_x+2, p1_y-1:p1_y+2])                             #(p1_x, p1_y)の近傍９点を足し合わせる
            if cutter > 2:                                                                      #足し合わせたものが２より大きければ、p1_yを更新
                p1_y = np.max(p1_ys)                                                            #p1_ysの中から最大のものを取り出す

            p2_x = np.max(line_pixel[0])                                                        #もう片方の端点のx座標を、x座標のリストの最大値として取得する
            p2_ys = line_pixel[1][np.where(line_pixel[0] == p2_x)]                              #x座標がp2_xとなるy座標を取得
            p2_y = np.max(p2_ys)                                                                #p2_ysの中から最小のものを取り出す
            cutter = np.sum(cut_line[p2_x-1:p2_x+2, p2_y-1:p2_y+2])                             #(p2_x, p2_y)の近傍９点を足し合わせる
            if cutter > 2:                                                                      #足し合わせたものが２より大きければ、p2_yを更新            
                p2_y = np.min(p2_ys)                                                            #p1_ysの中から最大のものを取り出す
            ###############################################################################################################

            ########################切り取られた線の長さを求める###########################################################
            dist = (p1_x-p2_x)**2 + (p1_y-p2_y)**2                                              #端点同士の距離を計算
            dists[i] = dist                                                                     #リストに距離を格納
            dist_ps[i] = [p1_x, p1_y, p2_x, p2_y]                                               #該当する端点座標をリストに格納
        sort_dists = np.argsort(dists)                                                          #距離を格納したリストをソートする
        ###############################################################################################################                                                                    

        ######################中心線に垂直となる線の方向(min_index)を取得#########################################################
        count = 0                                                                               #何番目に距離が小さい添字を参照するかをcountに入れておく
        while 1:
            if count == 16:
                return 0, 0, 0, 0, 0, -100, reg_color
            min_index = sort_dists[count]                                                       #count番目に距離が小さい添字を取り出す
            if len(dist_ps[min_index]) == 0:
                count += 1
                continue
            next_index = np.abs(pre_index-min_index)                                            #１つ前の中心点で選ばれた添字と現在の添字の差を求める
            if next_index == 1 or next_index == 15 or next_index == 0 or pre_index == -1:       #条件：１つ前の中心点で選ばれた添字と現在の添字の差が±1以内かpre_indexが-1
                break
            else:
                count += 1
                continue
        ################################################################################################################

        #####################min_indexとなる線分の情報取得、中心点の更新################################################
        try:
            min_x1 = dist_ps[min_index][0]
        except IndexError:
            print(len(dist_ps))
            print("dist_ps = ", dist_ps)
            print("min_index = ", min_index)                                                          #最小となる線の端点のx座標を取得
        min_x2 = dist_ps[min_index][2]                                                          #最小となる線の端点のx座標を取得
        min_y1 = dist_ps[min_index][1]                                                          #最小となる線の端点のy座標を取得
        min_y2 = dist_ps[min_index][3]                                                          #最小となる線の端点のy座標を取得
        mini_center_x = (min_x1+min_x2)//2                                                      #中心のx座標を取得した線分の中心のx座標に更新
        mini_center_y = (min_y1+min_y2)//2                                                      #中心のy座標を取得した線分の中心のy座標に更新
        min_dist = dists[min_index]                                                             #最小となる線の長さを取得
        ################################################################################################################

        ##############################最初の中心点における次の候補点の取得##############################################
        if pre_index == -1:
            theta1 = math.radians(180/self.Step*min_index+90)                                   #前方向の角度
            pmsign1 = 1                                                                         #前方向の符号(theta1式中の90の前についている符号)
            xsin1 = math.sin(theta1)                                                            #前方向の次の候補点へのxベクトル
            ycos1 = math.cos(theta1)                                                            #前方向の次の候補点へのyベクトル
            next_center_xy1, return_nextPdist1, _ = self.search_next_center_point([mini_center_x, mini_center_y], xsin1, ycos1, cut, [min_x1, min_y1], [min_x2, min_y2])

            theta2 = math.radians(180/self.Step*min_index-90)
            pmsign2 = -1
            xsin2 = math.sin(theta2)
            ycos2 = math.cos(theta2)
            next_center_xy2, return_nextPdist2, _ = self.search_next_center_point([mini_center_x, mini_center_y], xsin2, ycos2, cut, [min_x1, min_y1], [min_x2, min_y2])

           
            pre_index = min_index                                                               #pre_indexに現在の方向の添字min_indexを代入
            center_x1 = next_center_xy1[0] + left                                                   #前方向の次の中心点のx座標を取得
            center_x2 = next_center_xy2[0] + left                                                   #後方向の次の中心点のx座標を取得
            center_y1 = next_center_xy1[1] + upper                                                  #前方向の次の中心点のy座標を取得
            center_y2 = next_center_xy2[1] + upper                                                  #後ろ方向の次の中心点のy座標を取得
            
            reg_color[mini_center_x+left][mini_center_y+upper] = (255,0,0)                      #現在の中心点を赤点で表示

            ################################################
            # cut = gray2color(cut)
            # cut = cv2.arrowedLine(cut, (mini_center_y, mini_center_x), (next_center_xy1[1], next_center_xy1[0]), (0, 0, 255), 1)
            # cut = cv2.arrowedLine(cut, (mini_center_y, mini_center_x), (next_center_xy2[1], next_center_xy2[1]), (0, 255, 0), 1)
            # cut = cv2.line(cut, (min_y1, min_x1), (min_y2, min_x2), (255,255,0), 1)
            # Visual.visualize_3img(cut, line_angles[min_index], cut_line)
            ##################################################

            return [center_x1, center_y1], [center_x2, center_y2], return_nextPdist1, return_nextPdist2, pre_index, min_dist, pmsign1, pmsign2
        ################################################################################################################

        ###################################２点目以降の候補点の取得#####################################################
        if next_index == 15:                                                                                #pre_indexとmin_indexの差が15だと90の前につく符号が入れ替わる
            pre_pmsign *= -1                                                                                #pre_pmsignの符号を入れ替える
            theta = math.radians(180/self.Step*(min_index+1)+(90*pre_pmsign))                               #次の候補点が存在する方向を取得
        else:
            theta = math.radians(180/self.Step*(min_index+1)+(90*pre_pmsign))                               #符号を入れ替えずに次の候補点が存在する方向を取得

        xsin = math.sin(theta)                                                                              #次の候補点へのxベクトル
        ycos = math.cos(theta)                                                                              #次の候補点へのyベクトル
        next_center_xy, return_nextPdist, shift = self.search_next_center_point([mini_center_x, mini_center_y], xsin, ycos, cut, [min_x1, min_y1], [min_x2, min_y2])
        #################################################################################
        # cut = gray2color(cut)
        # cut = cv2.arrowedLine(cut, (mini_center_y, mini_center_x), (next_center_xy[1], next_center_xy[0]), (0, 0, 255), 1)
        # cut = cv2.line(cut, (min_y1, min_x1), (min_y2, min_x2), (255,255,0), 1)
        # cut[mini_center_x][mini_center_y] = (255, 0, 0)
        # cut[next_center_xy[0]][next_center_xy[1]] = (0, 255, 0)
        # Visual.visualize_3img(cut, line_angles[min_index], cut_line)

        # print("mini_center = {}".format((mini_center_x, mini_center_y)))
        # print("left, upper = {}".format((left, upper)))
        # print("center = {}".format((center_x, center_y)))
        # reg_color[mini_center_x+left][mini_center_y+upper] = (255, 0, 0)
        # Visual.visualize_3img(reg_color, reg_color, reg_color)
        ###################################################################################

        if next_center_xy == [-100, -100]:
            pre_index = -100
        else:
            pre_index = min_index                                                                               #pre_indexに現在の方向の添字min_indexを代入
        current_center_x = mini_center_x + left                                                             #現在の中心点のx座標を取得
        current_center_y = mini_center_y + upper                                                            #現在の中心点のy座標を取得
        next_center_x = next_center_xy[0] + left                                                                #次の中心点のx座標を取得
        next_center_y = next_center_xy[1] + upper                                                               #次の中心点のy座標を取得

        return [next_center_x, next_center_y], [current_center_x, current_center_y], return_nextPdist, pre_index, pre_pmsign, min_dist, shift, reg_color
        ###########################################################################################################

    def connect_point(self, xy, next_xy, line_xy):
        pre_x, pre_y = xy[0], xy[1]
        current_x, current_y = next_xy[0], next_xy[1]
        line_xy.append([pre_x, pre_y])
        if np.abs(pre_x - current_x) > np.abs(pre_y - current_y):
            if pre_x == current_x:
                tan = 0
                tan2 = 0
            else:
                tan = (pre_y - current_y) / (pre_x - current_x)
                # tan2 = (pre_z - current_z) / (pre_x - current_x)
            if pre_x <= current_x:
                for x in range(pre_x+1, current_x, 1):
                    xt = x - pre_x
                    y = int(tan * xt) + pre_y
                    # z = int(tan2 * xt) + pre_z
                    line_xy.append([x, y])
                    # line_z.append(z)           
                    # print("(x, y, z) = {}".format((x, y, z)))
            else:
                for x in range(pre_x-1, current_x, -1):
                    xt = x - pre_x
                    y = int(tan * xt) + pre_y
                    # z = int(tan2 * xt) + pre_z
                    line_xy.append([x, y])
                    # line_z.append(z)
                    # print("(x, y, z) = {}".format((x, y, z)))
        else:
            if pre_y == current_y:
                tan = 0
                tan2 = 0
            else:
                tan = (pre_x - current_x) / (pre_y - current_y)
                # tan2 = (pre_z - current_z) / (pre_y - current_y)
            if pre_y <= current_y:
                for y in range(pre_y+1, current_y, 1):
                    yt = y - pre_y
                    x = int(tan * yt) + pre_x
                    # z = int(tan2 * yt) + pre_z
                    line_xy.append([x, y])
                    # line_z.append(z)
                    # print("(x, y, z) = {}".format((x, y, z)))
            else:
                for y in range(pre_y-1, current_y, -1):
                    yt = y - pre_y
                    x = int(tan * yt) + pre_x
                    # z = int(tan2 * yt) + pre_z
                    line_xy.append([x, y])
                    # line_z.append(z)

        return line_xy

    def first_check(self,center_points_list):
        MI = MakeImage()
        V = Visualize()
        line_list = []
        correct_connection_index = []
        for center_points in center_points_list:
            line_xy = []
            for i in range(len(center_points)-1):
                pre_xy = [center_points[i][0], center_points[i][1]]
                current_xy = [center_points[i+1][0], center_points[i+1][1]]
                line_xy = self.connect_point(pre_xy, current_xy, line_xy)
            line_xy.append([current_xy[0], current_xy[1]])
            line_list.append(line_xy)
        first_check_result = self.check_connect_accuracy(line_list)

        [correct_connection_index.append(index) for index in range(len(first_check_result)) if first_check_result[index] == 1]

        return line_list, first_check_result, correct_connection_index

    #領域の中心点の情報から領域での接続を行う
    def connect_region(self, center_points_list, rregion, first_check_result):
        region_num = len(rregion)                                                                                   #region_numに領域の数を代入
        thetas, ends, combi, side, cost_list = [], [], [], [], []                                                            
        all_img, allow_img = np.zeros((height, width, 3)), np.zeros((height, width, 3))
        V = Visualize()
        MI = MakeImage()

        #中心点の配列から端点の位置、端点の向きを取得する
        for i in range(0, region_num):
            thetas.append([[] for t in range(0, 2)])                                                                        #端点の向きをthetasに格納
            ends.append([[] for t in range(0, 2)])                                                                          #端点の位置をendsに格納

            vx1 = center_points_list[i][0][0] - center_points_list[i][2][0]                                                 #端点1のx軸方向の向きを取得(端点と一つ前の点のみ参照)
            vy1 = center_points_list[i][0][1] - center_points_list[i][2][1]                                                 #端点1のy軸方向の向きを取得(端点と一つ前の点のみ参照)
            if abs(vx1)+abs(vy1) <= 2:
                vx1 = center_points_list[i][0][0] - center_points_list[i][3][0]
                vy1 = center_points_list[i][0][1] - center_points_list[i][3][1]
            theta1 = int(math.degrees(math.atan2(vy1, vx1)))                                                                #vx1,vy1から端点の向きを計算
            thetas[i][0] = theta1                                                                                           #端点1の向きを格納
            ends[i][0] = [center_points_list[i][0][0], center_points_list[i][0][1]]                                         #端点1の位置を格納

            vx2 = center_points_list[i][-1][0] - center_points_list[i][-2][0]                                               #端点2のx軸方向の向きを取得(端点と一つ前の点のみ参照)
            vy2 = center_points_list[i][-1][1] - center_points_list[i][-2][1]                                               #端点2のy軸方向の向きを取得(端点と一つ前の点のみ参照)
            if abs(vx2)+abs(vy2) <= 2:
                vx2 = center_points_list[i][-1][0] - center_points_list[i][-3][0]
                vy2 = center_points_list[i][-1][1] - center_points_list[i][-3][1]
            theta2 = int(math.degrees(math.atan2(vy2, vx2)))                                                                #vx2,vy2から端点の向きを計算
            thetas[i][1] = theta2                                                                                           #端点2の向きを格納
            ends[i][1] = [center_points_list[i][-1][0], center_points_list[i][-1][1]]                                       #端点2の位置を格納

            #####################中心点の位置と端点の向きを表示###########################
            # print((center_points_list[i][0][0], center_points_list[i][0][1]))
            # print((center_points_list[i][-1][0], center_points_list[i][-1][1]))
            # print(vx1, vy1)
            # print(vx2, vy2)
            # print(theta1)
        #     reg_img = MI.make_image(rregion[i], 1)
        #     reg_img = gray2color(reg_img)
        #     allow_img = cv2.arrowedLine(allow_img, (center_points_list[i][0][1], center_points_list[i][0][0]), (center_points_list[i][0][1]+vy1*2, center_points_list[i][0][0]+vx1*2), (255, 0, 0), thickness = 2, line_type=cv2.LINE_4, tipLength=0.1)
        #     allow_img = cv2.arrowedLine(allow_img, (center_points_list[i][-1][1], center_points_list[i][-1][0]), (center_points_list[i][-1][1]+vy2*2, center_points_list[i][-1][0]+vx2*2), (255, 0, 0), thickness = 2, line_type=cv2.LINE_4, tipLength=0.1)
        #     for xy in center_points_list[i]:
        #         reg_img = cv2.circle(reg_img, (xy[1], xy[0]), 1, (0, 0, 255), -1)
        #         # reg_img[xy[0]][xy[1]] = (0, 0, 255)
        #     all_img += reg_img
        # all_img[np.nonzero(allow_img)[0], np.nonzero(allow_img)[1]] = (0, 0, 0)
        # all_img += allow_img
        # V.visualize_3img(reg_img, all_img, allow_img)
        # all_img = np.clip(all_img * 255, a_min = 0, a_max = 255).astype(np.uint8)
        # cv2.imwrite("./result/arrowed_image.png", all_img)
        ###############################################################

        ###################################深度情報取得##########################################################
        center_point_depth = []                                                                                             #端点の深度情報をcenter_point_depthに格納
        depth_img = img_copy.copy()                                                                                         #深度画像をコピー(コピーしたものを使う)
        depth_img = cv2.morphologyEx(depth_img, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))                               #深度画像にクロージング処理(細かい虫食いを消す)
        #端点の位置における深度画像の値から端点の深度情報を取得
        for i in range(0, region_num):
            center_point_depth.append([])                                                                                   #対象となる領域の深度情報を格納するための空白を追加
            depth1 = depth_img[center_points_list[i][0][0]][center_points_list[i][0][1]]                                    #端点1の深度情報取得
            depth2 = depth_img[center_points_list[i][-1][0]][center_points_list[i][-1][1]]                                  #端点2の深度情報取得
            center_point_depth[i] = [int(depth1), int(depth2)]                                                              #center_point_depthに格納
        ##########################################################################################################

        ################################用語集######################################################
        #sum_theta : 2つの端点の向きの合計(互いに向き合っているときが最小となる)
        #dif_theta : 2つの端点を結んだ線の向きと端点の向きの差分
        #distance : 2つの端点の距離
        #dif_depth : 2つの端点の深度の差分
        ###########################################################################################

        ##################################接続条件############################################################
        sum_theta_threshold = 30                                                                                            #sum_thetaに対する閾値
        dif_theta_threshold = 50                                                                                            #dif_thetaに対する閾値
        distance_threshold = 75                                                                                             #distanceに対する閾値
        min_distance_threshold = 15                                                                                         #distanceに対する閾値
        cost_threshold = 200                                                                                                #costに対する閾値

        # with open("/home/takasu/ダウンロード/NN_test/test.csv", "w") as f:
        #     writer = csv.writer(f)

        #thetas,ends,center_point_depthの情報からsum_theta,dif_theta,distance,dif_depthを求め、コストを計算することで接続の有無を取得する
        for i in range(0, region_num-1):                                                                                    #対象となる1つ目の領域番号
            for j in (0, 1):                                                                                                #該当する領域の端点番号(端点は1つの領域に2つ)
                if first_check_result[i] == 1:
                    continue
                theta1 = thetas[i][j]                                                                                       
                end1 = ends[i][j]
                depth1 = center_point_depth[i][j]
                for m in range(i+1, region_num):                                                                            #対象となる2つ目の領域番号(全領域、端点を総当りで調べる)
                    for n in (0, 1):                                                                                        #該当する領域の端点番号(端点は1つの領域に2つ)
                        if first_check_result[m] == 1:
                            continue
                        theta2 = thetas[m][n]
                        end2 = ends[m][n]
                        depth2 = center_point_depth[m][n]
                        
                        sum_theta = np.abs(np.abs(theta1 - theta2) - 180)                                                   #sum_thetaの計算(互いの向きが向き合っているときが最小)

                        distance = int(np.sqrt((end1[0]-end2[0])**2 + (end1[1]-end2[1])**2))                                #distanceの計算
                        if distance > 50:
                            continue

                        #dif_thetaの計算
                        vx12 = end2[0] - end1[0]                                                                            #端点同士を結んだ線のx軸の向き
                        vy12 = end2[1] - end1[1]                                                                            #端点同士を結んだ線のy軸の向き
                        theta12 = int(math.degrees(math.atan2(vy12, vx12)))                                                 #端点1から見た端点同士を結んだ線の向きを計算
                        dif_theta1 = np.abs(theta12 - theta1)                                                               #端点1の向きと端点の同士を結んだ線の向きとの差分を求める
                        if dif_theta1 > 180:                                                                                #dif_tetha1を0~180に収める
                            dif_theta1  = 360 - dif_theta1
                        theta21 = int(math.degrees(math.atan2(-vy12, -vx12)))                                               #端点2から見た端点同士を結んだ線の向きを計算
                        dif_theta2 = np.abs(theta21 - theta2)                                                               #端点2の向きと端点の同士を結んだ線の向きとの差分を求める
                        if dif_theta2 > 180:                                                                                #dif_theta2を0~180に収める
                            dif_theta2 = 360 - dif_theta2
                        dif_theta = (dif_theta1 + dif_theta2)//2                                                            #dif_theta1とdif_theta2の平均を計算
                        
                        check_point_x = int((end1[0] + end2[0]) / 2)
                        check_point_y = int((end1[1] + end2[1]) / 2)
                        check_intensity = img_copy[check_point_x][check_point_y]                                        
                        
                        if dif_theta1 > 10 and dif_theta2 > 10 or check_intensity == 0:
                            continue

                        dif_depth = np.abs(depth1 - depth2)                                                                 #dif_depthの計算
                        
                        # writer.writerow([label, distance, sum_theta, dif_theta, dif_depth, i, j, m, n])

                        #costを計算(distanceが近すぎるとdif_depthが望ましくない結果を出すため、distanceに閾値を定める)
                        if distance <= min_distance_threshold:
                            cost = distance + sum_theta + dif_theta//3 + dif_depth*2                                        #distanceが閾値より小さければ、dif_depthを3分の1にする
                        else:
                            cost = distance + sum_theta + dif_theta + dif_depth*2

                        #costが閾値より小さければ接続の対象とする
                        if cost < cost_threshold:
                            flag = True

                            #得られた接続先がすでに格納されているかチェック
                            dupis = np.where(np.array(combi).flatten() == i)[0]
                            for dupi in dupis:
                                q, mod = divmod(dupi, 2)
                                if j == side[q][mod]:
                                    if cost < cost_list[q]:
                                        flag = True
                                        del combi[q]
                                        del side[q]
                                        del cost_list[q]
                                        break
                                    else:
                                        flag = False

                            dupis = np.where(np.array(combi).flatten() == m)[0]
                            for dupi in dupis:
                                q, mod = divmod(dupi, 2)
                                if n == side[q][mod]:
                                    if cost < cost_list[q]:
                                        flag = True
                                        del combi[q]
                                        del side[q]
                                        del cost_list[q]
                                        break
                                    else:
                                        flag = False

                            if flag == True:
                                cost_list.append(cost)                                                                      #cost_listにcostを格納
                                combi.append([i, m])                                                                        #combiに接続する領域番号を格納
                                side.append([j, n])                                                                         #sideに接続する端点番号を格納
        
        u, indiices = np.unique(combi, axis = 0, return_index = True)
        range_list = list(range(0, len(combi)))
        not_common_num = list(set(indiices)^set(range_list))
        for i in not_common_num:
            del combi[i]
            del side[i]

        index_form = [i for i, x in enumerate(first_check_result) if x == 1]

        if len(combi) == 0:
            not_combi = [i for i in range(0, region_num) if not i in index_form]
            return combi, side, not_combi, index_form, [], [], center_points_list
            raise ValueError("combiがありません connect_region内")
        #########################################################################################################          

        ###########################################接続する領域をリストに格納####################################
        #posはcombiの何番目と何番目がくっつくかを格納する
        #combiの何番目と何番目が接続されるか、更にその番号の接続を入れ子状に格納していく
        pos, new_combi, new_region_list, flat_combi, unique_pos = [], [], [], [], []
        side_pos = []
        count = 0
        output_combi_count = -1
        while 1:
            check = False                                                                       #同じ数字が２つあり、組み合わせとして得られたときTrueとなる
            pos.append([])                                                                      #posに接続する組み合わせを格納する
            side_pos.append([])
            if count == 0:                                                                      #はじめはiposにcombiを代入
                ipos = combi
            else:
                ipos = pos[count-1]                                                             #一つ前のposをiposとして取り出す
            flat_pos = list(chain(*ipos))                                                       #iposを１次元リストに変換

            for i in range(0, region_num):
                icount = flat_pos.count(i)                                                      #0~region_numまでfalt_posの中に何回登場するかチェック(0~2のどれかとなる)
                if icount == 2:                                                                 #icountが2だった場合はposに加える(3つの領域が接続される場合の橋渡しになる領域番号はicountが2である)
                    pos[count].append([y for y, row in enumerate(ipos) if i in row])            #iposの中身を順番に取り出していき、該当の領域番号が含まれるindexをyとして格納している
                    check = True                                                                #posに新たに格納された場合はcheckをTrueとする
            if not check :                                                                      #checkがFalseならば更新無しとして、終了
                del pos[-1]                                                                     #posの最後は空白の要素があるため削除
                break
            count += 1                                                                          #countを進める
            if count == 1:
                continue
            if len(pos[count-1]) == len(pos[count-2]):
                break 
        
        if not pos == []:
            #後に入れたものから順に接続関係を整列させていき、最終的にcombiの中身に対する接続情報を取得する
            for i in range(count-1, 0, -1):                                                             #countから逆順に進めていく
                sum_pos, flat_pos = [], []
                for ipos in pos[i]:                                                                     #posの中身をiposに入れていく
                    sum_pos.append(list(set([pos[i-1][p][t] for p in ipos for t in (0,1)])))            #iposの組み合わせ番号から一つ前のposの組み合わせ番号を取り出す
                    flat_pos.extend([k for k in ipos])                                                  #iposを1次元リストに変換
                [sum_pos.append(pos[i-1][j]) for j, _ in enumerate(pos[i-1]) if not j in flat_pos]      #一つ前のposに含まれていて、iposに含まれていない組み合わせ番号をsum_posに含める
                pos[i-1] = sum_pos                                                                      #sum_posを一つ前のposとして更新

            for elem in pos[0]:
                if not elem in unique_pos:
                    unique_pos.append(elem)
            #領域番号での接続情報をnew_combiとして取得する
            flat_pos = []
            for i in unique_pos:
                new_combi.append(list(set([combi[p][t] for p in i for t in (0,1)])))
                flat_pos.extend([k for k in i])
            [new_combi.append(combi[i]) for i, _ in enumerate(combi) if not i in flat_pos] 
            output_combi = []
            for x in new_combi:
                if not x in output_combi:
                    output_combi.append(x)

            for i, part in enumerate(output_combi):
                new_region_list.append([])
                for j in part:
                    new_region_list[i].extend(rregion[j])
            output_combi_count = i

            flat_combi = list(chain(*output_combi))

            not_combi = []
            [not_combi.append(i) for i in range(0, region_num) if not i in flat_combi]

        else:
            flat_combi = list(chain(*combi))
            for part in combi:
                new_region_list.append([])
                for j in part:
                    new_region_list.extend(rregion[j])
            not_combi = [i for i in range(0, region_num) if not i in flat_combi]

        [not_combi.remove(i) for i in index_form]

        for index in not_combi:
            new_region_list.append([])
            new_region_list[-1].extend(rregion[index])

   #########################################################################################################

        # color_img = V.visualize_region(new_region_list)
        # color_img = np.clip(color_img * 255, a_min = 0, a_max = 255).astype(np.uint8)
        # cv2.imwrite("./result/connected_color_img.png", color_img)
        # raise ValueError

        return combi, side, not_combi, index_form, unique_pos, new_region_list, center_points_list

    def connect_center_line(self, current_region, current_side, pre_side_point, center_points_list):
        image = np.zeros((height, width, 3))
        if not pre_side_point == []:
            current_side_point = center_points_list[current_region][-1 * current_side]
            image = cv2.line(image, (pre_side_point[1], pre_side_point[0]), (current_side_point[1], current_side_point[0]), (255,255,255), 1)
            pre_side_point = center_points_list[current_region][-1 * (1 - current_side)]
        else:
            pre_side_point = center_points_list[current_region][-1 * current_side]

        point_it = iter(center_points_list[current_region])
        point1 = next(point_it)
        while True:
            try:
                point2 = next(point_it)
                image = cv2.line(image, (point1[1], point1[0]), (point2[1], point2[0]), (255,255,255), 1)
                point1 = point2
            except StopIteration:
                break
        
        return pre_side_point, image

    def check_connect(self, combi, side, index_form, pos, line_list, check_result, correct_connection_index):
        V = Visualize()
        connection_index = []
        series_regions, series_sides = [], []
        for x in pos:
            series_side, series_region, flat_num, connection = [], [], [], []
            [flat_num.append(combi[num]) for num in x]
            flat_num = list(chain(*flat_num))
            counter = Counter(flat_num)
            region_num = [elem for elem in flat_num if counter[elem] < 2]
            if region_num == []:
                for index in x:
                    combi[index] = []
                continue
            else:
                region_num = region_num[0]
            for _ in range(len(x)):
                index = [elem for elem in x if region_num in combi[elem]][0]
                if region_num == combi[index][0]:
                    series_side.append(side[index])
                else:
                    side[index].reverse()
                    series_side.append(side[index])
                next_region_num = [elem for elem in combi[index] if not elem == region_num][0]
                series_region.append([region_num, next_region_num])
                connection.extend([region_num, next_region_num])
                region_num = next_region_num
                combi[index] = []
            series_regions.append(series_region)
            series_sides.append(series_side)
            connection_index.append(connection)
            # print("series_region = ", series_region)
            # print("series_side = ", series_side)

        connected_line_list, connected_index_list = [], []
        interpolate_list = []
        for series_region, series_side in zip(series_regions, series_sides):
            connected_line, connected_index = [], []
            interpolate = []
            first_region = series_region[0][0]
            first_side = series_side[0][0]
            first_region_line = line_list[first_region]
            if first_side == 0:
                pre_end_point = [first_region_line[0][0], first_region_line[0][1]]
                first_region_line.reverse()
            else:
                pre_end_point = [first_region_line[-1][0], first_region_line[-1][1]]
            connected_line.extend(first_region_line)
            connected_index.append(first_region)
            for region, region_side in zip(series_region, series_side):
                current_region = region[1]
                current_side = region_side[1]
                region_line = line_list[current_region]
                if current_side == 0:
                    end_point = [region_line[0][0], region_line[0][1]]
                    other_end_point = [region_line[-1][0], region_line[-1][1]]
                else:
                    end_point = [region_line[-1][0], region_line[-1][1]]
                    other_end_point = [region_line[0][0], region_line[0][1]]
                    region_line.reverse()
                connect_line = self.connect_point(pre_end_point, end_point, [])
                connected_line.extend(connect_line)
                connected_line.extend(region_line)
                connected_index.append(current_region)
                interpolate.extend(connect_line)
                pre_end_point = other_end_point
            connected_line = get_unique_list(connected_line)
            connected_line_list.append(connected_line)
            connected_index_list.append(connected_index)
            interpolate_list.append(interpolate)
                   
        for combi_elem, side_elem in zip(combi, side):
            if combi_elem == []:
                continue
            connection_index.append(combi_elem)
            connected_line, connected_index = [], []
            current_region1 = combi_elem[0]
            current_side1 = side_elem[0]
            region_line1 = line_list[current_region1]
            if current_side1 == 0:
                end_point1 = [region_line1[0][0], region_line1[0][1]]
                region_line1.reverse()
            else:
                end_point1 = [region_line1[-1][0], region_line1[-1][1]]
            connected_line.extend(region_line1)
            connected_index.append(current_region1)
            current_region2 = combi_elem[1]
            current_side2 = side_elem[1]
            region_line2 = line_list[current_region2]
            if current_side2 == 0:
                end_point2 = [region_line2[0][0], region_line2[0][1]]
            else:
                end_point2 = [region_line2[-1][0], region_line2[-1][1]]
                region_line2.reverse()
            connect_line = connect_line = self.connect_point(end_point1, end_point2, [])
            connected_line.extend(connect_line)
            connected_line.extend(region_line2)
            connected_index.append(current_region2)
            connected_line = get_unique_list(connected_line)
            connected_line_list.append(connected_line)
            connected_index_list.append(connected_index)
            interpolate_list.append(connect_line)

        # for elem in not_combi:
        #     connected_line_list.append(line_list[elem])
        correct_line, interpolate_list2 = [], []
        for elem in index_form:
            correct_line.append(line_list[elem])
            interpolate_list2.append([])

        sub_check_result = self.check_connect_accuracy(connected_line_list)
        correct_index = [i for i, x in enumerate(sub_check_result) if x == 1]
        check_index = []
        for index in correct_index:
            correct_line.append(connected_line_list[index])
            interpolate_list2.append(interpolate_list[index])
            check_index.extend(connected_index_list[index])
            correct_connection_index.append(connection_index[index])

        for index in check_index:
            check_result[index] = 1

        correct_line_zlist = []
        for skel in correct_line:
            correct_line_zlist.append([])
            for point in skel:
                z = int(img_copy[point[0]][point[1]])
                correct_line_zlist[-1].append(z)

        new_check_result = []
        count = 0
        for i, elem in enumerate(check_result):
            flag = 0
            while flag == 0:
                if i+count in skip_index:
                    new_check_result.append(0)
                    count += 1
                else:
                    new_check_result.append(elem)
                    flag = 1
            
        return new_check_result, correct_line, interpolate_list2, correct_line_zlist, correct_connection_index
        
    def check_connect_accuracy(self,line_list):
        curvature_list = []
        for line in line_list:
            length = len(line)
            curvature_list.append([])
            if length >= full_length - self.error_length and length <= full_length + self.error_length:
                for i in range(15, len(line)-15, 15):
                    point1 = line[i - 15]
                    point2 = line[i]
                    point3 = line[i + 15]
                    rho = self.curvature(point1, point2, point3)
                    curvature_list[-1].append(rho)
        
        check_result = []
        for curvature, line in zip(curvature_list, line_list):
            if curvature == []:
                check_result.append(0)
                continue
            peak = argrelmax(np.array(curvature))
            if len(peak):
                check_result.append(1)
            else:
                check_result.append(0)
            # fig = plt.figure()
            # curve_img = MI.make_image(line, 1)
            # ax1 = fig.add_subplot(1, 2, 1)
            # plt.imshow(curve_img)
            # x = list(range(len(curvature)))
            # ax2 = fig.add_subplot(1, 2, 2)
            # plt.plot(x, curvature)
            # print(curvature)
            # print(peak)
            # path = "./result/" + self.filename[0]
            # del self.filename[0] 
            # fig.savefig(path)
            # plt.show()
        
        return check_result

    def curvature(self, point1, point2, point3):
        x1, y1 = point1[0], point1[1]
        x2, y2 = point2[0], point2[1]
        x3, y3 = point3[0], point3[1]
        rho1 = 2*np.abs(x1*y2-x2*y1+x2*y3-x3*y2+x3*y1-x1*y3)
        rho2 = np.sqrt(((x1-x2)**2+(y1-y2)**2)*((x2-x3)**2+(y2-y3)**2)*((x3-x1)**2+(y3-y1)**2))
        rho = rho1/rho2
        
        return rho

    def cal_curvature(self, A, point2):
        x = point2[0]
        a, b = A[0], A[1]
        if a == 0:
            return 0
        first_order_derivative = 2*a*x + b
        second_order_derivative = 2*a
        R = np.abs(second_order_derivative) / (1 + first_order_derivative ** 2) ** 1.5
        
        return R

    def make_quadratic_function(self, point1, point2, point3):
        x1, y1 = point1[0], point1[1]
        x2, y2 = point2[0], point2[1]
        x3, y3 = point3[0], point3[1]
        Y = np.array([y1, y2, y3]).T
        X = np.array([[x1**2, x1, 1],
                      [x2**2, x2, 1],
                      [x3**2, x3, 1]])
        X_inv = np.linalg.pinv(X)
        A = np.dot(X_inv, Y)
        A = np.round(A, 2)
        
        return A

    def skip_by_length(self, region_list2, center_points_list):
        rregion, new_center_points_list, skip_index = [], [], []
        for i, (region, center_point) in enumerate(zip(region_list2, center_points_list)):
            if len(center_point) > 3:
                new_center_points_list.append(center_point)
                rregion.append(region)
            else:
                skip_index.append(i)
                
        return rregion, new_center_points_list, skip_index

class Skeletonize():

    def skeletonize_region_list(self, region_list, check_result):
        MI, V, D = MakeImage(), Visualize(), Detect()
        skel_list, branch_point_list, ad_branch_point_list, end_point_list = [], [], [], []
        # image = np.zeros((height, width))

        for region, check in zip(region_list, check_result):
            if check == 1:
                continue
            reg_img = MI.make_image(region, 1)
            reg_img = cv2.morphologyEx(reg_img, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            region_skel = skeletonize(reg_img, method="lee")
            region_skel = line_circle_delete(region_skel)
            ad_branch_point, branch_point, _ = D.detect_singularity(region_skel)
            # V.visualize_branch_point(region_skel, branch_point, ad_branch_point)
            region_skel, skel_list, branch_point_list, ad_branch_point_list, end_point_list = self.cut_branch(region_skel, ad_branch_point, branch_point, skel_list, branch_point_list, ad_branch_point_list, end_point_list)
            
            # image += region_skel
            
        # V.visualize_1img(image)
        # image = np.clip(image * 255, a_min = 0, a_max = 255).astype(np.uint8)
        # cv2.imwrite("./result/skeletonized_image.png", image)

        skel_list2, branch_point_list2, ad_branch_point_list2, end_point_list2 = [], [], [], []
        skel_skip_index = []
        for i, skel in enumerate(skel_list):
            if len(skel) <= 5:
                skel_skip_index.append(i)
                continue
            branch_point = branch_point_list[i]
            ad_branch_point = ad_branch_point_list[i]
            end_point = end_point_list[i]
            skel_list2.append(skel)
            branch_point_list2.append(branch_point)
            ad_branch_point_list2.append(ad_branch_point)
            end_point_list2.append(end_point)

        return skel_list2, branch_point_list2, ad_branch_point_list2, end_point_list2, skel_skip_index

    def cut_branch(self, skeleton, ad_branch_point, branch_point, skel_list, branch_point_list, ad_branch_point_list, end_point_list):
        ad_bra = list(points[0] for points in ad_branch_point) #ad_branch_pointから代表として1点取得
        index, ad_index = 0, 0 #branch_pointとad_branch_point用の添字
        last = []
        D = Detect()
        MI = MakeImage()

        #branch_pointとad_braに値がある間ループ
        while len(branch_point) > 0 or len(ad_bra) > 0:

            #branch_pointに値がある場合
            if len(branch_point) > 0:
                #indexがbranch_pointの数を超えていれば初期化
                if index >= len(branch_point):
                    index = 0
                xy = branch_point[index] #注目する1点連結の分岐点を取得
                skeleton, branch_point, ad_bra, flag, last, _ = self.cutting(skeleton, xy, branch_point, ad_bra, ad_branch_point, last) #cuttingで1点連結の分岐点の場合の枝切り
                index += flag #flagが1なら今回注目していた1点連結の分岐点は削除されなかった
                
            #ad_braに値がある場合
            if len(ad_bra) > 0:
                #ad_indexがad_braの数を超えていれば初期化
                if ad_index >= len(ad_bra):
                    ad_index = 0 
                xy = ad_bra[ad_index] #注目する3点連結の分岐点を取得
                skeleton, branch_point, ad_bra, ad_flag, last, _ = self.ad_cutting(skeleton, xy, branch_point, ad_bra, ad_branch_point, last) #ad_cuttingで3点連結の分岐点の場合の枝切り
                ad_index += ad_flag #ad_flagが1なら今回注目していた3点連結の分岐点は削除されなかった

        if len(last) > 0:
            for xy in last:
                skeleton[xy[0]-3:xy[0]+3, xy[1]-3:xy[1]+3] = 0
            skeleton2 = skeleton.copy()
            line_list = D.detect_line(skeleton)
            num = len(line_list)
        else:
            skeleton2 = skeleton.copy()
            num = 1

        for i in range(0, num):
            if len(last) > 0:
                skeleton = MI.make_image(line_list[i], 1)
            ad_branch_point, branch_point, end_point = D.detect_singularity(skeleton)
            if len(branch_point) > 0: 
                for count in range(0, len(branch_point)//2):
                    point, point2 = branch_point[count], branch_point[count+1]
                    for y in range(point[0]-1, point[0]+2):
                        for x in range(point[1]-1, point[1]+2):
                            if (y, x) == point:
                                continue
                            if abs(y-point2[0]) + abs(x-point2[1]) == 1:
                                skeleton[y][x] = 0
                                if point in branch_point:
                                    branch_point.remove(point)
                                    branch_point.remove(point2)
            
            skel = D.detect_line(skeleton)
            if len(skel) > 0:
                end_point_list.append(end_point)
                branch_point_list.append(branch_point)
                ad_branch_point_list.append(ad_branch_point)
                skel_list.extend(skel)

        return skeleton2, skel_list, branch_point_list, ad_branch_point_list, end_point_list

    def cutting(self, skeleton, xy, branch_point, ad_bra, ad_branch_point, last):
        check, count, flag = 0, 0, 0 #フラグ管理の変数(check:枝先の枝を削除したか、count:whileを回した回数、flag:今回の分岐点xyを飛ばしたか(cut_branch3のindexに影響))
        delete_length = 0
        if not xy in branch_point:
            return skeleton, branch_point, ad_bra, flag, last, 0
        
        #branch_pointに入っている座標xyがすでにskeletonから消えている場合は、branch_pointから削除して返却
        if skeleton[xy[0]][xy[1]] == 0:
            branch_point.remove(xy) #branch_pointから座標xyを削除
            return skeleton, branch_point, ad_bra, flag, last, 0

        line_img_copy = np.uint8(skeleton.copy()) #skeletonをコピーして、line_img_copyとする
        line_img_copy[xy[0]][xy[1]] = 0 #分岐点を削除
        _, line_labels, stats, _ = cv2.connectedComponentsWithStats(line_img_copy) #ラベリングする
        stats_copy = stats.copy() #stats(各ラベルの情報が載った配列)をコピー、これをしないと下のwhileで回したとき誤ったラベルを消しかねない

        #ラベルが3つより多ければその分岐点にはまだ枝が残っている
        while len(stats) > 3:
            #stats[:, 4]は各ラベルの画素数(statsはwhileの最後でmin_lable番目を消す, stats_copyはそのまま)
            min_label1 = np.where(stats[:, 4] == min(stats[:, 4]))[0][0] #最小の値を持つラベル番号を現在のstastsから取得
            current_xy = [stats[min_label1][0], stats[min_label1][1]]
            original_xy = [stats_copy[min_label1][0], stats_copy[min_label1][1]]
            if current_xy != original_xy:
                min_label = min_label1 + count
            else:
                min_label = min_label1
            min_labels = list(zip(*np.where(line_labels == min_label))) #ラベル番号がmin_labelの座標を全て取得
            brabranch = list(set(branch_point) & set(min_labels)) #min_labelの枝に1点連結の枝が含まれていれば、その座標をbrabranchに格納
            ad_brabranch = list(set(ad_bra) & set(min_labels)) #min_labelの枝に3点連結の枝が含まれていれば、その座標をad_brabranchに格納

            #brabranchに値が格納されている場合
            if len(brabranch) > 0:
                flag, check = 1, 1 
                line_img_copy[xy[0]][xy[1]] = 1 #削除した座標xyの値を1に戻す
                #brabranchに入っている座標の数だけ回す
                for xy in brabranch:
                    skeleton, branch_point, ad_bra, _, last, delete_length = self.cutting(skeleton, xy, branch_point, ad_bra, ad_branch_point, last) #cuttingにbrabranchの座標を分岐点として枝切り
                    stats[min_label1][4] -= delete_length
                    stats_copy[min_label][4] -= delete_length

            #ad_brabranchに値が格納されている場合
            if len(ad_brabranch) > 0:
                flag, check = 1, 1
                line_img_copy[xy[0]][xy[1]] = 1 #削除した座標xyの値を1に戻す
                #ad_brabranchに入っている座標の数だけ回す
                for xy in ad_brabranch:
                    skeleton, branch_point, ad_bra, _, last, delete_length = self.ad_cutting(skeleton, xy, branch_point, ad_bra, ad_branch_point, last) #ad_cuttingにad_brabranchの座標を分岐点として枝切り
                    stats[min_label1][4] -= delete_length
                    stats_copy[min_label][4] -= delete_length

            #枝先の枝が無い場合
            if check == 0:
                if min(stats[:, 4]) < cut_threshold:
                    line_labels_copy = line_labels.copy() #line_labelsをコピーし、line_labels_copyとする
                    line_labels_copy[line_labels_copy!=min_label] = 0 #長さが最小の線以外を削除
                    skeleton = skeleton - line_labels_copy #skeletonから長さが最小の枝を削除
                    skeleton[skeleton<0] = 0 #skeletonが負の値を持たないようにする
                else:
                    last.append(xy)

                #座標xyがbranch_pointに含まれている場合、branch_pointから削除する
                if xy in branch_point:
                    branch_point.remove(xy)
                delete_length = stats[min_label1][4]
                stats = np.delete(stats, min_label1, 0) #statsからmin_label1番目の行を削除
                count += 1
            check = 0

        #whileの条件であるstatsに3つより多くラベルが存在するを満たさない分岐点の処理(分岐点を削除しても線が2本しかできない場合)
        if xy in branch_point:
            nearby = 0 #1点分岐点の近くに1点分岐点があるかのフラグ

            #近傍点探索
            for i in range(-1, 2):
                for j in range(-1, 2):
                    #斜め方向に1点分岐点がある場合
                    if i != 0 and j != 0 and (xy[0]+i, xy[1]+j) in branch_point:
                        #隣接する1点分岐点と注目している1点分岐点の両方に隣接する点を削除
                        skeleton[xy[0]][xy[1]+j] = 0
                        skeleton[xy[0]+i][xy[1]] = 0
                        branch_point.remove((xy[0]+i, xy[1]+j)) #隣接する1点分岐点をbranch_pointから削除
                        nearby += 1

                    #上下左右に1点分岐点がある場合
                    if (i == 0 or j == 0) and (i, j) != (0, 0) and (xy[0]+i, xy[1]+j) in branch_point:
                        line_img_copy[xy[0]+i][xy[1]+j] = 0 #隣接する1点分岐点を削除

                        #通常の枝切り手法と同じ
                        _, line_labels, stats, _ = cv2.connectedComponentsWithStats(line_img_copy) 
                        stats_copy = stats.copy()
                        min_label = np.where(stats[:, 4] == min(stats[:, 4]))[0][0]
                        line_labels_copy = line_labels.copy()
                        line_labels_copy[line_labels_copy!=min_label] = 0 
                        skeleton = skeleton - line_labels_copy 
                        skeleton[skeleton<0] = 0

                        #2点の1点分岐点のどちらを削除するか
                        point1 = np.sum(skeleton[xy[0]-1:xy[0]+2, xy[1]-1:xy[1]+2]) #注目している1点分岐点の近傍点数を計算
                        point2 = np.sum(skeleton[xy[0]+i-1:xy[0]+i+2, xy[1]+j-1:xy[1]+j+2]) #隣接している1点分岐点の近傍点数を計算
                        #注目している1点分岐点の近傍点が3点なら消す
                        if point1 == 3 and point2 > 3:
                            skeleton[xy[0]][xy[1]] = 0
                        #隣接している1点分岐点の近傍点が3点なら消す
                        elif point2 == 3 and point1 > 3:
                            skeleton[xy[0]+i][xy[1]+j] = 0
                        branch_point.remove((xy[0]+i, xy[1]+j)) #隣接している1点分岐点をbranch_pointから削除
                        nearby += 1

            #1点分岐点が隣接していない場合
            if nearby == 0:
                #通常の枝切り手法と同じ
                min_label = np.where(stats[:, 4] == min(stats[:, 4]))[0][0]
                line_labels_copy = line_labels.copy()
                line_labels_copy[line_labels_copy!=min_label] = 0 
                skeleton = skeleton - line_labels_copy 
                skeleton[skeleton<0] = 0

            branch_point.remove(xy) #注目している1点近傍点をbranch_pointから削除
        
        return skeleton, branch_point, ad_bra, flag, last, delete_length

    def ad_cutting(self, skeleton, xy, branch_point, ad_bra, ad_branch_point, last):
        check, count, ad_flag = 0, 0, 0 #フラグ管理の変数(check:枝先の枝を削除したか、count:whileを回した回数、ad_flag:今回の分岐点xyを飛ばしたか(cut_branch3のad_indexに影響))
        delete_length = 0
        if not xy in ad_bra:
            return skeleton, branch_point, ad_bra, ad_flag, last, 0
        coods = ad_branch_point[ad_bra.index(xy)] #座標xyが含まれている3点分岐点をad_branch_pointから取得する

        #ad_bra, ad_branch_pointに入っている座標xyがすでにskeletonから消えている場合は、ad_bra, aad_branch_pointから削除して返却
        if skeleton[xy[0]][xy[1]] == 0:
            ad_bra.remove(xy) #ad_braから座標xyを削除
            ad_branch_point.remove(coods) #ad_branch_pointから座標群coodsを削除
            return skeleton, branch_point, ad_bra, ad_flag, last, 0

        x, y = [], []
        line_img_copy = np.uint8(skeleton.copy()) #skeletonをコピーして、line_img_copyとする
        #coodsに入っている座標をx座標とy座標に分け、各座標を削除
        for i in range(0, len(coods)):
            x.append(coods[i][0]) #x座標を格納
            y.append(coods[i][1]) #y座標を格納
            line_img_copy[x[i]][y[i]] = 0 #line_img_copyからcoodsの座標を削除

        _, line_labels, stats, _ = cv2.connectedComponentsWithStats(line_img_copy) #ラベリング
        stats_copy = stats.copy() #stats(各ラベルの情報が載った配列)をコピー、これをしないと下のwhileで回したとき誤ったラベルを消しかねない

        #ラベルが3つより多ければその分岐点にはまだ枝が残っている
        while len(stats) > 3:
            #stats[:, 4]は各ラベルの画素数(statsはwhileの最後でmin_lable番目を消す, stats_copyはそのまま)
            min_label1 = np.where(stats[:, 4] == min(stats[:, 4]))[0][0] #最小の値を持つラベル番号を現在のstastsから取得

            current_xy = [stats[min_label1][0], stats[min_label1][1]]
            original_xy = [stats_copy[min_label1][0], stats_copy[min_label1][1]]
            if current_xy != original_xy:
                min_label = min_label1 + count
            else:
                min_label = min_label1

            min_labels = list(zip(*np.where(line_labels == min_label))) #ラベル番号がmin_labelの座標を全て取得
            brabranch = list(set(branch_point) & set(min_labels)) #min_labelの枝に1点連結の枝が含まれていれば、その座標をbrabranchに格納
            ad_brabranch = list(set(ad_bra) & set(min_labels)) #min_labelの枝に3点連結の枝が含まれていれば、その座標をad_brabranchに格納

            #brabranchに値が格納されている場合
            if len(brabranch) > 0:
                ad_flag, check = 1, 1
                line_img_copy[xy[0]][xy[1]] = 1 #削除した座標xyの値を1に戻す
                #brabranchに入っている座標の数だけ回す
                for xy in brabranch:
                    skeleton, branch_point, ad_bra, _, last, delete_length = self.cutting(skeleton, xy, branch_point, ad_bra, ad_branch_point, last) #cuttingにbrabranchの座標を分岐点として枝切り
                    stats[min_label1][4] -= delete_length
                    stats_copy[min_label][4] -= delete_length

            #ad_brabranchに値が格納されている場合
            if len(ad_brabranch) > 0:
                ad_flag, check = 1, 1
                line_img_copy[xy[0]][xy[1]] = 1 #削除した座標xyの値を1に戻す
                #ad_brabranchに入っている座標の数だけ回す
                for xy in ad_brabranch:
                    #座標xyがad_braに含まれている場合のみ枝切り
                    if xy in ad_bra:
                        skeleton, branch_point, ad_bra, _, last, delete_length = self.ad_cutting(skeleton, xy, branch_point, ad_bra, ad_branch_point, last) #ad_cuttingにad_brabranchの座標を分岐点として枝切り
                        stats[min_label1][4] -= delete_length
                        stats_copy[min_label][4] -= delete_length
                    else:
                        continue

            #枝先の枝が無い場合
            if check == 0:   
                if min(stats[:, 4]) < cut_threshold:   
                    line_labels_copy = line_labels.copy() #line_labelsをコピーし、line_labels_copyとする
                    line_labels_copy[line_labels_copy!=min_label] = 0 #長さが最小の線以外を削除
                    skeleton = skeleton - line_labels_copy #skeletonから長さが最小の枝を削除
                    skeleton[skeleton<0] = 0 #skeletonが負の値を持たないようにする

                    value = []
                    #3点連結の各分岐点の近傍点を計算、valueに格納
                    for i in range(0, 3):
                        value.append(np.sum(skeleton[x[i]-1:x[i]+2, y[i]-1:y[i]+2]))
                    check_value = np.where(np.array(value) >= 4)[0] #値が4以上となるvalueの添字を取得
                    ad_value = np.where(np.array(value) == 3)[0] #値が3となるvalueの添字を取得
                    #値が4以上となる分岐点が2点で、値が3となる分岐点が存在する場合
                    if len(check_value) == 2 and len(ad_value) > 0:
                        x2, y2 = x[ad_value[0]], y[ad_value[0]] #値が3となる分岐点の座標を取得
                        skeleton[x2][y2] = 0 #値が3となる分岐点を削除
                else:
                    last.append(xy)
                
                #1周目のとき、ad_braとad_branch_pointからxyを削除
                if xy in ad_bra:  
                    ad_bra.remove(xy)
                    ad_branch_point.remove(coods)

                delete_length = stats[min_label1][4]
                stats = np.delete(stats, min_label1, 0) #statsからmin_label1番目の行を削除
                count += 1
            check = 0

        if xy in ad_bra:
            ad_bra.remove(xy)
            ad_branch_point.remove(coods)
        
        return skeleton, branch_point, ad_bra, ad_flag, last, delete_length

class Sort():

    def sort_skel_list(self, skel_list, branch_point_list, ad_branch_point_list, end_point_list):
        new_skel_list = [[] for i in range(0, len(skel_list))]
        for i, skel in enumerate(skel_list):
            ######################今回使うデータの整理#####################################
            try:
                flat_ad_branch = sum(ad_branch_point_list[i], [])
                flat_ad_branch = [list(e) for e in flat_ad_branch]
            except IndexError:
                flat_ad_branch = []
            try:
                branch_point = [list(e) for e in branch_point_list[i]]
            except IndexError:
                branch_point = []
            end_point = [list(e) for e in end_point_list[i]]
            ###############################################################################

            #########端点が分岐点に含まれているかを判断後、最初の注目点を決定##############
            if len(end_point) > 2:
                for ep in end_point:
                    _, _, flag = self.check_neighbor(ep, branch_point)
                    if flag == 0:
                        poi = ep
                        break
            else:
                poi = list(end_point[0])
            ###############################################################################
            # print("branch_point = ",branch_point)
            # print("flat_ad_branch = ",flat_ad_branch)
            #########################近傍点探索、リストに順番に格納########################
            skel = [list(elem) for elem in skel]
            skel.remove(poi)
            new_skel_list[i].append(poi)
            length = len(skel)
            while length > 0:
                #3点分岐がある場合
                if poi in flat_ad_branch:
                    flag = 0
                    poi, new_skel_list[i], flat_ad_branch, length, flag, next_poi = self.ad_sort(poi, skel, new_skel_list[i], flat_ad_branch, flag, [-1, -1], length)
                    if flag == 0:
                        new_skel_list[i].append(next_poi)
                        pre_poi = poi
                        poi = next_poi
                        poi, new_skel_list[i], flat_ad_branch, length, flag, next_poi = self.ad_sort(poi, skel, new_skel_list[i], flat_ad_branch, flag, pre_poi, length)

                #1点分岐がある場合
                if poi in branch_point:
                    branch_point.remove(poi)
                    poi, skel, new_skel_list[i], branch_point = self.branch_sort(poi, skel, new_skel_list[i], branch_point, end_point_list[i])
                    length = len(skel) 
  
                for count in range(0, length):
                    xy = skel[count]
                    check_x = abs(poi[0] - xy[0])
                    check_y = abs(poi[1] - xy[1])
                    if check_x <= 1 and check_y <= 1 and check_x + check_y <= 2:
                        new_skel_list[i].append(xy)
                        poi = xy
                        skel.remove(xy)
                        length -= 1
                        break

            if len(end_point_list[i]) == 1:
                end_point_list[i].append(new_skel_list[i][-1])

        return new_skel_list, end_point_list

    #近傍点がpoint_listに含まれているかの判断(注目点、point_listに含まれている近傍点、フラグが返される)
    def check_neighbor(self, poi, point_list):
        for i in range(-1, 2):
            for j in range(-1, 2):
                if [poi[0]+i, poi[1]+j] in point_list:
                    return poi, [poi[0]+i, poi[1]+j], 1
        
        return poi, [0, 0], 0
    
    def branch_sort(self, poi, skel, new_skel_list, branch_point, end_point_list):
        for t in range(-1, 2):
            for k in range(-1, 2):
                next_poi = [poi[0]+t, poi[1]+k]
                if next_poi in end_point_list:
                    skel.remove(next_poi)
                    continue
                if not next_poi in branch_point:
                    continue
                new_skel_list.append(next_poi)
                skel.remove(next_poi)
                branch_point.remove(next_poi)
                for l in range(-1, 2):
                    for m in range(-1, 2):
                        nearby_poi = [poi[0]+l, poi[1]+m]
                        if nearby_poi == poi or nearby_poi == next_poi:
                            continue
                        check_x = np.abs(next_poi[0] - nearby_poi[0])
                        check_y = np.abs(next_poi[1] - nearby_poi[1])
                        if check_x <= 1 and check_y <= 1 and check_x + check_y <= 2 and nearby_poi in skel:
                            skel.remove(nearby_poi)
                poi = next_poi
                break
            else:
                continue
            break
        
        return poi, skel, new_skel_list, branch_point

    # def ad_sort(self, poi, skel, new_skel_list, flat_ad_branch, flag, pre_poi, length):
    #     for t in (-1, 1):
    #         for k in (-1, 1):
    #             next_cood = [poi[0]+t, poi[1]+k]
    #             if next_cood in flat_ad_branch and not next_cood == pre_poi:
    #                 next_poi = next_cood
    #                 for a in range(-1, 2):
    #                     for b in range(-1, 2):
    #                         next_next_poi = [next_poi[0]+a, next_poi[1]+b]
    #                         if not next_next_poi in flat_ad_branch:
    #                             if next_next_poi in skel:
    #                                 new_skel_list.append(next_poi)
    #                                 new_skel_list.append(next_next_poi)
    #                                 skel.remove(next_next_poi)
    #                                 for ad in flat_ad_branch:
    #                                     if ad in skel:
    #                                         skel.remove(ad)
    #                                 flat_ad_branch = []
    #                                 length = len(skel)
    #                                 poi = next_next_poi
    #                                 flag += 1
    #                         if flag > 0:
    #                             break
    #                     if flag > 0:
    #                         break
    #             if flag > 0:
    #                 break
    #         if flag > 0:
    #             break

    #     return poi, new_skel_list, flat_ad_branch, length, flag, next_poi

    def ad_sort(self, poi, skel, new_skel_list, flat_ad_branch, flag, pre_poi, length):
        for t in (-1, 1):
            for k in (-1, 1):
                next_cood = [poi[0]+t, poi[1]+k]
                if not next_cood in flat_ad_branch or next_cood == pre_poi:
                    continue
                next_poi = next_cood

        next_next_poi_list = []
        for a in range(-1, 2):
            for b in range(-1, 2):
                next_next_poi = [next_poi[0]+a, next_poi[1]+b]
                if not next_next_poi in flat_ad_branch and next_next_poi in skel:
                    next_next_poi_list.append(next_next_poi)

        for next_next_poi in next_next_poi_list:
            for m in range(-1, 2):
                for n in range(-1, 2):
                    if m == 0 and n == 0:
                        continue
                    third_next_poi = [next_next_poi[0]+m, next_next_poi[1]+n]
                    if third_next_poi in flat_ad_branch or not third_next_poi in skel:
                        continue
                    new_skel_list.append(next_poi)
                    skel.remove(next_poi)
                    for ad in flat_ad_branch:
                        if ad in skel:
                            skel.remove(ad)
                    [skel.remove(elem) for elem in next_next_poi_list]
                    return next_next_poi, new_skel_list, [], len(skel), 1, next_poi

        return poi, new_skel_list, flat_ad_branch, length, 0, next_poi

class ConnectLine():

    def __main__(self, sorted_skel_list, end_point_list, correct_line, correct_line_zlist, interpolate_line, correct_connection_index, skel_skip_index, skip_index):
        count_list, switch_list, dis_data = self.choose_connect_line(end_point_list, sorted_skel_list)

        if not count_list == []:
            LC_skel_list, z_list, end_point_list, interpolate, correct_connection_index2 = CL.connect_line(sorted_skel_list, end_point_list, count_list, switch_list, dis_data, [], [])
            LC_region_list = region_img_index1(correct_connection_index, correct_connection_index2, skel_skip_index, skip_index)
            LC_skel_list.extend(correct_line)
            z_list.extend(correct_line_zlist)
            interpolate.extend(interpolate_line)
        else:
            LC_skel_list = []
            LC_skel_list.extend(sorted_skel_list)
            LC_skel_list.extend(correct_line)
            LC_region_list = region_img_index2(correct_connection_index, len(sorted_skel_list), skel_skip_index, skip_index)

            depth = img_copy.copy()
            depth = cv2.morphologyEx(depth, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            z_list = []
            for skel in sorted_skel_list:
                z_list.append([])
                for xy in skel:
                    z_list[-1].append(int(depth[xy[0]][xy[1]]))
            z_list.extend(correct_line_zlist)

            interpolate = interpolate_line
            for i in range(len(sorted_skel_list)):
                interpolate.append([])
            print("There are no line connections!")
        
        return LC_skel_list, z_list, interpolate, LC_region_list

    def connect_line(self, skel_list, end_point_list, combi_list, switch_list, not_combi, z_list, interpolate):
        depth = img_copy.copy() #深度画像をdepthとしてコピー
        depth = cv2.morphologyEx(depth, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) #深度画像にクロージング処理で穴埋め
        new_skel_list, new_end_point_list, new_z_list, connect, new_interpolate = [], [], [], [], []
        combi_list_copy = combi_list.copy()
        index = 0
        flag_z = 1 if len(z_list) > 0 else 0
        correct_connection_index = []
        MI = MakeImage()
        V = Visualize()

        #combi_list_copyの中を全て探索し終わるまで繰り返す
        while 1:
            combi_list_copy = np.array(combi_list_copy) #np.where、スライスを使うためにarray型に変える

            try:
                plus = np.where(combi_list_copy[:, 0:1] >= 0)[0] #combi_list_copyの各行で第１要素が正の値である行番号を取り出す
            except IndexError:
                raise ValueError("combi_listがありません connect_line内")

            #全ての要素が負であれば、whileから抜け出す
            if len(plus) == 0:
                break
            connect.append([]) #connectに新しく[]を加える
            i = plus[0] #適当に行番号を抜き出す
            count_info = combi_list_copy[i] #combi_list_copyからi行目の情報を抜き出す
            count, count2 = count_info[0], count_info[1] #count_infoには２本の線番号が含まれているのでそれぞれ抜き出す
            connect[index].extend([i]) #connectにiを追加
            combi_list_copy[i] = [-1, -1] #i行目はすでに探索済みなので負の値を格納しておく
            #繋げられる線がなくなるまで繰り返す
            while 1:
                #count, count2ともに更新されなければ、whileから抜け出す
                if count == -2 and count2 == -2 or count == -1 or count2 == -1:
                    break
                combi_list_copy = np.array(combi_list_copy) #np.where、スライスを使うためにarray型に変える
                check1 = np.where(combi_list_copy[:, 0:1] == count)[0] #combi_list_copyの各行で第１要素がcountと一致するものがあるか確かめる(つまり、線番号countとつながる線があるか探している)
                check2 = np.where(combi_list_copy[:, 1:2] == count)[0] #combi_list_copyの各行で第２要素がcountと一致するものがあるか確かめる
                check3 = np.where(combi_list_copy[:, 0:1] == count2)[0] #combi_list_copyの各行で第１要素がcount2と一致するものがあるか確かめる
                check4 = np.where(combi_list_copy[:, 1:2] == count2)[0] #combi_list_copyの各行で第２要素がcount2と一致するものがあるか確かめる
                combi_list_copy = list(combi_list_copy) #combi_list_copyをlist型に戻す
                count_copy, count2_copy = count, count2
                count, count2 = -2, -2 #count, count2を-2にしておく
                len_check1 = len(check1) #check1の要素数を取得
                len_check2 = len(check2) #check2の要素数を取得
                len_check3 = len(check3) #check3の要素数を取得
                len_check4 = len(check4) #check4の要素数を取得

                #check1に要素がある場合
                if len_check1 == 1:
                    check = check1[0] #check1の要素を取得
                    connect[index].append(check) #connectにcheckを追加
                    cou = combi_list_copy[check] #combi_list_copyのcheck行目を取得
                    #couのはじめの要素がcountである場合
                    if cou[0] == count_copy:
                        count = cou[1] #couの最後の要素が次のcountになる
                    else:
                        count = cou[0] #couの最初の要素が次のcountになる
                    combi_list_copy[check] = [-1, -1] #check行目は負の値にしておく
                #check2に要素がある場合    
                elif len_check2 == 1:
                    check = check2[0] #check2の要素を取得
                    connect[index].append(check) #connectにcheckを追加             
                    cou = combi_list_copy[check] #combi_list_copyのcheck行目を取得
                    #couのはじめの要素がcountである場合
                    if cou[0] == count_copy:
                        count = cou[1] #couの最後の要素が次のcountになる
                    else:
                        count = cou[0] #couの最初の要素が次のcountになる
                    combi_list_copy[check] = [-1, -1] #check行目は負の値にしておく

                #check3に要素がある場合
                if len_check3 == 1:
                    check = check3[0] #check3の要素を取得
                    connect[index].append(check) #connectにcheckを追加  
                    cou = combi_list_copy[check] #combi_list_copyのcheck行目を取得
                    #couのはじめの要素がcount2である場合
                    if cou[0] == count2_copy:
                        count2 = cou[1] #couの最後の要素が次のcount2になる
                    else:
                        count2 = cou[0] #couの最初の要素が次のcount2になる
                    combi_list_copy[check] = [-1, -1] #check行目は負の値にしておく
                #check4に要素がある場合               
                elif len_check4 == 1:
                    check = check4[0] #check4の要素を取得
                    connect[index].append(check) #connectにcheckを追加
                    cou = combi_list_copy[check] #combi_list_copyのcheck行目を取得
                    #couのはじめの要素がcount2である場合
                    if cou[0] == count2_copy:
                        count2 = cou[1] #couの最後の要素が次のcount2になる
                    else:
                        count2 = cou[0] #couの最初の要素が次のcount2になる
                    combi_list_copy[check] = [-1, -1] #check行目は負の値にしておく

            index += 1 #connectのindexを更新

        skel_index = 0 #接続後の線番号
        #connectに格納された線情報を基に格納する
        for precon in connect:
            new_skel_list.append([]) #new_skel_listに新たに線を格納するための[]を追加
            new_z_list.append([])
            new_interpolate.append([])
            precon = list(dict.fromkeys(precon)) #preconの重複要素を削除
            connect_combi_list, connect_switch_list = [], [] #接続する線情報、端点情報を格納するリストを作成
            count_order, switch_order = [], [] #順番を考慮してcount, switchを格納するリストを作成
            #preconの中の線・端点情報を取り出す
            for i in precon:
                connect_combi_list.append(combi_list[i]) #connect_combi_listに接続する線番号を格納
                connect_switch_list.append(switch_list[i]) #connect_switch_listに接続する端点番号を格納
            ccl = np.array(connect_combi_list) #flattenを使うためにarray型に変更
            csl = np.array(connect_switch_list) #cclに合わせてarray型に変更
            flat = list(ccl.flatten()) #cclを１次元リストに変更
            end_count = [i for i in list(set(flat)) if flat.count(i) == 1] #重複していない要素取り出す(end_countは接続する線が１つの線番号なので端線の番号となる)
            #一周してしまって端線が無い場合
            if len(end_count) == 0:
                start_count = ccl[0][0] #cclから適当にはじめの線を決める
                count_order.append(start_count) #count_orderにはじめの線番号を加える
                start_switch = csl[0][0] #はじめの線番号に合わせて、はじめの端点番号を決める
                switch_order.append(start_switch) #switch_orderにはじめの端点番号を加える
            #端線がある場合
            else:
                start_count = end_count[0] #end_countから適当にはじめの線を決める
                count_order.append(start_count) #count_orderにはじめの線番号を加える
                num = np.where(start_count == ccl)[0][0] #cclの中からstart_countが含まれる行番号を取得する
                num2 = np.where(start_count == ccl[num])[0][0] #ccl[num]の中からstart_countが含まれる列番号を取得する
                start_switch = csl[num][num2] #はじめの線番号に合わせて、はじめの端点番号を決める
                switch_order.append(start_switch) #switch_orderにはじめの端点番号を加える

            next_connect_index = np.where(start_count == ccl)[0][0] #はじめの線、端点に接続する線、端点番号を求める
            #count, switchはそれぞれ２つで１組として格納されているため添字も０か１
            for j in range(0, 2):
                #cclの中でstart_countが格納されている行のうち、start_countではない方を次のcountとする
                if not start_count == ccl[next_connect_index][j]:
                    next_count = ccl[next_connect_index][j] #次のcountである、next_countを定義
                    count_order.append(next_count) #count_orderにnext_countを格納する
                    next_switch = csl[next_connect_index][j] #次のswitchである、next_switchを定義
                    switch_order.append(next_switch) #switch_orderにnext_switchを格納する
                    break
            ccl[next_connect_index] = [-1, -1] #cclのnext_connect_index行目はすでに探索し終わったので、負の値にしておく
            #cclの中身を接続する順番に整頓していく
            for i in range(0, len(ccl)):
                pre_connect_index = next_connect_index #前のnext_connect_indexをpre_connect_indexに入れておく
                pre_j = j #前のjをpre_jに入れておく
                next_connect_index = np.where(next_count == ccl)[0] #next_connect_indexをcclの中でnext_countが含まれる行番号として更新
                #もしnext_count_indexに何も入っていなければ終了
                if len(next_connect_index) == 0:
                    break
                else:
                    next_connect_index = next_connect_index[0] #next_connect_indexから要素を抜き出す
                #count, switchはそれぞれ２つで１組として格納されているため添字も０か１
                for j in range(0, 2):
                    #cclの中でnext_countが格納されている行のうち、next_countではない方を次のcountとする
                    if not next_count == ccl[next_connect_index][j]:
                        next_count = ccl[next_connect_index][j] #次のcountである、next_countを定義
                        count_order.append(next_count) #count_orderにnext_countを格納する
                        pre_switch = 1 - csl[pre_connect_index][pre_j] #前の線における今回接続する端点番号を取得
                        next_switch = csl[next_connect_index][j] #今の線における今回接続する端点番号を取得
                        switch_order.append(pre_switch) #switch_orderにpre_switchを格納する
                        switch_order.append(next_switch) #switch_orderにnext_switchを格納する
                        break
                ccl[next_connect_index] = [-1, -1] #cclのnext_connect_index行目はすでに探索し終わったので、負の値にしておく 
            
            count = count_order[0]
            switch = switch_order[0]
            xy = [int(end_point_list[count][switch][0]), int(end_point_list[count][switch][1])]
            if xy == skel_list[count][0]:
                reverse = list(reversed(skel_list[count]))
                if flag_z == 1:
                    reverse_z = list(reversed(z_list[count]))
                for i, skel_xy in enumerate(reverse):
                    if flag_z == 1:
                        z = reverse_z[i]
                    else:
                        z = int(depth[skel_xy[0]][skel_xy[1]])
                    new_z_list[skel_index].append(z)
                new_skel_list[skel_index].extend(reverse)
            else:
                for i, skel_xy in enumerate(skel_list[count]):
                    if flag_z == 1:
                        z = z_list[count][i]
                    else:
                        z = int(depth[skel_xy[0]][skel_xy[1]])
                    new_z_list[skel_index].append(z)
                new_skel_list[skel_index].extend(skel_list[count])
            for i in range(1, len(count_order)):
                line_xy, line_z = [], []
                pre_count = count_order[i-1]
                current_count = count_order[i]
                pre_switch = switch_order[2*(i-1)]
                current_switch = switch_order[2*(i-1)+1]
                pre_x, pre_y, pre_z = int(end_point_list[pre_count][pre_switch][0]), int(end_point_list[pre_count][pre_switch][1]), int(end_point_list[pre_count][pre_switch][2])
                current_x, current_y, current_z = int(end_point_list[current_count][current_switch][0]), int(end_point_list[current_count][current_switch][1]), int(end_point_list[current_count][current_switch][2])
                # print("(pre_x, pre_y, pre_z) = {}".format((pre_x, pre_y, pre_z)))
                if np.abs(pre_x - current_x) > np.abs(pre_y - current_y):
                    if pre_x == current_x:
                        tan = 0
                        tan2 = 0
                    else:
                        tan = (pre_y - current_y) / (pre_x - current_x)
                        tan2 = (pre_z - current_z) / (pre_x - current_x)
                    if pre_x <= current_x:
                        for x in range(pre_x+1, current_x, 1):
                            xt = x - pre_x
                            y = int(tan * xt) + pre_y
                            z = int(tan2 * xt) + pre_z
                            line_xy.append([x, y])
                            line_z.append(z)           
                            # print("(x, y, z) = {}".format((x, y, z)))
                    else:
                        for x in range(pre_x-1, current_x, -1):
                            xt = x - pre_x
                            y = int(tan * xt) + pre_y
                            z = int(tan2 * xt) + pre_z
                            line_xy.append([x, y])
                            line_z.append(z)
                            # print("(x, y, z) = {}".format((x, y, z)))
                else:
                    if pre_y == current_y:
                        tan = 0
                        tan2 = 0
                    else:
                        tan = (pre_x - current_x) / (pre_y - current_y)
                        tan2 = (pre_z - current_z) / (pre_y - current_y)
                    if pre_y <= current_y:
                        for y in range(pre_y+1, current_y, 1):
                            yt = y - pre_y
                            x = int(tan * yt) + pre_x
                            z = int(tan2 * yt) + pre_z
                            line_xy.append([x, y])
                            line_z.append(z)
                            # print("(x, y, z) = {}".format((x, y, z)))
                    else:
                        for y in range(pre_y-1, current_y, -1):
                            yt = y - pre_y
                            x = int(tan * yt) + pre_x
                            z = int(tan2 * yt) + pre_z
                            line_xy.append([x, y])
                            line_z.append(z)
                            # print("(x, y, z) = {}".format((x, y, z)))
                # print("(current_x, current_y, current_z) = {}".format((current_x, current_y, current_z)))
                # print("")
                new_skel_list[skel_index].extend(line_xy)
                new_z_list[skel_index].extend(line_z)
                new_interpolate[skel_index].extend(line_xy)
                xy = [current_x, current_y]
                if xy == skel_list[current_count][0]:
                    for i, skel_xy in enumerate(skel_list[current_count]):
                        if flag_z == 1:
                            z = z_list[current_count][i]
                        else:
                            z = int(depth[skel_xy[0]][skel_xy[1]])
                        new_z_list[skel_index].append(z)
                    new_skel_list[skel_index].extend(skel_list[current_count])
                else:          
                    reverse = list(reversed(skel_list[current_count]))
                    if flag_z == 1:
                        reverse_z = list(reversed(z_list[current_count]))
                    for skel_xy in reverse:
                        if flag_z == 1:
                            z = reverse_z[i]
                        else:
                            z = int(depth[skel_xy[0]][skel_xy[1]])
                        new_z_list[skel_index].append(z)
                    new_skel_list[skel_index].extend(reverse)

            # image = MI.make_image(new_skel_list[skel_index], 1)
            # V.visualize_3img(image, image, image)
            skel_index += 1
        
        for i, skel in enumerate(new_skel_list):
            new_end_point_list.append([[skel[0][0], skel[0][1], new_z_list[i][0]], [skel[-1][0], skel[-1][1], new_z_list[i][-1]]])

        if len(interpolate) > 0:
            for index, choice in enumerate(connect):
                combis = []
                for i in choice:
                    combis.extend(combi_list[i])
                combis = list(set(combis))
                for i in combis:
                    new_interpolate[index].extend(interpolate[i])      


        for solo_index in not_combi:
            new_z_list.append([])
            for skel_xy in skel_list[solo_index]:
                z = int(depth[skel_xy[0]][skel_xy[1]])
                new_z_list[skel_index].append(z)
            new_skel_list.append(skel_list[solo_index])
            new_end_point_list.append(end_point_list[solo_index])
            if len(interpolate) > 0:
                new_interpolate.append(interpolate[solo_index])
            else:
                new_interpolate.append([])
            skel_index += 1
        
        for concon in connect:
            connection_index = []
            for index in concon:
                connection_index.extend(combi_list[index])
            correct_connection_index.append(connection_index)
        correct_connection_index.extend(not_combi)

        return new_skel_list, new_z_list, new_end_point_list, new_interpolate, correct_connection_index

    def line_orientation(self, skel_list):
        vector_xlist, vector_ylist = [], []
        count = 0
        MI = MakeImage()
        V = Visualize()

        # image = np.zeros((height, width))
        # for skel in skel_list:
        #     skel_img = MI.make_image(skel, 1)
        #     image += skel_img

        for skel in skel_list:
            vector_xlist.append([])
            vector_ylist.append([])
            max_len = len(skel)
            length, interval = 10, 5
            if max_len <= length:
                length = max_len - 1
                interval = max_len//5

            vector_x, vector_y = 0, 0
            poi = skel[0]
            if interval == 0:
                interval = 1
            for i in range(interval-1, length, interval):
                next_poi = skel[i]
                vector_x -= next_poi[0]-poi[0]
                vector_y -= next_poi[1]-poi[1]  
                poi = next_poi  
            vector_x, vector_y = int(vector_x/2), int(vector_y/2)
            vector_xlist[count].append(vector_x)
            vector_ylist[count].append(vector_y)
            # image = cv2.arrowedLine(image, (skel[0][1], skel[0][0]), (skel[0][1]+vector_y, skel[0][0]+vector_x), 1, thickness=2)

            vector_x, vector_y = 0, 0
            poi = skel[-1]
            for i in range(max_len-1-interval, max_len-1-length, -interval):
                next_poi = skel[i]
                vector_x -= next_poi[0]-poi[0]
                vector_y -= next_poi[1]-poi[1]
                poi = next_poi
            vector_x, vector_y = int(vector_x/2), int(vector_y/2)
            vector_xlist[count].append(vector_x)
            vector_ylist[count].append(vector_y)
            # image = cv2.arrowedLine(image, (skel[-1][1], skel[-1][0]), (skel[-1][1]+vector_y, skel[-1][0]+vector_x), 1, thickness=2) 

            count += 1

        # V.visualize_1img(image)
        
        return vector_xlist, vector_ylist

    def choose_connect_line(self, end_point_list, skel_list):
        count_list, switch_list = [], []
        depth = img_copy.copy() #深度画像をdepthとしてコピー
        depth = cv2.morphologyEx(depth, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) #深度画像にクロージング処理で穴埋め
        MI, V = MakeImage(), Visualize()

        vector_xlist, vector_ylist = self.line_orientation(skel_list)

        #終点のz座標を取得する
        for i in range(0, len(end_point_list)):
            for j in range(0, 2):
                z = depth[end_point_list[i][j][0]][end_point_list[i][j][1]] #深度画像で終点の座標の値を求める
                end_point_list[i][j] = list(np.append(end_point_list[i][j], round(z, 2))) #z座標の値を小数第二位で丸め、end_point_listに加える

        # image = np.zeros((height, width))
        # for skel in skel_list:
        #     image += MI.make_image(skel, 1)
        # image = gray2color(image)
        # for double_point in end_point_list:
        #     for point in double_point:
        #         # image_copy = image.copy()
        #         x = int(point[0])
        #         y = int(point[1])
        #         image_copy = cv2.circle(image, (y, x), 3, (255, 0, 0), -1)
        #         V.visualize_1img(image_copy)
        # raise ValueError

        cost, index = [], []
        min_distance_threshold = 15
        cost_threshold = 200
        alpha = 1
        beta = 3
        gamma = 3
        delta = 2

        # allimg = np.zeros((height, width, 3))
        # alpha = 2
        # for skel, end_point, vxs, vys in zip(skel_list, end_point_list, vector_xlist, vector_ylist):
        #     simg = MI.make_image(skel, 1)
        #     simg = gray2color(simg)
        #     x1 = int(end_point[0][0])
        #     y1 = int(end_point[0][1])
        #     x2 = int(end_point[1][0])
        #     y2 = int(end_point[1][1])
        #     simg = cv2.circle(simg, (y1, x1), 2, (255, 0, 0), -1)
        #     simg = cv2.circle(simg, (y2, x2), 2, (0, 0, 255), -1)
        #     vx1 = vxs[0]
        #     vy1 = vys[0]
        #     vx2 = vxs[1]
        #     vy2 = vys[1]
        #     simg = cv2.arrowedLine(simg, (y1, x1), (y1+alpha*vy1, x1+alpha*vx1), (255, 0, 0), 1)
        #     simg = cv2.arrowedLine(simg, (y2, x2), (y2+alpha*vy2, x2+alpha*vx2), (0, 0, 255), 1)
        #     # V.visualize_1img(simg)
        #     allimg += simg
        # V.visualize_1img(allimg)
        # raise ValueError

        #終点同士の距離を求める
        for count in range(0, len(end_point_list)): #countは細線の番号を表す(end_point_listの第1添字)
            cost.append([[] for t in range(0, 2)]) #costに細線の数だけ空の配列を加える
            index.append([[] for t in range(0, 2)]) #indexに細線の数だけ空の配列を加える
            for switch in range(0, 2): #switchは終点の番号を表す(終点は細線一本につき2点ある)(end_point_listの第2添字)
                end1 = [end_point_list[count][switch][0], end_point_list[count][switch][1]]
                theta1 = int(math.degrees(math.atan2(vector_ylist[count][switch], vector_xlist[count][switch])))
                depth1 = end_point_list[count][switch][2]
                for i in range(count+1, len(end_point_list)): #ペアとなる点が含まれる細線番号(end_point_listの第1添字)
                    for j in range(0, 2): #ペアとなる点の終点の番号(end_point_listの第2添字)
                        end2 = [end_point_list[i][j][0], end_point_list[i][j][1]]
                        theta2 = int(math.degrees(math.atan2(vector_ylist[i][j], vector_xlist[i][j])))
                        depth2 = end_point_list[i][j][2]

                        sum_theta = np.abs(np.abs(theta1 - theta2) - 180)

                        distance = int(np.sqrt((end1[0]-end2[0])**2 + (end1[1]-end2[1])**2))

                        vx12 = end2[0] - end1[0]
                        vy12 = end2[1] - end1[1]
                        theta12 = int(math.degrees(math.atan2(vy12, vx12)))
                        dif_theta1 = np.abs(theta12 - theta1)
                        if dif_theta1 > 180:
                            dif_theta1 = 360 - dif_theta1
                        theta21 = int(math.degrees(math.atan2(-vy12, -vx12)))
                        dif_theta2 = np.abs(theta21 - theta2)
                        if dif_theta2 > 180:
                            dif_theta2 = 360 - dif_theta2
                        dif_theta = (dif_theta1 + dif_theta2)//2

                        dif_depth = np.abs(depth1 - depth2)

                        if distance <= min_distance_threshold:
                            par_cost = alpha*distance + beta*sum_theta + gamma*dif_depth//3 + delta*dif_depth
                        else:
                            par_cost = alpha*distance + beta*sum_theta + gamma*dif_theta + delta*dif_depth
          
                        cost[count][switch].append(par_cost)
                        # print(cost[count][switch])
                        index[count][switch].append((i, j)) #countとswitchが注目画素の添字、iとjがペアとなる画素の添字

        image = np.zeros((height, width))
        flat_cost = sum(sum(cost, []), []) #costを一次元リストに変換
        if flat_cost == []:
            return [], [], []
        while 1:    
            min_cost = min(flat_cost) #最小となるコストを探索
            if min_cost > cost_threshold:
                break
            #min_costとなる終点を探索
            for count in range(0, len(cost)):
                for switch in range(0, 2):
                    #min_costとなる終点がend_point_list[count][switch]であった場合
                    if min_cost in cost[count][switch]:
                        count2 = np.where(np.array(cost[count][switch]) == min_cost)[0][0] #ある終点とペアになる点は複数あるため、最小のコストの点だけを取り出す
                        i = int(index[count][switch][count2][0]) #ペアとなる点のend_point_listでの添字を得る(countの方)
                        j = int(index[count][switch][count2][1]) #ペアとなる点のend_point_listでの添字を得る(switchの方)
                        #注目画素に対する削除
                        cost[count][switch] = [] #注目画素のコストをcostから削除
                        index[count][switch] = [] #注目画素の添字をindexから削除
                        #今回の注目画素がペアの画素となる場合のcost,index削除
                        delete_point = [(cou1, cou2, cou3) for cou1 in range(0, len(index)) for cou2 in [0, 1] for cou3 in range(0, len(index[cou1][cou2])) if index[cou1][cou2][cou3] == (count, switch)] #indexの中から(cost, switch)となる添字を見つける(つまり、一度結んでしまえばその終点は使わないので、ペアとなる点の添字をindexから削除しておく)
                        for p in delete_point:
                            del index[p[0]][p[1]][p[2]]
                            del cost[p[0]][p[1]][p[2]]

                        cost[i][j] = [] #今回ペアの画素が注目画素となる場合のコストをcostから削除
                        index[i][j] = [] #今回ペアの画素が注目画素となる場合の添字をindexから削除
                        #ペアの画素のcost,index削除                   
                        delete_point = [(cou1, cou2, cou3) for cou1 in range(0, len(index)) for cou2 in [0, 1] for cou3 in range(0, len(index[cou1][cou2])) if index[cou1][cou2][cou3] == (i, j)] #indexの中から(i, j)となる添字を見つける(つまり、一度結んでしまえばその終点は使わないので、ペアとなる点の添字をindexから削除しておく)
                        for p in delete_point:
                            del index[p[0]][p[1]][p[2]]
                            del cost[p[0]][p[1]][p[2]]

                        break
                else:
                    continue
                break

            # x0 = int(end_point_list[count][switch][0]) #注目画素のx座標を取得
            # y0 = int(end_point_list[count][switch][1]) #注目画素のy座標を取得
            # x = int(end_point_list[i][j][0]) #ペアの画素のx座標を取得
            # y = int(end_point_list[i][j][1]) #ペアの画素のy座標を取得
            count_list.append([count, i]) #count_listは結ぶべき線が２コ１で入ったリスト
            switch_list.append([switch, j]) #switch_listは結ぶべき端点が２コ１で入ったリスト
    
            # image = cv2.line(image, (y0, x0), (y, x), (1, 0, 0), 1) #終点を結ぶ
            # image = cv2.circle(image, (y0, x0), 3, (0,0,1), -1)
            # image = cv2.circle(image, (y, x), 3, (0,0,1), -1)

            flat_cost = sum(sum(cost, []), []) #costを一次元リストに変換
            #costの中身が空なら終了
            if not flat_cost:
                break

        # V.visualize_1img(image)

        # depth = gray2color(depth/255)
        # image[image[:,:,0]>0] = (0,1,1)
        # depth = depth - image
        # visualize_skeleton(image, depth, depth)
        data = list(itertools.chain.from_iterable(count_list))
        data = list(set(data))
        check_data = list(range(len(skel_list)))
        dis_data = list(set(check_data) - set(data))

        return count_list, switch_list, dis_data

    def match_end_point(self, end_point_list, center_end_point_list):
        depth = img_copy.copy()

        for i, end_points in enumerate(end_point_list):
            end1 = list(end_points[0])
            end2 = list(end_points[1])
            z1 = int(depth[end1[0]][end1[1]])
            z2 = int(depth[end2[0]][end2[1]])
            end1.append(z1)
            end2.append(z2)
            end_point_list[i][0] = end1
            end_point_list[i][1] = end2            

        return end_point_list

def insert2list(target_list, insert_index):
    for x in insert_index:
        for i, li in enumerate(target_list):
            if type(li) is list:
                for j, y in enumerate(li):
                    if y >= x:
                        target_list[i][j] += 1
            else:
                if li >= x:
                    target_list[i] += 1
    return target_list

def make_1and2DimentionalList2flat(target_list):
    flat = []
    for li in target_list:
        if type(li) is list:
            for x in li:
                flat.append(x)
        else:
            flat.append(li)
    return flat

def region_img_index1(index_list1, index_list2, skel_skip_index, skip_index):
    skel_skip_index.sort()
    skip_index.sort()
    index_list2 = insert2list(index_list2, skel_skip_index)
    index_list1 = insert2list(index_list1, skip_index)
    flat_index1 = make_1and2DimentionalList2flat(index_list1)
    flat_index1.sort()
    index_list2 = insert2list(index_list2, flat_index1)
    index_list2.extend(index_list1)

    return index_list2

def region_img_index2(index_list1, len_skel_list, skel_skip_index, skip_index):
    index_list2 = []
    for i in range(len_skel_list):
        index_list2.append(i)
    skel_skip_index.sort()
    skip_index.sort()
    index_list2 = insert2list(index_list2, skel_skip_index)
    index_list1 = insert2list(index_list1, skip_index)
    flat_index1 = make_1and2DimentionalList2flat(index_list1)
    flat_index1.sort()
    index_list2 = insert2list(index_list2, flat_index1)
    index_list2.extend(index_list1)

    return index_list2

class GaussLinkingIintegral():

    def calculate_GLI(self, skel_list, z_list):
        len_skel = len(skel_list)
        GLI = np.zeros((len_skel, len_skel))
        objs_GLI = np.zeros((len_skel))
        image = np.zeros((height, width, 3))
        interval = 10
        MI = MakeImage()
        V = Visualize()

        if len_skel == 1:
            return [0], [0]

        ###########線iと線jからintervalごとに点を取り出しGLIを計算####################################
        for i in range(0, len_skel-1):
            skel1 = skel_list[i]
            z1 = z_list[i]
            for j in range(i+1, len_skel):
                num = 0
                skel2 = skel_list[j]
                z2 = z_list[j]
                for k in range(0, len(skel1)-interval, interval):
                    r1 = np.array([skel1[k][0], skel1[k][1], z1[k]])
                    next_r1 = np.array([skel1[k+interval][0], skel1[k+interval][1], z1[k+interval]])
                    dr1 = next_r1 - r1
                    # image = cv2.line(image, (r1[1], r1[0]), (next_r1[1], next_r1[0]), (1,0,0), 1)
                    # image = cv2.circle(image, (r1[1], r1[0]), 3, (0,0,1), -1)
                    for l in range(0, len(skel2)-interval, interval):
                        r2 = np.array([skel2[l][0], skel2[l][1], z2[l]])
                        next_r2 = np.array([skel2[l+interval][0], skel2[l+interval][1], z2[l+interval]])
                        dr2 = next_r2 - r2
                        r12 = r2 - r1
                        norm_r12 = np.linalg.norm(r12)
                        num += np.dot(np.cross(dr1, dr2), r12) / (norm_r12**3)
                GLI[i][j] = abs(num)
                GLI[j][i] = abs(num)       
            # V.visualize_skeleton(image, image, image)
            # image = np.zeros((height, width, 3))

        for i in range(0, len_skel): 
            objs_GLI[i] = np.sum(GLI[i]) / ((len(skel1)-interval)//interval+1)
        
        for i in range(0, len_skel):
            ske = MI.make_image(skel_list[i],1)
            if i == 0:
                image[ske>0] = (255, 0, 0)
            elif i == 1:
                image[ske>0] = (0, 255, 0)
            elif i == 2:
                image[ske>0] = (0, 0, 255)
            elif i == 3:
                image[ske>0] = (255, 255, 0)
            elif i == 4:
                image[ske>0] = (255, 0, 255)
            elif i == 5:
                image[ske>0] = (0, 255, 255)
            elif i == 6:
                image[ske>0] = (255, 255, 255)
            elif i == 7:
                image[ske>0] = (0.5, 0, 0.8)
            elif i == 8:
                image[ske>0] = (0.2, 0.7, 0.5)
                
        # V.visualize_3img(image,image,image)

        return objs_GLI, GLI

    def recalculate_GLI(self, skel_list, interpolate, GLI, z_list):
        MI, V = MakeImage(), Visualize()
        image_list, dilate_image_list = [], []
        cross_matrix = [[[] for i in range(0, len(skel_list))] for j in range(0, len(skel_list))]
        for skel in skel_list:
            image = MI.make_image(skel, 1)
            image_list.append(image)
            dilate_image = cv2.dilate(image, np.ones((3, 3)), iterations=1)
            dilate_image_list.append(dilate_image)

        for i in range(0, len(skel_list)-1):
            dilate_image1 = dilate_image_list[i]
            image1 = image_list[i]
            for j in range(i+1, len(skel_list)):
                dilate_image2 = dilate_image_list[j]
                image2 = image_list[j]
                sum_image = dilate_image1 + dilate_image2
                sum_image[sum_image<2] = 0
                nlabels, labels, stats, _ = cv2.connectedComponentsWithStats(sum_image)
                for num in range(1, nlabels):
                    if stats[num][4] > 5:
                        label = labels.copy()
                        label[label != num] = 0
                        label[label > 0] = 1
                        sum_label1 = label + image1
                        sum_label2 = label + image2
                        # V.visualize_2img(sum_label1, sum_label2)
                        cross_point1 = np.argwhere(sum_label1 == 2)
                        try :
                            cross_point1 = list(cross_point1[0])
                        except IndexError:
                            continue                       
                        cross_point2 = np.argwhere(sum_label2 == 2)
                        try:
                            cross_point2 = list(cross_point2[0])
                        except IndexError:
                            continue

                        # sample = np.zeros((height, width, 3))
                        # sample = cv2.circle(sample, (cross_point1[1], cross_point1[0]), 3, (255, 0, 0), -1)
                        # inter = MI.make_image(interpolate[i], 1)
                        # sample[inter>0] = (0, 0, 255)

                        # sample2 = np.zeros((height, width, 3))
                        # sample2 = cv2.circle(sample2, (cross_point2[1], cross_point2[0]), 3, (255, 0, 0), -1)
                        # inter2 = MI.make_image(interpolate[j], 1)
                        # sample2[inter2>0] = (0, 0, 255)

                        # V.visualize_3img(image1+image2, sample, sample2)

                        if cross_point1 in interpolate[i] and cross_point2 in interpolate[j]:
                            index1 = skel_list[i].index(cross_point1)
                            index2 = skel_list[j].index(cross_point2)
                            z1 = z_list[i][index1]
                            z2 = z_list[j][index2]
                            if z1 > z2:
                                cross_matrix[i][j].append([cross_point1[0], cross_point1[1], 0])
                                cross_matrix[j][i].append([cross_point2[0], cross_point2[1], 1])
                            else:
                                cross_matrix[j][i].append([cross_point2[0], cross_point2[1], 0])
                                cross_matrix[i][j].append([cross_point1[0], cross_point1[1], 1])
                        elif cross_point1 in interpolate[i]:
                            cross_matrix[j][i].append([cross_point2[0], cross_point2[1], 0])
                            cross_matrix[i][j].append([cross_point1[0], cross_point1[1], 1])
                        elif cross_point2 in interpolate[j]:
                            cross_matrix[i][j].append([cross_point1[0], cross_point1[1], 0])
                            cross_matrix[j][i].append([cross_point2[0], cross_point2[1], 1])
                        else:
                            continue
                            # raise ValueError("二物体の位置関係が不明瞭です。recalculate_GLI")

        for i in range(0, len(cross_matrix)):
            for j in range(0, len(cross_matrix[i])):
                if i == j:
                    continue
                cross_points = cross_matrix[i][j]
                count, sum_check = 0, 0
                for point in cross_points:
                    sum_check += point[2]
                    count += 1
                if sum_check == 0:
                    GLI[i][j] /= 100
        
        return GLI , cross_matrix

    def select_obj(self, GLI, skel_list, z_list):
        objs_GLI = []
        for i in range(0, len(GLI)):
            objs_GLI.append(0)
            for j in range(0, len(GLI)):
                objs_GLI[i] += GLI[i][j]
        normalize_objs_GLI = min_max_x(objs_GLI)

        len_list, dif_len = [], []
        for skel, z in zip(skel_list, z_list):
            length = len(skel)
            dif = np.abs(full_length - length)
            normalize_dif = dif / full_length
            len_list.append(length)
            dif_len.append(normalize_dif)
        dif_len = np.array(dif_len)

        ave_z_list = []
        for z, length in zip(z_list, len_list):
            sum_z = np.array(z).sum()
            ave_z = sum_z / length 
            ave_z_list.append(ave_z)
        normalize_ave_z = 1 - min_max_x(ave_z_list)
  
        alpha, beta, gamma = 2, 2, 1
        result_value = alpha*normalize_objs_GLI + beta*dif_len + gamma*normalize_ave_z

        sort_value = np.argsort(result_value)

        top_5_index = []
        top_5_max_GLI_index = []
        for i in range(5):
            try:
                index = sort_value[i]
            except IndexError:
                return top_5_index, top_5_max_GLI_index, objs_GLI
            top_5_index.append(index)
            top_5_max_GLI_index.append(np.argmax(GLI[index]))

        return top_5_index, top_5_max_GLI_index, objs_GLI

def min_max_x(x):
    x = np.array(x)
    max_x = x.max(keepdims=True)
    min_x = x.min(keepdims=True)
    min_max_x = (x - min_x) / (max_x - min_x)
    return min_max_x

class Graspability():

    def __init__(self):
        self.TEMPLATE_SIZE = 100
        self.HAND_WIDTH = 10
        self.HAND_THICKNESS_X = 2
        self.HAND_THICKNESS_Y = 5
        self.BEFORE_TO_AFTER = 500/100
        self.cutsize = self.TEMPLATE_SIZE/2
        self.margin = 50
        self.GaussianKernelSize = 75
        self.GaussianSigma = 25
        self.InitialHandDepth = 0
        self.FinalHandDepth = 201
        self.HandDepthStep = 25
        self.GripperD = 15
        self.interval = 15
        self.error_width = 50

    def __main__(self, objs_GLI, LC_skel_list, LC_region_list, interpolate, GLI_matrix, z_list, region_list2):
        GLI = GaussLinkingIintegral()

        if len(objs_GLI) > 1:
            GLI_matrix, cross_matrix = GLI.recalculate_GLI(LC_skel_list, interpolate, GLI_matrix, z_list)
            top_5_index, top_5_max_GLI_index, objs_GLI = GLI.select_obj(GLI_matrix, LC_skel_list, z_list)

            print("GLI caluculation is succeeded!")

            all_region = np.zeros((height, width))
            for region in region_list2:
                rimg = MI.make_image(region, 1)
                all_region += rimg

            count_iterations = 0
            warn_area = 30
            for obj_index, max_GLI_index in zip(top_5_index, top_5_max_GLI_index):
                optimal_grasp = self.find_grasp_point(LC_skel_list, obj_index, interpolate, cross_matrix, region_list2, LC_region_list, all_region)
                if optimal_grasp == []:
                    count_iterations += 1
                    continue
                elif optimal_grasp[0][3][0] < warn_area or optimal_grasp[0][3][0] > height-warn_area or optimal_grasp[0][3][1] < warn_area or optimal_grasp[0][3][1] > width-warn_area:
                    count_iterations += 1
                    print("This grasp position is danger, so next coodinate will be calculate!")
                    continue 
                else:
                    optimal_grasp2 = self.find_grasp_point2(LC_skel_list, obj_index, max_GLI_index, cross_matrix, region_list2, LC_region_list, all_region)
                    break

            if count_iterations == 5:
                raise ValueError("掴むことができる物体がありません")

            return optimal_grasp, optimal_grasp2, obj_index
        else:
            optimal_grasp = self.find_grasp_point(LC_skel_list, 0, [[]], [[]], region_list2, LC_region_list, all_region)
            return optimal_grasp, [], 0

    def make_template(self):
        L1x = int((self.TEMPLATE_SIZE / 2) - (((self.HAND_WIDTH / 2) + self.HAND_THICKNESS_X) * self.BEFORE_TO_AFTER))
        L3x = int((self.TEMPLATE_SIZE / 2) - (((self.HAND_WIDTH / 2) + 0) * self.BEFORE_TO_AFTER))
        R1x = int((self.TEMPLATE_SIZE / 2) + (((self.HAND_WIDTH / 2) + 0) * self.BEFORE_TO_AFTER))
        R3x = int((self.TEMPLATE_SIZE / 2) + (((self.HAND_WIDTH / 2) + self.HAND_THICKNESS_X) * self.BEFORE_TO_AFTER))

        L1y = int((self.TEMPLATE_SIZE / 2) - ((self.HAND_THICKNESS_Y / 2) * self.BEFORE_TO_AFTER))
        L3y = int((self.TEMPLATE_SIZE / 2) + ((self.HAND_THICKNESS_Y / 2) * self.BEFORE_TO_AFTER))
        R1y = int((self.TEMPLATE_SIZE / 2) - ((self.HAND_THICKNESS_Y / 2) * self.BEFORE_TO_AFTER))
        R3y = int((self.TEMPLATE_SIZE / 2) + ((self.HAND_THICKNESS_Y / 2) * self.BEFORE_TO_AFTER))
 
        Hc_original = np.zeros((self.TEMPLATE_SIZE, self.TEMPLATE_SIZE))
        cv2.rectangle(Hc_original, (L1x, L1y), (L3x, L3y), (255, 255, 255), -1)
        cv2.rectangle(Hc_original, (R1x, R1y), (R3x, R3y), (255, 255, 255), -1)

        Hc_original_list = []

        for i in range(3):
            Hc_original = np.zeros((self.TEMPLATE_SIZE, self.TEMPLATE_SIZE))
            cv2.rectangle(Hc_original, (L1x+5*i, L1y), (L3x+5*i, L3y), (255, 255, 255), -1)
            cv2.rectangle(Hc_original, (R1x-5*i, R1y), (R3x-5*i, R3y), (255, 255, 255), -1)
            Hc_original_list.append(Hc_original)

        return Hc_original_list

    def cv2pil(self, image):
        new_image = image.copy()
        if new_image.ndim == 2:  # モノクロ
            pass
        elif new_image.shape[2] == 3:  # カラー
            new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
        elif new_image.shape[2] == 4:  # 透過
            new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
        new_image = Image.fromarray(new_image)

        return new_image

    #把持位置決定
    def find_grasp_point(self, skel_list, obj_index, interpolate, cross_matrix, region_list, LC_region_list, all_region):
        grasp_obj = skel_list[obj_index]                                            #obj_indexは把持対象物の番号
        grasp_region_index = LC_region_list[obj_index]
        length = len(grasp_obj)                                                     #grasp_objの画素数
        obj_interpolate = interpolate[obj_index]                                    #grasp_objの補間部分
        optimal_grasp = []                    #最適な５つの把持位置がoptimal_graspに保存される
        Hc_original_list = self.make_template()                             #グリッパーのテンプレート

        ##############把持対象物が他の物体の下側に存在する場合、その交点を見つける#######################
        underlapped_point = []
        obj_cross_points = cross_matrix[obj_index]                                  
        for points in obj_cross_points:                                             
            for point in points:
                if point[2] == 1:
                    index = grasp_obj.index([point[0], point[1]])
                    underlapped_point.append([point[0], point[1], index])
        #################################################################################################

        ##############把持位置探索の範囲を絞る(start_index, finish_index)################################
        underlapped_point.append([grasp_obj[0][0], grasp_obj[0][1], 0])                              #把持対象物の始点の座標を格納
        underlapped_point.append([grasp_obj[length-1][0], grasp_obj[length-1][1], length-1])         #把持対象物の終点の座標を格納
        underlapped_point = np.array(underlapped_point)
        sorted_underlapped_point = underlapped_point[np.argsort(underlapped_point[:, 2])]            #始点からの距離でソート

        point_distance = []
        for i in range(len(sorted_underlapped_point)-1):
            point_distance.append(sorted_underlapped_point[i+1][2] - sorted_underlapped_point[i][2]) #sorted_underlapped_pointに格納されている点と点の距離を求める
        max_underlapped_point_index = np.argmax(point_distance)                                      #点間の距離が最長の部分を見つける
        start_index = sorted_underlapped_point[max_underlapped_point_index][2]                       #該当の点の一つをstart_indexとする
        finish_index = sorted_underlapped_point[max_underlapped_point_index+1][2]                    #該当の点の一つをfinish_indexとする
        if finish_index == length - 1:
            finish_index -= (self.interval + 1)
        #################################################################################################

        if type(grasp_region_index) is list:
            contact_image = np.zeros((height, width))
            for rind in grasp_region_index:
                rimg = MI.make_image(region_list[rind], 1)
                contact_image += rimg
        else:
            contact_image = MI.make_image(region_list[grasp_region_index], 1)

        if not type(grasp_region_index) is list:
            grasp_region_index = [grasp_region_index]

        depth = img_copy.copy()
        depth[all_region==0] *= 0
        depth[contact_image>0] = 0
        contact_depth = depth * contact_image
        # V.visualize_2img(contact_image, depth)

        for i in range(start_index, finish_index - 1, self.interval):
            poi = grasp_obj[i]
            if poi in obj_interpolate:
                continue
            try:
                next_poi = grasp_obj[i+self.interval]
            except IndexError:
                next_poi = grasp_obj[length-1]
            z = contact_depth[poi[0]][poi[1]]+30
            optimal_grasp = self.graspability(poi, next_poi, optimal_grasp, Hc_original_list, depth, z, i//self.interval)
            # depth = cv2.circle(depth, (poi[1], poi[0]), 1, (255, 0, 0), -1)

        ##############################################################################
        # vec = np.array(next_poi) - np.array(poi)
        # poi = grasp_obj[-1]
        # next_poi = poi + vec
        # if not poi in obj_interpolate:
        #     optimal_grasp = self.graspability(poi, next_poi, optimal_grasp, Hc_original_list, contact_image, depth)
        #     depth = cv2.circle(depth, (poi[1], poi[0]), 1, (255, 0, 0), -1)
        ##############################################################################

        return optimal_grasp

    def find_grasp_point2(self, skel_list, obj_index, max_GLI_index, cross_matrix, region_list, LC_region_list, all_region):
        grasp_obj = skel_list[max_GLI_index]
        grasp_region_index = LC_region_list[obj_index]
        Hc_original_list = self.make_template()
        optimal_grasp = []

        points = cross_matrix[max_GLI_index][obj_index]

        print("要注意:2652 交点がないときの場合が必要 find_grasp_point2")
        if points == []:
           print("交点なし")
           return []

        for point in points:
            if point[2] == 0:
                overlapped_point = [point[0], point[1]]
                overlapped_point_index = grasp_obj.index(overlapped_point)

        try:
            print(overlapped_point_index)
        except UnboundLocalError:
            return []

        if type(grasp_region_index) is list:
            contact_image = np.zeros((height, width))
            for rind in grasp_region_index:
                rimg = MI.make_image(region_list[rind], 1)
                contact_image += rimg
        else:
            contact_image = MI.make_image(region_list[grasp_region_index], 1)

        depth = img_copy.copy()
        depth[all_region==0] *= 0
        depth[contact_image>0] = 0
        contact_depth = depth * contact_image

        for i in range(overlapped_point_index - self.interval*2, overlapped_point_index + self.interval*2 + 1, self.interval):
            if i < 0 or i + self.interval >= len(grasp_obj):
                continue
            poi = grasp_obj[i]
            next_poi = grasp_obj[i+self.interval]
            z = contact_depth[poi[0]][poi[1]]+30
            optimal_grasp = self.graspability(poi, next_poi, optimal_grasp, Hc_original_list, depth, z, i//self.interval)
            # depth = cv2.circle(depth, (poi[1], poi[0]), 1, (255, 0, 0), -1)

        return optimal_grasp

    def graspability(self, poi, next_poi, optimal_grasp, Hc_original_list, depth, z, index):
        if z == 0:
            raise ValueError("2945行目、把持点が不適当です。")        

        depth = min_max_x(depth)
        depth *= 255

        Hc_rotate_list = []
        vector2next = np.array([next_poi[0], next_poi[1]]) - np.array([poi[0], poi[1]])
        for Hc_original in Hc_original_list:
            Hc_original = self.cv2pil(Hc_original)
            initial_rotate = math.degrees(math.atan2(vector2next[1], vector2next[0]))
            Hc_rotate_list.append(Hc_original.rotate(initial_rotate))

        num_template = len(Hc_rotate_list)
        _, Wc = cv2.threshold(depth,z-30,255,cv2.THRESH_BINARY)
        Wc = self.cv2pil(Wc)
        Wc = Wc.crop((poi[1]-self.cutsize, poi[0]-self.cutsize, poi[1]+self.cutsize, poi[0]+self.cutsize))
        for angle in np.arange(-22.5, 33.75, 22.5):
            count = 0
            for Hc_rotate in Hc_rotate_list:
                Hc_rotate = Hc_rotate.rotate(angle)
                C = np.array(Wc) * np.array(Hc_rotate)
                C[C>0] = 1
                if np.sum(C) <= 10:
                    count += 1
            if count == num_template:
                grasp = []
                grasp.append(index)
                grasp.append(initial_rotate+angle)
                grasp.append(z)
                grasp.append(poi)
                optimal_grasp.append(grasp)
        
        return optimal_grasp

def make_motionfile(optimal_grasp, goal_potion):
    #motion = [start, end, option, x, y, z, roll, pitch, yaw]
    #optimal_grasp = [graspability, orientation, z, [x, y]]
    motion_list = []
    time, time_interval = 0, 3

    best_grasp = optimal_grasp[0]
    obj_x = 0.25 + best_grasp[3][0]/330*0.37 + 0.02
    obj_y = -0.18 + (best_grasp[3][1]-70)/420*0.47 - 0.02 + (height - best_grasp[3][1])*0.00007

    obj_z = -((1200-best_grasp[2]*40/255)*1118/1200-(1118-115))/1000 + 0.115
    print("best_z = ", best_grasp[2] )
    print("obj_z = ", obj_z)
        
    obj_orientation = best_grasp[1]
    obj_roll, obj_pitch, obj_yaw = 0, 0, round(obj_orientation, 1)

    goal_x = goal_potion[0]
    goal_y = goal_potion[1]
    goal_z = goal_potion[2]

    upper_z = 0.2

    motion_list.append([time, time+3, "LHAND_JNT_OPEN"]) #ハンドを開く
    time += time_interval
    motion_list.append([time, time+time_interval, "LARM_XYZ_ABS", obj_x, obj_y, upper_z, obj_roll, obj_pitch, obj_yaw]) #対象物の上に移動
    time += time_interval
    motion_list.append([time, time+time_interval+7, "LARM_XYZ_ABS", obj_x, obj_y, obj_z, obj_roll, obj_pitch, obj_yaw]) #対象物の高さに下ろす
    time += time_interval
    time += 7
    motion_list.append([time, time+time_interval, "LHAND_JNT_CLOSE"]) #ハンドを閉じる
    time += time_interval
    motion_list.append([time, time+time_interval, "LARM_XYZ_ABS", obj_x, obj_y, upper_z, obj_roll, obj_pitch, obj_yaw]) #対象物を真上へ持ち上げ
    time += time_interval
    motion_list.append([time, time+time_interval, "LARM_XYZ_ABS", goal_x, goal_y, upper_z, 0, 0, 0]) #対象物を置く地点の上に移動
    time += time_interval
    motion_list.append([time, time+time_interval, "LARM_XYZ_ABS", goal_x, goal_y, goal_z, 0, 0, 0]) #対象物を下ろす
    time += time_interval
    motion_list.append([time, time+time_interval, "LHAND_JNT_OPEN"]) #ハンドを開く
    time += time_interval
    motion_list.append([time, time+time_interval, "LARM_XYZ_ABS", goal_x, goal_y, upper_z, 0, 0, 0]) #ハンドを上げる
    time += time_interval

    with open("/home/takasu/ダウンロード/calculate_ik/motionfile/motionfile.dat", "w") as f:
        for motion in motion_list:
            length = len(motion)
            for i in range(length):
                f.write(str(motion[i]))
                if not i == length - 1:
                    f.write(" ")
                else:
                    f.write("\n")
    f.close()

    with open("/home/takasu/ダウンロード/calculate_ik/motionfile/motionfile_csv.csv", "w") as f:
        writer = csv.writer(f)
        for motion in motion_list:
            writer.writerow(motion)
    f.close()


if __name__ == "__main__":
    start = time.time()

    MI = MakeImage()
    V = Visualize()
    SI = SaveImage()
    
    ##################################セグメンテーション############################################
    RG = RegionGrowing(img)
    region_list2, RG_edge_list = RG.search_seed()
    SI.save_region(region_list2, "RG")
    print("Segmentation is succeeded!")
    ################################################################################################

    # V.visualize_region(region_list2)

    ##################################領域接続######################################################
    CR = ConnectRegion(region_list2)
    center_points_list, center_end_point_list = CR.search_center_points()
    rregion, new_center_points_list, skip_index = CR.skip_by_length(region_list2, center_points_list)
    line_list, check_result1, correct_connection_index = CR.first_check(new_center_points_list)
    combi, side, not_combi, index_form, pos, new_region_list, center_points_list = CR.connect_region(new_center_points_list, rregion, check_result1)
    print("not_combi, new_region_listは必要ありません connect_region")
    check_result, correct_line, interpolate_line, correct_line_zlist, correct_connection_index = CR.check_connect(combi, side, index_form,  pos, line_list, check_result1, correct_connection_index)
    SI.save_centerpoints(center_points_list)
    print("Center point process is succeeded!")
    ################################################################################################
    
    ##################################細線化########################################################
    Skel = Skeletonize()
    S = Sort()
    skel_list, branch_point_list, ad_branch_point_list, end_point_list, skel_skip_index = Skel.skeletonize_region_list(region_list2, check_result)
    SI.save_region(skel_list, "skel")
    sorted_skel_list, end_point_list = S.sort_skel_list(skel_list, branch_point_list, ad_branch_point_list, end_point_list)
    print("Skeletonize is succeeded!")
    ################################################################################################

    ##################################細線接続######################################################
    CL = ConnectLine()
    LC_skel_list, z_list, interpolate, LC_region_list = CL.__main__(sorted_skel_list, end_point_list, correct_line, correct_line_zlist, interpolate_line, correct_connection_index, skel_skip_index, skip_index)
    SI.save_region(LC_skel_list, "connect")
    print("Connect line process is succeeded!")
    ################################################################################################

    ################################GLIの計算#######################################################
    GLI = GaussLinkingIintegral()
    objs_GLI, GLI_matrix = GLI.calculate_GLI(LC_skel_list, z_list)
    ################################################################################################

    ###############################把持位置の取得###################################################
    GA = Graspability()
    optimal_grasp, optimal_grasp2, obj_index = GA.__main__(objs_GLI, LC_skel_list, LC_region_list, interpolate, GLI_matrix, z_list, region_list2)
    print("optimal_grasp = {}".format(optimal_grasp))
    # print("opitmal_grasp2 = {}".format(optimal_grasp2))
    SI.save_grasp_position(LC_skel_list, obj_index, optimal_grasp)
    print("Decision of grasp position is succeeded!")
    ################################################################################################

    ##############################モーションファイルの作成##########################################
    goal_position = [0.4, 0.33, 0]
    make_motionfile(optimal_grasp, goal_position)
    print("Making motion file is succeeded! : file name is motionfile.dat and motionfile_csv.csv")
    ################################################################################################

    print("test")

    elapsed_time = time.time() - start#処理の終了時間を取得
    print("実行時間は{}秒でした．".format(elapsed_time))
