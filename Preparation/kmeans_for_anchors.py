import numpy as np
import xml.etree.ElementTree as ET
import glob
import random
import argparse
from tqdm import tqdm

# python3 kmeans_for_anchors.py -size (416, 416) -anchors_num 9 -path D:/BDD100K/Annotations_18/train -save_path data_txt/BDD100K_yolov4_anchors_416_416.txt

def parse_arguments():
    parser = argparse.ArgumentParser(description='This a script using K-means to generate the anchors for the target dataset')
    parser.add_argument('-size', default=(416, 416), help="Input image size", action="store_true")
    parser.add_argument('-anchors_num',default=9, help="Proposed anchor numbers", action="store_true") #9
    parser.add_argument('-path', default='G:\Datasets\yolov5_voc/voc/train\VOCdevkit\VOC2012\Annotations', help="Read dataset annotations", action="store_true")
    parser.add_argument('-save_path', default="data_txt/voc_obj/yolov4_anchors_416_416.txt", help="Txt file",
                        action="store_true")
    args = parser.parse_args()
    return args


# 608
# 12, 16,  19, 36,  40, 28,  36, 75,  76, 55,  72, 146,  142, 110,  192, 243,  459, 401

# voc2012 416 x 416 
# 14,27, 34,47, 42,99, 67,186, 104,101, 122,261, 212,326, 232,170, 363,361
# 15,24, 28,59, 48,133, 67,63, 88,222, 128,123, 164,307, 266,193, 344,366
# 14,26, 29,48, 37,112, 67,65, 74,180, 128,117, 143,279, 255,184, 336,345
# voc2012 608 x 608
# 23,37, 40,97, 72,194, 86,72, 132,322, 169,162, 243,451, 364,267, 503,529

def cas_iou(box,cluster):
    '''
    计算iou: 
    '''
    x = np.minimum(cluster[:,0],box[0])
    y = np.minimum(cluster[:,1],box[1])

    intersection = x * y
    area1 = box[0] * box[1]
    area2 = cluster[:,0] * cluster[:,1]
    iou = intersection / (area1 + area2 -intersection)

    return iou

def avg_iou(box,cluster):
    return np.mean([np.max(cas_iou(box[i],cluster)) for i in range(box.shape[0])])

def kmeans(box,k):
    # k = anchor numbers
    # 取出一共有多少框
    row = box.shape[0]
    
    # 每个框各个点的位置
    distance = np.empty((row,k))
    
    # 最后的聚类位置
    last_clu = np.zeros((row,))

    np.random.seed()

    # 随机选k个当聚类中心
    cluster = box[np.random.choice(row,k,replace = False)]
    # cluster = random.sample(row, k)
    while True:
        # 计算每一行距离k个点的iou情况。
        for i in range(row):
            distance[i] = 1 - cas_iou(box[i],cluster)
        
        # 取出最小点
        near = np.argmin(distance,axis=1)

        if (last_clu == near).all():
            break
        
        # 求每一个类的中位点
        for j in range(k):
            cluster[j] = np.median(
                box[near == j],axis=0)

        last_clu = near

    return cluster

def load_data(path):
    data = []
    # 对于每一个xml都寻找box
    for xml_file in tqdm(glob.glob('{}/*xml'.format(path))):
        tree = ET.parse(xml_file)
        height = int(tree.findtext('./size/height'))
        width = int(tree.findtext('./size/width'))
        # 对于每一个目标都获得它的宽高
        # normalized 
        for obj in tree.iter('object'):
            xmin = int(float(obj.findtext('bndbox/xmin'))) / width
            ymin = int(float(obj.findtext('bndbox/ymin'))) / height
            xmax = int(float(obj.findtext('bndbox/xmax'))) / width
            ymax = int(float(obj.findtext('bndbox/ymax'))) / height

            xmin = np.float64(xmin)
            ymin = np.float64(ymin)
            xmax = np.float64(xmax)
            ymax = np.float64(ymax)
            # 得到宽高 width, height
            data.append([xmax-xmin,ymax-ymin])
    return np.array(data)


if __name__ == '__main__':
    args = parse_arguments()
    # 运行该程序会计算'./VOCdevkit/VOC2007/Annotations'的xml
    # 会生成yolo_anchors.txt
    SIZE = args.size
    anchors_num = args.anchors_num
    # 载入数据集，可以使用VOC的xml
    path = args.path
    
    # 载入所有的xml
    # 存储格式为转化为比例后的width,height
    data = load_data(path) # 每一个xml每一个目标宽高
    
    # 使用k聚类算法 
    # 9个聚类框都是归一化的
    out = kmeans(data, anchors_num)
    print('normalized anchors: \n', out)
    # 根据 width 给所有anchor排序
    out = out[np.argsort(out[:,0])]
    print('IOU acc:{:.2f}%'.format(avg_iou(data,out) * 100))
    print('Argsorted normalized anchors: \n', out)
    data=[[out[i, 0] * SIZE[0], out[i,1]*SIZE[1]] for i in range(len(out))] # 宽高
    print('Argsorted unnormalized anchors: \n', data)
    # data = out*SIZE
    f = open(args.save_path, 'w')
    row = np.shape(data)[0]
    for i in range(row):
        if i == 0:
            x_y = "%d,%d" % (data[i][0], data[i][1])
        else:
            x_y = ", %d,%d" % (data[i][0], data[i][1])
        f.write(x_y)
    f.close()