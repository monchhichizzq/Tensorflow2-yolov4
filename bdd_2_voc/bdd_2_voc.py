# -*- coding: utf-8 -*-
# @Time    : 2021/1/29 21:55
# @Author  : Zeqi@@
# @FileName: bdd_2_voc.py
# @Software: PyCharm
import os
import os.path as osp

import argparse
import logging

import json

from xml.etree.ElementTree import Element, SubElement
from xml.etree import ElementTree
from xml.dom import minidom

from PIL import Image

from tqdm import tqdm


logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger('BDD2VOC')
logger.setLevel(logging.DEBUG)

# DEBUG = False
#
# BDD_FOLDER = "./bdd100k"
#
# if DEBUG:
#     XML_PATH = "./xml"
# else:
#     XML_PATH = BDD_FOLDER + "/xml"

# target_labels = ['bus','traffic light','traffic sign','pedestrian','bike','truck','moter','car','train','rider']
# ignore = ['other vehicle', 'other person', ]
# {'traffic light', 'bus', 'other vehicle', 'truck', 'train', 'bicycle', 'traffic sign', 'car', 'trailer', 'motorcycle', 'other person', 'rider', 'pedestrian'} 2020
# {'rider', 'car', 'bike', 'person', 'train', 'traffic light', 'motor', 'bus', 'truck', 'traffic sign'} 2018

# 记得区分 detection18 和 annotations 18

def bdd_to_voc(bdd_folder, xml_folder, ImageSets_folder):
    image_path = bdd_folder + "/images/100k/%s"
    label_path = bdd_folder + "/labels/detection18/bdd100k_labels_images_%s.json"
    # label_path = bdd_folder + "/labels/detection20/det_v2_%s_release.json"
    os.makedirs(ImageSets_folder, exist_ok=True)
    main_txt_path = ImageSets_folder + "/%s.txt"

    classes = set()

    for trainval in ['train', 'val']:
        image_folder = image_path % trainval
        json_path = label_path % trainval
        logger.info(image_path)
        logger.info(label_path)

        xml_folder_ = osp.join(xml_folder, trainval)
        logger.info(xml_folder_)

        if not os.path.exists(xml_folder_):
            os.makedirs(xml_folder_)

        with open(json_path) as f:
            j = f.read()

        data = json.loads(j)

        # ImageSets/Main/trainval.txt
        # ImageSets/Main/test.txt
        txt_path = main_txt_path % trainval
        ids = []

        for datum in tqdm(data):
            tmp_list = []
            annotation = Element('annotation')
            SubElement(annotation, 'folder').text ='VOC2007'
            SubElement(annotation, 'filename').text = datum['name']
            source = get_source()
            owner = get_owner()
            annotation.append(source)
            annotation.append(owner)
            size = get_size(osp.join(image_folder, datum['name']))
            annotation.append(size)
            SubElement(annotation, 'segmented').text ='0'
            # additional information
            #for key, item in datum['attributes'].items():
            #    SubElement(annotation, key).text = item

            if datum['labels'] is None:
                continue

            # bounding box
            for label in datum['labels']:

                # if label['category'] != "traffic light":
                #     continue
                # else:
                #     tmp_list.append(1)
                # # if label['category'] not in target_labels:
                # #     continue
                # # else:
                # #     tmp_list.append(1)
                # color = label['attributes']["trafficLightColor"]
                try:
                    box2d = label['box2d']
                except KeyError:
                    continue
                else:
                    bndbox = get_bbox(box2d)

                object_ = Element('object')

                SubElement(object_, 'name').text = label['category']
                # SubElement(object_, 'name').text = color
                SubElement(object_, 'pose').text = "Unspecified"
                SubElement(object_, 'truncated').text = '0'
                SubElement(object_, 'difficult').text = '0'
                classes.add(label['category'])
                # classes.add(color)

                object_.append(bndbox)
                annotation.append(object_)

            # if len(tmp_list) == 0:
            #     continue
            xml_filename = osp.splitext(datum['name'])[0] + '.xml'
            ids.append( osp.splitext(datum['name'])[0] + '\n')
            with open(osp.join(xml_folder_, xml_filename), 'w') as f:
                f.write(prettify(annotation))

        with open(osp.join(txt_path), 'w') as fid:
            fid.writelines(ids)
        fid.close()

    logger.info(classes)

def get_owner():
    owner = Element('owner')
    SubElement(owner, 'flickrid').text ='NULL'
    SubElement(owner, 'name').text ='Zeqi'
    return owner

def get_source():
    source = Element('source')
    SubElement(source, 'database').text ='voc_bdd'
    SubElement(source, 'annotation').text ='VOC2007'
    SubElement(source, 'image').text ='flickr'
    SubElement(source, 'flickrid').text ='NULL'
    return source

def get_size(image_path):
    i = Image.open(image_path)
    sz = Element('size')
    SubElement(sz, 'width').text = str(i.width)
    SubElement(sz, 'height').text = str(i.height)
    SubElement(sz, 'depth').text = str(3)
    return sz


def get_bbox(box2d):
    bndbox = Element('bndbox')
    SubElement(bndbox, 'xmin').text = str(int(round(box2d['x1'])))
    SubElement(bndbox, 'ymin').text = str(int(round(box2d['y1'])))
    SubElement(bndbox, 'xmax').text = str(int(round(box2d['x2'])))
    SubElement(bndbox, 'ymax').text = str(int(round(box2d['y2'])))
    return bndbox


def prettify(elem):
    rough_string = ElementTree.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="\t")

def parse_arguments():
    parser = argparse.ArgumentParser(description='This a script to convert the bdd100k data to voc format')
    parser.add_argument('-bdd_folder', default="../../BDD100K", help="The raw folder", action="store_true")
    parser.add_argument('-annotation',default="Annotations_18", help="Directory for generated xml files", action="store_true")
    parser.add_argument('-image_ids', default="ImageSets/Main", help="Directory for image id txt files", action="store_true")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()

    BDD_FOLDER = args.bdd_folder
    XML_PATH = os.path.join(BDD_FOLDER, args.annotation)
    ImageSets_folder = os.path.join(BDD_FOLDER, args.image_ids)
    bdd_to_voc(BDD_FOLDER, XML_PATH, ImageSets_folder)


# python3 bdd_2_voc.py -bdd_folder ../../BDD100K -annotation Annotations_18 -image_ids ImageSets/Main