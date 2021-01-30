import os
import logging
import argparse
import xml.etree.ElementTree as ET
from tqdm import tqdm

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger('voc2txt')
logger.setLevel(logging.DEBUG)


# 改annotations

# python3 voc2txt_annotation.py -name bdd100k_obj -input_dir D:/BDD100K/ -save data_txt

def parse_arguments():
    parser = argparse.ArgumentParser(description='This a script to generate train, test, val dataset, the generated txt file will be used for yolo training')
    parser.add_argument('-name', default="bdd100k_obj", help="Dataset name", action="store_true")
    parser.add_argument('-input_dir',default="D:/BDD100K/",
                        help="Read dataset annotations", action="store_true")
    parser.add_argument('-save', default='data_txt', help="Txt file generated for yolo training and test", action="store_true")
    args = parser.parse_args()
    return args

def convert_annotation(image_id, list_file, image_set, input_dir_path):
    # print(os.path.join(input_dir_path, 'Annotations/%s.xml'%(image_id)))
    in_file = open(os.path.join(input_dir_path, 'Annotations_18/%s/%s.xml'%(image_set, image_id)))
    tree=ET.parse(in_file)
    root = tree.getroot()
    list_file.write(os.path.join(current_path, input_dir_path, 'images/100k/%s/%s.jpg'%(image_set, image_id)))
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

    list_file.write('\n')

def save_data_txt(input_dir_path, sets):
    for name, image_set in sets:
        # print(os.path.join(input_dir_path, 'ImageSets','Main','%s.txt'%(image_set)))
        logger.info(input_dir_path)
        image_ids = open(os.path.join(input_dir_path, 'ImageSets', 'Main','%s.txt'%(image_set))).read().strip().split()
        list_file = open(os.path.join(save_path, '%s_%s.txt'%(args.name, image_set)), 'w')
        for image_id in tqdm(image_ids):
            convert_annotation(image_id, list_file, image_set, input_dir_path)
        list_file.close()


if __name__=='__main__':
    current_path = os.getcwd()
    args = parse_arguments()

    save_path = args.save
    os.makedirs(save_path, exist_ok=True)
    classes = ['rider', 'car', 'bike', 'person', 'train', 'traffic light', 'motor', 'bus', 'truck', 'traffic sign']

    # Train/Val
    # trainval_sets=[(args.name, 'train'), (args.name, 'trainval'), (args.name, 'val')]
    trainval_sets = [(args.name, 'train')]
    trainval_input_dir_path = args.input_dir
    save_data_txt(trainval_input_dir_path, trainval_sets)

    # Test
    test_sets = [(args.name, 'val')]
    test_input_dir_path = args.input_dir
    save_data_txt(test_input_dir_path, test_sets)
