'''
Author: snowy
Email: 1447048804@qq.com
Date: 2025-07-11 11:36:33
Version: 0.0.1
LastEditors: snowy 1447048804@qq.com
LastEditTime: 2025-07-11 15:13:58
Description: VisDrone 数据集的标注文件转换为YOLO格式
'''

import os
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def visdrone2yolo(dir):
    def convertbox(size, box):
        dw = 1./size[0]
        dh = 1./size[1]
        return (box[0] + box[2]/2.0) * dw, (box[1] + box[3] / 2.0) * dh, box[2] * dw, box[3] * dh

    dir_labels = os.path.join(dir, 'labels')
    os.makedirs(dir_labels, exist_ok=True)
    dir_images = os.path.join(dir, 'images')
    dir_annotations = os.path.join(dir, 'annotations')

    anno_txt_names = [f for f in os.listdir(dir_annotations) if f.endswith('.txt')]
    num_anno_txt = len(anno_txt_names)
    
    pbar = tqdm(range(num_anno_txt), desc=f'Converting {dir}', leave=False)

    for f in pbar:
        img_file = os.path.join(dir_images, anno_txt_names[f].replace('.txt', '.jpg'))
        img_size = Image.open(img_file).size
        lines = []

        with open(os.path.join(dir_annotations, anno_txt_names[f]), 'r') as file:
            for row in [x.split(',') for x in file.read().strip().splitlines()]:
                if row[4] == '0': # 排除忽略的标注0，这在数据集中表示忽略
                    continue
                cls = int(row[5]) - 1 # 类别减1
                box = convertbox(img_size, tuple(map(int, row[0:4])))
                lines.append(f"{cls} {' '.join(f'{x:.6f}' for x in box)}\n")
                with open(os.path.join(dir_labels, anno_txt_names[f]), 'w') as label_file:
                    label_file.writelines(lines)

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_path', type=str, default=r'E:\Dataset\visdrone', help='VisDrone数据集的路径')
    args = parser.parse_args()

    dir = Path(args.dir_path)

    for d in 'VisDrone2019-DET-train', 'VisDrone2019-DET-val', 'VisDrone2019-DET-test-dev':
        visdrone2yolo(os.path.join(dir, d))
