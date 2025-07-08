"""
    目标检测结果，水平框或旋转框转成 labelPlog 生成的 xml 格式文件
    文件放在yolov8的输出目录下，可运行
"""
import os
import shutil
import xml.etree.ElementTree as ET
import cv2
import yaml
from infer import Yolov8_Inference  # 连接到 yolov8 的推理程序


class ImageInfo:
    img: None
    path: str = None
    filename: str = None
    img_type: str = None
    bboxes: list = None


class ObjectBoxToXml:
    def __init__(self, save_dir, class_file_path, is_sparated=False, is_rotated=False):
        """
            将目标检测结果（以yolov8格式）转成 labelPlog 生成的 xml 格式文件
        参数：
            save_dir：图像和标注存储文件目录
            class_file_path: 类别文件路径 (yolo 的 yaml格式)
            is_sparated：bool，是否将图像和标注文件分开存储
            is_rotated：bool，是否为旋转框
        返回：
            xml 格式文件内容
        """
        self.save_dir = save_dir
        self.class_file_path = class_file_path
        self.is_sparated = is_sparated
        self.is_rotated = is_rotated
        self.root_node = self._annotation_()
        os.makedirs(os.path.join(self.save_dir, "images"), exist_ok=True)
        if self.is_sparated:
            os.makedirs(os.path.join(self.save_dir, "annotations"), exist_ok=True)


    def _annotation_(self):
        """
            创建文件的根节点
        """
        annotation = ET.Element("annotation", {"verified": "no"})
        return annotation
    
    def _folder_(self):
        """
            创建 <folder> 标签, 存放图像所在的文件夹
        """
        folder = ET.SubElement(self.root_node, "folder")
        folder.text = "images"
        return
    
    def _filename_(self, img_info: ImageInfo):
        """
            创建 <filename> 标签, 存放图像文件名
        """
        filename = ET.SubElement(self.root_node, "filename")
        filename.text = img_info.filename
        return
    
    def _path_(self, img_info: ImageInfo):
        """
            创建 <path> 标签, 存放图像文件路径    
        """
        path = ET.SubElement(self.root_node, "path")
        save_path = os.path.join(self.save_dir, os.path.join("images", img_info.filename + img_info.img_type))
        path.text = save_path
        return
    
    def _source_(self):
        """
            创建 <source> 标签, 存放图像来源信息, 来自哪个数据库
        """
        source = ET.SubElement(self.root_node, "source")
        database = ET.SubElement(source, "database")
        database.text = "Unknown"
        return
    
    def _size_(self, img_info: ImageInfo):
        """
            创建 <size> 标签, 存放图像尺寸信息
        """
        size = ET.SubElement(self.root_node, "size")
        width = ET.SubElement(size, "width")
        width.text = str(img_info.img.shape[1])
        height = ET.SubElement(size, "height")
        height.text = str(img_info.img.shape[0])
        depth = ET.SubElement(size, "depth")
        depth.text = str(img_info.img.shape[2])
        return
    
    def _segmented_(self):
        """
            创建 <segmented> 标签, 标明图像是否分割过, 0 表示未分割
        """
        segmented = ET.SubElement(self.root_node, "segmented")
        segmented.text = "0"
        return

    def trans_classes(self, class_file_path):
        """
            转换类别名称
        """
        with open(class_file_path, "r") as f:
            data = yaml.safe_load(f)
        classes = data["names"]
        return classes

    def trans_box(self, box, class_file_path, obj):
        """
            转换目标检测框的格式
        """
        obj_type = ET.SubElement(obj, "type")
        obj_name = ET.SubElement(obj, "name")
        obj_pose = ET.SubElement(obj, "pose")
        obj_truncated = ET.SubElement(obj, "truncated")
        obj_difficult = ET.SubElement(obj, "difficult")
        classes = self.trans_classes(class_file_path)
        obj_name.text = classes[int(box[0])]
        obj_pose.text = "Unspecified"
        obj_truncated.text = "0"
        obj_difficult.text = "0"
        if self.is_rotated:  # 旋转框
            obj_type.text = "robndbox"
            obj_robndbox = ET.SubElement(obj, "robndbox")
            cx = ET.SubElement(obj_robndbox, "cx")
            cy = ET.SubElement(obj_robndbox, "cy")
            w = ET.SubElement(obj_robndbox, "w")
            h = ET.SubElement(obj_robndbox, "h")
            angle = ET.SubElement(obj_robndbox, "angle")
            cx.text = str(format(box[2], ".4f"))
            cy.text = str(format(box[3], ".4f"))
            w.text = str(format(box[4], ".4f"))
            h.text = str(format(box[5], ".4f"))
            angle.text = str(box[6])
        else:               # 水平框
            obj_type.text = "bndbox"
            obj_bndbox = ET.SubElement(obj, "bndbox")
            xmin = ET.SubElement(obj_bndbox, "xmin")
            ymin = ET.SubElement(obj_bndbox, "ymin")
            xmax = ET.SubElement(obj_bndbox, "xmax")
            ymax = ET.SubElement(obj_bndbox, "ymax")
            xmin.text = str(int(box[2]))
            ymin.text = str(int(box[3]))
            xmax.text = str(int(box[4]))
            ymax.text = str(int(box[5]))
        return obj

    def _object_(self, img_info: ImageInfo):
        """
            创建 <object> 标签, 存放目标检测信息
        """
        for bbox in img_info.bboxes:
            obj = ET.SubElement(self.root_node, "object")
            obj = self.trans_box(bbox, self.class_file_path, obj)
        return
    
    def save_xml(self, img_info: ImageInfo):
        """
            保存 xml 格式文件
        """
        self._folder_()
        self._filename_(img_info)
        self._path_(img_info)
        self._source_()
        self._size_(img_info)
        self._segmented_()
        self._object_(img_info)
        
        tree = ET.ElementTree(self.root_node)
        if self.is_sparated:
            xml_save_path = os.path.join(self.save_dir, os.path.join("annotations", img_info.filename + ".xml"))
        else:
            xml_save_path = os.path.join(self.save_dir, os.path.join("images", img_info.filename + ".xml"))
        shutil.copy(img_info.path, os.path.join(self.save_dir, "images"))
        tree.write(xml_save_path, encoding="utf-8", xml_declaration=True)
        return
    

# 结合 yolov8 推理程序，将目标检测结果转成 xml 格式文件
def run_dir(model_params, pending_dir, trans_params):   # 处理一个文件夹
    infer = Yolov8_Inference(
        model_path=model_params['model_path'],
        model_type=model_params['model_type'], 
        input_size=model_params['input_size'], 
        conf_thres=model_params['conf_thres'], 
        iou_thres=model_params['iou_thres'], 
        max_det=model_params['max_det'], 
        device=model_params['device'],
        cls_path=model_params['cls_path']
    )
    
    obj_box_to_xml = ObjectBoxToXml(
        save_dir=trans_params['save_dir'], 
        class_file_path=trans_params['class_file_path'], 
        is_sparated=trans_params['is_sparated'], 
        is_rotated=trans_params['is_rotated']
    )
    
    for img_name in os.listdir(pending_dir):
        if img_name.endswith('.jpg') or img_name.endswith('.png'):
            img_path = os.path.join(pending_dir, img_name)
            img = cv2.imread(img_path)
            bboxes = infer.predict(img, mode=1).tolist()
            img_info = ImageInfo()
            img_info.img = img
            img_info.path = img_path
            filename, ext = os.path.splitext(img_name)
            img_info.filename = filename
            img_info.img_type = ext
            img_info.bboxes = bboxes
            obj_box_to_xml.save_xml(img_info)
    print("Done!")


def run_file(model_params, img_path, trans_params):   # 处理一个文件
    infer = Yolov8_Inference(
        model_path=model_params['model_path'],
        model_type=model_params['model_type'], 
        input_size=model_params['input_size'], 
        conf_thres=model_params['conf_thres'], 
        iou_thres=model_params['iou_thres'], 
        max_det=model_params['max_det'], 
        device=model_params['device'],
        cls_path=model_params['cls_path']
    )
    
    obj_box_to_xml = ObjectBoxToXml(
        save_dir=trans_params['save_dir'], 
        class_file_path=trans_params['class_file_path'], 
        is_sparated=trans_params['is_sparated'], 
        is_rotated=trans_params['is_rotated']
    )
    
    if img_path.endswith('.jpg') or img_path.endswith('.png'):
        img = cv2.imread(img_path)
        bboxes = infer.predict(img, mode=1).tolist()
        img_info = ImageInfo()
        img_info.img = img
        img_info.path = img_path
        filename, ext = os.path.splitext(os.path.basename(img_path))
        img_info.filename = filename
        img_info.img_type = ext
        img_info.bboxes = bboxes
        obj_box_to_xml.save_xml(img_info)
    print("Done!")


if __name__ == "__main__":
    model_params = {
        "model_path": r'/home/prolog/Datasets/huangyueyu/model/ultralytics/runs/detect/rock_and_arm/weights/best.onnx',
        "model_type": 'onnx',
        "input_size": 640,
        "conf_thres": 0.3,
        "iou_thres": 0.5,
        "max_det": 100,
        "device": 'gpu',
        "cls_path": r'/home/prolog/Datasets/huangyueyu/model/ultralytics/datasets/rock_arm/rock_arm.yaml'
    }
    trans_params = {
        "save_dir": r'/home/prolog/Datasets/huangyueyu/model/ultralytics/datasets/auto_annotation/rock_arm',
        "class_file_path": r'/home/prolog/Datasets/huangyueyu/model/ultralytics/datasets/rock_arm/rock_arm.yaml',
        "is_sparated": False,
        "is_rotated": False
    }
    pending_dir = r'/home/prolog/Datasets/huangyueyu/model/ultralytics/datasets/rock_arm/test'
    img_path = r'/home/prolog/Datasets/huangyueyu/model/ultralytics/datasets/rock_arm/images/val/2.jpg'

    # run_dir(model_params, pending_dir, trans_params)
    run_file(model_params, img_path, trans_params)
