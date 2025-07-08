"""
    训练文件，配置好路径后，将文件复制到工程主目录下使用, 当前只能使用单卡训练
"""
import sys
sys.path.append(r'E:\Common_scripts\pytorch-model\ObjectDetection\yolo\ultralytics\ultralytics') # 你自己的文件夹
from ultralytics import YOLO

# Load a model
model = YOLO(r"E:\Common_scripts\pytorch-model\ObjectDetection\yolo\ultralytics\datasets\sod_person\yolov8.yaml")
model = model.load(r"E:\Common_scripts\pytorch-model\ObjectDetection\yolo\ultralytics\yolov8n.pt") 

# Train the model
train_results = model.train(
    data=r"E:\Common_scripts\pytorch-model\ObjectDetection\yolo\ultralytics\datasets\sod_person\tiny_person.yaml",  # path to dataset YAML
    epochs=200,  # number of training epochs
    imgsz=640,  # training image size
    device="0",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
    name="tiny_person",
    workers=0, 
    batch=16,
)