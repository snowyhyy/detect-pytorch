"""
    训练文件，配置好路径后，将文件复制到工程主目录下使用, 当前只能使用单卡训练
"""
import sys
sys.path.append(r'E:\Common_scripts\pytorch-model\Detect\yolo') # 你自己的文件夹
from ultralytics import YOLO

# Load a model
# model = YOLO(r"E:\Common_scripts\pytorch-model\Detect\yolo\datasets\visdrone\fbrt_yolov8.yaml") # 从头训练
model = YOLO(r"E:\Common_scripts\pytorch-model\Detect\yolo\yolov8n.pt")
# model = model.load(r"E:\Common_scripts\pytorch-model\Detect\yolo\runs\detect\visdrone\weights\last.pt") 

# Train the model
train_results = model.train(
    data=r"E:\Common_scripts\pytorch-model\Detect\yolo\datasets\visdrone\visdrone.yaml", # dataset path
    epochs=300,  # number of training epochs
    imgsz=640,  # training image size
    device="0",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
    name="visdrone",
    workers=0, 
    batch=8,
    optimizer="SGD",
    # resume=True, # 断点续训
)