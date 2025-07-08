"""
    将训练好的模型导出为ONNX格式
"""
import sys
sys.path.append('/home/prolog/Datasets/huangyueyu/model/ultralytics') # 你自己的文件夹
from ultralytics import YOLO


model = YOLO(r'runs\detect\tiny_person\weights\best.pt')  # 加载模型
model.export(format="onnx", imgsz=640, device=0, half=True, nms=True, workspace=4, dynamic=False)
