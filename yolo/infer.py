"""
    yolov8 模型推理
"""
import random
import torch
import cv2
import numpy as np
import onnx
import onnxruntime
import pycuda.driver as cuda
import pycuda.autoinit
import yaml
import time
import os
import sys


class Yolov8_Inference:
    def __init__(self, model_path, model_type, input_size, conf_thres=0.5, 
                 iou_thres=0.5, max_det=100, device='gpu', cls_path=None):
        self.model_path = model_path
        self.model_type = model_type
        self.input_size = input_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.model = None
        self.device = torch.device('cuda' if device == 'gpu' and torch.cuda.is_available() else 'cpu')

        if self.model_type == 'onnx':
            self.model = self._load_onnx_model()
        elif self.model_type == 'pt':
            self.model = self._load_pt_model()
        elif self.model_type == 'trt' and self.device == 'cuda':
            self.model = self._load_trt_model()
        else:
            raise ValueError('Unsupported model type: {}'.format(self.model_type))
        
        self.cls_names = None
        if cls_path is not None:
            with open(cls_path, 'r') as f:
                data = yaml.safe_load(f)
            self.cls_names = data['names']
        
    def _load_onnx_model(self):
        if self.device.type == 'cuda':
            providers = ['CUDAExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
        return onnxruntime.InferenceSession(self.model_path, providers=providers)
    
    def _load_trt_model(self, trt):
        import tensorrt as trt
        from collections import OrderedDict, namedtuple

        trt.init_libnvinfer_plugins(None, '')
        trt_logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(trt_logger)
        with open(self.model_path, 'rb') as f:
            engine_data = f.read()
        engine = self.runtime.deserialize_cuda_engine(engine_data)
        context = engine.create_execution_context()

        # 创建 Binding 对象
        Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        bindings = OrderedDict()
        fp16 = False
        # 输入输出绑定
        for i in range(engine.num_bindings):
            name = engine.get_binding_name(i)
            dtype = trt.nptype(engine.get_binding_dtype(i))
            shape = engine.get_binding_shape(i)
            data = torch.from_numpy(np.empty(shape, dtype=np.dtype(dtype))).to(self.device)
            # Tensor.data_ptr 该tensor首个元素的地址即指针，为int类型
            bindings[name] = Binding(name, dtype, shape, data, int(data.data_ptr()))
            if engine.binding_is_input(i) and dtype == np.float16:
                fp16 = True

        # 记录输入输出绑定的指针地址
        binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
    
        return context, bindings, binding_addrs
    
    def _load_pt_model(self):
        model = torch.load(self.model_path)
        model.to(self.device)
        model.eval()
        return model
    
    def _infer(self, img):
        """
            模型推理
        """
        if self.model_type == 'onnx':
            img = np.expand_dims(img, axis=0)
            input_type = self.model.get_inputs()[0].type
            if 'tensor(float16)' in input_type:
                img = img.astype(np.float16)
            elif 'tensor(float)' in input_type:
                img = img.astype(np.float32)
            else:
                raise ValueError(f"The onnx model unsupport this input type: {input_type}")
            input_name = self.model.get_inputs()[0].name
            output = self.model.run(None, {input_name: img})
            return output[0]
        elif self.model_type == 'pt':
            img = torch.from_numpy(img)
            img = img.to(self.device)
            with torch.no_grad():
                output = self.model(img)
            return output
        elif self.model_type == 'trt':
            img = torch.from_numpy(img)
            img = img.to(self.device)
            if self.device == 'cpu':
                raise ValueError('Error: TensorRT engine can only be used on GPU.')
            
            context, bindings, binding_addrs = self.model
            binding_addrs['images'] = int(img.data_ptr())
            context.execute_v2(list(binding_addrs.values()))

            output = bindings['output0'].data
            return output
            
    def _preprocess(self, img):
        """
            图像预处理(前处理)
            将图像调整到 self.input_size 指定大小, 缺少部分填充灰条
        """
        scale = min((self.input_size / img.shape[1]), (self.input_size / img.shape[0]))
        offset_x = (self.input_size - img.shape[1] * scale) / 2
        offset_y = (self.input_size - img.shape[0] * scale) / 2
        M = np.array([scale, 0, offset_x, 0, scale, offset_y], dtype=np.float32)
        M = M.reshape(2, 3)  # 缩放平移矩阵
        img_pre = cv2.warpAffine(img, M, (self.input_size, self.input_size), flags=cv2.INTER_LINEAR, 
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(114, 114, 114))
        IM = cv2.invertAffineTransform(M)  # 缩放平移逆矩阵
        img_pre = (img_pre[..., ::-1] / 255.0).astype(np.float32)  # BGR转RGB, 归一化
        img_pre = img_pre.transpose(2, 0, 1)  # HWC转CHW
        # img_pre = torch.from_numpy(img_pre)  # 转为张量
        return img_pre, IM
    
    def horizontal_iou(self, box1, box2):
        """
            计算两个box的iou, 水平框的iou计算方式, box的格式为 [cls_id, conf, cx, cy, w, h]
        """
        # 计算 box 的面积，写一个嵌套函数
        def box_area(box):
            return box[2] * box[3]
        
        # [cx, cy, w, h] -> [x1, y1, x2, y2]
        def xywh2xyxy(box):
            x1 = box[0] - box[2] / 2
            y1 = box[1] - box[3] / 2
            x2 = box[0] + box[2] / 2
            y2 = box[1] + box[3] / 2
            return [x1, y1, x2, y2]
        
        # 计算两个 box 的交集
        area_box1 = box_area(box1[2:])
        area_box2 = box_area(box2[2:])
        x1_box1, y1_box1, x2_box1, y2_box1 = xywh2xyxy(box1[2:])
        x1_box2, y1_box2, x2_box2, y2_box2 = xywh2xyxy(box2[2:])
        if x1_box1 >= x2_box2 or x2_box1 <= x1_box2 or y1_box1 >= y2_box2 or y2_box1 <= y1_box2:
            return 0.0
        x_list = sorted([x1_box1, x2_box1, x1_box2, x2_box2])
        y_list = sorted([y1_box1, y2_box1, y1_box2, y2_box2])
        intersection = (x_list[2] - x_list[1]) * (y_list[2] - y_list[1])
        
        iou = intersection / (area_box1 + area_box2 - intersection)
        return iou

    def horizontal_nms(self, boxes):
        """
            水平框非极大值抑制(NMS)
            anchor 框的数量根据分辨率而来，分辨率是 32 的倍数（这里用的正方形的框，普通矩形类似，其实就是最后3层特征图大小之和）
            min_feature_map_size = 分辨率 / 32
            每个框的维度为 [cx, cy, w, h, classes]
            一幅图框的总数 anchor_num = min_feature_map_size ** 2 + (min_feature_map_size * 2) ** 2 + (min_feature_map_size * 4) ** 2
        参数：
            boxes(np.array): yolo输出的预测框列表，每个预测框为 [cx, cy, w, h, classes]   shape: (1, 4+classes, anchor_num)
        返回：
            keep_boxes(np.array): 经过NMS后的框列表，每个框为 [class_id, confidence, cx, cy, w, h]   shape: (max_det, 6)
        """
        keep_boxes = []
        # 取出当前图的所有预测框
        boxes_i = boxes.squeeze(0)  # shape: (4+classes, anchor_num)
        boxes_i = np.transpose(boxes_i, (1, 0))  # shape: (anchor_num, 4+classes)
        # 解析每个预测框
        boxes_loc = boxes_i[..., :4]  # shape: (anchor_num, 4) -- cx, cy, w, h
        boxes_prob = boxes_i[..., 4:]  # 框对于所有类别的预测概率 shape: (anchor_num, classes)
        boxes_cls = np.argmax(boxes_prob, axis=-1)  # 框的预测类别 shape: (anchor_num,)
        boxes_conf = np.max(boxes_prob, axis=-1)   # 框的预测类别对应的概率(即是框的置信度) shape: (anchor_num,)
        # 合并成新数据
        new_boxes = np.concatenate((boxes_cls[:, None], boxes_conf[:, None], boxes_loc), axis=-1)    # shape: (anchor_num, 6)
        # 置信度筛选
        selected_conf_boxes = new_boxes[boxes_conf > self.conf_thres]
        # nms前，先将框按类别分组
        grouped_boxes = {}
        for box in selected_conf_boxes:
            if box[0] not in grouped_boxes:
                grouped_boxes[box[0]] = []
            grouped_boxes[box[0]].append(box)
        # 进行nms
        for cls_id, boxes_cls in grouped_boxes.items():
            boxes_cls_np = np.array(boxes_cls)
            sorted_boxes_cls = sorted(boxes_cls_np, key=lambda x: -x[1])    # 按置信度排序, 降序

            while len(sorted_boxes_cls) > 0:
                max_conf_box = sorted_boxes_cls[0]
                keep_boxes.append(max_conf_box)
                ious = np.array([self.horizontal_iou(max_conf_box, box) for box in sorted_boxes_cls[1:]])
                sorted_boxes_cls = np.array(sorted_boxes_cls[1:])
                sorted_boxes_cls = sorted_boxes_cls[(ious < self.iou_thres).astype(bool)].tolist()   # 计算iou, 保留小于iou_thres的框
        
        # 限制最大检测框数
        if len(keep_boxes) > self.max_det:
            keep_boxes = np.array(keep_boxes)
            keep_boxes = keep_boxes[np.argsort(-keep_boxes[:, 1])[:self.max_det]]
            return keep_boxes
        return  np.array(keep_boxes)
    
    def horizontal_postprocess(self, outputs, IM):
        """
            水平框后处理
        参数：
            outputs(np.array): nms后的预测框列表，每个预测框为 [cls_id, conf, cx, cy, w, h]   shape: (box_num, 6)
            IM: 缩放平移逆矩阵
        返回：
            outputs(np.array): 经过坐标还原后的框列表，每个框为 [cls_id, conf, cx, cy, w, h]   shape: (box_num, 6)
        """
        cx = outputs[:, 2]
        cy = outputs[:, 3]
        w = outputs[:, 4]
        h = outputs[:, 5]
        outputs[:, 2] = IM[0][0] * cx + IM[0][2]
        outputs[:, 3] = IM[1][1] * cy + IM[1][2]
        outputs[:, 4] = w * IM[0][0]
        outputs[:, 5] = h * IM[1][1]
        return outputs
    
    def obb_iou(self, box1, box2, eps=1e-7):
        """
            计算两个box的iou, 旋转框的iou计算方式, box的格式为 [cls_id, conf, cx, cy, w, h, angle]
        """
        def box_area(box):
            return box[2] * box[3]
        
        rect1 = [(box1[0], box1[1]), (box1[2], box1[3]), box1[4]]
        rect2 = [(box2[0], box2[1]), (box2[2], box2[3]), box2[4]]
        area_box1 = box_area(box1[2:])
        area_box2 = box_area(box2[2:])
        inter_pts = cv2.rotatedRectangleIntersection(rect1, rect2)[1]
        if inter_pts is not None:
            order_pts = cv2.convexHull(inter_pts, returnPoints=True)
            int_area = cv2.contourArea(order_pts)
            iou = int_area / (area_box1 + area_box2 - int_area + eps)
        else:
            iou = 0.0
        return iou
    
    def obb_nms(self, boxes):
        """
            旋转框非极大值抑制(NMS)
        参数：
            boxes(np.array): yolo输出的预测框列表，每个预测框为 [cls_id, conf, cx, cy, w, h, angle]   shape: (1, 5+classes, anchor_num)
        返回：
            keep_boxes(np.array): 经过NMS后的框列表，每个框为 [class_id, confidence, cx, cy, w, h, angle]   shape: (max_det, 7)
        """
        keep_boxes = []
        # 取出当前图的所有预测框
        boxes_i = boxes.squeeze(0)  # shape: (5+classes, anchor_num)
        boxes_i = np.transpose(boxes_i, (1, 0))  # shape: (anchor_num, 5+classes)
        # 解析每个预测框
        boxes_loc = boxes_i[..., :4]  # shape: (anchor_num, 4) -- cx, cy, w, h
        boxes_prob = boxes_i[..., 4:-1]  # 框对于所有类别的预测概率 shape: (anchor_num, classes)
        boxes_cls = np.argmax(boxes_prob, axis=-1)  # 框的预测类别 shape: (anchor_num,)
        boxes_conf = np.max(boxes_prob, axis=-1)   # 框的预测类别对应的概率(即是框的置信度) shape: (anchor_num,)
        boxes_angle = boxes_i[..., -1]  # shape: (anchor_num,)
        # 合并成新数据
        new_boxes = np.concatenate((boxes_cls[:, None], boxes_conf[:, None], boxes_loc, boxes_angle[:, None]), axis=-1)    # shape: (anchor_num, 7)
        # 置信度筛选
        selected_conf_boxes = new_boxes[boxes_conf > self.conf_thres]
        # nms前，先将框按类别分组
        grouped_boxes = {}
        for box in selected_conf_boxes:
            if box[0] not in grouped_boxes:
                grouped_boxes[box[0]] = []
            grouped_boxes[box[0]].append(box)
        # 进行nms
        for cls_id, boxes_cls in grouped_boxes.items():
            boxes_cls_np = np.array(boxes_cls)
            sorted_boxes_cls = sorted(boxes_cls_np, key=lambda x: -x[1])    # 按置信度排序, 降序

            while len(sorted_boxes_cls) > 0:
                max_conf_box = sorted_boxes_cls[0]
                keep_boxes.append(max_conf_box)
                ious = np.array([self.obb_iou(max_conf_box, box) for box in sorted_boxes_cls[1:]])
                sorted_boxes_cls = np.array(sorted_boxes_cls[1:])
                sorted_boxes_cls = sorted_boxes_cls[(ious < self.iou_thres).astype(bool)].tolist()   # 计算iou, 保留小于iou_thres的框

        # 限制最大检测框数
        if len(keep_boxes) > self.max_det:
            keep_boxes = np.array(keep_boxes)
            keep_boxes = keep_boxes[np.argsort(-keep_boxes[:, 1])[:self.max_det]]
            return keep_boxes
        return  np.array(keep_boxes)
    
    def obb_postprocess(self, outputs, IM):
        """
            旋转框后处理
        参数：
            outputs(np.array): nms后的预测框列表，每个预测框为 [cls_id, conf, cx, cy, w, h, angle]   shape: (box_num, 7)
            IM: 缩放平移逆矩阵
        返回：
            outputs(np.array): 经过坐标还原后的框列表，每个框为 [cls_id, conf, cx, cy, w, h, angle]   shape: (box_num, 7)
        """
        cx = outputs[:, 2]
        cy = outputs[:, 3]
        w = outputs[:, 4]
        h = outputs[:, 5]
        outputs[:, 2] = IM[0][0] * cx + IM[0][2]
        outputs[:, 3] = IM[1][1] * cy + IM[1][2]
        outputs[:, 4] = w * IM[0][0]
        outputs[:, 5] = h * IM[1][1]
        return outputs

    def predict(self, img, mode=0, save_path=None):
        """
            预测
        参数：
            img: 输入图像
            save_path: 保存预测结果的路径
        """
        def rotate_point(cx, cy, px, py, angle):
            """
                计算旋转后的点坐标, 支持数组
            """
            xoff = px - cx
            yoff = py - cy
            cos_theta = np.cos(angle)
            sin_theta = np.sin(angle)
            pResx = cos_theta * xoff + sin_theta * yoff
            pResy = -sin_theta * xoff + cos_theta * yoff
            return pResx + cx, pResy + cy
        
        def xywh2xyxy(outputs):
            # 将 [cx, cy, w, h] 格式的 box 转换为 [x1, y1, x2, y2] 格式
            cx = outputs[:, 2]
            cy = outputs[:, 3]
            w = outputs[:, 4]
            h = outputs[:, 5]
            outputs[:, 2] = cx - w / 2
            outputs[:, 3] = cy - h / 2
            outputs[:, 4] = cx + w / 2
            outputs[:, 5] = cy + h / 2
            return outputs
        
        def xywh2xyxyxyxy(outputs):
            # 将 [cx, cy, w, h, angle] 格式的 box 转换为 [x1, y1, x2, y2, x3, y3, x4, y4] 格式
            class_id_conf = outputs[:, :2]
            cx = outputs[:, 2]
            cy = outputs[:, 3]
            w = outputs[:, 4]
            h = outputs[:, 5]
            angle = outputs[:, 6]

            x1, y1 = rotate_point(cx, cy, cx - w / 2, cy - h / 2, -angle)
            x2, y2 = rotate_point(cx, cy, cx + w / 2, cy - h / 2, -angle)
            x3, y3 = rotate_point(cx, cy, cx + w / 2, cy + h / 2, -angle)
            x4, y4 = rotate_point(cx, cy, cx - w / 2, cy + h / 2, -angle)
            outputs = np.concatenate((class_id_conf, x1[:, None], y1[:, None], x2[:, None], y2[:, None], x3[:, None], y3[:, None], x4[:, None], y4[:, None]), axis=-1)
            return outputs
            
        def tensor2numpy(data):
            if isinstance(data, torch.Tensor):
                return data.detach().cpu().numpy()
            else:
                return data
        
        def random_color(hue_min=0, hue_max=179):
            hue = random.randint(hue_min, hue_max)  # 色相从 0 到 179 均匀分布
            saturation = 255  # 饱和度设置为最大值
            value = 255  # 亮度设置为最大值
            color = cv2.cvtColor(np.uint8([[[hue, saturation, value]]]), cv2.COLOR_HSV2BGR)[0][0]
            return tuple(int(c) for c in color)
        
        def draw_hbb(img, outputs, save_path, cls_names):
            # 画框
            if save_path is None:
                print("Warning: save_path is None, no image will be saved.")
                return
            if cls_names is None:
                print("Warning: cls_names is None, class names will replaced by class id.")
                cls_names = [str(cls_id) for cls_id in outputs[:, 0]]
            rect_colors = {cls_id: random_color(0, 100) for cls_id in cls_names.keys()}
            font_colors = {cls_id: random_color(101, 179) for cls_id in cls_names.keys()}
            for box in outputs:
                cls_id, conf, cx, cy, w, h = box
                x1, y1, x2, y2 = int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2)
                text = "{}:{}".format(cls_names[int(cls_id)], round(conf, 2))
                font_size = min(h / img.shape[0], w / img.shape[1]) * 3
                cv2.rectangle(img, (x1, y1), (x2, y2), rect_colors[int(cls_id)], 1)
                cv2.putText(img, text, (max(10, int(x1)), max(10, int(y1))), cv2.FONT_HERSHEY_SIMPLEX, max(font_size, 0.5), font_colors[int(cls_id)], 1)
            cv2.imwrite(save_path, img)
            return

        def draw_obb(img, outputs, save_path, cls_names):
            # 画框
            if save_path is None:
                print("Warning: save_path is None, no image will be saved.")
                return
            if cls_names is None:
                print("Warning: cls_names is None, class names will replaced by class id.")
                cls_names = [str(cls_id) for cls_id in outputs[:, 0]]
            rect_colors = {cls_id: random_color(0, 100) for cls_id in cls_names.keys()}
            font_colors = {cls_id: random_color(101, 179) for cls_id in cls_names.keys()}
            for box in outputs:
                cls_id, conf, cx, cy, w, h, angle = box
                text = "{}:{}".format(cls_names[int(cls_id)], round(conf, 2))
                font_size = min(h / img.shape[0], w / img.shape[0]) * 3
                x1, y1 = rotate_point(cx, cy, cx - w / 2, cy - h / 2, -angle)
                x2, y2 = rotate_point(cx, cy, cx + w / 2, cy - h / 2, -angle)
                x3, y3 = rotate_point(cx, cy, cx + w / 2, cy + h / 2, -angle)
                x4, y4 = rotate_point(cx, cy, cx - w / 2, cy + h / 2, -angle)
                cv2.polylines(img, [np.array([[int(x1), int(y1)], [int(x2), int(y2)], [int(x3), int(y3)], [int(x4), int(y4)]], np.int32)], True, rect_colors[int(cls_id)], 1)
                cv2.putText(img, text, (max(10, int(x1)), max(10, int(y1))), cv2.FONT_HERSHEY_SIMPLEX, max(font_size, 0.5), font_colors[int(cls_id)], 1)
            cv2.imwrite(save_path, img)
            return

        def hbb_infer(outputs, IM):     # 水平框预测
            outputs = self.horizontal_nms(outputs)
            outputs = self.horizontal_postprocess(outputs, IM)
            outputs = tensor2numpy(outputs)  # shape: (box_num, 6)
            if mode == 0:
                return outputs
            elif mode == 1:
                outputs = xywh2xyxy(outputs)
            elif mode == 2:
                draw_hbb(img.copy(), outputs, save_path, self.cls_names)
            return outputs
        
        def obb_infer(outputs, IM):     # 旋转框预测
            outputs = self.obb_nms(outputs)
            outputs = self.obb_postprocess(outputs, IM)
            outputs = tensor2numpy(outputs)  # shape: (box_num, 7)
            if mode == 0:
                return outputs
            elif mode == 1:
                outputs = xywh2xyxyxyxy(outputs)
            elif mode == 2:
                draw_obb(img.copy(), outputs, save_path, self.cls_names)
            return outputs

        img_pre, IM = self._preprocess(img)  # img_pre(torch.Tensor): 预处理后的图像, IM(np.array(2x3)): 缩放平移逆矩阵
        outputs = self._infer(img_pre)
        if outputs.shape[1] == 4 + len(self.cls_names):
            outputs = hbb_infer(outputs, IM)
        elif outputs.shape[1] == 5 + len(self.cls_names):
            outputs = obb_infer(outputs, IM)
        else:
            raise ValueError("Warning: outputs.shape[1] is not 4 or 5, no infer result will be returned.")

        return outputs
    

if __name__ == '__main__':
    infer = Yolov8_Inference(
        model_path=r'E:\Common_scripts\pytorch-model\ObjectDetection\yolo\ultralytics\runs\detect\tiny_person\weights\best.onnx', 
        model_type='onnx', 
        input_size=640, 
        conf_thres=0.3, 
        iou_thres=0.5, 
        max_det=100, 
        device='gpu',
        cls_path=r'E:\Common_scripts\pytorch-model\ObjectDetection\yolo\ultralytics\datasets\sod_person\tiny_person.yaml'
    )
    # img = cv2.imread(r'E:\Dataset\tinyperson\yolo_type\images\test\baidu_P000_5.jpg')
    # infer.predict(
    #     img=img, 
    #     mode=2, 
    #     save_path=r'infer_result.jpg'
    # )

    pathlist = os.listdir(r'E:\Dataset\tinyperson\yolo_type\yolo_train_data\images\test')
    all_time = 0
    for i, path in enumerate(pathlist):
        
        img = cv2.imread(os.path.join(r'E:\Dataset\tinyperson\yolo_type\yolo_train_data\images\test', path))
        start_time = time.time()
        try:
            infer.predict(
                img=img, 
            )
        except:
            print("Error: ", path)

        end_time = time.time()
        print("Time cost: {:.6f}s".format(end_time - start_time))
        if i != 0:
            all_time += (end_time - start_time)
    print("Average Time cost: {:.6f}s".format(all_time / len(pathlist)))