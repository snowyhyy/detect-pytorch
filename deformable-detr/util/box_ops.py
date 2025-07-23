'''
Author: snowy
Email: 1447048804@qq.com
Date: 2025-06-05 14:54:39
Version: 0.0.1
LastEditors: snowy 1447048804@qq.com
LastEditTime: 2025-07-11 18:22:27
Description: 根据官方的实现，添加中文注释，源码：https://github.com/facebookresearch/detr.git
'''
"""
用于边界框和GIou操作
"""
import torch
from torchvision.ops.boxes import box_area


def box_cxcywh_to_xyxy(x):
    """
    将边界框的中心点坐标和宽高转换为左上角和右下角坐标
    """
    x_c, y_c, w, h = x.unbind(-1)
    b = [
        (x_c - 0.5 * w), (y_c - 0.5 * h),
        (x_c + 0.5 * w), (y_c + 0.5 * h)
    ]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    """
    将边界框的左上角和右下角坐标转换为中心点坐标和宽高
    """
    x0, y0, x1, y1 = x.unbind(-1)
    b = [
        (x0 + x1) / 2, (y0 + y1) / 2,
        (x1 - x0), (y1 - y0)
    ]
    return torch.stack(b, dim=-1)


def box_iou(boxes1, boxes2):
    """
    计算两个边界框的IoU
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou


def generalized_box_iou(boxes1, boxes2):
    """
    计算两个边界框的GIoU
    boxes 的数据格式为 [x0, y0, x1, y1]
    返回 [N, M] 维度的矩阵，其中 N 为 boxes1 的数量，M 为 boxes2 的数量
    """
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2], 闭包矩形的宽高
    area = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    return iou - (area - union) / area
    

def masks_to_boxes(masks): # 输入 [N, H, W] 输出 [N, 4]
    """
    计算一组二值掩码对应的边界框
    masks 的数据格式为 [N, H, W]，其中 N 为掩码的数量，H 为掩码的高度，W 为掩码的宽度
    返回 [N, 4] 维度的矩阵，其中 N 为掩码的数量，4 为边界框的左上角和右下角坐标 [xyxy]
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)  # 空掩码处理
    
    h, w = masks.shape[-2:] # 掩码的高度和宽度
    
    y = torch.arange(0, h, dtype=torch.float)   # 创建从 0 到 h-1 的y坐标序列 [H]
    x = torch.arange(0, w, dtype=torch.float)   # 创建从 0 到 w-1 的x坐标序列 [W]
    y, x = torch.meshgrid(y, x) # 生成坐标网格矩阵  [H, W]
    # 计算每个掩码在 x 方向的最小和最大值
    x_mask = (masks * x.unsqueeze(0))   # [N, H, W] * [1, H, W] = [N, H, W]
    x_max = x_mask.flatten(1).max(-1)[0] # [N, H, W] -> [N, H*W] -> [N]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0] # 非掩码区域填充 1e8
    # 计算每个掩码在 y 方向的最小和最大值
    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)  # [N, 4]


if __name__ == '__main__':
    # 测试
    x = torch.rand(3, 10, 10)
    y = masks_to_boxes(x)
    print(y)