import torch
import time
import numpy as np

from torchvision.ops import batched_nms
from typing import List, Optional, Union, Tuple


def non_max_suppression(
    predicted_labels: torch.Tensor, 
    predicted_boxes: torch.Tensor, 
    predicted_scores: torch.Tensor, 
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    classes: Optional[Union[List[int], torch.Tensor]] = None,
    agnostic: bool = False,
    max_det: int = 300,
    max_nms: int = 30000,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """
    自定义非极大抑制（NMS）函数，适用于新模型的输出格式。

    参数：
        predicted_labels (torch.Tensor): [batch_size, num_boxes] 每个框的类别标签
        predicted_boxes (torch.Tensor): [batch_size, num_boxes, 4] 每个框的坐标 [x1, y1, x2, y2]
        predicted_scores (torch.Tensor): [batch_size, num_boxes] 每个框的置信度得分
        conf_thres (float): 置信度阈值，低于此值的框将被过滤
        iou_thres (float): IoU 阈值，用于NMS，重叠度高于此值的框将被抑制
        classes (List[int] 或 torch.Tensor, optional): 指定需要过滤的类别索引
        agnostic (bool): 是否忽略类别进行NMS
        max_det (int): 每张图片最多保留的检测框数量
        max_nms (int): NMS前最多处理的预测框数量

    返回：
        Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
            - List of filtered boxes for each image in the batch, shape [num_selected_boxes, 4]
            - List of filtered labels for each image in the batch, shape [num_selected_boxes]
            - List of filtered scores for each image in the batch, shape [num_selected_boxes]
    """

    # 获取批量大小
    batch_size: int = predicted_boxes.shape[0]
    
    # 初始化输出列表，用于存储每张图片的过滤后结果
    output_boxes: List[torch.Tensor] = []
    output_labels: List[torch.Tensor] = []
    output_scores: List[torch.Tensor] = []

    # 设定NMS的时间限制，防止处理时间过长
    time_limit: float = 2.0 + 0.05 * batch_size  # 以秒为单位
    t_start: float = time.time()  # 记录开始时间

    # 遍历每一张图片
    for i in range(batch_size):
        # 提取当前图片的预测框、得分和标签
        boxes: torch.Tensor = predicted_boxes[i]          # [num_boxes, 4]
        scores: torch.Tensor = predicted_scores[i]        # [num_boxes]
        labels: torch.Tensor = predicted_labels[i]        # [num_boxes]

        mask: torch.Tensor = scores > conf_thres
        boxes = boxes[mask]
        scores = scores[mask]
        labels = labels[mask]

        # 如果过滤后没有剩余的预测框，添加空的结果并继续
        if boxes.numel() == 0:
            output_boxes.append(torch.empty((0, 4), device=boxes.device))
            output_scores.append(torch.empty((0,), device=boxes.device))
            output_labels.append(torch.empty((0,), dtype=torch.int64, device=boxes.device))
            continue

        if classes is not None:
            # 如果classes是列表，将其转换为与labels相同设备的Tensor
            if isinstance(classes, list):
                classes = torch.tensor(classes, device=labels.device)
            # 创建一个布尔掩码，保留labels中属于classes的预测框
            mask = (labels.unsqueeze(1) == classes).any(1)
            boxes = boxes[mask]
            scores = scores[mask]
            labels = labels[mask]
            # 如果过滤后没有剩余的预测框，添加空的结果并继续
            if boxes.numel() == 0:
                output_boxes.append(torch.empty((0, 4), device=boxes.device))
                output_scores.append(torch.empty((0,), device=boxes.device))
                output_labels.append(torch.empty((0,), dtype=torch.int64, device=boxes.device))
                continue

        if boxes.shape[0] > max_nms:
            # 保留置信度最高的max_nms个预测框
            scores, idx = scores.topk(max_nms)
            boxes = boxes[idx]
            labels = labels[idx]

        # 如果agnostic=True，则忽略类别信息，所有预测框视为同一类别
        if agnostic:
            nms_labels: torch.Tensor = torch.zeros_like(labels)
        else:
            nms_labels = labels

        # 执行NMS，返回保留的框的索引
        keep: torch.Tensor = batched_nms(boxes, scores, nms_labels, iou_thres)

        if keep.shape[0] > max_det:
            keep = keep[:max_det]

        # 根据保留的索引提取最终的预测框、得分和标签
        selected_boxes: torch.Tensor = boxes[keep]
        selected_scores: torch.Tensor = scores[keep]
        selected_labels: torch.Tensor = labels[keep]

        # 将过滤后的结果添加到输出列表中
        output_boxes.append(selected_boxes)
        output_scores.append(selected_scores)
        output_labels.append(selected_labels)

        if (time.time() - t_start) > time_limit:
            print(f"WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded")
            break

    # 返回过滤后的预测框、标签和得分
    return output_boxes, output_labels, output_scores