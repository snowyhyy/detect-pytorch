"""
用于可视化训练日志的绘图工具
"""
import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path, PurePath


def plot_logs(logs, fields=('class_error', 'loss_bbox_unscaled', 'mAP'), ewm_col=0, log_name='log.txt'):
    '''
    函数用于绘制训练日志中的特定字段。绘制培训和测试结果。

    参数：
        logs = 包含 Path 对象的列表，每个对象都指向一个带日志文件的单独目录
        fields = 从每个日志文件中绘制哪些结果——为每个字段绘制训练和测试结果。
        ewm_col = 可选，使用哪一列作为图的指数加权平滑
        log_name = 可选，日志文件的名称

    返回：
        字段结果的matplotlib图，每个日志文件都有颜色编码
        实线是训练结果，虚线是测试结果
    '''
    func_name = "plot_utils.py::plot_logs"

    # 验证日志是路径列表（list[Paths]）或单个Pathlib对象Path，将单个Path转换为list以避免“不可迭代”错误

    if not isinstance(logs, list):
        if isinstance(logs, PurePath):
            logs = [logs]
            print(f"{func_name} info: logs param expects a list  argument, converted to list[Path].")
        else:
            raise ValueError(f"{func_name} - invalid argument for logs parameter.\n \
            Expect list[Path] or single Path obj, received {type(logs)}")
    
    # 质量检查 —— 验证有效的目录，列表中的每个项目都是 Path 对象，并且每个目录都存在 log_name 文件
    for i, dir in enumerate(logs):
        if not isinstance(dir, PurePath):
            raise ValueError(f"{func_name} - non-Path object in logs argument of {type(dir)}: \n{dir}")
        if not dir.exists():
            raise ValueError(f"{func_name} - invalid directory in logs argument: \n{dir}")
        # 验证 log_name 存在
        fn = Path(dir / log_name)
        if not fn.exists():
            print(f"-> missing {log_name}. Have you gotten to Epoch 1 in training?")
            print(f"--> full path of missing log file: {fn}")
            return

    # 载入日志文件并绘制
    dfs = [pd.read_json(Path(p) / log_name, lines=True) for p in logs]

    fig, axs = plt.subplots(ncols=len(fields), figsize=(16, 5))

    for df, color in zip(dfs, sns.color_palette(n_colors=len(logs))):
        for j, field in enumerate(fields):
            if field == 'mAP':
                coco_eval = pd.DataFrame(
                    np.stack(df.test_coco_eval_bbox.dropna().values)[:, 1]
                ).ewm(com=ewm_col).mean()
                axs[j].plot(coco_eval, c=color)
            else:
                df.interpolate().ewm(com=ewm_col).mean().plot(
                    y=[f'train_{field}', f'test_{field}'],
                    ax = axs[j],
                    color=[color] * 2,
                    style=['-', '--']
                )
    for ax, field in zip(axs, fields):
        ax.legend([Path(p).name for p in logs])
        ax.set_title(field)


# 绘制准确率召回率
def plot_precision_recall(files, naming_scheme='iter'):
    if naming_scheme == 'exp_id':
        # 名称变为 exp_id
        names = [f.parts[-3] for f in files]
    elif naming_scheme == 'iter':
        names = [f.stem for f in files]
    else:
        raise ValueError(f'not supported {naming_scheme}')
    fig, axs = plt.subplots(ncols=2, figsize=(16, 5))
    for f, color, name in zip(files, sns.color_palette("Blues", n_colors=len(files)), names):
        data = torch.load(f)
        # 精度为n_iou、n_points、n_cat、n_area、max_det
        precision = data['precision']
        recall = data['params'].recThrs
        scores = data['scores']
        # 对所有类别、所有区域和100次检测进行精确测量
        precision = precision[0, :, :, 0, -1].mean(1)
        scores = scores[0, :, :, 0, -1].mean(1)
        prec = precision.mean()
        rec = data['recall'][0, :, 0, -1].mean()
        print(
            f'{naming_scheme} {name}: mAP@50={prec * 100: 05.1f}, ' + 
            f'score={scores.mean():0.3f}, ' + 
            f'f1={2 * prec * rec / (prec + rec + 1e-8):0.3f}'
        )
        axs[0].plot(recall, precision, c=color)
        axs[1].plot(recall, scores, c=color)

    axs[0].set_title('Precision / Recall')
    axs[0].legend(names)
    axs[1].set_title('Score / Recall')
    axs[1].legend(names)
    return fig, axs
    