from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

import numpy as np


def _infer_regions(lead_names):
    """根据导联名称动态推断属于哪个参考类型"""
    ear = []
    bio = []
    avg = []
    for idx, name in enumerate(lead_names):
        if name.endswith('-A1') or name.endswith('-A2'):
            ear.append(idx)
        elif name.endswith('-AVG'):
            avg.append(idx)
        else:
            bio.append(idx)
    regions = {}
    if ear:
        regions['EAR'] = ear
    if bio:
        regions['BIO'] = bio
    if avg:
        regions['AVG'] = avg
    return regions


def plot_probs(prob_array, lead_names=None):
    """
    绘制伪迹概率热力图

    参数
    -----
    prob_array : ndarray, shape (N_lead, N_time, 3)
        每个导联在每个半秒窗口的 BKG/ART/ALPHA 概率
    lead_names : list of str, optional
        导联名称列表，用于推断脑区分组。不传则不分区域。

    返回
    -----
    fig : matplotlib.figure.Figure
    """
    print(np.shape(prob_array))
    # 输入是 (N_lead, N_time, 3) (N通道, 时间轴0.5秒一个, 3种可能性)
    _lead_len = np.shape(prob_array)[0]
    _array_len = np.shape(prob_array)[1]

    # 时序平滑
    smooth_data = gaussian_filter1d(prob_array, sigma=1, axis=1)
    matrix = smooth_data / smooth_data.sum(axis=2, keepdims=True)

    # 准备时间轴 (单位：分钟)
    time_axis = np.arange(_array_len) * 0.5 / 60

    # 推断脑区
    if lead_names is not None:
        regions = _infer_regions(lead_names)
    else:
        regions = {'All': list(range(_lead_len))}

    # 按照脑区顺序重新排列导联
    reordered_indices = []
    region_bounds = []
    current_idx = 0
    for name, ch_list in regions.items():
        reordered_indices.extend(ch_list)
        current_idx += len(ch_list)
        region_bounds.append((current_idx - len(ch_list), current_idx, name))

    reordered_matrix = matrix[reordered_indices, :, :]

    # 核心绘图：3个类型纵向展开 (3行1列)
    fig, axes = plt.subplots(3, 1, figsize=(22, 14), sharex=True)
    fig.subplots_adjust(hspace=0.25)

    class_names = ['BKG (背景脑电)', 'ART (伪迹)', 'ALPHA (α节律)']
    cmaps = ['Reds', 'Blues', 'Greens']

    # X轴时间刻度（每5分钟一个刻度）
    tick_spacing = max(1, int(600 / 0.5))
    xticks = np.arange(0, _array_len, tick_spacing)
    xtick_labels = [f"{x * 0.5 / 60:.0f}" for x in xticks]

    for idx in range(3):
        ax = axes[idx]
        plot_data = reordered_matrix[:, :, idx]

        im = ax.imshow(plot_data, cmap=cmaps[idx], aspect='auto',
                       vmin=0, vmax=1, extent=[time_axis[0], time_axis[-1], _lead_len, 0])

        # 绘制脑区分界线
        for start, end, name in region_bounds:
            if end < _lead_len and len(regions) > 1:
                ax.axhline(y=end, color='black', linestyle='-', linewidth=1.2, alpha=0.7)
            ax.text(time_axis[-1] + 0.2, (start + end) / 2, name,
                    va='center', ha='left', fontsize=11, fontweight='bold', color='dimgray')

        if idx == 2:
            ax.set_xticks([x * 0.5 / 60 for x in xticks])
            ax.set_xticklabels(xtick_labels, fontsize=10)
            ax.set_xlabel("时间 (分钟)", fontsize=13, fontweight='bold', labelpad=8)
        ax.set_xlim(time_axis[0], time_axis[-1])
        ax.set_ylabel("通道", fontsize=11)
        ax.set_yticks([])

        ax.set_title(class_names[idx], fontsize=14, fontweight='bold', pad=10, loc='left')

        cbar = fig.colorbar(im, ax=ax, orientation='vertical', pad=0.08, shrink=0.85)
        cbar.set_label('概率', fontsize=10)

    fig.suptitle("伪迹概率热力图 (按导联类型分组)", fontsize=16, fontweight='bold', y=0.97)
    fig.tight_layout()

    return fig