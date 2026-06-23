import streamlit as st
import plotly.graph_objects as go
import mne
import pickle
import numpy as np
import json
import os
import glob
import tempfile
import pandas as pd
from io import BytesIO
from scipy import signal
from a_montage_tools import *
from a_psd_stat_tool import *
from plot_prob_array import plot_probs

# ========== 自包含的伪迹检测模型  ==========
# 使用纯 PyTorch 避免 fastai 依赖
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

class _Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class _GAP1d(nn.Module):
    def __init__(self, output_size=1):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool1d(output_size)
    def forward(self, x):
        return self.gap(x).view(x.size(0), -1)

class _TemporalBlock(nn.Module):
    def __init__(self, ni, nf, ks, stride, dilation, padding, dropout=0.):
        super().__init__()
        self.conv1 = weight_norm(nn.Conv1d(ni, nf, ks, stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = _Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = weight_norm(nn.Conv1d(nf, nf, ks, stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = _Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(ni, nf, 1) if ni != nf else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

def _TemporalConvNet(c_in, layers, ks=2, dropout=0.):
    temp_layers = []
    for i in range(len(layers)):
        dilation_size = 2 ** i
        ni = c_in if i == 0 else layers[i-1]
        nf = layers[i]
        temp_layers += [_TemporalBlock(ni, nf, ks, stride=1, dilation=dilation_size,
                                       padding=(ks-1) * dilation_size, dropout=dropout)]
    return nn.Sequential(*temp_layers)

class _TCN(nn.Module):
    def __init__(self, c_in, c_out, layers=8*[25], ks=7, conv_dropout=0., fc_dropout=0.):
        super().__init__()
        self.tcn = _TemporalConvNet(c_in, layers, ks=ks, dropout=conv_dropout)
        self.gap = _GAP1d()
        self.dropout = nn.Dropout(fc_dropout) if fc_dropout else None
        self.linear = nn.Linear(layers[-1], c_out)
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        x = self.tcn(x)
        x = self.gap(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.linear(x)


def normalize_electrode_name(name):
    """
    标准化电极名称大小写
    """
    if not name:
        return name
    
    name_upper = name.upper()
    
    # 特殊处理：Fp (额极) 系列 - 第二个字母 p 小写
    # 包括: Fp1, Fp2, Fpz 等
    if name_upper.startswith('FP') and len(name) > 2:
        third_char = name[2].lower()  # 第3个字符小写
        return 'Fp' + third_char + (name[3:] if len(name) > 3 else '')
    
    # 其他中线电极：最后一个 z 小写
    # 包括: Fz, Cz, Pz, Oz 等
    if name_upper.endswith('Z') and len(name) > 1:
        return name[:-1] + 'z'
    
    # 其他电极：首字母大写，其余保持
    # 例如 F3, C4, O1, T5 等
    return name.capitalize() if len(name) > 1 else name.upper()


# 设置页面
st.set_page_config(page_title="PSD 可视化工具", layout="wide")

# ========== 缓存函数 ==========
@st.cache_data(max_entries=3, ttl=3600)
def load_eeg_data(eeg_bytes: bytes, file_name: str, file_type: str):
    """加载EEG文件（支持EDF和FIF格式）"""
    temp_dir = tempfile.gettempdir()
    ext = file_type.lower()
    temp_path = os.path.join(temp_dir, f"temp_eeg_{hash(eeg_bytes)}_{file_name}.{ext}")
    
    try:
        with open(temp_path, "wb") as f:
            f.write(eeg_bytes)
        
        if ext == "fif":
            raw = mne.io.read_raw_fif(temp_path, preload=True)
        else:
            raw = mne.io.read_raw_edf(temp_path, preload=True)
        
        raw._data *= 1e6  # 转为微伏
        raw.resample(256)
        
        data_dict = {
            'data': raw.get_data(),
            'ch_names': raw.ch_names,
            'sfreq': raw.info['sfreq'],
            'n_times': raw.n_times,
            'times': raw.times.tolist(),
            'duration_sec': raw.times[-1]
        }
        
        return data_dict
        
    finally:
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except:
            pass


@st.cache_data(max_entries=2, ttl=3600)
def load_prob_data(prob_bytes: bytes, file_name: str):
    """加载Prob文件"""
    if prob_bytes is None:
        return None
    return pickle.loads(prob_bytes)


@st.cache_resource(show_spinner="🧠 加载伪迹检测模型...")
def load_artifact_model():
    """加载预训练的伪迹检测TCN模型"""
    model = _TCN(1, 3, layers=5 * [16], ks=7, conv_dropout=0.3, fc_dropout=0.3)
    model_path = os.path.join(os.path.dirname(__file__),
                              'Normal_Reference', 'arch', 'tcn_state_20260529_BAA')
    if not os.path.exists(model_path):
        st.warning(f"⚠️ 模型文件不存在: {model_path}")
        return None
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    return model


def auto_generate_prob_dict(full_montage_dict, fs=256):
    """自动生成伪迹概率字典 (无需用户上传 .pkl 文件)

    使用预训练的TCN模型对每个导联的信号逐秒预测伪迹概率，
    输出与 pickle 文件格式完全一致的 prob_dict。
    """
    model = load_artifact_model()
    if model is None:
        return None

    prob_dict = {}
    for lead_name, signal_array in full_montage_dict.items():
        second_len = int(len(signal_array) / fs)
        if second_len < 2:
            continue

        # 每1秒一个窗口，加通道维度 → (second_len, 1, 256)
        lead_x = signal_array[:second_len * fs].reshape(second_len, fs)
        lead_x = lead_x[:, np.newaxis, :].astype(np.float32)

        # 错位0.5秒再取一次 → 得到0.5秒分辨率
        lead_x_128 = signal_array[128:second_len * fs - 128].reshape(second_len - 1, fs)
        lead_x_128 = lead_x_128[:, np.newaxis, :].astype(np.float32)

        with torch.no_grad():
            x = torch.from_numpy(np.ascontiguousarray(lead_x))
            prob = model(x)
            prob = torch.nn.functional.softmax(prob, dim=1).cpu().numpy()

            x2 = torch.from_numpy(np.ascontiguousarray(lead_x_128))
            prob2 = model(x2)
            prob2 = torch.nn.functional.softmax(prob2, dim=1).cpu().numpy()

        # 交织得到半秒分辨率 (second_len*2, 3)
        out_prob_array = np.zeros((second_len * 2, 3))
        out_prob_array[0::2, :] = prob
        out_prob_array[1:-1:2, :] = prob2

        prob_dict[lead_name] = out_prob_array

    return prob_dict


@st.cache_data(max_entries=1, ttl=3600)
def load_normal_reference_data(json_dir_hash: str):
    """加载正常参考数据（新版 Normal_Reference 格式）"""
    # 尝试多个可能的路径
    possible_paths = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Normal_Reference", "normal", "combined_result_0611.json"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "Normal_Reference", "normal", "combined_result_0611.json"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Normal_Reference", "normal", "combined_result_0611.json"),
    ]
    
    json_path = None
    for path in possible_paths:
        if os.path.exists(path):
            json_path = path
            break
    
    if json_path is None:
        st.warning(f"⚠️ 参考数据文件不存在")
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Normal_Reference", "arch", "tcn_state_20260529_BAA")
        if not os.path.exists(model_path):
            st.warning(f"⚠️ 模型文件不存在: {model_path}")
        return None
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            all_data = json.load(f)
        return all_data
    except Exception as e:
        st.warning(f"加载参考数据失败: {e}")
        return None


def load_sp_reference_data():
    """加载 SP 参照值数据（show_sp.py 输出的 JSON）"""
    # 尝试多个可能的路径
    possible_paths = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sp_reference_*.json"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "sp_reference_*.json"),
    ]
    
    for pattern in possible_paths:
        sp_files = sorted(glob.glob(pattern))
        if sp_files:
            break
    
    if not sp_files:
        return None
    
    try:
        combined = {}
        for sp_file in sp_files:
            with open(sp_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                combined.update(data)
        return combined
    except Exception as e:
        st.warning(f"加载 SP 参照数据失败: {e}")
        return None


def get_sp_ref(sp_ref_data, band, lead, stat_type='mean'):
    """获取 SP 参照值
    
    SP 参照值不需要 age_group 和 window_sizes，直接通过 lead__band 查找。
    """
    if sp_ref_data is None:
        return None
    # 键名映射（与 get_normal_ref 保持一致）
    band_key = band
    if band == 'DTAR':
        band_key = 'DT_AR'
    new_key = f"{lead}__{band_key}"
    for key in sp_ref_data.keys():
        g_info = sp_ref_data[key].get('g_info', {})
        if new_key in g_info:
            val = g_info[new_key].get(stat_type)
            if val is not None:
                return val
    return None


def get_normal_ref(all_ref_data, age_group, band, lead, stat_type='mean', window_sizes=None):
    """获取正常参考值（适配新版 Normal_Reference 数据格式）

    Parameters
    ----------
    window_sizes : list, optional
        用户选择的窗口大小列表，用于筛选对应窗口的常模数据。
        例如 [2, 5, 10]。为 None 或空时不过滤（保留旧行为）。
    """
    if all_ref_data is None:
        return None
    
    # 年龄组映射（旧格式 -> 新格式前缀）
    age_group_mapping = {
        '0-6': 'age_0_6',
        '7-13': 'age_7_13',
        '14-18': 'age_14_18',
        '19-44': 'age_19_44',
        '45-59': 'age_45_59',
        '60-80': 'age_60_80',
        '80': 'age_80',
        'total': 'age_19_44'
    }
    
    # 获取年龄组前缀
    age_prefix = age_group_mapping.get(age_group, 'age_19_44')
    
    # 将用户选择的窗口大小转为 set，方便快速查找
    target_window_sizes = set(window_sizes) if window_sizes else set()
    
    # 新格式 key: age_19_44_10_normal_ref_0526
    # 年龄前缀有几段（如 age_19_44 → 3段），窗口就在第几段
    prefix_parts_count = len(age_prefix.split('_'))
    
    values = []
    for key in all_ref_data.keys():
        if not key.startswith(age_prefix):
            continue
        parts = key.split('_')
        if len(parts) > prefix_parts_count:
            try:
                ws_in_key = int(parts[prefix_parts_count])
            except:
                continue
        else:
            continue
        if target_window_sizes and ws_in_key not in target_window_sizes:
            continue
        band_key = band
        if band == 'DTAR':
            band_key = 'DT_AR'
        new_key = f"{lead}__{band_key}"
        ref_data = all_ref_data[key]
        g_info = ref_data.get('g_info', {})
        if new_key in g_info:
            val = g_info[new_key].get(stat_type)
            if val is not None:
                values.append(val)
    return np.mean(values) if values else None


def calculate_zscore(value, mean, std):
    if std == 0 or mean is None or std is None:
        return 0.0
    return (value - mean) / std


def calculate_alpha_asymmetry(spec_dict_all, left_lead, right_lead):
    """计算 Alpha 不对称指数
    
    公式: (L - R) / (L + R)
    L = 左侧电极 alpha 功率
    R = 右侧电极 alpha 功率
    """
    if left_lead not in spec_dict_all or right_lead not in spec_dict_all:
        return None
    
    left_alpha = np.mean(spec_dict_all[left_lead].get('alpha', [0]))
    right_alpha = np.mean(spec_dict_all[right_lead].get('alpha', [0]))
    
    if left_alpha + right_alpha == 0:
        return 0.0
    
    return (left_alpha - right_alpha) / (left_alpha + right_alpha)


def calculate_alpha_asymmetry_series(spec_dict_all, left_lead, right_lead):
    """计算每个epoch的Alpha不对称指数时序
    
    公式: (L - R) / (L + R)
    """
    if left_lead not in spec_dict_all or right_lead not in spec_dict_all:
        return None
    
    left_alpha = np.array(spec_dict_all[left_lead].get('alpha', []))
    right_alpha = np.array(spec_dict_all[right_lead].get('alpha', []))
    
    if len(left_alpha) == 0 or len(right_alpha) == 0:
        return None
    
    if len(left_alpha) != len(right_alpha):
        min_len = min(len(left_alpha), len(right_alpha))
        left_alpha = left_alpha[:min_len]
        right_alpha = right_alpha[:min_len]
    
    asymmetry_series = (left_alpha - right_alpha) / (left_alpha + right_alpha + 1e-8)
    return asymmetry_series


def get_asymmetry_ref(all_ref_data, age_group, asymmetry_key, ref_type='mean'):
    """获取不对称指数的正常参考值"""
    if all_ref_data is None:
        return None, None
    
    age_group_mapping = {
        '0-6': 'age_0_6', '7-13': 'age_7_13', '14-18': 'age_14_18',
        '19-44': 'age_19_44', '45-59': 'age_45_59', '60-80': 'age_60_80',
        '80': 'age_80', 'total': 'age_19_44'
    }
    
    age_prefix = age_group_mapping.get(age_group, 'age_19_44')
    
    for key in all_ref_data.keys():
        if not key.startswith(age_prefix):
            continue
        
        f3478_info = all_ref_data[key].get('f3478_info', {})
        if asymmetry_key in f3478_info:
            mean = f3478_info[asymmetry_key].get('mean')
            std = f3478_info[asymmetry_key].get('std')
            return mean, std
    
    return None, None


# ========== 主程序 ==========
CODE_VERSION = "v2"  # 代码变更时递增此值以强制清除旧缓存

def main():
    # 初始化session_state
    if 'psd_results' not in st.session_state:
        st.session_state.psd_results = None
    if 'analysis_params' not in st.session_state:
        st.session_state.analysis_params = {}
    
    st.title("🧠 EEG PSD 可视化工具")
    
    with st.sidebar:
        st.header("⚙️ 参数设置")
        
        # EEG 文件格式选择
        eeg_format = st.selectbox(
            "选择EEG文件格式",
            ["EDF", "FIF"],
            key="eeg_format"
        )
        
        if eeg_format == "EDF":
            edf_file = st.file_uploader("选择EDF文件", type=["edf"], key="edf_uploader")
        else:
            edf_file = st.file_uploader("选择FIF文件", type=["fif"], key="edf_uploader")
        
        prob_file = st.file_uploader("选择Prob文件（可选，不传则自动检测伪迹）", type=["pkl"], key="prob_uploader")
        
        st.divider()
        
        st.subheader("📊 分析参数")
        epoch_len_sec = st.slider("Epoch长度(秒)", 1, 10, 5, key="epoch_len")
        nperseg_len = st.slider("Welch窗口长度(秒)", 1, 5, 2, key="nperseg")
        art_threshold = st.slider("伪迹过滤阈值", 0.0, 1.0, 0.0, 0.1, key="art_threshold")
        # art_fallback_mode 不在侧边栏显示，触发时才弹窗确认
        
        st.subheader("🔗 导联选择")
        
        # 导联类型选择
        lead_type = st.selectbox(
            "选择导联类型",
            ["双极导联 (Bipolar)", "耳电极参考 (Ear)", "平均参考 (Average)"],
            key="lead_type"
        )
        
        # 根据类型定义导联列表
        ear_leads = [
            'Fp1-A1', 'Fp2-A2', 'F3-A1', 'F4-A2', 'C3-A1', 'C4-A2',
            'P3-A1', 'P4-A2', 'O1-A1', 'O2-A2', 'F7-A1', 'F8-A2',
            'T3-A1', 'T4-A2', 'T5-A1', 'T6-A2'
        ]
        bipolar_leads = [
            'Fp1-F3', 'Fp2-F4', 'F3-C3', 'F4-C4', 'C3-P3', 'C4-P4',
            'P3-O1', 'P4-O2', 'Fp1-F7', 'Fp2-F8', 'F7-T3', 'T3-T5', 'T5-O1',
            'F8-T4', 'T4-T6', 'T6-O2', 'Fz-Pz', 'Cz-Pz', 'Pz-Oz'
        ]
        avg_leads = [
            'Fp1-AVG', 'Fp2-AVG', 'F3-AVG', 'F4-AVG', 'C3-AVG', 'C4-AVG',
            'P3-AVG', 'P4-AVG', 'O1-AVG', 'O2-AVG', 'F7-AVG', 'F8-AVG',
            'T3-AVG', 'T4-AVG', 'T5-AVG', 'T6-AVG', 'Fz-AVG', 'Cz-AVG', 'Pz-AVG', 'Fpz-AVG', 'Oz-AVG'
        ]
        
        # 根据选择显示对应的导联列表
        if "耳电极" in lead_type:
            lead_options = ear_leads
            default_leads = ['Fp1-A1', 'F3-A1']
        elif "平均" in lead_type:
            lead_options = avg_leads
            default_leads = ['Fp1-AVG', 'F3-AVG']
        else:
            lead_options = bipolar_leads
            default_leads = ['Fp1-F3', 'F3-C3']
        
        # 定义左脑/右脑导联
        def _is_left_lead(name):
            """判断是否为左脑导联（奇数编号或A1参考）"""
            n = name.upper()
            return any(x in n for x in ['1-', '3-', '5-', '7-', '-A1', 'F7', 'T3', 'T5', 'O1', 'FP1'])
        
        def _is_right_lead(name):
            """判断是否为右脑导联（偶数编号或A2参考）"""
            n = name.upper()
            return any(x in n for x in ['2-', '4-', '6-', '8-', '-A2', 'F8', 'T4', 'T6', 'O2', 'FP2'])
        
        left_leads = [l for l in lead_options if _is_left_lead(l)]
        right_leads = [l for l in lead_options if _is_right_lead(l)]
        
        # 在 multiselect 选项中添加 L/R/全脑 快捷组（作为虚拟导联，表示该组均值）
        lead_options_with_groups = ["🧠 全脑 (均值)", "🧠 左脑 L (均值)", "🧠 右脑 R (均值)"] + lead_options
        
        selected_leads = st.multiselect("选择导联", lead_options_with_groups, default=default_leads, key="selected_leads")
        
        st.divider()
        
        st.subheader("📈 正常参考对比")
        enable_zscore = st.checkbox("启用Z-score对比", value=True, key="enable_zscore")
        
        age_group_options = ['0-6', '7-13', '14-18', '19-44', '45-59', '60-80', '80', 'total']
        selected_age_group = st.selectbox("选择参考年龄组", age_group_options, index=3, key="age_group")
        
        window_sizes = st.multiselect(
            "选择窗口大小", 
            [2, 5, 10, 15], 
            default=[2, 5, 10], 
            key="window_sizes"
        )
        
        zscore_threshold = st.slider("Z-score异常阈值", 1.0, 3.0, 2.0, 0.1, key="zscore_threshold")
        
        ref_source = st.selectbox(
            "选择参考数据源",
            ["常模 (Normal_Reference)", "SP 参照值 (SOLAR2000)", "双参照对比 (常模 + SP)"],
            index=0,
            key="ref_source",
            help="常模：基于 Normal_Reference 的统计常模\nSP 参照值：基于 show_sp.py 输出的 SOLAR2000 参考值\n双参照对比：同时显示两组参照值"
        )
    
    # 构建当前参数
    current_params = {
        'code_version': CODE_VERSION,
        'edf_name': edf_file.name if edf_file else None,
        'prob_name': prob_file.name if prob_file else None,
        'epoch_len_sec': epoch_len_sec,
        'nperseg_len': nperseg_len,
        'art_threshold': art_threshold,
        'art_fallback_mode': st.session_state.get('art_fallback_mode', '保留原始数据'),
        'selected_leads': tuple(sorted(selected_leads)),
        'enable_zscore': enable_zscore,
        'selected_age_group': selected_age_group,
        'window_sizes': tuple(sorted(window_sizes)),
        'zscore_threshold': zscore_threshold,
        'ref_source': ref_source
    }
    
    # 检查是否需要重新计算（edf_file存在且参数变化时自动计算）
    need_compute = False
    
    if edf_file:
        if st.session_state.analysis_params != current_params:
            need_compute = True
    
    # 执行分析
    # 当用户手动调整参数时，重置导联剔除模式（弹窗将在新的分析条件下重新触发）
    if need_compute and edf_file:
        st.session_state.pop('_fb_dismissed', None)
        st.session_state['art_fallback_mode'] = '保留原始数据'
        st.session_state.analysis_params = current_params
        
        with st.status("正在分析数据...", expanded=True) as status:
            try:
                # 1. 加载EEG数据
                status.update(label=f"📂 加载{eeg_format}文件...")
                file_ext = "fif" if eeg_format == "FIF" else "edf"
                edf_data = load_eeg_data(edf_file.getvalue(), edf_file.name, file_ext)
                
                if edf_data is None:
                    st.error(f"❌ {eeg_format}文件加载失败")
                    return
                
                # 2. 加载Prob文件（如有上传则加载，否则稍后自动生成）
                status.update(label="📋 加载Prob文件...")
                prob_dict = None
                if prob_file is not None:
                    prob_dict = load_prob_data(prob_file.getvalue(), prob_file.name)
                    if prob_dict is not None:
                        st.info(f"✅ 已加载用户上传的Prob文件 ({prob_file.name})")
                
                # 3. 加载正常参考数据
                status.update(label="📚 加载参考数据...")
                all_ref_data = None
                sp_ref_data = None
                if enable_zscore:
                    all_ref_data = load_normal_reference_data(None)
                    if all_ref_data is None:
                        st.warning("⚠️ 正常参考数据加载失败！Z-score功能将不可用。")
                    # 如果选择了 SP 参照或双参照，额外加载 SP 参照
                    if ref_source in ["SP 参照值 (SOLAR2000)", "双参照对比 (常模 + SP)"]:
                        sp_ref_data = load_sp_reference_data()
                        if sp_ref_data is None:
                            st.warning("⚠️ SP 参照值加载失败！请先运行 show_sp.py 生成 JSON 文件。")
                
                # 4. 准备数据
                status.update(label="🔧 准备数据...")
                all_data = {}
                for idx, ch_name in enumerate(edf_data['ch_names']):
                    all_data[ch_name] = edf_data['data'][idx]
                
                # 通道名称归一化：剥离前缀和后缀，使 montage 函数能匹配标准电极名
                # 处理格式: "EEG FP1-REF" -> "Fp1"
                _prefixes = ['EEG ', 'eeg ', 'EEG-', 'eeg-']
                _ref_suffixes = ['-REF', '-ref', '-LE', '-le', '-AVG', '-avg', '-A1', '-a1', '-A2', '-a2']
                all_data_normalized = dict(all_data)
                for ch_name, ch_data in list(all_data.items()):
                    norm_name = ch_name
                    # 1. 去掉前缀 (如 "EEG ")
                    for prefix in _prefixes:
                        if norm_name.upper().startswith(prefix.upper()):
                            norm_name = norm_name[len(prefix):]
                            break
                    # 2. 去掉后缀 (如 "-REF")
                    for suffix in _ref_suffixes:
                        if norm_name.upper().endswith(suffix.upper()):
                            norm_name = norm_name[:-len(suffix)]
                            break
                    # 3. 标准化电极名大小写 (FP1 -> Fp1, FP2 -> Fp2, 等)
                    norm_name = normalize_electrode_name(norm_name)
                    if norm_name != ch_name:
                        all_data_normalized[norm_name] = ch_data
                
                # 5. 计算完整 montage（始终获取全部三种导联类型）
                status.update(label="🔗 计算导联数据...")
                full_montage_dict = get_bipolar_data_caueeg(all_data_normalized, 1.5, 70)
                
                # 5b. 如果没有上传Prob文件，自动生成伪迹概率
                if prob_dict is None:
                    status.update(label="🧠 自动检测伪迹...")
                    with st.spinner("使用神经网络模型自动检测伪迹..."):
                        prob_dict = auto_generate_prob_dict(full_montage_dict, int(edf_data['sfreq']))
                    if prob_dict is not None:
                        st.info(f"✅ 已自动生成伪迹概率（{len(prob_dict)}个导联）")
                    else:
                        st.warning("⚠️ 自动伪迹检测不可用，将不使用伪迹过滤")
                # 过滤当前选中类型用于显示
                if "耳电极" in lead_type:
                    leads_montage_dict = {k: v for k, v in full_montage_dict.items() if k.endswith('-A1') or k.endswith('-A2')}
                elif "平均" in lead_type:
                    leads_montage_dict = {k: v for k, v in full_montage_dict.items() if k.endswith('-AVG')}
                else:
                    leads_montage_dict = {k: v for k, v in full_montage_dict.items() if '-A1' not in k and '-A2' not in k and '-AVG' not in k}
                
                # 6. 计算PSD（保持原有的epoch计算逻辑）
                status.update(label="📊 计算PSD...")
                fs = int(edf_data['sfreq'])
                duration_sec = edf_data['duration_sec']
                
                
                epoch_count = int(duration_sec / epoch_len_sec)
                
                if epoch_count == 0:
                    st.error("❌ Epoch长度超过数据总时长")
                    return
                
                
                epoch_times = list(range(epoch_count))
                
                spec_dict_all = {}
                leads_list = []
                all_psds = []
                fallback_leads = []  # 记录所有epoch被伪迹过滤剔除的导联
                all_psds_raw_dict = {}  # lead_name -> 全量(过滤前)PSD
                keep_masks_dict = {}     # lead_name -> epoch保留掩码
                
                # 进度条
                progress_bar = st.progress(0)
                total_leads = len(leads_montage_dict.keys())
                
                for idx, lead_name in enumerate(leads_montage_dict.keys()):
                    progress_bar.progress((idx + 1) / total_leads)
                    
                    one_signal = leads_montage_dict[lead_name]
                    
                    
                    total_samples = epoch_count * epoch_len_sec * fs
                    
                    if len(one_signal) < total_samples:
                        one_signal = np.pad(one_signal, (0, total_samples - len(one_signal)), 'constant')
                    
                    one_signal_reshape = one_signal[:total_samples].reshape(epoch_count, epoch_len_sec * fs)
                    freqs_raw, psds_raw = signal.welch(one_signal_reshape, fs=fs, window='hann', nperseg=fs * nperseg_len)

                    # 插值到标准 1Hz 频率网格 (0 ~ fs/2 Hz)，使列索引 = 频率 Hz 值
                    # 以适配 get_spec_stat_info 等引用代码的硬编码索引约定
                    freqs = np.arange(0, fs // 2 + 1)
                    psds = np.zeros((psds_raw.shape[0], len(freqs)))
                    for i in range(psds_raw.shape[0]):
                        psds[i, :] = np.interp(freqs, freqs_raw, psds_raw[i, :])
                    psds = np.log1p(psds)
                    all_psds_raw_dict[lead_name] = psds  # 保存全量PSD（过滤前）

                    if prob_dict is not None and lead_name in prob_dict:
                        _prob = prob_dict[lead_name]
                        # 保持原有的prob处理逻辑
                        if epoch_count * epoch_len_sec != np.shape(_prob)[0] * 2:
                            _prob = np.pad(_prob, ((0, 1), (0, 0)), mode='constant', constant_values=0)
                        art_prob_index = _prob[:, 2] + _prob[:, 1]
                        prob_len = epoch_count * epoch_len_sec * 2
                        if prob_len <= len(art_prob_index):
                            mean_art_prob = np.max(art_prob_index[:prob_len].reshape(epoch_count, epoch_len_sec * 2), axis=1)
                        else:
                            mean_art_prob = np.max(art_prob_index, axis=0) * np.ones(epoch_count)
                        keep_mask = mean_art_prob < (1 - art_threshold)
                        psds_without_art = psds[keep_mask, :]
                    else:
                        psds_without_art = psds
                        keep_mask = np.ones(epoch_count, dtype=bool)
                    keep_masks_dict[lead_name] = keep_mask

                    # ====== 检测并处理"所有epoch被过滤" ======
                    if len(psds_without_art) == 0 and len(psds) > 0:
                        fallback_leads.append(lead_name)
                        _fb_mode = st.session_state.get('art_fallback_mode', '保留原始数据')
                        if _fb_mode == "剔除该导联":
                            continue

                    if len(psds_without_art) > 0:
                        leads_list.append(lead_name)
                        all_psds.append(psds_without_art)
                    elif len(psds) > 0:
                        leads_list.append(lead_name)
                        all_psds.append(psds)
                
                progress_bar.empty()
                
                if len(all_psds) == 0:
                    st.error("❌ 没有有效的PSD数据，请降低伪迹过滤严格度")
                    return
                
                # 7. 计算统计信息（与 PSD_calculate_full_EDF.py 保持一致）
                status.update(label="📈 计算统计指标...")
                spec_dict = get_spec_stat_info(all_psds)
                
                # 计算未过滤的全量PSD统计信息（用于过滤前后对比图）
                if prob_dict is not None:
                    all_psds_full = [all_psds_raw_dict[ln] for ln in leads_list]
                    spec_dict_full = get_spec_stat_info(all_psds_full)
                else:
                    spec_dict_full = spec_dict
                
                for lead_idx, lead_name in enumerate(leads_list):
                    if lead_idx < len(all_psds):
                        # 保留原始PSD数据 + 所有计算指标
                        # get_spec_stat_info返回的是{key: [array1, array2, ...]}格式
                        # 其中每个列表的长度应该等于len(all_psds)
                        lead_features = {}
                        for k, v in spec_dict.items():
                            if isinstance(v, list) and lead_idx < len(v):
                                lead_features[k] = v[lead_idx]
                            elif isinstance(v, np.ndarray) and len(v.shape) > 1 and lead_idx < v.shape[0]:
                                lead_features[k] = v[lead_idx]
                        
                        # 构建过滤前后对比数据
                        _full_extra = {}
                        if prob_dict is not None and lead_name in keep_masks_dict:
                            for k in lead_features.keys():
                                if k in spec_dict_full and isinstance(spec_dict_full[k], list) and lead_idx < len(spec_dict_full[k]):
                                    full_vals = np.array(spec_dict_full[k][lead_idx])  # G_: 全量epoch
                                    _full_extra[f'{k}_full'] = full_vals
                                    filt_vals = np.full(len(full_vals), np.nan)  # E_: NaN填充
                                    mask = keep_masks_dict[lead_name]
                                    if np.any(mask):
                                        filt_vals[mask] = lead_features[k]
                                    _full_extra[f'{k}_filt'] = filt_vals
                        
                        spec_dict_all[lead_name] = {
                            'psd': all_psds[lead_idx],  # 原始线性功率谱 (epoch×freq)
                            'psd_db': 10 * np.log10(all_psds[lead_idx]),  # dB版本
                            **lead_features,  # 所有计算指标
                            **_full_extra  # 过滤前后对比数据
                        }
                
                # 8. 存储结果（包含原始montage数据供导出使用）
                st.session_state.psd_results = {
                    'spec_dict_all': spec_dict_all,
                    'epoch_times': epoch_times,
                    'leads_list': leads_list,
                    'all_ref_data': all_ref_data,
                    'sp_ref_data': sp_ref_data,
                    'freqs': freqs,
                    'full_montage_dict': full_montage_dict,
                    'epoch_count': epoch_count,
                    'fs': fs,
                    'nperseg_len': nperseg_len,
                    'art_threshold': art_threshold,
                    'prob_dict': prob_dict,
                    'fallback_leads': fallback_leads,
                    'duration_sec': edf_data['duration_sec'],
                    'sfreq_val': edf_data['sfreq'],
                    'analysis_params': current_params.copy()
                }
                
                status.update(label="✅ 分析完成！", state="complete")
                st.success("分析完成！")
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ 分析过程中出错: {str(e)}")
                import traceback
                with st.expander("查看详细错误信息"):
                    st.code(traceback.format_exc())
                return
    
    # 显示结果
    if st.session_state.psd_results:
        results = st.session_state.psd_results
        spec_dict_all = results['spec_dict_all']
        epoch_times = results['epoch_times']
        all_ref_data = results.get('all_ref_data')
        sp_ref_data = results.get('sp_ref_data')
        if enable_zscore and not all_ref_data:
            all_ref_data = load_normal_reference_data(None)
        if enable_zscore and sp_ref_data is None and ref_source in ["SP 参照值 (SOLAR2000)", "双参照对比 (常模 + SP)"]:
            sp_ref_data = load_sp_reference_data()
        freqs = results.get('freqs')  # 获取频率轴数据
        
        valid_leads = [l for l in selected_leads if l in spec_dict_all or l in ["🧠 全脑 (均值)", "🧠 左脑 L (均值)", "🧠 右脑 R (均值)"]]
        
        if not valid_leads:
            st.warning("⚠️ 选中的导联无有效数据，请重新选择")
            return
        
        # 虚拟组导联名称集合（这些导联在正常参考数据中无对应条目）
        _VIRTUAL_LEADS = {"🧠 全脑 (均值)", "🧠 左脑 L (均值)", "🧠 右脑 R (均值)"}
        
        # 为虚拟组导联（全脑/左脑/右脑）计算组内均值
        _group_virtual = {
            "🧠 全脑 (均值)": lead_options,
            "🧠 左脑 L (均值)": left_leads,
            "🧠 右脑 R (均值)": right_leads,
        }
        for virt_name, group_leads in _group_virtual.items():
            if virt_name not in valid_leads:
                continue
            group_leads_in_data = [l for l in group_leads if l in spec_dict_all]
            if not group_leads_in_data:
                continue
            # 聚合 spec_dict_all 中该组所有导联的数据
            virt_data = {}
            sample_keys = spec_dict_all[group_leads_in_data[0]].keys()
            for k in sample_keys:
                arrays = []
                for l in group_leads_in_data:
                    v = spec_dict_all[l].get(k)
                    if v is not None and isinstance(v, (np.ndarray, list)):
                        arrays.append(np.array(v))
                if arrays:
                    min_len = min(len(a) for a in arrays)
                    stacked = np.array([a[:min_len] for a in arrays])
                    virt_data[k] = np.mean(stacked, axis=0)
            spec_dict_all[virt_name] = virt_data
        
        tab1, tab2, tab3 = st.tabs(["📊 频段功率", "📈 功率比率", "📋 统计汇总"])
        
        with tab1:
            st.subheader("各频段功率分布")
            
            # 频段选择器
            _all_band_names = [
                "Delta (1-4Hz)", "Theta (4-8Hz)",
                "Alpha (8-13Hz)", "Alpha₁ (8-9Hz)", "Alpha₂ (9-11Hz)", "Alpha₃ (11-13Hz)",
                "Beta (13-30Hz)", "Beta₁ (13-20Hz)", "Beta₂ (20-30Hz)",
                "Gamma (30-70Hz)", "Gamma₁ (30-50Hz)", "Gamma₂ (50-70Hz)",
            ]
            selected_bands = st.multiselect(
                "选择要显示的频段",
                _all_band_names,
                default=_all_band_names,
                key="band_select"
            )
            
            _band_keys = [
                ('delta',   "Delta (1-4Hz)",   '#e74c3c'),
                ('theta',   "Theta (4-8Hz)",   '#f1c40f'),
                ('alpha',   "Alpha (8-13Hz)",  '#27ae60'),
                ('alpha_1', "Alpha₁ (8-9Hz)",  '#58d68d'),
                ('alpha_2', "Alpha₂ (9-11Hz)", '#1e8449'),
                ('alpha_3', "Alpha₃ (11-13Hz)",'#0e6655'),
                ('beta',    "Beta (13-30Hz)",  '#3498db'),
                ('beta_1',  "Beta₁ (13-20Hz)", '#85c1e9'),
                ('beta_2',  "Beta₂ (20-30Hz)", '#2471a3'),
                ('gamma',   "Gamma (30-70Hz)", '#9b59b6'),
                ('gamma_1', "Gamma₁ (30-50Hz)",'#c39bd3'),
                ('gamma_2', "Gamma₂ (50-70Hz)",'#7d3c98'),
            ]
            
            show_filter_compare = st.checkbox(
                "📊 显示伪迹过滤前后对比（G_: 全量, E_: 过滤NaN→0）",
                value=False,
                key="show_filter_compare",
                disabled=(results.get('prob_dict') is None)
            )

            for lead in valid_leads:
                sd = spec_dict_all[lead]
                if sd.get('delta') is None:
                    continue
                fig = go.Figure()
                for key, name, color in _band_keys:
                    if name not in selected_bands:
                        continue
                    power_data = sd.get(key)
                    if power_data is None:
                        continue
                    fig.add_trace(go.Scatter(
                        x=epoch_times[:len(power_data)],
                        y=power_data,
                        name=name,
                        mode='lines',
                        line=dict(color=color, width=1.2)
                    ))

                # 添加正常参考值阴影
                if enable_zscore and all_ref_data:
                    for key, name, color in _band_keys:
                        if name not in selected_bands:
                            continue
                        if lead not in _VIRTUAL_LEADS:
                            normal_mean = get_normal_ref(all_ref_data, selected_age_group, key, lead, 'mean', window_sizes)
                            normal_std = get_normal_ref(all_ref_data, selected_age_group, key, lead, 'std', window_sizes)
                        else:
                            group_leads_for_ref = _group_virtual.get(lead, [])
                            g_means = []
                            g_stds = []
                            for gl in group_leads_for_ref:
                                rm = get_normal_ref(all_ref_data, selected_age_group, key, gl, 'mean', window_sizes)
                                rs = get_normal_ref(all_ref_data, selected_age_group, key, gl, 'std', window_sizes)
                                if rm is not None:
                                    g_means.append(rm)
                                if rs is not None:
                                    g_stds.append(rs)
                            normal_mean = np.mean(g_means) if g_means else None
                            normal_std = np.mean(g_stds) if g_stds else None
                        if normal_mean is None or normal_std is None or normal_std == 0:
                            pass
                        else:
                            upper = normal_mean + zscore_threshold * normal_std
                            lower = normal_mean - zscore_threshold * normal_std
                            n_epochs = len(epoch_times)
                            # hex颜色转 rgba 半透明
                            r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
                            fig.add_trace(go.Scatter(
                                x=list(range(n_epochs)) + list(range(n_epochs))[::-1],
                                y=[upper] * n_epochs + [lower] * n_epochs,
                                fill='toself',
                                fillcolor=f'rgba({r},{g},{b},0.1)',
                                line=dict(color='rgba(0,0,0,0)'),
                                name=f'{name} 参考(常模)',
                                showlegend=False,
                                hoverinfo='skip'
                            ))
                        
                        # SP 参照值阴影
                        if ref_source in ["SP 参照值 (SOLAR2000)", "双参照对比 (常模 + SP)"] and sp_ref_data:
                            if lead not in _VIRTUAL_LEADS:
                                sp_mean = get_sp_ref(sp_ref_data, key, lead, 'mean')
                                sp_std = get_sp_ref(sp_ref_data, key, lead, 'std')
                            else:
                                group_leads_for_ref = _group_virtual.get(lead, [])
                                g_means = []
                                g_stds = []
                                for gl in group_leads_for_ref:
                                    rm = get_sp_ref(sp_ref_data, key, gl, 'mean')
                                    rs = get_sp_ref(sp_ref_data, key, gl, 'std')
                                    if rm is not None:
                                        g_means.append(rm)
                                    if rs is not None:
                                        g_stds.append(rs)
                                sp_mean = np.mean(g_means) if g_means else None
                                sp_std = np.mean(g_stds) if g_stds else None
                            if sp_mean is not None and sp_std is not None and sp_std > 0:
                                sp_upper = sp_mean + zscore_threshold * sp_std
                                sp_lower = sp_mean - zscore_threshold * sp_std
                                n_epochs = len(epoch_times)
                                r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
                                fig.add_trace(go.Scatter(
                                    x=list(range(n_epochs)) + list(range(n_epochs))[::-1],
                                    y=[sp_upper] * n_epochs + [sp_lower] * n_epochs,
                                    fill='toself',
                                    fillcolor=f'rgba({r},{g},{b},0.05)',
                                    line=dict(color=f'rgba({r},{g},{b},0.4)', width=1, dash='dash'),
                                    name=f'{name} 参考(SP)',
                                    showlegend=False,
                                    hoverinfo='skip'
                                ))
                fig.update_layout(
                    title=dict(text=f"<b>{lead}</b> 频段功率分布", x=0.5),
                    xaxis_title="Epoch",
                    yaxis_title="功率 (μV²/Hz)",
                    height=500,
                    legend=dict(x=1.02, y=1, bgcolor='rgba(255,255,255,0.9)', font=dict(size=10)),
                    margin=dict(r=150)
                )
                st.plotly_chart(fig, width="stretch")

                # ====== 过滤前后对比图（同一张图，两种线型）======
                if show_filter_compare and results.get('prob_dict') is not None:
                    _has_full = any(sd.get(f'{k}_full') is not None for k, _, _ in _band_keys)
                    if _has_full:
                        cmp_fig = go.Figure()
                        for key, name, color in _band_keys:
                            if name not in selected_bands:
                                continue
                            g_data = sd.get(f'{key}_full')
                            e_data = sd.get(f'{key}_filt')
                            if g_data is None:
                                continue
                            # 过滤前: 浅色虚线
                            cmp_fig.add_trace(go.Scatter(
                                x=epoch_times, y=g_data,
                                name=f'{name} 过滤前',
                                mode='lines',
                                line=dict(color=color, width=1, dash='dash'),
                                opacity=0.5
                            ))
                            # 过滤后: 深色实线（NaN→0）
                            if e_data is not None:
                                e_plot = np.nan_to_num(e_data, nan=0.0)
                                cmp_fig.add_trace(go.Scatter(
                                    x=epoch_times, y=e_plot,
                                    name=f'{name} 过滤后',
                                    mode='lines',
                                    line=dict(color=color, width=1.5)
                                ))
                        cmp_fig.update_layout(
                            title=dict(text=f"<b>{lead}</b> 伪迹过滤前后对比", x=0.5),
                            xaxis_title="Epoch",
                            yaxis_title="功率 (μV²/Hz)",
                            height=400,
                            legend=dict(x=1.02, y=1, font=dict(size=9)),
                            margin=dict(r=150)
                        )
                        st.plotly_chart(cmp_fig, width="stretch")

        with tab2:
            st.subheader("📈 功率比率指标")
            
            ratio_options = st.multiselect(
                "选择要显示的比率",
                ["TBR", "DAR", "DTR", "ABR", "ATR", "DTAR"],
                default=["TBR", "DAR"],
                key="ratio_select"
            )
            
            for lead in valid_leads:
                sd = spec_dict_all[lead]
                
                fig = go.Figure()
                colors = {
                    "TBR": "blue", "DAR": "orange", "DTR": "purple",
                    "ABR": "green", "ATR": "red", "DTAR": "brown"
                }
                
                for ratio in ratio_options:
                    # 键名映射
                    ratio_key_map = {
                        "TBR": "TBR", "DAR": "DAR", "DTR": "DTR",
                        "ABR": "ABR", "ATR": "ATR",
                        "DTAR": "DT_AR"
                    }
                    ratio_key = ratio_key_map.get(ratio, ratio)
                    
                    if ratio_key in sd:
                        data = np.array(sd[ratio_key])
                        epochs = list(range(len(data)))
                        
                        # 初始化参考范围
                        upper = None
                        lower = None
                        
                        # 添加参考范围（常模）
                        if enable_zscore and all_ref_data and ref_source in ["常模 (Normal_Reference)", "双参照对比 (常模 + SP)"]:
                            if lead not in _VIRTUAL_LEADS:
                                normal_mean = get_normal_ref(all_ref_data, selected_age_group, ratio, lead, 'mean', window_sizes)
                                normal_std = get_normal_ref(all_ref_data, selected_age_group, ratio, lead, 'std', window_sizes)
                            else:
                                group_leads_for_ref = _group_virtual.get(lead, [])
                                g_means = []
                                g_stds = []
                                for gl in group_leads_for_ref:
                                    rm = get_normal_ref(all_ref_data, selected_age_group, ratio, gl, 'mean', window_sizes)
                                    rs = get_normal_ref(all_ref_data, selected_age_group, ratio, gl, 'std', window_sizes)
                                    if rm is not None:
                                        g_means.append(rm)
                                    if rs is not None:
                                        g_stds.append(rs)
                                normal_mean = np.mean(g_means) if g_means else None
                                normal_std = np.mean(g_stds) if g_stds else None
                            if normal_mean is not None and normal_std is not None:
                                upper = normal_mean + zscore_threshold * normal_std
                                lower = normal_mean - zscore_threshold * normal_std
                                data_len = len(data)
                                fig.add_trace(go.Scatter(
                                    x=list(range(data_len)) + list(range(data_len))[::-1],
                                    y=[upper] * data_len + [lower] * data_len,
                                    fill='toself',
                                    fillcolor='rgba(255,0,0,0.1)',
                                    line=dict(color='rgba(255,0,0,0)'),
                                    name=f'{ratio}常模范围',
                                    showlegend=True
                                ))
                        
                        # 添加参考范围（SP参照值）
                        if enable_zscore and sp_ref_data and ref_source in ["SP 参照值 (SOLAR2000)", "双参照对比 (常模 + SP)"]:
                            if lead not in _VIRTUAL_LEADS:
                                sp_mean = get_sp_ref(sp_ref_data, ratio, lead, 'mean')
                                sp_std = get_sp_ref(sp_ref_data, ratio, lead, 'std')
                            else:
                                group_leads_for_ref = _group_virtual.get(lead, [])
                                g_means = []
                                g_stds = []
                                for gl in group_leads_for_ref:
                                    rm = get_sp_ref(sp_ref_data, ratio, gl, 'mean')
                                    rs = get_sp_ref(sp_ref_data, ratio, gl, 'std')
                                    if rm is not None:
                                        g_means.append(rm)
                                    if rs is not None:
                                        g_stds.append(rs)
                                sp_mean = np.mean(g_means) if g_means else None
                                sp_std = np.mean(g_stds) if g_stds else None
                            if sp_mean is not None and sp_std is not None and sp_std > 0:
                                sp_upper = sp_mean + zscore_threshold * sp_std
                                sp_lower = sp_mean - zscore_threshold * sp_std
                                data_len = len(data)
                                fig.add_trace(go.Scatter(
                                    x=list(range(data_len)) + list(range(data_len))[::-1],
                                    y=[sp_upper] * data_len + [sp_lower] * data_len,
                                    fill='toself',
                                    fillcolor='rgba(0,0,255,0.08)',
                                    line=dict(color='rgba(0,0,255,0.3)', width=1, dash='dash'),
                                    name=f'{ratio}SP范围',
                                    showlegend=True
                                ))
                        
                        # 显示数据线条（无论是否有参考数据都显示）
                        fig.add_trace(go.Scatter(
                            x=epochs, y=data,
                            name=ratio,
                            mode='lines+markers',
                            marker=dict(size=3),
                            line=dict(color=colors.get(ratio, "gray"), width=1.5)
                        ))
                
                fig.update_layout(
                    title=dict(text=f"<b>{lead}</b> 功率比率", x=0.5),
                    xaxis_title="Epoch",
                    yaxis_title="比率值",
                    height=350,
                    legend=dict(x=1.02, y=1, bgcolor='rgba(255,255,255,0.8)'),
                    margin=dict(r=100),
                    hovermode='x unified'
                )
                st.plotly_chart(fig, width="stretch")
        
        with tab3:
            st.subheader("📋 统计汇总")
            
            # ========== 显示正常参考值 ==========
            if enable_zscore and (all_ref_data or sp_ref_data):
                with st.expander(f"📚 查看正常参考值（均值±{zscore_threshold}σ 范围）", expanded=False):
                    ref_df_data = []
                    for lead in valid_leads[:5]:  # 只显示前5个导联避免太长
                        for band in ["TBR", "DAR", "DTR", "ABR", "ATR", "DTAR"]:
                            band_key_map = {"DTAR": "DT_AR"}
                            band_key = band_key_map.get(band, band)
                            if ref_source in ["常模 (Normal_Reference)", "双参照对比 (常模 + SP)"] and all_ref_data:
                                if lead not in _VIRTUAL_LEADS:
                                    mean_val = get_normal_ref(all_ref_data, selected_age_group, band, lead, 'mean', window_sizes)
                                    std_val = get_normal_ref(all_ref_data, selected_age_group, band, lead, 'std', window_sizes)
                                else:
                                    group_leads_for_ref = _group_virtual.get(lead, [])
                                    g_means = []
                                    g_stds = []
                                    for gl in group_leads_for_ref:
                                        rm = get_normal_ref(all_ref_data, selected_age_group, band, gl, 'mean', window_sizes)
                                        rs = get_normal_ref(all_ref_data, selected_age_group, band, gl, 'std', window_sizes)
                                        if rm is not None:
                                            g_means.append(rm)
                                        if rs is not None:
                                            g_stds.append(rs)
                                    mean_val = np.mean(g_means) if g_means else None
                                    std_val = np.mean(g_stds) if g_stds else None
                                if mean_val is not None and std_val is not None:
                                    ref_df_data.append({
                                        "导联": lead,
                                        "指标": band,
                                        "参考源": "常模",
                                        "均值": f"{mean_val:.4f}",
                                        "标准差": f"{std_val:.4f}",
                                        "正常范围": f"[{mean_val - zscore_threshold*std_val:.4f}, {mean_val + zscore_threshold*std_val:.4f}]"
                                    })
                            if ref_source in ["SP 参照值 (SOLAR2000)", "双参照对比 (常模 + SP)"] and sp_ref_data:
                                if lead not in _VIRTUAL_LEADS:
                                    sp_mean = get_sp_ref(sp_ref_data, band, lead, 'mean')
                                    sp_std = get_sp_ref(sp_ref_data, band, lead, 'std')
                                else:
                                    group_leads_for_ref = _group_virtual.get(lead, [])
                                    g_means = []
                                    g_stds = []
                                    for gl in group_leads_for_ref:
                                        rm = get_sp_ref(sp_ref_data, band, gl, 'mean')
                                        rs = get_sp_ref(sp_ref_data, band, gl, 'std')
                                        if rm is not None:
                                            g_means.append(rm)
                                        if rs is not None:
                                            g_stds.append(rs)
                                    sp_mean = np.mean(g_means) if g_means else None
                                    sp_std = np.mean(g_stds) if g_stds else None
                                if sp_mean is not None and sp_std is not None:
                                    ref_df_data.append({
                                        "导联": lead,
                                        "指标": band,
                                        "参考源": "SP",
                                        "均值": f"{sp_mean:.4f}",
                                        "标准差": f"{sp_std:.4f}",
                                        "正常范围": f"[{sp_mean - zscore_threshold*sp_std:.4f}, {sp_mean + zscore_threshold*sp_std:.4f}]"
                                    })
                    if ref_df_data:
                        ref_df = pd.DataFrame(ref_df_data)
                        st.dataframe(ref_df, use_container_width=True, hide_index=True)
            
            # ========== 1. 汇总表格（所有导联的Z-score一览）==========
            if enable_zscore and (all_ref_data or sp_ref_data):
                ref_source_label = "常模" if ref_source == "常模 (Normal_Reference)" else ("SP" if ref_source == "SP 参照值 (SOLAR2000)" else "常模+SP")
                st.markdown(f"**📊 正常参考对比** ({ref_source_label}, 阈值: Z-score > {zscore_threshold})")
                
                # 构建汇总数据
                summary_data = []
                for lead in valid_leads:
                    sd = spec_dict_all[lead]
                    row = {"导联": lead}
                    for band in ["TBR", "DAR", "DTR", "ABR", "ATR", "DTAR"]:
                        band_key_map = {"DTAR": "DT_AR"}
                        band_key = band_key_map.get(band, band)
                        band_data = sd.get(band_key, [])
                        val = np.mean(band_data) if len(band_data) > 0 else 0
                        
                        # 常模 Z-score
                        if ref_source in ["常模 (Normal_Reference)", "双参照对比 (常模 + SP)"] and all_ref_data:
                            if lead not in _VIRTUAL_LEADS:
                                ref_mean = get_normal_ref(all_ref_data, selected_age_group, band, lead, 'mean', window_sizes)
                                ref_std = get_normal_ref(all_ref_data, selected_age_group, band, lead, 'std', window_sizes)
                            else:
                                group_leads_for_ref = _group_virtual.get(lead, [])
                                group_means = []
                                group_stds = []
                                for gl in group_leads_for_ref:
                                    rm = get_normal_ref(all_ref_data, selected_age_group, band, gl, 'mean', window_sizes)
                                    rs = get_normal_ref(all_ref_data, selected_age_group, band, gl, 'std', window_sizes)
                                    if rm is not None and rs is not None and rs > 0:
                                        group_means.append(rm)
                                        group_stds.append(rs)
                                ref_mean = np.mean(group_means) if group_means else None
                                ref_std = np.mean(group_stds) if group_stds else None
                            if ref_mean is not None and ref_std is not None and ref_std > 0:
                                z = (val - ref_mean) / ref_std
                                row[f"{band}(常模)"] = f"{z:.2f}" if abs(z) <= zscore_threshold else f"🔴{z:.2f}"
                            else:
                                row[f"{band}(常模)"] = "N/A"
                        
                        # SP 参照值 Z-score
                        if ref_source in ["SP 参照值 (SOLAR2000)", "双参照对比 (常模 + SP)"] and sp_ref_data:
                            if lead not in _VIRTUAL_LEADS:
                                sp_mean = get_sp_ref(sp_ref_data, band, lead, 'mean')
                                sp_std = get_sp_ref(sp_ref_data, band, lead, 'std')
                            else:
                                group_leads_for_ref = _group_virtual.get(lead, [])
                                group_means = []
                                group_stds = []
                                for gl in group_leads_for_ref:
                                    rm = get_sp_ref(sp_ref_data, band, gl, 'mean')
                                    rs = get_sp_ref(sp_ref_data, band, gl, 'std')
                                    if rm is not None and rs is not None and rs > 0:
                                        group_means.append(rm)
                                        group_stds.append(rs)
                                sp_mean = np.mean(group_means) if group_means else None
                                sp_std = np.mean(group_stds) if group_stds else None
                            if sp_mean is not None and sp_std is not None and sp_std > 0:
                                z = (val - sp_mean) / sp_std
                                row[f"{band}(SP)"] = f"{z:.2f}" if abs(z) <= zscore_threshold else f"🔴{z:.2f}"
                            else:
                                row[f"{band}(SP)"] = "N/A"
                    summary_data.append(row)
                
                # 显示汇总表格
                st.markdown("#### 📈 Z-score 汇总表")
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True, hide_index=True)
            
            # ========== 伪迹概率热力图 ==========
            prob_dict = results.get('prob_dict')
            if prob_dict is not None and len(prob_dict) > 0:
                with st.expander("🎨 Artifact Probability Heatmap", expanded=False):
                    st.caption("Each 0.5s EEG segment is classified into 3 categories: **BKG (Background)** / **ART (Artifact)** / **ALPHA (Alpha rhythm)**. Darker color = higher probability.")
                    all_probs = np.stack(list(prob_dict.values()), axis=0)
                    lead_names = list(prob_dict.keys())
                    fig = plot_probs(all_probs, lead_names)
                    st.pyplot(fig)
                    st.caption(f"{len(lead_names)} leads | 1 row per lead | 0.5s time resolution")

            # ========== 导联完全过滤确认弹窗 ==========
            fallback_leads = results.get('fallback_leads', [])
            _fb_mode = st.session_state.get('art_fallback_mode', '保留原始数据')
            _fb_dismissed = st.session_state.get('_fb_dismissed', False)

            # ====== 弹窗：导联完全过滤，让用户选择 ======
            if fallback_leads and _fb_mode == '保留原始数据' and not _fb_dismissed:
                st.error(f"⚠️ 伪迹过滤导致 {len(fallback_leads)} 个导联的全部epoch被剔除", icon="🚨")
                st.markdown(
                    f"以下导联的所有epoch均因伪迹过滤被剔除：\n\n"
                    f"\n".join(fallback_leads)
                )
                st.caption("请选择处理方式：")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🔧 调整阈值重新分析", use_container_width=True):
                        st.session_state['_fb_dismissed'] = True
                        st.rerun()
                with col2:
                    if st.button("❌ 剔除这些导联", type="primary", use_container_width=True):
                        st.session_state['art_fallback_mode'] = '剔除该导联'
                        st.session_state.pop('_fb_dismissed', None)
                        st.rerun()
                st.stop()

            # ====== 已剔除：显示提示信息 ======
            elif fallback_leads and _fb_mode == '剔除该导联':
                st.info(f"ℹ️ 已从分析中剔除 {len(fallback_leads)} 个导联：{'、'.join(fallback_leads)}")

            # ====== 已关闭弹窗（点击了调整阈值）：结果上方显示引导提示 ======
            elif fallback_leads and _fb_mode == '保留原始数据' and _fb_dismissed:
                st.warning(
                    f"部分导联使用了未过滤的原始PSD（数据可能不准确）\n\n"
                    f"受影响导联：{'、'.join(fallback_leads)}\n\n"
                    f"请调整侧边栏「伪迹过滤阈值」滑块后重新分析"
                )

            # ========== 2. 可视化选项 ==========
            st.divider()
            st.divider()
            st.markdown("#### 📊 可视化选项")
            zscore_viz_options = st.multiselect(
                "选择要显示的Z-score可视化图表（可多选）",
                ["📊 柱状图", "📈 时序图", "🕸️ 雷达图"],
                default=["📊 柱状图"],
                key="zscore_viz"
            )
            
            # ========== 3. 每个导联的详细信息（折叠菜单）==========
            st.divider()
            st.markdown("#### 📌 各导联详情")
            
            for lead in valid_leads:
                with st.expander(f"**{lead}**", expanded=False):
                    sd = spec_dict_all[lead]
                    
                    # 指标卡片
                    metrics_data = [
                        ("TBR (θ/β)", np.mean(sd['TBR']) if 'TBR' in sd and len(sd['TBR']) > 0 else 0, "TBR"),
                        ("DAR (δ/α)", np.mean(sd['DAR']) if 'DAR' in sd and len(sd['DAR']) > 0 else 0, "DAR"),
                        ("DTR (δ/θ)", np.mean(sd['DTR']) if 'DTR' in sd and len(sd['DTR']) > 0 else 0, "DTR"),
                        ("ABR (α/β)", np.mean(sd['ABR']) if 'ABR' in sd and len(sd['ABR']) > 0 else 0, "ABR"),
                        ("ATR (α/θ)", np.mean(sd['ATR']) if 'ATR' in sd and len(sd['ATR']) > 0 else 0, "ATR"),
                        ("DTAR", np.mean(sd['DT_AR']) if 'DT_AR' in sd and len(sd['DT_AR']) > 0 else 0, "DTAR")
                    ]
                    
                    # 计算Z-score
                    metrics_with_z = []
                    for name, val, band in metrics_data:
                        z = None
                        if enable_zscore and all_ref_data:
                            if lead not in _VIRTUAL_LEADS:
                                ref_mean = get_normal_ref(all_ref_data, selected_age_group, band, lead, 'mean', window_sizes)
                                ref_std = get_normal_ref(all_ref_data, selected_age_group, band, lead, 'std', window_sizes)
                                if ref_mean is not None and ref_std is not None and ref_std > 0:
                                    z = (val - ref_mean) / ref_std
                            else:
                                group_leads_for_ref = _group_virtual.get(lead, [])
                                group_means = []
                                group_stds = []
                                for gl in group_leads_for_ref:
                                    rm = get_normal_ref(all_ref_data, selected_age_group, band, gl, 'mean', window_sizes)
                                    rs = get_normal_ref(all_ref_data, selected_age_group, band, gl, 'std', window_sizes)
                                    if rm is not None and rs is not None and rs > 0:
                                        group_means.append(rm)
                                        group_stds.append(rs)
                                if group_means:
                                    agg_mean = np.mean(group_means)
                                    agg_std = np.mean(group_stds)
                                    if agg_std > 0:
                                        z = (val - agg_mean) / agg_std
                        metrics_with_z.append((name, val, z))
                    
                    # 显示指标卡片
                    cols = st.columns(4)
                    for idx, (name, val, z) in enumerate(metrics_with_z[:4]):
                        with cols[idx]:
                            if z is None:
                                st.metric(name, f"{val:.2f}")
                            else:
                                icon = "🔴" if abs(z) > zscore_threshold else "🟢"
                                st.metric(name, f"{val:.2f}", f"Z={z:.2f} {icon}")
                    
                    cols2 = st.columns(3)
                    for idx, (name, val, z) in enumerate(metrics_with_z[4:]):
                        with cols2[idx]:
                            if z is None:
                                st.metric(name, f"{val:.2f}")
                            else:
                                icon = "🔴" if abs(z) > zscore_threshold else "🟢"
                                st.metric(name, f"{val:.2f}", f"Z={z:.2f} {icon}")
                    
                    # Z-score 可视化
                    if zscore_viz_options:
                        st.divider()
                        
                        # 柱状图
                        if "📊 柱状图" in zscore_viz_options:
                            valid_zscores = [(n.split(" ")[0], z) for n, v, z in metrics_with_z if isinstance(z, (int, float))]
                            if valid_zscores:
                                names = [n for n, z in valid_zscores]
                                zscores = [z for n, z in valid_zscores]
                                colors = ['red' if abs(z) > zscore_threshold else 'steelblue' for z in zscores]
                                
                                fig_bar = go.Figure()
                                fig_bar.add_trace(go.Bar(x=names, y=zscores, marker_color=colors,
                                    text=[f"{z:.2f}" for z in zscores], textposition='outside'))
                                fig_bar.add_hline(y=zscore_threshold, line_dash="dash", line_color="red")
                                fig_bar.add_hline(y=-zscore_threshold, line_dash="dash", line_color="red")
                                fig_bar.update_layout(title=dict(text=f"{lead} Z-score", x=0.5), height=250)
                                st.plotly_chart(fig_bar, width="stretch")
                        
                        # 时序图
                        if "📈 时序图" in zscore_viz_options:
                            epoch_count = len(sd.get('TBR', []))
                            if epoch_count > 0:
                                epochs = list(range(epoch_count))
                                for ratio_name, ratio_key in [("TBR", "TBR"), ("DAR", "DAR"), ("DTR", "DTR")]:
                                    if ratio_key in sd and len(sd[ratio_key]) > 0:
                                        fig_ts = go.Figure()
                                        fig_ts.add_trace(go.Scatter(x=epochs, y=sd[ratio_key], mode='lines', name=ratio_name, line=dict(width=1.5)))
                                        
                                        # 添加参考线
                                        if enable_zscore and all_ref_data:
                                            normal_mean = get_normal_ref(all_ref_data, selected_age_group, ratio_name, lead, 'mean', window_sizes)
                                            normal_std = get_normal_ref(all_ref_data, selected_age_group, ratio_name, lead, 'std', window_sizes)
                                            if normal_mean is not None:
                                                fig_ts.add_hline(y=normal_mean, line_dash="solid", line_color="green", annotation_text="正常均值")
                                                if normal_std is not None and normal_std > 0:
                                                    upper = normal_mean + zscore_threshold * normal_std
                                                    lower = normal_mean - zscore_threshold * normal_std
                                                    fig_ts.add_hline(y=upper, line_dash="dash", line_color="red", annotation_text=f"+{zscore_threshold}σ")
                                                    fig_ts.add_hline(y=lower, line_dash="dash", line_color="red", annotation_text=f"-{zscore_threshold}σ")
                                        
                                        fig_ts.update_layout(title=dict(text=f"{lead} {ratio_name}", x=0.5), height=200)
                                        st.plotly_chart(fig_ts, width="stretch")
                        
                        # 雷达图
                        if "🕸️ 雷达图" in zscore_viz_options:
                            valid_zscores = [(n.split(" ")[0], z) for n, v, z in metrics_with_z if isinstance(z, (int, float))]
                            if len(valid_zscores) >= 3:
                                labels = [n for n, z in valid_zscores]
                                values = [z for n, z in valid_zscores]
                                
                                fig_radar = go.Figure()
                                fig_radar.add_trace(go.Scatterpolar(
                                    r=values + [values[0]], theta=labels + [labels[0]],
                                    fill='toself', fillcolor='rgba(0,100,255,0.2)', line=dict(color='blue', width=2)))
                                fig_radar.update_layout(title=dict(text=f"{lead} 雷达图", x=0.5), height=300)
                                st.plotly_chart(fig_radar, width="stretch")
            
            # ========== 4. Alpha \u4e0d\u5bf9\u79f0\u6307\u6570\uff08\u5355\u72ec\u533a\u57df\uff09==========
            st.divider()
            st.subheader("🧠 Alpha 不对称指数")
            
            asymmetry_pairs = []
            if "耳电极" in lead_type:
                asymmetry_pairs = [("F3/F4 额叶", "F3-A1", "F4-A2", "E__F3_F4_E"), ("F7/F8 前颞叶", "F7-A1", "F8-A2", "E__F7_F8_E")]
            elif "平均" in lead_type:
                asymmetry_pairs = [("F3/F4 额叶", "F3-AVG", "F4-AVG", "A__F3_F4_E"), ("F7/F8 前颞叶", "F7-AVG", "F8-AVG", "A__F7_F8_E")]
            else:
                asymmetry_pairs = [("F3/F4 额叶", "F3-C3", "F4-C4", "B__F3_F4_E"), ("F7/F8 前颞叶", "F7-T3", "F8-T4", "B__F7_F8_E")]
            
            asymmetry_cols = st.columns(len(asymmetry_pairs))
            asymmetry_series_dict = {}
            
            for idx, (name, left_lead, right_lead, ref_key) in enumerate(asymmetry_pairs):
                with asymmetry_cols[idx]:
                    # 计算时序
                    asymmetry_series = calculate_alpha_asymmetry_series(spec_dict_all, left_lead, right_lead)
                    asymmetry_series_dict[name] = (asymmetry_series, left_lead, right_lead, ref_key)
                    
                    # 计算平均值用于卡片显示
                    asymmetry_val = np.mean(asymmetry_series) if asymmetry_series is not None and len(asymmetry_series) > 0 else None
                    
                    if asymmetry_val is not None and enable_zscore and all_ref_data:
                        normal_mean, normal_std = get_asymmetry_ref(all_ref_data, selected_age_group, ref_key)
                        if normal_mean is not None and normal_std is not None and normal_std > 0:
                            z = (asymmetry_val - normal_mean) / normal_std
                            icon = "🔴" if abs(z) > zscore_threshold else "🟢"
                            st.metric(name, f"{asymmetry_val:.3f}", f"Z={z:.2f} {icon}")
                        else:
                            st.metric(name, f"{asymmetry_val:.3f}")
                    elif asymmetry_val is not None:
                        st.metric(name, f"{asymmetry_val:.3f}")
                    else:
                        st.metric(name, "N/A")
            
            # 时序图
            if "📈 时序图" in zscore_viz_options:
                st.markdown("#### 不对称指数时序图")
                for name, (asymmetry_series, left_lead, right_lead, ref_key) in asymmetry_series_dict.items():
                    if asymmetry_series is not None and len(asymmetry_series) > 0:
                        epochs = list(range(len(asymmetry_series)))
                        fig_asym = go.Figure()
                        fig_asym.add_trace(go.Scatter(
                            x=epochs, y=asymmetry_series,
                            mode='lines', name=name.split(' ')[0],
                            line=dict(width=1.5)
                        ))
                        fig_asym.add_hline(y=0, line_dash="solid", line_color="gray")
                        
                        if enable_zscore and all_ref_data:
                            normal_mean, normal_std = get_asymmetry_ref(all_ref_data, selected_age_group, ref_key)
                            if normal_mean is not None and normal_std is not None:
                                fig_asym.add_hline(y=normal_mean, line_dash="solid", line_color="green", annotation_text="正常均值")
                                upper = normal_mean + zscore_threshold * normal_std
                                lower = normal_mean - zscore_threshold * normal_std
                                fig_asym.add_hline(y=upper, line_dash="dash", line_color="red", annotation_text=f"+{zscore_threshold}σ")
                                fig_asym.add_hline(y=lower, line_dash="dash", line_color="red", annotation_text=f"-{zscore_threshold}σ")
                        
                        fig_asym.update_layout(
                            title=dict(text=f"<b>{name}</b> 不对称指数时序图", x=0.5),
                            xaxis_title="Epoch", yaxis_title="Asymmetry Index",
                            height=250
                        )
                        st.plotly_chart(fig_asym, width="stretch")
            
             # ========== 导出 Excel（三种导联类型、含频段Z-score、诊断表）==========
            st.divider()
            st.subheader("📥 导出数据")
            
            def _compute_psd_for_leads(montage_dict, lead_filter_fn, ec, fs_val, np_len, art_th, prob, fallback_mode="保留原始数据（当前）"):
                """为指定导联计算PSD并返回spec_dict"""
                flt_leads = {k: v for k, v in montage_dict.items() if lead_filter_fn(k)}
                if not flt_leads:
                    return {}, []
                leads_list_local = []
                all_psds_local = []
                for lead_name, one_signal in flt_leads.items():
                    total_samples = ec * epoch_len_sec * fs_val
                    if len(one_signal) < total_samples:
                        one_signal = np.pad(one_signal, (0, total_samples - len(one_signal)), 'constant')
                    one_signal_reshape = one_signal[:total_samples].reshape(ec, epoch_len_sec * fs_val)
                    freqs_raw, psds_raw = signal.welch(one_signal_reshape, fs=fs_val, window='hann', nperseg=fs_val * np_len)
                    freqs_grid = np.arange(0, fs_val // 2 + 1)
                    psds_i = np.zeros((psds_raw.shape[0], len(freqs_grid)))
                    for i in range(psds_raw.shape[0]):
                        psds_i[i, :] = np.interp(freqs_grid, freqs_raw, psds_raw[i, :])
                    psds_i = np.log1p(psds_i)
                    if prob is not None and lead_name in prob:
                        _prob = prob[lead_name]
                        if ec * epoch_len_sec != np.shape(_prob)[0] * 2:
                            _prob = np.pad(_prob, ((0, 1), (0, 0)), mode='constant', constant_values=0)
                        art_prob_index = _prob[:, 2] + _prob[:, 1]
                        prob_len = ec * epoch_len_sec * 2
                        if prob_len <= len(art_prob_index):
                            mean_art_prob = np.max(art_prob_index[:prob_len].reshape(ec, epoch_len_sec * 2), axis=1)
                        else:
                            mean_art_prob = np.max(art_prob_index, axis=0) * np.ones(ec)
                        psds_i_clean = psds_i[mean_art_prob < (1 - art_th), :]
                    else:
                        psds_i_clean = psds_i
                    # 检测并处理"所有epoch被过滤"
                    if len(psds_i_clean) == 0 and len(psds_i) > 0:
                        if fallback_mode == "剔除该导联":
                            continue
                        else:
                            psds_i_clean = psds_i
                    if len(psds_i_clean) > 0:
                        leads_list_local.append(lead_name)
                        all_psds_local.append(psds_i_clean)
                    elif len(psds_i) > 0:
                        leads_list_local.append(lead_name)
                        all_psds_local.append(psds_i)
                if not all_psds_local:
                    return {}, leads_list_local
                spec_dict_local = get_spec_stat_info(all_psds_local)
                result = {}
                for idx, ln in enumerate(leads_list_local):
                    feat = {}
                    for k, v in spec_dict_local.items():
                        if isinstance(v, list) and idx < len(v):
                            feat[k] = v[idx]
                        elif isinstance(v, np.ndarray) and len(v.shape) > 1 and idx < v.shape[0]:
                            feat[k] = v[idx]
                    result[ln] = {'psd': all_psds_local[idx], 'psd_db': 10 * np.log10(all_psds_local[idx]), **feat}
                return result, leads_list_local
            
            def _zscore_for_lead(lead, band_key, val, _ard, _age, _virt_leads, _grp_virtual, _ws=None):
                """计算单个Z-score，返回 (z_value, ref_mean, ref_std)"""
                if lead not in _virt_leads:
                    rm = get_normal_ref(_ard, _age, band_key, lead, 'mean', _ws)
                    rs = get_normal_ref(_ard, _age, band_key, lead, 'std', _ws)
                else:
                    gl_list = _grp_virtual.get(lead, [])
                    g_means, g_stds = [], []
                    for gl in gl_list:
                        r1 = get_normal_ref(_ard, _age, band_key, gl, 'mean', _ws)
                        r2 = get_normal_ref(_ard, _age, band_key, gl, 'std', _ws)
                        if r1 is not None: g_means.append(r1)
                        if r2 is not None: g_stds.append(r2)
                    rm = np.mean(g_means) if g_means else None
                    rs = np.mean(g_stds) if g_stds else None
                if rm is not None and rs is not None and rs > 0 and val is not None:
                    return (val - rm) / rs, rm, rs
                return None, rm, rs
            
            def _zscore_for_lead_sp(lead, band_key, val, _srd, _virt_leads, _grp_virtual):
                """使用 SP 参照值计算 Z-score，返回 (z_value, ref_mean, ref_std)"""
                if _srd is None:
                    return None, None, None
                if lead not in _virt_leads:
                    rm = get_sp_ref(_srd, band_key, lead, 'mean')
                    rs = get_sp_ref(_srd, band_key, lead, 'std')
                else:
                    gl_list = _grp_virtual.get(lead, [])
                    g_means, g_stds = [], []
                    for gl in gl_list:
                        r1 = get_sp_ref(_srd, band_key, gl, 'mean')
                        r2 = get_sp_ref(_srd, band_key, gl, 'std')
                        if r1 is not None: g_means.append(r1)
                        if r2 is not None: g_stds.append(r2)
                    rm = np.mean(g_means) if g_means else None
                    rs = np.mean(g_stds) if g_stds else None
                if rm is not None and rs is not None and rs > 0 and val is not None:
                    return (val - rm) / rs, rm, rs
                return None, rm, rs
            
            def _build_export_spec(montage_dict, type_label, lead_opts, left_l, right_l, ec, fs_val, np_len, art_th, prob):
                """为一种导联类型构建 spec_dict 和虚拟组"""
                def _flt(name):
                    if "耳电极" in type_label:
                        return name.endswith('-A1') or name.endswith('-A2')
                    elif "平均" in type_label:
                        return name.endswith('-AVG')
                    else:
                        return '-A1' not in name and '-A2' not in name and '-AVG' not in name
                sd, _ = _compute_psd_for_leads(montage_dict, _flt, ec, fs_val, np_len, art_th, prob, st.session_state.get('art_fallback_mode', '保留原始数据'))
                grp_v = {
                    "🧠 全脑 (均值)": lead_opts,
                    "🧠 左脑 L (均值)": left_l,
                    "🧠 右脑 R (均值)": right_l,
                }
                for vn, gl in grp_v.items():
                    gli = [l for l in gl if l in sd]
                    if not gli:
                        continue
                    vd = {}
                    for k in sd[gli[0]].keys():
                        arrs = [np.array(sd[l][k]) for l in gli if sd[l].get(k) is not None]
                        if arrs:
                            min_len = min(len(a) for a in arrs)
                            vd[k] = np.mean(np.array([a[:min_len] for a in arrs]), axis=0)
                    sd[vn] = vd
                return sd, grp_v
            
            fm_dict = results.get('full_montage_dict')
            if fm_dict is None:
                st.warning("⚠️ 缺少完整导联数据，将仅导出当前类型")
                fm_dict = leads_montage_dict
            
            TYPE_CONFIGS = [
                ("双极导联 (Bipolar)", bipolar_leads,
                 [l for l in bipolar_leads if _is_left_lead(l)],
                 [l for l in bipolar_leads if _is_right_lead(l)]),
                ("耳电极参考 (Ear)", ear_leads,
                 [l for l in ear_leads if _is_left_lead(l)],
                 [l for l in ear_leads if _is_right_lead(l)]),
                ("平均参考 (Average)", avg_leads,
                 [l for l in avg_leads if _is_left_lead(l)],
                 [l for l in avg_leads if _is_right_lead(l)]),
            ]
            
            def generate_export_excel(_ez, _ard, _age, _zth, _duration_sec, _sfreq, _srd=None, _ref_src="常模 (Normal_Reference)"):
                """生成导出Excel（三种导联类型，含Z-score和诊断）"""
                # 导出时始终计算所有可用的参考值，不受界面选择限制
                # 如果常模数据未传入（比如从旧 session 恢复），重新加载
                if _ez and _ard is None:
                    _ard = load_normal_reference_data(None)
                _has_sp = _srd is not None
                _has_norm = _ard is not None
                buf = BytesIO()
                band_cols = [
                    ("delta", "δ (1-4Hz)"), ("theta", "θ (4-8Hz)"),
                    ("alpha", "α (8-13Hz)"), ("alpha_1", "α₁ (8-9Hz)"),
                    ("alpha_2", "α₂ (9-11Hz)"), ("alpha_3", "α₃ (11-13Hz)"),
                    ("beta", "β (13-30Hz)"), ("beta_1", "β₁ (13-20Hz)"),
                    ("beta_2", "β₂ (20-30Hz)"), ("gamma", "γ (30-70Hz)"),
                    ("gamma_1", "γ₁ (30-50Hz)"), ("gamma_2", "γ₂ (50-70Hz)"),
                ]
                ratio_cols = [
                    ("TBR", "TBR (θ/β)"), ("DAR", "DAR (δ/α)"),
                    ("DTR", "DTR (δ/θ)"), ("ABR", "ABR (α/β)"),
                    ("ATR", "ATR (α/θ)"), ("DT_AR", "DTAR ((δ+θ)/α)"),
                ]
                rel_cols = [
                    ("relative_delta", "相对δ"), ("relative_theta", "相对θ"),
                    ("relative_alpha", "相对α"), ("relative_beta", "相对β"),
                    ("relative_gamma", "相对γ"),
                ]
                _data_key_to_ref = {"DT_AR": "DTAR"}
                _VTL = {"🧠 全脑 (均值)", "🧠 左脑 L (均值)", "🧠 右脑 R (均值)"}
                all_rows = []
                ec = results.get('epoch_count', int(_duration_sec / epoch_len_sec))
                fs_val = results.get('fs', int(_sfreq))
                np_len = results.get('nperseg_len', nperseg_len)
                art_th = results.get('art_threshold', art_threshold)
                prob = results.get('prob_dict')
                
                for type_idx, (type_label, lead_opts, left_l, right_l) in enumerate(TYPE_CONFIGS):
                    if type_idx > 0:
                        all_rows.append({})
                    sd_local, grp_v_local = _build_export_spec(
                        fm_dict, type_label, lead_opts, left_l, right_l,
                        ec, fs_val, np_len, art_th, prob
                    )
                    exp_leads = lead_opts + ["🧠 全脑 (均值)", "🧠 左脑 L (均值)", "🧠 右脑 R (均值)"]
                    valid_l = [l for l in exp_leads if l in sd_local]
                    for lead in valid_l:
                        sd = sd_local[lead]
                        row = {"导联": lead, "导联类型": type_label}
                        for key, label in band_cols:
                            v = sd.get(key)
                            val = float(np.mean(v)) if v is not None and len(v) > 0 else None
                            row[label] = f"{val:.4f}" if val is not None else "N/A"
                            # 常模 Z
                            if _ez and _has_norm:
                                z, rm, rs = _zscore_for_lead(lead, key, val, _ard, _age, _VTL, grp_v_local, window_sizes)
                                row[f"{label} Z"] = f"{z:.2f}" if z is not None else "N/A"
                            else:
                                row[f"{label} Z"] = "N/A"
                            # SP 参照 Z
                            if _ez and _has_sp:
                                z_sp, _, _ = _zscore_for_lead_sp(lead, key, val, _srd, _VTL, grp_v_local)
                                row[f"{label} Z(SP)"] = f"{z_sp:.2f}" if z_sp is not None else "N/A"
                            else:
                                row[f"{label} Z(SP)"] = "N/A"
                        for key, label in ratio_cols:
                            ref_key = _data_key_to_ref.get(key, key)
                            v = sd.get(key)
                            val = float(np.mean(v)) if v is not None and len(v) > 0 else None
                            row[label] = f"{val:.4f}" if val is not None else "N/A"
                            # 常模 Z
                            if _ez and _has_norm:
                                z, rm, rs = _zscore_for_lead(lead, ref_key, val, _ard, _age, _VTL, grp_v_local, window_sizes)
                                row[f"{label} Z"] = f"{z:.2f}" if z is not None else "N/A"
                            else:
                                row[f"{label} Z"] = "N/A"
                            # SP 参照 Z
                            if _ez and _has_sp:
                                z_sp, _, _ = _zscore_for_lead_sp(lead, ref_key, val, _srd, _VTL, grp_v_local)
                                row[f"{label} Z(SP)"] = f"{z_sp:.2f}" if z_sp is not None else "N/A"
                            else:
                                row[f"{label} Z(SP)"] = "N/A"
                        for key, label in rel_cols:
                            v = sd.get(key)
                            val = float(np.mean(v)) if v is not None and len(v) > 0 else None
                            row[label] = f"{val:.4f}" if val is not None else "N/A"
                            # 常模 Z
                            if _ez and _has_norm:
                                z, rm, rs = _zscore_for_lead(lead, key, val, _ard, _age, _VTL, grp_v_local, window_sizes)
                                row[f"{label} Z"] = f"{z:.2f}" if z is not None else "N/A"
                            else:
                                row[f"{label} Z"] = "N/A"
                            # SP 参照 Z
                            if _ez and _has_sp:
                                z_sp, _, _ = _zscore_for_lead_sp(lead, key, val, _srd, _VTL, grp_v_local)
                                row[f"{label} Z(SP)"] = f"{z_sp:.2f}" if z_sp is not None else "N/A"
                            else:
                                row[f"{label} Z(SP)"] = "N/A"
                        total_v = sd.get("DT_total_R")
                        row["总功率"] = f"{np.mean(total_v):.4f}" if total_v is not None and len(total_v) > 0 else "N/A"
                        all_rows.append(row)
                
                with pd.ExcelWriter(buf, engine='openpyxl') as writer:
                    if all_rows:
                        pd.DataFrame(all_rows).to_excel(writer, sheet_name="频段功率与比率", index=False)
                    pd.DataFrame({
                        "参数": ["文件名", "Epoch长度(秒)", "Welch窗口(秒)", "伪迹阈值", "年龄组", "Z-score阈值", "总Epoch数"],
                        "值": [
                            edf_file.name, epoch_len_sec, nperseg_len, art_threshold,
                            _age if _ez else "未启用", _zth if _ez else "未启用", len(epoch_times),
                        ]
                    }).to_excel(writer, sheet_name="分析参数", index=False)
                buf.seek(0)
                return buf
            
            if enable_zscore:
                has_norm = all_ref_data is not None
                has_sp = sp_ref_data is not None
                need_norm = ref_source in ["常模 (Normal_Reference)", "双参照对比 (常模 + SP)"]
                need_sp = ref_source in ["SP 参照值 (SOLAR2000)", "双参照对比 (常模 + SP)"]
                if (need_norm and not has_norm) and (need_sp and not has_sp):
                    st.warning("⚠️ 常模和SP参照均未加载，无法计算Z-score")
            
            if st.button("📥 生成导出文件", type="primary", key="gen_excel_btn"):
                with st.spinner("正在计算全部三种导联类型（双极/耳电极/平均参考）..."):
                    excel_buf = generate_export_excel(enable_zscore, all_ref_data, selected_age_group, zscore_threshold, results.get('duration_sec', 0), results.get('sfreq_val', 256), sp_ref_data, ref_source)
                st.session_state.export_buf = excel_buf
                st.session_state.export_ready = True
            
            if st.session_state.get('export_ready') and st.session_state.get('export_buf'):
                st.download_button(
                    label="⬇️ 点击下载 Excel",
                    data=st.session_state.export_buf,
                    file_name=f"EEG_PSD_{edf_file.name.replace('.edf','').replace('.fif','')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="dl_excel_btn"
                )
                st.caption("已生成：包含双极导联、耳电极参考、平均参考共三种导联类型")

if __name__ == "__main__":
    main()