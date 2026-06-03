import streamlit as st
import plotly.graph_objects as go
import mne
import pickle
import numpy as np
import json
import os
import tempfile
import pandas as pd
from scipy import signal
from a_montage_tools import *
from a_psd_stat_tool import *

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


@st.cache_data(max_entries=1, ttl=3600)
def load_normal_reference_data(json_dir_hash: str):
    """加载正常参考数据（新版 Normal_Reference 格式）"""
    # 尝试多个可能的路径
    possible_paths = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Normal_Reference", "normal", "combined_result.json"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "Normal_Reference", "normal", "combined_result.json"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Normal_Reference", "normal", "combined_result.json"),
    ]
    
    json_path = None
    for path in possible_paths:
        if os.path.exists(path):
            json_path = path
            break
    
    if json_path is None:
        st.warning(f"⚠️ 参考数据文件不存在")
        return None
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            all_data = json.load(f)
        return all_data
    except Exception as e:
        st.warning(f"加载参考数据失败: {e}")
        return None


def get_normal_ref(all_ref_data, age_group, band, lead, stat_type='mean'):
    """获取正常参考值（适配新版 Normal_Reference 数据格式）"""
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
    
    # 窗口大小映射
    ws_mapping = {
        1: '2',   # 窗口1秒用2秒的数据代替
        2: '2',
        5: '5',
        10: '10',
        15: '15'
    }
    
    # 获取年龄组前缀
    age_prefix = age_group_mapping.get(age_group, 'age_19_44')
    
    # 查找匹配的年龄组 key（根据窗口大小）
    values = []
    
    # 新格式 key: age_19_44_10_normal_ref_0526
    # 遍历所有 key 找匹配的
    for key in all_ref_data.keys():
        # 检查是否匹配年龄组
        if not key.startswith(age_prefix):
            continue
        
        # 从 key 中提取窗口大小
        # 格式: age_19_44_10_normal_ref_0526
        parts = key.split('_')
        if len(parts) >= 3:
            try:
                ws_in_key = int(parts[2])
            except:
                continue
        else:
            continue
        
        # 键名格式转换: delta_Fp1-A1 -> Fp1-A1__delta
        # 需要保持原始大小写，因为JSON中的键名如DT_AR不是全小写的
        band_key = band
        # 特殊情况处理
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
        
        prob_file = st.file_uploader("选择Prob文件（可选）", type=["pkl"], key="prob_uploader")
        
        st.divider()
        
        st.subheader("📊 分析参数")
        epoch_len_sec = st.slider("Epoch长度(秒)", 1, 10, 5, key="epoch_len")
        nperseg_len = st.slider("Welch窗口长度(秒)", 1, 5, 2, key="nperseg")
        art_threshold = st.slider("伪迹过滤阈值", 0.0, 1.0, 0.5, 0.1, key="art_threshold")
        
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
            'P3-O1', 'P4-O2', 'Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1',
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
        
        selected_leads = st.multiselect("选择导联", lead_options, default=default_leads, key="selected_leads")
        
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
    
    # 构建当前参数
    current_params = {
        'code_version': CODE_VERSION,
        'edf_name': edf_file.name if edf_file else None,
        'prob_name': prob_file.name if prob_file else None,
        'epoch_len_sec': epoch_len_sec,
        'nperseg_len': nperseg_len,
        'art_threshold': art_threshold,
        'selected_leads': tuple(sorted(selected_leads)),
        'enable_zscore': enable_zscore,
        'selected_age_group': selected_age_group,
        'window_sizes': tuple(sorted(window_sizes)),
        'zscore_threshold': zscore_threshold
    }
    
    # 检查是否需要重新计算（edf_file存在且参数变化时自动计算）
    need_compute = False
    
    if edf_file:
        if st.session_state.analysis_params != current_params:
            need_compute = True
    
    # 执行分析
    if need_compute and edf_file:
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
                
                # 2. 加载Prob文件
                status.update(label="📋 加载Prob文件...")
                prob_dict = None
                if prob_file is not None:
                    prob_dict = load_prob_data(prob_file.getvalue(), prob_file.name)
                
                # 3. 加载正常参考数据
                status.update(label="📚 加载参考数据...")
                all_ref_data = None
                if enable_zscore:
                    all_ref_data = load_normal_reference_data(None)
                    if all_ref_data is None:
                        st.warning("⚠️ 正常参考数据加载失败！Z-score功能将不可用。")
                
                # 4. 准备数据
                status.update(label="🔧 准备数据...")
                all_data = {}
                for idx, ch_name in enumerate(edf_data['ch_names']):
                    all_data[ch_name] = edf_data['data'][idx]
                
                # 5. 根据导联类型计算数据
                if "耳电极" in lead_type:
                    status.update(label="👂 计算耳电极参考...")
                    ear_format_channels = [ch for ch in all_data.keys() if 'fp1' in ch.lower()]
                    if len(ear_format_channels) > 0:
                        leads_montage_dict = {}
                        for ch_name in ear_format_channels:
                            data = butter_bandpass_filter(all_data[ch_name], 0.5, 70, fs=256)
                            leads_montage_dict[ch_name] = norch_50(np.array(data))
                    else:
                        full_dict = get_bipolar_data_caueeg(all_data, 0.5, 70)
                        leads_montage_dict = {k: v for k, v in full_dict.items() if k.endswith('-A1') or k.endswith('-A2')}
                elif "平均" in lead_type:
                    status.update(label="📊 计算平均参考...")
                    full_dict = get_bipolar_data_caueeg(all_data, 0.5, 70)
                    leads_montage_dict = {k: v for k, v in full_dict.items() if k.endswith('-AVG')}
                else:
                    status.update(label="🔗 计算双极导联...")
                    leads_montage_dict = get_bipolar_data_caueeg(all_data, 0.5, 70)
                
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
                        psds_without_art = psds[mean_art_prob < art_threshold, :]
                    else:
                        psds_without_art = psds
                    
                    if len(psds_without_art) > 0:
                        leads_list.append(lead_name)
                        all_psds.append(psds_without_art)
                
                progress_bar.empty()
                
                if len(all_psds) == 0:
                    st.error("❌ 没有有效的PSD数据，请调整伪迹过滤阈值")
                    return
                
                # 7. 计算统计信息（与 PSD_calculate_full_EDF.py 保持一致）
                status.update(label="📈 计算统计指标...")
                spec_dict = get_spec_stat_info(all_psds)
                
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
                        
                        spec_dict_all[lead_name] = {
                            'psd': all_psds[lead_idx],  # 原始线性功率谱 (epoch×freq)
                            'psd_db': 10 * np.log10(all_psds[lead_idx]),  # dB版本
                            **lead_features  # 所有计算指标
                        }
                
                # 8. 存储结果
                st.session_state.psd_results = {
                    'spec_dict_all': spec_dict_all,
                    'epoch_times': epoch_times,
                    'leads_list': leads_list,
                    'all_ref_data': all_ref_data,
                    'freqs': freqs,  # 保存频率轴数据
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
        freqs = results.get('freqs')  # 获取频率轴数据
        
        valid_leads = [l for l in selected_leads if l in spec_dict_all]
        
        if not valid_leads:
            st.warning("⚠️ 选中的导联无有效数据，请重新选择")
            return
        
        tab1, tab2, tab3 = st.tabs(["📊 频段功率", "📈 功率比率", "📋 统计汇总"])
        
        with tab1:
            st.subheader("各频段功率分布")
            
            for lead in valid_leads:
                sd = spec_dict_all[lead]
                
                # 使用dB版PSD计算频段功率时序
                if 'psd_db' in sd:
                    psd_data = sd['psd_db']  # 形状: (epoch数, 频率点数)，单位 dB
                    
                    delta_power = np.mean(psd_data[:, 1:5], axis=1)   # 1-4Hz
                    theta_power = np.mean(psd_data[:, 4:9], axis=1)   # 4-8Hz
                    alpha_power = np.mean(psd_data[:, 8:14], axis=1)  # 8-13Hz
                    beta_power = np.mean(psd_data[:, 13:31], axis=1)  # 13-30Hz
                    gamma_power = np.mean(psd_data[:, 30:71], axis=1) # 30-70Hz
                    
                    # 创建图表
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=epoch_times[:len(delta_power)],
                        y=delta_power,
                        name="Delta (1-4Hz)",
                        mode='lines',
                        line=dict(color='purple', width=1.5)
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=epoch_times[:len(theta_power)],
                        y=theta_power,
                        name="Theta (4-8Hz)",
                        mode='lines',
                        line=dict(color='blue', width=1.5)
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=epoch_times[:len(alpha_power)],
                        y=alpha_power,
                        name="Alpha (8-13Hz)",
                        mode='lines',
                        line=dict(color='red', width=1.5)
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=epoch_times[:len(beta_power)],
                        y=beta_power,
                        name="Beta (13-30Hz)",
                        mode='lines',
                        line=dict(color='green', width=1.5)
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=epoch_times[:len(gamma_power)],
                        y=gamma_power,
                        name="Gamma (30-70Hz)",
                        mode='lines',
                        line=dict(color='orange', width=1.5)
                    ))
                    
                    fig.update_layout(
                        title=dict(text=f"<b>{lead}</b> 频段功率", x=0.5),
                        xaxis_title="Epoch",
                        yaxis_title="功率 (dB/Hz)",
                        height=400,
                        legend=dict(x=1.02, y=1, bgcolor='rgba(255,255,255,0.8)'),
                        margin=dict(r=100)
                    )
                    st.plotly_chart(fig, width="stretch")
        
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
                        
                        # 添加参考范围
                        if enable_zscore and all_ref_data:
                            normal_mean = get_normal_ref(all_ref_data, selected_age_group, ratio, lead, 'mean')
                            normal_std = get_normal_ref(all_ref_data, selected_age_group, ratio, lead, 'std')
                            
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
                                    name=f'{ratio}参考范围',
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
            if enable_zscore and all_ref_data:
                with st.expander("📚 查看正常参考值（均值±标准差）", expanded=False):
                    ref_df_data = []
                    for lead in valid_leads[:5]:  # 只显示前5个导联避免太长
                        for band in ["TBR", "DAR", "DTR", "ABR", "ATR", "DTAR"]:
                            band_key_map = {"DTAR": "DT_AR"}
                            band_key = band_key_map.get(band, band)
                            mean_val = get_normal_ref(all_ref_data, selected_age_group, band, lead, 'mean')
                            std_val = get_normal_ref(all_ref_data, selected_age_group, band, lead, 'std')
                            if mean_val is not None:
                                ref_df_data.append({
                                    "导联": lead,
                                    "指标": band,
                                    "正常均值": f"{mean_val:.4f}",
                                    "正常标准差": f"{std_val:.4f}",
                                    "正常范围": f"[{mean_val - 2*std_val:.4f}, {mean_val + 2*std_val:.4f}]"
                                })
                    if ref_df_data:
                        ref_df = pd.DataFrame(ref_df_data)
                        st.dataframe(ref_df, use_container_width=True, hide_index=True)
            
            # ========== 1. 汇总表格（所有导联的Z-score一览）==========
            if enable_zscore and all_ref_data:
                st.markdown(f"**📊 正常参考对比** (年龄组: `{selected_age_group}`, 阈值: Z-score > {zscore_threshold})")
                
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
                        mean = get_normal_ref(all_ref_data, selected_age_group, band, lead, 'mean')
                        std = get_normal_ref(all_ref_data, selected_age_group, band, lead, 'std')
                        if mean is not None and std is not None and std > 0:
                            z = (val - mean) / std
                            row[band] = f"{z:.2f}" if abs(z) <= zscore_threshold else f"🔴{z:.2f}"
                        else:
                            row[band] = "N/A"
                    summary_data.append(row)
                
                # 显示汇总表格
                st.markdown("#### 📈 Z-score 汇总表")
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True, hide_index=True)
            
            # ========== 2. 可视化选项 ==========
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
                            mean = get_normal_ref(all_ref_data, selected_age_group, band, lead, 'mean')
                            std = get_normal_ref(all_ref_data, selected_age_group, band, lead, 'std')
                            if mean is not None and std is not None and std > 0:
                                z = (val - mean) / std
                        metrics_with_z.append((name, val, z))
                    
                    # 显示指标卡片
                    cols = st.columns(4)
                    for idx, (name, val, z) in enumerate(metrics_with_z[:4]):
                        with cols[idx]:
                            if z is not None:
                                icon = "🔴" if abs(z) > zscore_threshold else "🟢"
                                st.metric(name, f"{val:.2f}", f"Z={z:.2f} {icon}")
                            else:
                                st.metric(name, f"{val:.2f}")
                    
                    cols2 = st.columns(3)
                    for idx, (name, val, z) in enumerate(metrics_with_z[4:]):
                        with cols2[idx]:
                            if z is not None:
                                icon = "🔴" if abs(z) > zscore_threshold else "🟢"
                                st.metric(name, f"{val:.2f}", f"Z={z:.2f} {icon}")
                            else:
                                st.metric(name, f"{val:.2f}")
                    
                    # Z-score 可视化
                    if zscore_viz_options:
                        st.divider()
                        
                        # 柱状图
                        if "📊 柱状图" in zscore_viz_options:
                            valid_zscores = [(n.split(" ")[0], z) for n, v, z in metrics_with_z if z is not None]
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
                                            normal_mean = get_normal_ref(all_ref_data, selected_age_group, ratio_name, lead, 'mean')
                                            normal_std = get_normal_ref(all_ref_data, selected_age_group, ratio_name, lead, 'std')
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
                            valid_zscores = [(n.split(" ")[0], z) for n, v, z in metrics_with_z if z is not None]
                            if len(valid_zscores) >= 3:
                                labels = [n for n, z in valid_zscores]
                                values = [z for n, z in valid_zscores]
                                
                                fig_radar = go.Figure()
                                fig_radar.add_trace(go.Scatterpolar(
                                    r=values + [values[0]], theta=labels + [labels[0]],
                                    fill='toself', fillcolor='rgba(0,100,255,0.2)', line=dict(color='blue', width=2)))
                                fig_radar.update_layout(title=dict(text=f"{lead} 雷达图", x=0.5), height=300)
                                st.plotly_chart(fig_radar, width="stretch")
            
            # ========== 4. Alpha 不对称指数（单独区域）==========
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
    
    elif edf_file is None:
        st.info("👈 请上传EEG文件")
    elif not need_compute:
        st.info("👈 设置参数")

if __name__ == "__main__":
    main()