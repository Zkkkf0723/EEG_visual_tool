import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import mne
import pickle
import numpy as np
import json
import os
import tempfile
from scipy import signal
from functools import lru_cache
from a_montage_tools import *
from a_psd_stat_tool import *


def load_normal_reference(json_dir, window_sizes):
    """加载所有窗口大小的正常参考数据，计算综合均值和标准差"""
    all_data = {}
    for ws in window_sizes:
        json_path = os.path.join(json_dir, f"{ws}_stat_info_dict_ALL_AVG_0514.json")
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                all_data[ws] = json.load(f)
    return all_data


def calculate_zscore(patient_value, normal_mean, normal_std):
    """计算Z-score"""
    if normal_std == 0 or np.isnan(normal_std) or normal_mean is None:
        return 0.0
    return (patient_value - normal_mean) / normal_std


def get_normal_ref_for_lead(all_ref_data, age_group, band, lead, stat_type='mean'):
    """从所有窗口大小的参考数据中获取均值或标准差"""
    values = []
    for ws, ref_data in all_ref_data.items():
        key = f"{band}_{lead}_{stat_type}"
        if age_group in ref_data and key in ref_data[age_group]:
            values.append(ref_data[age_group][key])
    if values:
        return np.mean(values)
    return None


# 使用st.cache_data缓存数据加载
@st.cache_data(ttl=3600, show_spinner=False)
def load_edf_data(edf_bytes, file_hash):
    """加载并预处理EDF数据"""
    # 写入临时文件
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, f"temp_edf_{file_hash}.edf")
    
    with open(temp_path, "wb") as f:
        f.write(edf_bytes)
    
    try:
        raw = mne.io.read_raw_edf(temp_path, preload=True)
        duration_sec = raw.times[-1]
        
        raw._data *= 1e6
        raw.resample(256)
        
        all_data = {}
        for ch_name in raw.ch_names:
            all_data[ch_name] = raw.get_data(picks=ch_name)[0]
        
        # 清理临时文件
        try:
            os.remove(temp_path)
        except:
            pass
        
        return all_data, duration_sec, raw.ch_names
    except Exception as e:
        try:
            os.remove(temp_path)
        except:
            pass
        raise e


# 使用st.cache_data缓存PSD计算
@st.cache_data(ttl=3600, show_spinner=False)
def compute_psd(all_data_hash, epoch_len_sec, nperseg_len, art_threshold, prob_bytes=None, prob_hash=None):
    """计算PSD数据"""
    # 这个函数需要根据实际的数据结构重新实现
    # 由于all_data无法直接hash，我们需要传入已经处理好的数据
    pass


st.set_page_config(page_title="PSD 可视化工具", layout="wide")

# 初始化session_state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'last_params' not in st.session_state:
    st.session_state.last_params = {}

st.title("🧠 EEG PSD 可视化工具")

with st.sidebar:
    st.header("⚙️ 参数设置")
    
    st.markdown("**📁 上传EDF文件**")
    edf_file = st.file_uploader("选择EDF文件", type=["edf"], key="edf_uploader")
    
    st.markdown("**📁 上传Prob文件（可选）**")
    prob_file = st.file_uploader("选择Prob文件（用于伪迹过滤）", type=["pkl"], key="prob_uploader")
    
    epoch_len_sec = st.slider("Epoch长度(秒)", 1, 10, 5)
    nperseg_len = st.slider("Welch窗口长度(秒)", 1, 5, 2)
    art_threshold = st.slider("伪迹过滤阈值", 0.0, 1.0, 0.5, 0.1)
    
    lead_options = ['Fp1-F3', 'Fp2-F4', 'F3-C3', 'F4-C4', 'C3-P3', 'C4-P4', 
                    'P3-O1', 'P4-O2', 'Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1',
                    'F7-T3', 'F8-T4', 'T4-T6', 'T6-O2', 'Fz-Pz', 'Cz-Pz', 'Pz-Oz']
    selected_leads = st.multiselect("选择导联", lead_options, default=['Fp1-F3', 'F3-C3'])
    
    st.divider()
    st.subheader("📊 正常参考对比")
    
    enable_zscore = st.checkbox("启用Z-score对比", value=True)
    age_group_options = ['0-6', '7-13', '14-18', '19-44', '45-59', '60-80', '80', 'total']
    selected_age_group = st.selectbox("选择参考年龄组", age_group_options, index=3)
    window_sizes = st.multiselect("选择窗口大小", [1, 2, 5, 10, 15], default=[1, 2, 5, 10, 15])
    zscore_threshold = st.slider("Z-score异常阈值", 1.0, 3.0, 2.0, 0.1)
    
    run_button = st.button("🚀 运行分析", type="primary", use_container_width=True)

# 检查参数是否改变
current_params = {
    'edf_file_name': edf_file.name if edf_file else None,
    'prob_file_name': prob_file.name if prob_file else None,
    'epoch_len_sec': epoch_len_sec,
    'nperseg_len': nperseg_len,
    'art_threshold': art_threshold,
    'selected_leads': selected_leads,
    'enable_zscore': enable_zscore,
    'selected_age_group': selected_age_group,
    'window_sizes': window_sizes,
    'zscore_threshold': zscore_threshold
}

# 判断是否需要重新分析
need_reanalysis = (run_button or 
                   not st.session_state.analysis_complete or
                   current_params != st.session_state.get('last_params', {}))

if need_reanalysis and edf_file is not None:
    # 更新最后使用的参数
    st.session_state.last_params = current_params
    st.session_state.analysis_complete = False
    
    with st.spinner("正在加载和分析数据..."):
        try:
            # 1. 加载EDF数据（带缓存）
            progress_text = st.empty()
            progress_text.info("📂 加载EDF文件...")
            
            # 计算文件hash用于缓存
            edf_bytes = edf_file.getvalue()
            file_hash = hash(edf_bytes)
            
            all_data, duration_sec, ch_names = load_edf_data(edf_bytes, file_hash)
            
            # 2. 计算双极导联
            progress_text.info("🔗 计算双极导联...")
            leads_montage_dict = get_bipolar_data_caueeg(all_data, 0.5, 70)
            
            # 3. 加载prob文件（如果有）
            prob_dict = None
            prob_bytes = None
            if prob_file is not None:
                progress_text.info("📋 加载Prob文件...")
                prob_bytes = prob_file.getvalue()
                prob_dict = pickle.loads(prob_bytes)
            
            # 4. 计算PSD
            progress_text.info("📊 计算PSD...")
            fs = 256
            epoch_count = int(duration_sec / epoch_len_sec)
            
            if epoch_count == 0:
                st.error("❌ Epoch长度超过数据总时长")
                st.stop()
            
            epoch_times = list(range(epoch_count))
            spec_dict_all = {}
            PSD_array_no_art = []
            leads_list = []
            
            # 使用进度条
            progress_bar = st.progress(0)
            total_leads = len(leads_montage_dict.keys())
            
            for idx, lead_name in enumerate(leads_montage_dict.keys()):
                progress_bar.progress((idx + 1) / total_leads)
                
                one_signal = leads_montage_dict[lead_name]
                total_samples = epoch_count * epoch_len_sec * fs
                
                if len(one_signal) < total_samples:
                    st.warning(f"导联 {lead_name} 信号长度不足，将截断处理")
                    one_signal = one_signal[:total_samples]
                
                one_signal_reshape = one_signal[0:total_samples].reshape(epoch_count, epoch_len_sec * fs)
                freqs, psds = signal.welch(one_signal_reshape, fs=fs, window='hann', nperseg=fs * nperseg_len)
                
                # 伪迹过滤
                if prob_dict is not None and lead_name in prob_dict:
                    _prob = prob_dict[lead_name]
                    if epoch_count * epoch_len_sec != np.shape(_prob)[0] * 2:
                        _prob = np.pad(_prob, ((0, 1), (0, 0)), mode='constant', constant_values=0)
                    art_prob_index = _prob[:, 2] + _prob[:, 1]
                    prob_len = epoch_count * epoch_len_sec * 2
                    mean_art_prob = np.max(art_prob_index[:prob_len].reshape(epoch_count, epoch_len_sec * 2), axis=1)
                    psds_without_art = psds[mean_art_prob < art_threshold, :]
                else:
                    psds_without_art = psds
                
                if len(psds_without_art) > 0:
                    leads_list.append(lead_name)
                    PSD_array_no_art.append(psds_without_art)
            
            progress_bar.empty()
            
            if len(PSD_array_no_art) == 0:
                st.error("❌ 没有有效的PSD数据，请调整伪迹过滤阈值")
                st.stop()
            
            # 5. 计算统计信息
            progress_text.info("📈 计算统计指标...")
            spec_dict = get_spec_stat_info(PSD_array_no_art)
            
            for lead_idx, lead_name in enumerate(leads_list):
                if lead_idx < len(PSD_array_no_art):
                    spec_dict_all[lead_name] = {k: v[lead_idx] for k, v in spec_dict.items()}
            
            # 6. 加载参考数据（如果需要）
            all_ref_data = None
            if enable_zscore:
                progress_text.info("📚 加载正常参考数据...")
                current_dir = os.path.dirname(os.path.abspath(__file__))
                json_dir = os.path.join(current_dir, "json")
                if os.path.exists(json_dir):
                    all_ref_data = load_normal_reference(json_dir, window_sizes)
                else:
                    st.warning(f"⚠️ 参考数据目录不存在: {json_dir}")
            
            # 存储到session_state
            st.session_state.spec_dict_all = spec_dict_all
            st.session_state.epoch_times = epoch_times
            st.session_state.all_ref_data = all_ref_data
            st.session_state.leads_list = leads_list
            st.session_state.selected_leads = [l for l in selected_leads if l in spec_dict_all]
            st.session_state.analysis_complete = True
            
            progress_text.success("✅ 分析完成！")
            progress_text.empty()
            
        except Exception as e:
            st.error(f"❌ 分析过程中出错: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            st.stop()

# 显示结果
if st.session_state.analysis_complete:
    spec_dict_all = st.session_state.spec_dict_all
    epoch_times = st.session_state.epoch_times
    all_ref_data = st.session_state.all_ref_data
    leads_list = st.session_state.leads_list
    selected_leads = [l for l in st.session_state.selected_leads if l in spec_dict_all]
    
    # 如果没有选中的导联，显示警告
    if not selected_leads:
        st.warning("⚠️ 没有选中的有效导联，请在侧边栏选择导联")
    else:
        tab1, tab2, tab3 = st.tabs(["📊 频段功率", "📈 功率比率", "📋 统计汇总"])
        
        with tab1:
            st.subheader("各频段功率分布")
            
            for lead in selected_leads:
                if lead in spec_dict_all:
                    sd = spec_dict_all[lead]
                    
                    col_chart, col_info = st.columns([4, 1])
                    
                    with col_chart:
                        fig_single = go.Figure()
                        fig_single.add_trace(go.Scatter(
                            x=epoch_times[:len(sd["theta"])],
                            y=sd["theta"],
                            name="Theta (4-8Hz)",
                            mode='lines',
                            line=dict(color='blue', width=1.5)
                        ))
                        fig_single.add_trace(go.Scatter(
                            x=epoch_times[:len(sd["alpha"])],
                            y=sd["alpha"],
                            name="Alpha (8-13Hz)",
                            mode='lines',
                            line=dict(color='red', width=1.5)
                        ))
                        fig_single.add_trace(go.Scatter(
                            x=epoch_times[:len(sd["beta"])],
                            y=sd["beta"],
                            name="Beta (13-30Hz)",
                            mode='lines',
                            line=dict(color='green', width=1.5)
                        ))
                        
                        fig_single.update_layout(
                            title=dict(text=f"<b>{lead}</b> 频段功率", x=0.5),
                            xaxis_title="Epoch",
                            yaxis_title="功率 (dB)",
                            height=280,
                            legend=dict(x=1.02, y=1, bgcolor='rgba(255,255,255,0.5)'),
                            margin=dict(r=80)
                        )
                        st.plotly_chart(fig_single, use_container_width=True)
                    
                    with col_info:
                        st.markdown(f"**📊 {lead} 统计**")
                        if 'TBR' in sd:
                            st.write(f"TBR: {np.mean(sd['TBR']):.2f}")
                        if 'DAR' in sd:
                            st.write(f"DAR: {np.mean(sd['DAR']):.2f}")
                        if 'DTR' in sd:
                            st.write(f"DTR: {np.mean(sd['DTR']):.2f}")
                        if 'ABR' in sd:
                            st.write(f"ABR: {np.mean(sd['ABR']):.2f}")
                        if 'ATR' in sd:
                            st.write(f"ATR: {np.mean(sd['ATR']):.2f}")
                        if 'DT_AR' in sd:
                            st.write(f"DTAR: {np.mean(sd['DT_AR']):.2f}")
        
        with tab2:
            st.subheader("📈 功率比率指标")
            
            ratio_options = st.multiselect(
                "选择要显示的比率", 
                ["TBR", "DAR", "DTR", "ABR", "ATR", "DTAR", "DTPWR"],
                default=["TBR", "DAR"],
                key="ratio_select"
            )
            
            for lead in selected_leads:
                if lead in spec_dict_all:
                    sd = spec_dict_all[lead]
                    
                    col_chart, col_info = st.columns([4, 1])
                    
                    with col_chart:
                        fig_ratio = go.Figure()
                        colors = {
                            "TBR": "blue", "DAR": "orange", "DTR": "purple",
                            "ABR": "green", "ATR": "red", "DTAR": "brown", "DTPWR": "pink"
                        }
                        for ratio in ratio_options:
                            if ratio in sd:
                                if enable_zscore and all_ref_data and ratio not in ["DTAR", "DTPWR"]:
                                    normal_mean = get_normal_ref_for_lead(all_ref_data, selected_age_group, ratio, lead, 'mean')
                                    normal_std = get_normal_ref_for_lead(all_ref_data, selected_age_group, ratio, lead, 'std')
                                    if normal_mean is not None and normal_std is not None:
                                        upper_bound = normal_mean + zscore_threshold * normal_std
                                        lower_bound = normal_mean - zscore_threshold * normal_std
                                        data_len = len(sd[ratio])
                                        x_range = list(range(data_len))
                                        fig_ratio.add_trace(go.Scatter(
                                            x=x_range + x_range[::-1],
                                            y=[upper_bound] * data_len + [lower_bound] * data_len,
                                            fill='toself',
                                            fillcolor='rgba(255, 0, 0, 0.15)',
                                            line=dict(color='rgba(255, 0, 0, 0)'),
                                            name=f'{ratio}正常范围',
                                            showlegend=True
                                        ))
                                
                                fig_ratio.add_trace(go.Scatter(
                                    x=epoch_times[:len(sd[ratio])],
                                    y=sd[ratio],
                                    name=ratio,
                                    mode='lines+markers',
                                    marker=dict(size=3),
                                    line=dict(color=colors.get(ratio, "gray"), width=1.5)
                                ))
                        
                        fig_ratio.update_layout(
                            title=dict(text=f"<b>{lead}</b> 功率比率", x=0.5),
                            xaxis_title="Epoch",
                            yaxis_title="比率值",
                            height=300,
                            legend=dict(x=1.02, y=1, bgcolor='rgba(255,255,255,0.5)'),
                            margin=dict(r=80),
                            hovermode='x unified'
                        )
                        st.plotly_chart(fig_ratio, use_container_width=True)
                    
                    with col_info:
                        st.markdown(f"**📊 {lead} 均值**")
                        for ratio in ratio_options:
                            if ratio in sd:
                                st.write(f"{ratio}: {np.mean(sd[ratio]):.2f}")
        
        with tab3:
            st.subheader("📋 统计汇总")
            
            if enable_zscore and all_ref_data:
                st.markdown(f"**📊 正常参考对比** (年龄组: `{selected_age_group}`, 阈值: Z-score > {zscore_threshold})")
            
            for lead in selected_leads:
                if lead in spec_dict_all:
                    sd = spec_dict_all[lead]
                    
                    st.markdown(f"### {lead}")
                    
                    metrics_data = [
                        ("TBR (θ/β)", np.mean(sd['TBR']) if 'TBR' in sd else 0, "TBR", lead, "TBR"),
                        ("DAR (δ/α)", np.mean(sd['DAR']) if 'DAR' in sd else 0, "DAR", lead, "DAR"),
                        ("DTR (δ/θ)", np.mean(sd['DTR']) if 'DTR' in sd else 0, "DTR", lead, "DTR"),
                        ("ABR (α/β)", np.mean(sd['ABR']) if 'ABR' in sd else 0, "ABR", lead, "ABR"),
                        ("ATR (α/θ)", np.mean(sd['ATR']) if 'ATR' in sd else 0, "ATR", lead, "ATR"),
                        ("DTAR (δ+θ)/α", np.mean(sd['DT_AR']) if 'DT_AR' in sd else 0, "DTAR", lead, "DTAR"),
                        ("(δ+θ)/总功率", np.mean(sd['DT_total_R']) if 'DT_total_R' in sd else 0, "DTPWR", lead, "DTPWR")
                    ]
                    
                    cols = st.columns(4)
                    for idx, (name, val, band, l, ratio) in enumerate(metrics_data):
                        with cols[idx % 4]:
                            if enable_zscore and all_ref_data and ratio not in ["DTAR", "DTPWR"]:
                                normal_mean = get_normal_ref_for_lead(all_ref_data, selected_age_group, band, l, 'mean')
                                normal_std = get_normal_ref_for_lead(all_ref_data, selected_age_group, band, l, 'std')
                                if normal_mean is not None and normal_std is not None:
                                    zscore = calculate_zscore(val, normal_mean, normal_std)
                                    is_abnormal = abs(zscore) > zscore_threshold
                                    color = "🔴" if is_abnormal else "🟢"
                                    st.metric(name, f"{val:.2f}", f"Z={zscore:.2f} {color}")
                                else:
                                    st.metric(name, f"{val:.2f}")
                            else:
                                st.metric(name, f"{val:.2f}")

elif edf_file is None:
    st.info("👈 在左侧上传EDF文件后点击 **运行分析** 按钮开始")
else:
    st.info("👈 在左侧设置参数后点击 **运行分析** 按钮开始")

# 更新弃用警告
# 将所有 use_container_width=True 改为 width="stretch"
# 将所有 use_container_width=False 改为 width="content"

st.markdown("""
### 使用说明
    
    1. **上传EDF文件** - 选择你的脑电数据文件
    2. **调整参数** - 根据需求设置分析参数
    3. **选择导联** - 选择要查看的脑电导联
    4. **点击运行** - 开始分析并查看结果
    
    💡 **提示**: 第一次分析后，调整参数（如阈值、导联选择）不会重新加载EDF文件，响应更快！
    """)