import streamlit as st
import plotly.graph_objects as go
import mne
import pickle
import numpy as np
import json
import os
import tempfile
from scipy import signal
from a_montage_tools import *
from a_psd_stat_tool import *

# 设置页面
st.set_page_config(page_title="PSD 可视化工具", layout="wide")

# 初始化session_state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'last_params_hash' not in st.session_state:
    st.session_state.last_params_hash = None

st.title("🧠 EEG PSD 可视化工具")

def get_params_hash(params_dict):
    """生成参数的哈希值，用于检测是否变化"""
    import hashlib
    params_str = str(sorted(params_dict.items()))
    return hashlib.md5(params_str.encode()).hexdigest()

# 缓存EDF文件路径而不是数据
@st.cache_resource
def save_uploaded_file(uploaded_file):
    """保存上传的文件并返回路径"""
    if uploaded_file is None:
        return None
    
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, f"temp_{uploaded_file.name}_{hash(uploaded_file.name)}")
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    
    return file_path

# 缓存MNE Raw对象
@st.cache_resource
def load_raw_edf(edf_path):
    """加载EDF文件为MNE Raw对象"""
    if edf_path is None or not os.path.exists(edf_path):
        return None
    
    raw = mne.io.read_raw_edf(edf_path, preload=True)
    raw._data *= 1e6
    raw.resample(256)
    return raw

with st.sidebar:
    st.header("⚙️ 参数设置")
    
    edf_file = st.file_uploader("选择EDF文件", type=["edf"], key="edf_uploader")
    prob_file = st.file_uploader("选择Prob文件（可选）", type=["pkl"], key="prob_uploader")
    
    epoch_len_sec = st.slider("Epoch长度(秒)", 1, 10, 5, key="epoch_len")
    nperseg_len = st.slider("Welch窗口长度(秒)", 1, 5, 2, key="nperseg")
    art_threshold = st.slider("伪迹过滤阈值", 0.0, 1.0, 0.5, 0.1, key="art_threshold")
    
    lead_options = ['Fp1-F3', 'Fp2-F4', 'F3-C3', 'F4-C4', 'C3-P3', 'C4-P4', 
                    'P3-O1', 'P4-O2', 'Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1',
                    'F8-T4', 'T4-T6', 'T6-O2', 'Fz-Pz', 'Cz-Pz', 'Pz-Oz']
    selected_leads = st.multiselect("选择导联", lead_options, default=['Fp1-F3', 'F3-C3'], key="selected_leads")
    
    st.divider()
    st.subheader("📊 正常参考对比")
    
    enable_zscore = st.checkbox("启用Z-score对比", value=True, key="enable_zscore")
    age_group_options = ['0-6', '7-13', '14-18', '19-44', '45-59', '60-80', '80', 'total']
    selected_age_group = st.selectbox("选择参考年龄组", age_group_options, index=3, key="age_group")
    window_sizes = st.multiselect("选择窗口大小", [1, 2, 5, 10, 15], default=[1, 2, 5, 10, 15], key="window_sizes")
    zscore_threshold = st.slider("Z-score异常阈值", 1.0, 3.0, 2.0, 0.1, key="zscore_threshold")
    
    run_button = st.button("🚀 运行分析", type="primary", width="stretch")

# 检查是否需要重新分析
current_params = {
    'edf_name': edf_file.name if edf_file else None,
    'prob_name': prob_file.name if prob_file else None,
    'epoch_len_sec': epoch_len_sec,
    'nperseg_len': nperseg_len,
    'art_threshold': art_threshold,
    'selected_leads': selected_leads,
    'enable_zscore': enable_zscore,
    'selected_age_group': selected_age_group,
    'window_sizes': window_sizes,
    'zscore_threshold': zscore_threshold
}

current_hash = get_params_hash(current_params)

# 如果参数变化或点击运行按钮，重新分析
if run_button or (edf_file is not None and current_hash != st.session_state.last_params_hash):
    st.session_state.last_params_hash = current_hash
    st.session_state.analysis_complete = False
    
    if edf_file is None:
        st.error("❌ 请先上传EDF文件")
        st.stop()
    
    with st.spinner("正在分析数据..."):
        try:
            # 1. 保存文件
            edf_path = save_uploaded_file(edf_file)
            prob_path = save_uploaded_file(prob_file) if prob_file else None
            
            # 2. 加载Raw对象
            raw = load_raw_edf(edf_path)
            if raw is None:
                st.error("❌ 无法加载EDF文件")
                st.stop()
            
            duration_sec = raw.times[-1]
            
            # 3. 获取数据字典（转换为列表以支持缓存）
            all_data = {}
            for ch_name in raw.ch_names:
                all_data[ch_name] = raw.get_data(picks=ch_name)[0].tolist()  # 转换为列表
            
            # 4. 计算双极导联
            leads_montage_dict = get_bipolar_data_caueeg(all_data, 0.5, 70)
            
            # 5. 加载prob文件
            prob_dict = None
            if prob_path and os.path.exists(prob_path):
                with open(prob_path, 'rb') as f:
                    prob_dict = pickle.load(f)
            
            # 6. 计算PSD
            fs = 256
            epoch_count = int(duration_sec / epoch_len_sec)
            
            if epoch_count == 0:
                st.error("❌ Epoch长度超过数据总时长")
                st.stop()
            
            epoch_times = list(range(epoch_count))
            spec_dict_all = {}
            PSD_array_no_art = []
            leads_list = []
            
            progress_bar = st.progress(0)
            total_leads = len(leads_montage_dict.keys())
            
            for idx, lead_name in enumerate(leads_montage_dict.keys()):
                progress_bar.progress((idx + 1) / total_leads)
                
                one_signal = np.array(leads_montage_dict[lead_name])  # 转回numpy数组
                total_samples = epoch_count * epoch_len_sec * fs
                
                if len(one_signal) < total_samples:
                    one_signal = np.pad(one_signal, (0, total_samples - len(one_signal)), 'constant')
                
                one_signal_reshape = one_signal[:total_samples].reshape(epoch_count, epoch_len_sec * fs)
                freqs, psds = signal.welch(one_signal_reshape, fs=fs, window='hann', nperseg=fs * nperseg_len)
                
                # 伪迹过滤
                if prob_dict is not None and lead_name in prob_dict:
                    _prob = prob_dict[lead_name]
                    expected_len = epoch_count * epoch_len_sec
                    if _prob.shape[0] != expected_len:
                        # 调整prob长度
                        if _prob.shape[0] > expected_len:
                            _prob = _prob[:expected_len]
                        else:
                            _prob = np.pad(_prob, ((0, expected_len - _prob.shape[0]), (0, 0)), 'constant')
                    
                    art_prob = _prob[:, 2] + _prob[:, 1]
                    psds_without_art = psds[art_prob < art_threshold, :]
                else:
                    psds_without_art = psds
                
                if len(psds_without_art) > 0:
                    leads_list.append(lead_name)
                    PSD_array_no_art.append(psds_without_art)
            
            progress_bar.empty()
            
            if len(PSD_array_no_art) == 0:
                st.error("❌ 没有有效的PSD数据，请调整伪迹过滤阈值")
                st.stop()
            
            # 7. 计算统计信息
            spec_dict = get_spec_stat_info(PSD_array_no_art)
            
            for lead_idx, lead_name in enumerate(leads_list):
                if lead_idx < len(PSD_array_no_art):
                    spec_dict_all[lead_name] = {k: v[lead_idx] for k, v in spec_dict.items()}
            
            # 8. 加载参考数据
            all_ref_data = None
            if enable_zscore:
                current_dir = os.path.dirname(os.path.abspath(__file__))
                json_dir = os.path.join(current_dir, "json")
                if os.path.exists(json_dir):
                    all_ref_data = {}
                    for ws in window_sizes:
                        json_path = os.path.join(json_dir, f"{ws}_stat_info_dict_ALL_AVG_0514.json")
                        if os.path.exists(json_path):
                            with open(json_path, 'r', encoding='utf-8') as f:
                                all_ref_data[ws] = json.load(f)
                else:
                    st.warning(f"⚠️ 参考数据目录不存在: {json_dir}")
            
            # 存储到session_state
            st.session_state.spec_dict_all = spec_dict_all
            st.session_state.epoch_times = epoch_times
            st.session_state.all_ref_data = all_ref_data
            st.session_state.leads_list = leads_list
            st.session_state.analysis_complete = True
            
            st.success("✅ 分析完成！")
            st.rerun()
            
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
    selected_leads = [l for l in st.session_state.get('selected_leads', []) if l in spec_dict_all]
    
    if not selected_leads:
        st.warning("⚠️ 没有选中的有效导联，请在侧边栏选择导联")
    else:
        tab1, tab2, tab3 = st.tabs(["📊 频段功率", "📈 功率比率", "📋 统计汇总"])
        
        def load_normal_ref(band, lead, stat_type='mean'):
            """加载正常参考值"""
            if all_ref_data is None:
                return None
            values = []
            for ws, ref_data in all_ref_data.items():
                key = f"{band}_{lead}_{stat_type}"
                if selected_age_group in ref_data and key in ref_data[selected_age_group]:
                    values.append(ref_data[selected_age_group][key])
            return np.mean(values) if values else None
        
        with tab1:
            st.subheader("各频段功率分布")
            
            for lead in selected_leads:
                if lead in spec_dict_all:
                    sd = spec_dict_all[lead]
                    
                    col_chart, col_info = st.columns([4, 1])
                    
                    with col_chart:
                        fig_single = go.Figure()
                        
                        for band, color, name in [("theta", "blue", "Theta (4-8Hz)"),
                                                   ("alpha", "red", "Alpha (8-13Hz)"),
                                                   ("beta", "green", "Beta (13-30Hz)")]:
                            if band in sd:
                                fig_single.add_trace(go.Scatter(
                                    x=epoch_times[:len(sd[band])],
                                    y=sd[band],
                                    name=name,
                                    mode='lines',
                                    line=dict(color=color, width=1.5)
                                ))
                        
                        fig_single.update_layout(
                            title=dict(text=f"<b>{lead}</b> 频段功率", x=0.5),
                            xaxis_title="Epoch",
                            yaxis_title="功率 (dB)",
                            height=280,
                            legend=dict(x=1.02, y=1, bgcolor='rgba(255,255,255,0.5)'),
                            margin=dict(r=80)
                        )
                        st.plotly_chart(fig_single, width="stretch")
                    
                    with col_info:
                        st.markdown(f"**📊 {lead} 统计**")
                        metrics = ['TBR', 'DAR', 'DTR', 'ABR', 'ATR', 'DT_AR']
                        for metric in metrics:
                            if metric in sd:
                                st.write(f"{metric}: {np.mean(sd[metric]):.2f}")
        
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
                    
                    fig_ratio = go.Figure()
                    colors = {"TBR": "blue", "DAR": "orange", "DTR": "purple",
                             "ABR": "green", "ATR": "red", "DTAR": "brown", "DTPWR": "pink"}
                    
                    for ratio in ratio_options:
                        ratio_key = "DT_AR" if ratio == "DTAR" else ratio
                        if ratio_key in sd:
                            # 添加参考范围
                            if enable_zscore and all_ref_data and ratio not in ["DTAR", "DTPWR"]:
                                normal_mean = load_normal_ref(ratio, lead, 'mean')
                                normal_std = load_normal_ref(ratio, lead, 'std')
                                if normal_mean is not None and normal_std is not None:
                                    upper_bound = normal_mean + zscore_threshold * normal_std
                                    lower_bound = normal_mean - zscore_threshold * normal_std
                                    data_len = len(sd[ratio_key])
                                    fig_ratio.add_trace(go.Scatter(
                                        x=list(range(data_len)) + list(range(data_len))[::-1],
                                        y=[upper_bound] * data_len + [lower_bound] * data_len,
                                        fill='toself',
                                        fillcolor='rgba(255, 0, 0, 0.15)',
                                        line=dict(color='rgba(255, 0, 0, 0)'),
                                        name=f'{ratio}正常范围',
                                        showlegend=True
                                    ))
                            
                            fig_ratio.add_trace(go.Scatter(
                                x=epoch_times[:len(sd[ratio_key])],
                                y=sd[ratio_key],
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
                    st.plotly_chart(fig_ratio, width="stretch")
        
        with tab3:
            st.subheader("📋 统计汇总")
            
            if enable_zscore and all_ref_data:
                st.markdown(f"**📊 正常参考对比** (年龄组: `{selected_age_group}`, 阈值: Z-score > {zscore_threshold})")
            
            for lead in selected_leads:
                if lead in spec_dict_all:
                    sd = spec_dict_all[lead]
                    
                    st.markdown(f"### {lead}")
                    
                    metrics_data = [
                        ("TBR", np.mean(sd.get('TBR', [0]))),
                        ("DAR", np.mean(sd.get('DAR', [0]))),
                        ("DTR", np.mean(sd.get('DTR', [0]))),
                        ("ABR", np.mean(sd.get('ABR', [0]))),
                        ("ATR", np.mean(sd.get('ATR', [0]))),
                        ("DTAR", np.mean(sd.get('DT_AR', [0]))),
                    ]
                    
                    cols = st.columns(4)
                    for idx, (name, val) in enumerate(metrics_data[:4]):
                        with cols[idx]:
                            if enable_zscore and all_ref_data and name != "DTAR":
                                normal_mean = load_normal_ref(name, lead, 'mean')
                                normal_std = load_normal_ref(name, lead, 'std')
                                if normal_mean is not None and normal_std is not None:
                                    zscore = (val - normal_mean) / normal_std if normal_std != 0 else 0
                                    is_abnormal = abs(zscore) > zscore_threshold
                                    color = "🔴" if is_abnormal else "🟢"
                                    st.metric(name, f"{val:.2f}", f"Z={zscore:.2f} {color}")
                                else:
                                    st.metric(name, f"{val:.2f}")
                            else:
                                st.metric(name, f"{val:.2f}")
                    
                    cols2 = st.columns(2)
                    for idx, (name, val) in enumerate(metrics_data[4:]):
                        with cols2[idx]:
                            st.metric(name, f"{val:.2f}")

elif edf_file is None:
    st.info("👈 在左侧上传EDF文件后点击 **运行分析** 按钮开始")
else:
    st.info("👈 在左侧设置参数后点击 **运行分析** 按钮开始")

st.markdown("""
### 使用说明
    
    1. **上传EDF文件** - 选择你的脑电数据文件
    2. **调整参数** - 根据需求设置分析参数
    3. **选择导联** - 选择要查看的脑电导联
    4. **点击运行** - 开始分析并查看结果

    """)