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

# ========== 缓存函数 ==========
@st.cache_data(max_entries=3, ttl=3600)
def load_edf_data(edf_bytes: bytes, file_name: str):
    """加载EDF文件并返回可序列化的数据字典"""
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, f"temp_edf_{hash(edf_bytes)}_{file_name}.edf")
    
    try:
        with open(temp_path, "wb") as f:
            f.write(edf_bytes)
        
        raw = mne.io.read_raw_edf(temp_path, preload=True)
        raw._data *= 1e6
        raw.resample(256)
        
        # 获取原始数据（不截断）
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
    """加载正常参考数据"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_dir = os.path.join(current_dir, "json")
    
    if not os.path.exists(json_dir):
        st.warning(f"⚠️ 参考数据目录不存在: {json_dir}")
        return None
    
    all_data = {}
    for ws in [1, 2, 5, 10, 15]:
        json_path = os.path.join(json_dir, f"{ws}_stat_info_dict_ALL_AVG_0514.json")
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    all_data[str(ws)] = json.load(f)
            except Exception as e:
                st.warning(f"加载{ws}窗口数据失败: {e}")
    
    return all_data if all_data else None


# ========== 辅助函数 ==========
def get_normal_ref(all_ref_data, age_group, band, lead, stat_type='mean'):
    """获取正常参考值"""
    if all_ref_data is None:
        return None
    
    values = []
    for ws, ref_data in all_ref_data.items():
        key = f"{band}_{lead}_{stat_type}"
        if age_group in ref_data and key in ref_data[age_group]:
            values.append(ref_data[age_group][key])
    
    return np.mean(values) if values else None


def calculate_zscore(value, mean, std):
    if std == 0 or mean is None or std is None:
        return 0.0
    return (value - mean) / std


# ========== 主程序 ==========
def main():
    # 初始化session_state
    if 'psd_results' not in st.session_state:
        st.session_state.psd_results = None
    if 'analysis_params' not in st.session_state:
        st.session_state.analysis_params = {}
    
    st.title("🧠 EEG PSD 可视化工具")
    
    with st.sidebar:
        st.header("⚙️ 参数设置")
        
        edf_file = st.file_uploader("选择EDF文件", type=["edf"], key="edf_uploader")
        prob_file = st.file_uploader("选择Prob文件（可选）", type=["pkl"], key="prob_uploader")
        
        st.divider()
        
        st.subheader("📊 分析参数")
        epoch_len_sec = st.slider("Epoch长度(秒)", 1, 10, 5, key="epoch_len")
        nperseg_len = st.slider("Welch窗口长度(秒)", 1, 5, 2, key="nperseg")
        art_threshold = st.slider("伪迹过滤阈值", 0.0, 1.0, 0.5, 0.1, key="art_threshold")
        
        st.subheader("🔗 导联选择")
        lead_options = [
            'Fp1-F3', 'Fp2-F4', 'F3-C3', 'F4-C4', 'C3-P3', 'C4-P4',
            'P3-O1', 'P4-O2', 'Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1',
            'F8-T4', 'T4-T6', 'T6-O2', 'Fz-Pz', 'Cz-Pz', 'Pz-Oz'
        ]
        selected_leads = st.multiselect("选择导联", lead_options, default=['Fp1-F3', 'F3-C3'], key="selected_leads")
        
        st.divider()
        
        st.subheader("📈 正常参考对比")
        enable_zscore = st.checkbox("启用Z-score对比", value=True, key="enable_zscore")
        
        age_group_options = ['0-6', '7-13', '14-18', '19-44', '45-59', '60-80', '80', 'total']
        selected_age_group = st.selectbox("选择参考年龄组", age_group_options, index=3, key="age_group")
        
        window_sizes = st.multiselect(
            "选择窗口大小", 
            [1, 2, 5, 10, 15], 
            default=[1, 2, 5], 
            key="window_sizes"
        )
        
        zscore_threshold = st.slider("Z-score异常阈值", 1.0, 3.0, 2.0, 0.1, key="zscore_threshold")
        
        st.divider()
        run_button = st.button("🚀 运行分析", type="primary", width="stretch")
    
    # 构建当前参数
    current_params = {
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
    
    # 检查是否需要重新计算
    need_compute = run_button
    
    if not need_compute and edf_file:
        if st.session_state.analysis_params != current_params:
            need_compute = True
    
    # 执行分析
    if need_compute and edf_file:
        st.session_state.analysis_params = current_params
        
        with st.status("正在分析数据...", expanded=True) as status:
            try:
                # 1. 加载EDF数据
                status.update(label="📂 加载EDF文件...")
                edf_data = load_edf_data(edf_file.getvalue(), edf_file.name)
                
                if edf_data is None:
                    st.error("❌ EDF文件加载失败")
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
                    json_dir_hash = str(hash(os.path.dirname(os.path.abspath(__file__))))
                    all_ref_data = load_normal_reference_data(json_dir_hash)
                
                # 4. 准备数据
                status.update(label="🔧 准备数据...")
                all_data = {}
                for idx, ch_name in enumerate(edf_data['ch_names']):
                    all_data[ch_name] = edf_data['data'][idx]
                
                # 5. 计算双极导联
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
                    freqs, psds = signal.welch(one_signal_reshape, fs=fs, window='hann', nperseg=fs * nperseg_len)
                    
                    
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
                        PSD_dict_10log = 10 * np.log10(psds_without_art)
                        spec_dict_all[lead_name] = {'psd': PSD_dict_10log}
                
                progress_bar.empty()
                
                if len(all_psds) == 0:
                    st.error("❌ 没有有效的PSD数据，请调整伪迹过滤阈值")
                    return
                
                # 7. 计算统计信息
                status.update(label="📈 计算统计指标...")
                spec_dict = get_spec_stat_info(all_psds)
                
                for lead_idx, lead_name in enumerate(leads_list):
                    if lead_idx < len(all_psds):
                        spec_dict_all[lead_name] = {k: v[lead_idx] for k, v in spec_dict.items()}
                
                # 8. 存储结果
                st.session_state.psd_results = {
                    'spec_dict_all': spec_dict_all,
                    'epoch_times': epoch_times,
                    'leads_list': leads_list,
                    'all_ref_data': all_ref_data,
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
        
        valid_leads = [l for l in selected_leads if l in spec_dict_all]
        
        if not valid_leads:
            st.warning("⚠️ 选中的导联无有效数据，请重新选择")
            return
        
        tab1, tab2, tab3 = st.tabs(["📊 频段功率", "📈 功率比率", "📋 统计汇总"])
        
        with tab1:
            st.subheader("各频段功率分布")
            
            for lead in valid_leads:
                sd = spec_dict_all[lead]
                
                # 创建图表
                fig = go.Figure()
                
                # Theta
                if "theta" in sd:
                    fig.add_trace(go.Scatter(
                        x=epoch_times[:len(sd["theta"])],
                        y=sd["theta"],
                        name="Theta (4-8Hz)",
                        mode='lines',
                        line=dict(color='blue', width=1.5)
                    ))
                
                # Alpha
                if "alpha" in sd:
                    fig.add_trace(go.Scatter(
                        x=epoch_times[:len(sd["alpha"])],
                        y=sd["alpha"],
                        name="Alpha (8-13Hz)",
                        mode='lines',
                        line=dict(color='red', width=1.5)
                    ))
                
                # Beta
                if "beta" in sd:
                    fig.add_trace(go.Scatter(
                        x=epoch_times[:len(sd["beta"])],
                        y=sd["beta"],
                        name="Beta (13-30Hz)",
                        mode='lines',
                        line=dict(color='green', width=1.5)
                    ))
                
                fig.update_layout(
                    title=dict(text=f"<b>{lead}</b> 频段功率", x=0.5),
                    xaxis_title="Epoch",
                    yaxis_title="功率 (dB)",
                    height=350,
                    legend=dict(x=1.02, y=1, bgcolor='rgba(255,255,255,0.8)'),
                    margin=dict(r=100)
                )
                st.plotly_chart(fig, width="stretch")
        
        with tab2:
            st.subheader("📈 功率比率指标")
            
            ratio_options = st.multiselect(
                "选择要显示的比率",
                ["TBR", "DAR", "DTR", "ABR", "ATR", "DTAR", "DTPWR"],
                default=["TBR", "DAR"],
                key="ratio_select"
            )
            
            for lead in valid_leads:
                sd = spec_dict_all[lead]
                
                fig = go.Figure()
                colors = {
                    "TBR": "blue", "DAR": "orange", "DTR": "purple",
                    "ABR": "green", "ATR": "red", "DTAR": "brown", "DTPWR": "pink"
                }
                
                for ratio in ratio_options:
                    ratio_key = "DT_AR" if ratio == "DTAR" else ratio
                    if ratio_key in sd:
                        # 添加参考范围
                        if enable_zscore and all_ref_data and ratio not in ["DTAR", "DTPWR"]:
                            normal_mean = get_normal_ref(all_ref_data, selected_age_group, ratio, lead, 'mean')
                            normal_std = get_normal_ref(all_ref_data, selected_age_group, ratio, lead, 'std')
                            
                            if normal_mean is not None and normal_std is not None:
                                upper = normal_mean + zscore_threshold * normal_std
                                lower = normal_mean - zscore_threshold * normal_std
                                data_len = len(sd[ratio_key])
                                
                                fig.add_trace(go.Scatter(
                                    x=list(range(data_len)) + list(range(data_len))[::-1],
                                    y=[upper] * data_len + [lower] * data_len,
                                    fill='toself',
                                    fillcolor='rgba(255,0,0,0.1)',
                                    line=dict(color='rgba(255,0,0,0)'),
                                    name=f'{ratio}参考范围',
                                    showlegend=True
                                ))
                        
                        fig.add_trace(go.Scatter(
                            x=epoch_times[:len(sd[ratio_key])],
                            y=sd[ratio_key],
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
            
            if enable_zscore and all_ref_data:
                st.markdown(f"**📊 正常参考对比** (年龄组: `{selected_age_group}`, 阈值: Z-score > {zscore_threshold})")
            
            for lead in valid_leads:
                sd = spec_dict_all[lead]
                st.markdown(f"### {lead}")
                
                metrics_data = [
                    ("TBR (θ/β)", np.mean(sd['TBR']) if 'TBR' in sd else 0, "TBR", lead),
                    ("DAR (δ/α)", np.mean(sd['DAR']) if 'DAR' in sd else 0, "DAR", lead),
                    ("DTR (δ/θ)", np.mean(sd['DTR']) if 'DTR' in sd else 0, "DTR", lead),
                    ("ABR (α/β)", np.mean(sd['ABR']) if 'ABR' in sd else 0, "ABR", lead),
                    ("ATR (α/θ)", np.mean(sd['ATR']) if 'ATR' in sd else 0, "ATR", lead),
                    ("DTAR (δ+θ)/α", np.mean(sd['DT_AR']) if 'DT_AR' in sd else 0, "DTAR", lead),
                    ("(δ+θ)/总功率", np.mean(sd['DT_total_R']) if 'DT_total_R' in sd else 0, "DTPWR", lead)
                ]
                
                cols = st.columns(4)
                for idx, (name, val, band, l) in enumerate(metrics_data[:4]):
                    with cols[idx]:
                        if enable_zscore and all_ref_data and band not in ["DTAR", "DTPWR"]:
                            mean = get_normal_ref(all_ref_data, selected_age_group, band, l, 'mean')
                            std = get_normal_ref(all_ref_data, selected_age_group, band, l, 'std')
                            if mean is not None and std is not None:
                                z = calculate_zscore(val, mean, std)
                                is_abnormal = abs(z) > zscore_threshold
                                icon = "🔴" if is_abnormal else "🟢"
                                st.metric(name, f"{val:.2f}", f"Z={z:.2f} {icon}")
                            else:
                                st.metric(name, f"{val:.2f}")
                        else:
                            st.metric(name, f"{val:.2f}")
                
                cols2 = st.columns(3)
                for idx, (name, val, band, l) in enumerate(metrics_data[4:]):
                    with cols2[idx]:
                        if enable_zscore and all_ref_data and band not in ["DTAR", "DTPWR"]:
                            mean = get_normal_ref(all_ref_data, selected_age_group, band, l, 'mean')
                            std = get_normal_ref(all_ref_data, selected_age_group, band, l, 'std')
                            if mean is not None and std is not None:
                                z = calculate_zscore(val, mean, std)
                                is_abnormal = abs(z) > zscore_threshold
                                icon = "🔴" if is_abnormal else "🟢"
                                st.metric(name, f"{val:.2f}", f"Z={z:.2f} {icon}")
                            else:
                                st.metric(name, f"{val:.2f}")
                        else:
                            st.metric(name, f"{val:.2f}")
    
    elif edf_file is None:
        st.info("👈 请上传EDF文件")
    elif not need_compute:
        st.info("👈 设置参数")

if __name__ == "__main__":
    main()