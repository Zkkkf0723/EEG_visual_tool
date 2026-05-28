import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import glob
import mne
import pickle
import numpy as np
import json
import os
import tempfile
import shutil
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
    if normal_std == 0 or np.isnan(normal_std):
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

st.set_page_config(page_title="PSD 可视化工具", layout="wide")

st.title("🧠 EEG PSD 可视化工具")

with st.sidebar:
    st.header("⚙️ 参数设置")
    
    st.markdown("**📁 上传EDF文件**")
    edf_file = st.file_uploader("选择EDF文件", type=["edf"])
    
    st.markdown("**📁 上传Prob文件（可选）**")
    prob_file = st.file_uploader("选择Prob文件（用于伪迹过滤）", type=["pkl"])
    
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
    
    run_button = st.button("🚀 运行分析", type="primary")

if run_button or 'data_loaded' in st.session_state:
    if not run_button and 'data_loaded' in st.session_state:
        edf_path = st.session_state.edf_path
        epoch_len_sec = st.session_state.epoch_len_sec
        nperseg_len = st.session_state.nperseg_len
        art_threshold = st.session_state.art_threshold
        selected_leads = st.session_state.selected_leads
        enable_zscore = st.session_state.get('enable_zscore', True)
        selected_age_group = st.session_state.get('selected_age_group', '19-44')
        window_sizes = st.session_state.get('window_sizes', [1, 2, 5, 10, 15])
        zscore_threshold = st.session_state.get('zscore_threshold', 2.0)
    
    if edf_file is None:
        st.info("👆 请先上传EDF文件")
        st.stop()
    
    temp_dir = tempfile.gettempdir()
    edf_path = os.path.join(temp_dir, edf_file.name)
    with open(edf_path, "wb") as f:
        f.write(edf_file.getvalue())
    
    prob_path = None
    if prob_file is not None:
        prob_path = os.path.join(temp_dir, prob_file.name)
        with open(prob_path, "wb") as f:
            f.write(prob_file.getvalue())
    
    if prob_path is None:
        st.warning("⚠️ 未上传Prob文件，将跳过伪迹过滤")
    
    if not os.path.exists(edf_path):
        st.error(f"❌ EDF文件不存在: {edf_path}")
    elif prob_path is not None and not os.path.exists(prob_path):
        st.error(f"❌ Prob文件不存在: {prob_path}")
    else:
        with st.spinner("正在加载数据..."):
            raw = mne.io.read_raw_edf(edf_path, preload=True)
            duration_sec = raw.times[-1]
            
            raw._data *= 1e6
            raw.resample(256)
            
            all_data = {}
            for ch_name in raw.ch_names:
                all_data[ch_name] = raw.get_data(picks=ch_name)[0]
            
            leads_montage_dict = get_bipolar_data_caueeg(all_data, 0.5, 70)
            
            prob_dict = None
            if prob_path is not None and os.path.exists(prob_path):
                with open(prob_path, 'rb') as f:
                    prob_dict = pickle.load(f)
        
        with st.spinner("正在计算PSD..."):
            fs = 256
            epoch_count = int(duration_sec / epoch_len_sec)            
            epoch_times = list(range(epoch_count))
            
            PSD_dict = {}
            PSD_array_no_art_dict = {}  # 存储每个导联过滤后的PSD数组
            spec_dict_all = {}
            
            PSD_array_no_art = []  # 用于get_spec_stat_info的列表格式
            
            for lead_name in leads_montage_dict.keys():
                one_signal = leads_montage_dict[lead_name]
                one_signal_reshape = one_signal[0:epoch_count * epoch_len_sec * fs].reshape(epoch_count, epoch_len_sec * fs)
                freqs, psds = signal.welch(one_signal_reshape, fs=fs, window='hann', nperseg=fs * nperseg_len)
                
                if prob_dict is not None:
                    _prob = prob_dict[lead_name]
                    if epoch_count * epoch_len_sec != np.shape(_prob)[0] * 2:
                        _prob = np.pad(_prob, ((0, 1), (0, 0)), mode='constant', constant_values=0)
                    art_prob_index = _prob[:, 2] + _prob[:, 1]
                    prob_len = epoch_count * epoch_len_sec * 2
                    mean_art_prob = np.max(art_prob_index[:prob_len].reshape(epoch_count, epoch_len_sec * 2), axis=1)
                    psds_without_art = psds[mean_art_prob < art_threshold, :]
                else:
                    psds_without_art = psds
                
                PSD_dict[lead_name] = 10 * np.log10(psds_without_art)
                PSD_array_no_art.append(psds_without_art)
            
            spec_dict = get_spec_stat_info(PSD_array_no_art)
            for lead_idx, lead_name in enumerate(leads_montage_dict.keys()):
                spec_dict_all[lead_name] = {k: v[lead_idx] for k, v in spec_dict.items()}
            
            for lead_name in leads_montage_dict.keys():
                one_signal = leads_montage_dict[lead_name]
                one_signal_reshape = one_signal[0:epoch_count * epoch_len_sec * fs].reshape(epoch_count, epoch_len_sec * fs)
                freqs, psds_raw = signal.welch(one_signal_reshape, fs=fs, window='hann', nperseg=fs * nperseg_len)
                PSD_dict[lead_name] = 10 * np.log10(psds_raw)
            
            F3478_dict = get_F3478(PSD_dict, nperseg_len)
            
            F3478_dict = get_F3478(PSD_dict, nperseg_len)
            
        st.session_state['data_loaded'] = True
        st.session_state.edf_path = edf_path
        st.session_state.epoch_len_sec = epoch_len_sec
        st.session_state.nperseg_len = nperseg_len
        st.session_state.art_threshold = art_threshold
        st.session_state.selected_leads = selected_leads
        st.session_state.enable_zscore = enable_zscore
        st.session_state.selected_age_group = selected_age_group
        st.session_state.window_sizes = window_sizes
        st.session_state.zscore_threshold = zscore_threshold
        
        if enable_zscore:
            with st.spinner("正在加载正常参考数据..."):
                json_dir = os.path.join(os.path.dirname(__file__), "json")
                all_ref_data = load_normal_reference(json_dir, window_sizes)
        else:
            all_ref_data = None
        
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
                            x=epoch_times,
                            y=sd["theta"],
                            name="Theta (4-8Hz)",
                            mode='lines',
                            line=dict(color='blue', width=1.5)
                        ))
                        fig_single.add_trace(go.Scatter(
                            x=epoch_times,
                            y=sd["alpha"],
                            name="Alpha (8-13Hz)",
                            mode='lines',
                            line=dict(color='red', width=1.5)
                        ))
                        fig_single.add_trace(go.Scatter(
                            x=epoch_times,
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
                        st.write(f"TBR: {np.mean(sd['TBR']):.2f}")
                        st.write(f"DAR: {np.mean(sd['DAR']):.2f}")
                        st.write(f"DTR: {np.mean(sd['DTR']):.2f}")
                        st.write(f"ABR: {np.mean(sd['ABR']):.2f}")
                        st.write(f"ATR: {np.mean(sd['ATR']):.2f}")
                        st.write(f"DTAR: {np.mean(sd['DT_AR']):.2f}")
        
        with tab2:
            st.subheader("📈 功率比率指标")
            
            ratio_options = st.multiselect(
                "选择要显示的比率", 
                ["TBR", "DAR", "DTR", "ABR", "ATR", "DTAR", "DTPWR"],
                default=["TBR", "DAR"]
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
                            if ratio in sd and enable_zscore and all_ref_data and ratio not in ["DTAR", "DTPWR"]:
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
                            if ratio in sd:
                                fig_ratio.add_trace(go.Scatter(
                                    x=epoch_times,
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
            
            zscore_viz_options = st.multiselect(
                "选择Z-score可视化方式",
                ["📊 柱状图", "📈 带参考线的时序图", "🕸️ 雷达图", "🔥 热力图"],
                default=["📊 柱状图"]
            )
            
            for lead in selected_leads:
                if lead in spec_dict_all:
                    sd = spec_dict_all[lead]
                    
                    st.markdown(f"### {lead}")
                    
                    metrics_data = [
                        ("TBR (θ/β)", np.mean(sd['TBR']), "TBR", lead, "TBR"),
                        ("DAR (δ/α)", np.mean(sd['DAR']), "DAR", lead, "DAR"),
                        ("DTR (δ/θ)", np.mean(sd['DTR']), "DTR", lead, "DTR"),
                        ("ABR (α/β)", np.mean(sd['ABR']), "ABR", lead, "ABR"),
                        ("ATR (α/θ)", np.mean(sd['ATR']), "ATR", lead, "ATR"),
                        ("DTAR (δ+θ)/α", np.mean(sd['DT_AR']), "DTAR", lead, "DTAR"),
                        ("(δ+θ)/总功率", np.mean(sd['DT_total_R']), "DTPWR", lead, "DTPWR")
                    ]
                    
                    metrics_with_zscore = []
                    for name, val, band, l, ratio in metrics_data:
                        if ratio in ["DTAR", "DTPWR"]:
                            normal_mean = None
                            normal_std = None
                        else:
                            normal_mean = get_normal_ref_for_lead(all_ref_data, selected_age_group, band, l, 'mean')
                            normal_std = get_normal_ref_for_lead(all_ref_data, selected_age_group, band, l, 'std')
                        
                        if normal_mean is not None and normal_std is not None:
                            zscore = calculate_zscore(val, normal_mean, normal_std)
                        else:
                            zscore = None
                        metrics_with_zscore.append((name, val, zscore, normal_mean, normal_std, ratio))
                    
                    if "📊 柱状图" in zscore_viz_options:
                        valid_zscores = [(n, z) for n, v, z, m, s, r in metrics_with_zscore if z is not None]
                        if valid_zscores:
                            names = [n for n, z in valid_zscores]
                            zscores = [z for n, z in valid_zscores]
                            colors = ['red' if abs(z) > zscore_threshold else 'steelblue' for z in zscores]
                            
                            fig_bar = go.Figure()
                            fig_bar.add_trace(go.Bar(
                                x=names, y=zscores,
                                marker_color=colors,
                                text=[f"{z:.2f}" for z in zscores],
                                textposition='outside'
                            ))
                            fig_bar.add_hline(y=zscore_threshold, line_dash="dash", line_color="red", annotation_text=f"+{zscore_threshold}σ")
                            fig_bar.add_hline(y=-zscore_threshold, line_dash="dash", line_color="red", annotation_text=f"-{zscore_threshold}σ")
                            fig_bar.update_layout(
                                title=dict(text=f"<b>{lead}</b> Z-score", x=0.5),
                                yaxis_title="Z-score",
                                height=300
                            )
                            st.plotly_chart(fig_bar, use_container_width=True)
                    
                    if "📈 带参考线的时序图" in zscore_viz_options:
                        ratio_list = [("TBR", "TBR"), ("DAR", "DAR"), ("DTR", "DTR"), ("ABR", "ABR"), ("ATR", "ATR")]
                        for ratio_name, ratio_key in ratio_list:
                            normal_mean = get_normal_ref_for_lead(all_ref_data, selected_age_group, ratio_name, lead, 'mean')
                            normal_std = get_normal_ref_for_lead(all_ref_data, selected_age_group, ratio_name, lead, 'std')
                            if normal_mean is not None and normal_std is not None:
                                upper_bound = normal_mean + zscore_threshold * normal_std
                                lower_bound = normal_mean - zscore_threshold * normal_std
                                
                                fig_ts = go.Figure()
                                fig_ts.add_trace(go.Scatter(
                                    x=epoch_times, y=sd[ratio_key],
                                    mode='lines', name=ratio_name,
                                    line=dict(color='blue', width=1.5)
                                ))
                                fig_ts.add_hline(y=normal_mean, line_dash="solid", line_color="green", annotation_text="正常均值")
                                fig_ts.add_hline(y=upper_bound, line_dash="dash", line_color="red", annotation_text=f"+{zscore_threshold}σ")
                                fig_ts.add_hline(y=lower_bound, line_dash="dash", line_color="red", annotation_text=f"-{zscore_threshold}σ")
                                fig_ts.update_layout(
                                    title=dict(text=f"<b>{lead}</b> {ratio_name} 时序图", x=0.5),
                                    xaxis_title="Epoch", yaxis_title=ratio_name,
                                    height=250
                                )
                                st.plotly_chart(fig_ts, use_container_width=True)
                    
                    if "🕸️ 雷达图" in zscore_viz_options:
                        valid_zscores = [(n.split(" ")[0], z) for n, v, z, m, s, r in metrics_with_zscore if z is not None]
                        if len(valid_zscores) >= 3:
                            labels = [n for n, z in valid_zscores]
                            values = [z for n, z in valid_zscores]
                            
                            fig_radar = go.Figure()
                            fig_radar.add_trace(go.Scatterpolar(
                                r=values + [values[0]],
                                theta=labels + [labels[0]],
                                fill='toself',
                                fillcolor='rgba(0,100,255,0.2)',
                                line=dict(color='blue', width=2),
                                name='Z-score'
                            ))
                            fig_radar.add_trace(go.Scatterpolar(
                                r=[zscore_threshold]*len(labels) + [zscore_threshold],
                                theta=labels + [labels[0]],
                                mode='lines',
                                line=dict(color='red', width=1, dash='dash'),
                                name=f'±{zscore_threshold}σ阈值'
                            ))
                            fig_radar.update_layout(
                                polar=dict(radialaxis=dict(range=[-3, max(values)+1])),
                                title=dict(text=f"<b>{lead}</b> Z-score 雷达图", x=0.5),
                                height=350
                            )
                            st.plotly_chart(fig_radar, use_container_width=True)
                    
                    if "🔥 热力图" in zscore_viz_options:
                        all_leads_zscore = []
                        all_leads_names = []
                        for l in selected_leads:
                            if l in spec_dict_all:
                                sd_l = spec_dict_all[l]
                                row = []
                                for name, val, band, lead_l, ratio in metrics_data:
                                    if ratio in ["DTAR", "DTPWR"]:
                                        row.append(None)
                                    else:
                                        normal_mean = get_normal_ref_for_lead(all_ref_data, selected_age_group, band, lead_l, 'mean')
                                        normal_std = get_normal_ref_for_lead(all_ref_data, selected_age_group, band, lead_l, 'std')
                                        if normal_mean is not None and normal_std is not None:
                                            row.append(calculate_zscore(np.mean(sd_l[ratio]), normal_mean, normal_std))
                                        else:
                                            row.append(None)
                                all_leads_zscore.append(row)
                                all_leads_names.append(l)
                        
                        metric_names = [n.split(" ")[0] for n, v, z, m, s, r in metrics_with_zscore]
                        
                        fig_heatmap = go.Figure(data=go.Heatmap(
                            z=all_leads_zscore,
                            x=metric_names,
                            y=all_leads_names,
                            colorscale='RdBu_r',
                            zmid=0,
                            text=[[f"{z:.2f}" if z is not None else "N/A" for z in row] for row in all_leads_zscore],
                            texttemplate="%{text}",
                            colorbar=dict(title="Z-score")
                        ))
                        fig_heatmap.update_layout(
                            title=dict(text="<b>各导联Z-score热力图</b>", x=0.5),
                            height=max(300, len(selected_leads) * 50)
                        )
                        st.plotly_chart(fig_heatmap, use_container_width=True)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        for name, val, band, l, ratio in metrics_data[:2]:
                            if enable_zscore and all_ref_data:
                                zscore_info = next((z for n, v, z, m, s, r in metrics_with_zscore if n == name and r == ratio), None)
                                if zscore_info is not None and zscore_info != 0:
                                    is_abnormal = abs(zscore_info) > zscore_threshold
                                    color = "🔴" if is_abnormal else "🟢"
                                    st.metric(name, f"{val:.2f}", f"Z={zscore_info:.2f} {color}")
                                else:
                                    st.metric(name, f"{val:.2f}")
                            else:
                                st.metric(name, f"{val:.2f}")
                    with col2:
                        for name, val, band, l, ratio in metrics_data[2:4]:
                            if enable_zscore and all_ref_data:
                                zscore_info = next((z for n, v, z, m, s, r in metrics_with_zscore if n == name and r == ratio), None)
                                if zscore_info is not None and zscore_info != 0:
                                    is_abnormal = abs(zscore_info) > zscore_threshold
                                    color = "🔴" if is_abnormal else "🟢"
                                    st.metric(name, f"{val:.2f}", f"Z={zscore_info:.2f} {color}")
                                else:
                                    st.metric(name, f"{val:.2f}")
                            else:
                                st.metric(name, f"{val:.2f}")
                    with col3:
                        for name, val, band, l, ratio in metrics_data[4:6]:
                            if enable_zscore and all_ref_data:
                                zscore_info = next((z for n, v, z, m, s, r in metrics_with_zscore if n == name and r == ratio), None)
                                if zscore_info is not None and zscore_info != 0:
                                    is_abnormal = abs(zscore_info) > zscore_threshold
                                    color = "🔴" if is_abnormal else "🟢"
                                    st.metric(name, f"{val:.2f}", f"Z={zscore_info:.2f} {color}")
                                else:
                                    st.metric(name, f"{val:.2f}")
                            else:
                                st.metric(name, f"{val:.2f}")
                    with col4:
                        for name, val, band, l, ratio in metrics_data[6:]:
                            if enable_zscore and all_ref_data:
                                zscore_info = next((z for n, v, z, m, s, r in metrics_with_zscore if n == name and r == ratio), None)
                                if zscore_info is not None and zscore_info != 0:
                                    is_abnormal = abs(zscore_info) > zscore_threshold
                                    color = "🔴" if is_abnormal else "🟢"
                                    st.metric(name, f"{val:.2f}", f"Z={zscore_info:.2f} {color}")
                                else:
                                    st.metric(name, f"{val:.2f}")
                            else:
                                st.metric(name, f"{val:.2f}")
        
else:
    st.info("👈 在左侧设置参数后点击 **运行分析** 按钮开始")
    
    st.markdown("""
    ### 使用说明
    
    1. **设置EDF文件路径** - 指向你的脑电数据文件
    2. **调整参数**:
       - Epoch长度: 每次分析的时长(秒)
       - Welch窗口: PSD计算的窗口大小
       - 伪迹阈值: 过滤伪迹的概率阈值
    3. **选择导联** - 选择要查看的脑电导联
    4. **点击运行** - 开始分析并查看结果
    
    ### 功能介绍
    
    | 标签页 | 内容 |
    |-------|------|
    | 频段功率 | Theta/Alpha/Beta各频段功率随时间变化 |
    | TBR时间序列 | Theta/Beta比率，常用于评估注意力 |
    | Alpha不对称指数 | F3/F4和F7/F8的偏侧化分析 |
    """)