import glob
import matplotlib

matplotlib.use('Qt5Agg')

import pickle

from scipy import signal
import json
import mne

from pathlib import Path

# 获取文件名
script_name = Path(__file__).name
print(f"文件名: {script_name}")

from tqdm import tqdm
import age_group_list
from parameter_tools import *

import parameter_tools as pt







def trans_raw_2_show(in_array):
    high_th = 1.5
    in_array = np.log10(in_array + 1.002)
    gp = np.ma.where(in_array > high_th, high_th + np.log2(in_array), in_array)
    return gp



def get_lead_dict_fif(fif_path,lead_type,sig_len):

    raw = mne.io.read_raw_fif(fif_path, preload=True)
    duration_sec = raw.times[-1]
    sfreq = raw.info['sfreq']

    print(sfreq)
    cut_sec = 0

    if sig_len > duration_sec:
        cut_sec = duration_sec
    elif sig_len == 0:
        cut_sec = duration_sec
    else:
        cut_sec = sig_len


    #V转成正常uV
    #调整采样率
    raw._data *= 1e6
    #raw.resample(256)

    all_data = {}
    # 遍历导联名称和索引
    print(raw.ch_names)

    #读取所有数据
    for i, ch_name in enumerate(raw.ch_names):
        # get_data 可以指定 picks（索引或名称）
        # 返回的是二维数组，取 [0] 变成一维
        _signal = raw.get_data(picks=ch_name)[0]
        all_data[ch_name] = _signal[:int(cut_sec*sfreq)]

    return all_data




def calculate_trimmed_stats(data_list):
    # 1. 边界条件：如果输入为空，直接返回 None 或自定义值
    if not data_list:
        return {"mean": None, "std": None, "message": "Input list is empty"}

    # 转换为 numpy 数组并排序
    arr = np.array(data_list)
    sorted_arr = np.sort(arr)
    n = len(sorted_arr)

    # 2. 计算需要剔除的元素数量（使用 floor 或 round，这里用 int 自动向下取整）
    low_idx = int(n * 0.10)  # 最低 10% 的截止索引
    high_idx = n - int(n * 0.20)  # 最高 20% 的开始索引

    # 3. 边界条件：确保剔除后至少还剩下一个元素
    if low_idx >= high_idx:
        return {
            "mean": None,
            "std": None,
            "message": "List is too short to trim 10% low and 20% high"
        }

    # 4. 切片获取中间有效的数据
    trimmed_data = sorted_arr[low_idx:high_idx]

    # 5. 计算 mean 和 std
    mean_val = np.mean(trimmed_data)
    std_val = np.std(trimmed_data)

    return {
        "mean": mean_val,
        "std": std_val,
        "trimmed_data_length": len(trimmed_data)
    }



def calculate_eeg_asymmetry_dict(SPEC_DICT, name):
    """
    计算EEG频谱的左右非对称指数，并以词典格式返回。
    参数:
    SPEC_G (numpy.ndarray): 输入的频谱矩阵 [通道, 时间, 频率]
    返回:
    dict: 包含所有计算结果的词典
    """


    def get_asymmetry(idx_left, idx_right):
        # 提取特定通道和频率区间 (8:13)
        left = np.sum(SPEC_DICT[idx_left][:, 8:13],axis=1)
        right = np.sum(SPEC_DICT[idx_right][:, 8:13],axis=1)
        # 计算非对称指数: (L - R) / (L + R)
        # 加上 1e-8 是为了防止分母为 0 导致报错
        return (left - right) / (left + right + 1e-8)

    # 构造结果词典
    results = {
        "E__F3_F4_"+name: get_asymmetry("F3-A1", "F4-A2"),
        "E__F7_F8_"+name: get_asymmetry("F7-A1", "F8-A2"),

        "B__F3_F4_"+name: get_asymmetry("F3-C3", "F4-C4"),
        "B__F7_F8_"+name: get_asymmetry("F7-T3", "F8-T4"),

        "A__F3_F4_"+name: get_asymmetry("F3-AVG", "F4-AVG"),
        "A__F7_F8_"+name: get_asymmetry("F7-AVG", "F8-AVG")
    }



    return results

def get_feature_dict_by_lead(SPEC_second):

    # print(np.shape(SPEC_second))

    #_delta = np.mean(SPEC_second[:,:, 1:4], axis=2)
    _delta = np.mean(SPEC_second[:, 1:4], axis=1)
    _theta = np.mean(SPEC_second[:, 4:8], axis=1)
    _alpha = np.mean(SPEC_second[:, 8:13], axis=1)
    _alpha_1 = np.mean(SPEC_second[:, 8:9], axis=1)
    _alpha_2 = np.mean(SPEC_second[:, 9:11], axis=1)
    _alpha_3 = np.mean(SPEC_second[:, 11:13], axis=1)
    _beta = np.mean(SPEC_second[:, 13:30], axis=1)
    _beta_1 = np.mean(SPEC_second[:, 13:20], axis=1)
    _beta_2 = np.mean(SPEC_second[:, 20:30], axis=1)

    _gamma = np.mean(SPEC_second[:, 30:70], axis=1)
    _gamma_1 = np.mean(SPEC_second[:, 30:50], axis=1)
    _gamma_2 = np.mean(SPEC_second[:, 50:70], axis=1)


    _total = _alpha + _theta + _beta + _delta + _gamma + 1e-6

    #print(np.shape(_alpha))

    _r_delta = _delta / _total
    _r_theta = _theta / _total
    _r_alpha = _alpha / _total
    _r_alpha_1 = _alpha_1 / _total
    _r_alpha_2 = _alpha_2 / _total
    _r_beta = _beta / _total
    _r_beta_1 = _beta_1 / _total
    _r_beta_2 = _beta_2 / _total
    _r_gamma = _gamma / _total
    _r_gamma_1 = _gamma_1/_total
    _r_gamma_2 = _gamma_2/_total

    TBR = _theta  / (_beta+1e-6)
    DAR = _delta  / (_alpha+1e-6)
    DTR =  _delta / (_theta+1e-6)
    ABR = _alpha  / (_beta+1e-6)
    ATR = _alpha  /(_theta+1e-6)
    DT_AR = (_delta+_theta)/(_alpha+1e-6)
    DT_total_R = (_delta+_theta)/_total


    spec_features_dict = {
        "delta": _delta,
        "theta": _theta,
        "alpha": _alpha,
        "alpha_1": _alpha_1,
        "alpha_2": _alpha_2,
        "alpha_3": _alpha_3,
        "beta": _beta,
        "beta_1": _beta_1,
        "beta_2": _beta_2,
        "gamma": _gamma,
        "gamma_1": _gamma_1,
        "gamma_2": _gamma_2,
        "relative_delta": _r_delta,
        "relative_theta": _r_theta,
        "relative_alpha": _r_alpha,
        "relative_beta": _r_beta,
        "relative_gamma": _r_gamma,
        "TBR": TBR,
        "DAR": DAR,
        "DTR": DTR,
        "ABR": ABR,
        "ATR": ATR,
        "DT_AR": DT_AR,
        "DT_total_R": DT_total_R
    }

    out_dict ={}
    for k,v in spec_features_dict.items():
        clean_arr = v[~np.isnan(v)]
        out_dict[k] = clean_arr
        #print(k,np.isnan(v).sum(),np.shape(v),np.shape(clean_arr))


    return  out_dict



def hl_envelopes_idx(s, dmin=2, dmax=2, split=False):
    """
    Input :
    s: 1d-array, data signal from which to extract high and low envelopes
    dmin, dmax: int, optional, size of chunks, use this if the size of the input signal is too big
    split: bool, optional, if True, split the signal in half along its mean, might help to generate the envelope in some cases
    Output :
    lmin,lmax : high/low envelope idx of input signal s
    """
    # locals min
    lmin = (np.diff(np.sign(np.diff(s))) > 0).nonzero()[0] + 1
    # locals max
    lmax = (np.diff(np.sign(np.diff(s))) < 0).nonzero()[0] + 1

    if split:
        # s_mid is zero if s centered around x-axis or more generally mean of signal
        s_mid = np.mean(s)
        # pre-sorting of locals min based on relative position with respect to s_mid
        lmin = lmin[s[lmin] < s_mid]
        # pre-sorting of local max based on relative position with respect to s_mid
        lmax = lmax[s[lmax] > s_mid]

    # global max of dmax-chunks of locals max
    lmin = lmin[[i + np.argmin(s[lmin[i:i + dmin]]) for i in range(0, len(lmin), dmin)]]
    # global min of dmin-chunks of locals min
    lmax = lmax[[i + np.argmax(s[lmax[i:i + dmax]]) for i in range(0, len(lmax), dmax)]]

    return lmin, lmax


def get_pp_value(cur_signal):
    signal_len = 256
    # 获取幅值
    high_idx, low_idx = hl_envelopes_idx(cur_signal)
    high_signal = np.zeros(signal_len)
    low_signal = np.zeros(signal_len)

    high_index_end = signal_len
    for i_high in np.flipud(high_idx):
        high_index_begin = i_high
        high_signal[high_index_begin:high_index_end] = cur_signal[high_index_begin]
        high_index_end = i_high

    low_index_end = signal_len
    for i_low in np.flipud(low_idx):
        low_index_begin = i_low
        low_signal[low_index_begin:low_index_end] = cur_signal[low_index_begin]
        low_index_end = i_low

    p2p_range = low_signal - high_signal
    mask = p2p_range > np.mean(p2p_range)
    pp_mean_value = p2p_range[mask].mean()

    return  pp_mean_value



eeg_path = "J:\\NORMAL\\*\\*\\*\\*\\*.eeg"
eeg_path_list = glob.glob(eeg_path)


sample_rate = 256
LEAD_TYPE = '21'



epoch_len_sec = 2
print("epoch len:",epoch_len_sec)


for age_group_name,age_group in age_group_list.age_dict.items():

    print(age_group_name,len(age_group))


    g_process_dict = {}
    e_process_dict = {}
    alpha_process_dict = {k: [] for k in core_alpha_list}
    f3478_process_dict = {k: [] for k in F3478_lead_list}

    for _lead_name in Total_lead_name_list:
        for _feature in spec_features_list:
            _key = _lead_name + '__' + _feature
            g_process_dict[_key] = []
            e_process_dict[_key] = []


    for eeg_path in tqdm(age_group):

        #解析raw数据
        #解析NG 如果有的话
        #解析prob_array
        PROB_array = []
        pro_pkl_path = eeg_path.replace(".eeg", "_prob_230516_with_softmax_half_second_EBA.pkl")

        one_fif = eeg_path.replace(".eeg","_raw.fif")
        with open(pro_pkl_path, 'rb') as file:
            prob_dict_with_softmax = pickle.load(file)

        print(len(prob_dict_with_softmax.keys()))


        for _k in prob_dict_with_softmax.keys():
            _prob = prob_dict_with_softmax[_k]
            PROB_array.append(_prob)


        #设置为零，即获取所有的数据
        seg_len = 0

        eeg_dict = get_lead_dict_fif(one_fif,LEAD_TYPE,seg_len)

        print(eeg_dict.keys())


        leads_montage_dict = get_montage_data_from_dict(eeg_dict, "EBA")
        leads_montage_dict_raw = get_montage_data_from_dict(eeg_dict, "EBA_RAW")

        signal_second_length = int(np.shape(leads_montage_dict['Fp1-A1'])[0] / 256)

        # print(signal_second_length)
        # print(len(leads_montage_dict.keys()))

        nperseg = 256
        noverlap = 128
        fs = 256

        PSD_dict = {}
        PSD_array = []
        PSD_array_no_art = []

        lead_name_list = []
        alpha_signal = []
        alpha_psd =[]


        lead_count = 0
        nperseg_len = 1

        epoch_len = epoch_len_sec * fs
        duration_sec = signal_second_length
        epoch_count = int(duration_sec / epoch_len_sec)


        F3478_PSD_dict = {}
        F3478_PSD_no_ART_dict = {}


        for lead_name, one_signal in leads_montage_dict.items():

            lead_name_list.append(lead_name)
            one_signal_raw = leads_montage_dict_raw[lead_name]


            #用raw数据计算spec
            one_signal_reshape = one_signal_raw[0:epoch_count * epoch_len_sec * fs].reshape(epoch_count, epoch_len_sec * fs)
            freqs, psds = signal.welch(one_signal_reshape, fs=fs, window='hann', nperseg= fs * nperseg_len)
            #_prob = prob_dict_with_softmax[_k]



            #获取prob相关信息
            _prob = prob_dict_with_softmax[lead_name]
            if epoch_count*epoch_len_sec != np.shape(_prob)[0]*2:
                _prob = np.pad(_prob, ((0, 1), (0, 0)), mode='constant', constant_values=0)

            #这里使用load出来的prob 无非是节省时间
            art_prob_index = _prob[ :,2] + _prob[ :,1]
            prob_len = epoch_count*epoch_len_sec*2
            #print(prob_len, epoch_count, epoch_len_sec, signal_second_length)
            mean_art_prob = np.max(art_prob_index[:prob_len].reshape(epoch_count,epoch_len_sec*2),axis=1)
            art_th = 0.6
            mask = (mean_art_prob < art_th).reshape(-1, 1)
            true_count = np.sum(mask)


            #获取带ART和不带ART的prob
            psds_without_art = np.where(mask, psds, np.nan)
            PSD_array_no_art.append(psds_without_art)
            PSD_array.append(psds)


            if lead_name in F3478_name_list:
                F3478_PSD_dict[lead_name] = psds
                F3478_PSD_no_ART_dict[lead_name] = psds_without_art
            #计算参数

            g_feature_dict = get_feature_dict_by_lead(psds)

            e_feature_dict = get_feature_dict_by_lead(psds_without_art)


            for _k,_v in g_feature_dict.items():
                _name = lead_name+"__"+_k
                g_process_dict[_name].extend(_v)


            for _k,_v in e_feature_dict.items():
                _name = lead_name+"__"+_k
                e_process_dict[_name].extend(_v)

            if lead_name in avg_keys:
                _one_lead_alpha_prob = prob_dict_with_softmax[lead_name][:, 7]
                indices = np.argwhere(_one_lead_alpha_prob > 0.60).flatten()

                for _idx in indices:
                    _alpha_signal = leads_montage_dict[lead_name][_idx * 128:(_idx + 2) * 128]
                    pp_value = get_pp_value(_alpha_signal)
                    freqs, psds_alpha = signal.welch(_alpha_signal, fs=fs, window='hann', nperseg=256)
                    f_int = np.arange(6, 13)
                    max_power_index = np.argmax(psds_alpha[6:13])
                    max_frequency = f_int[max_power_index]
                    power_30 = np.sum(psds_alpha[:30]) + 10e-5
                    power_alpha = np.sum(psds_alpha[6:13]) + 10e-5
                    if 6 <= max_frequency < 13 and power_alpha / power_30 > 0.45:
                        alpha_process_dict["A__core_HZ"].append(max_frequency)
                        alpha_process_dict["A__core_power_ratio"].append(power_alpha / power_30)
                        alpha_process_dict["A__pp_value"].append(pp_value)

            if lead_name in bipolar_keys:
                _one_lead_alpha_prob = prob_dict_with_softmax[lead_name][:, 7]
                indices = np.argwhere(_one_lead_alpha_prob > 0.70).flatten()
                for _idx in indices:
                    _alpha_signal = leads_montage_dict[lead_name][_idx * 128:(_idx + 2) * 128]
                    pp_value = get_pp_value(_alpha_signal)
                    freqs, psds_alpha = signal.welch(_alpha_signal, fs=fs, window='hann', nperseg=256)
                    f_int = np.arange(6, 13)
                    max_power_index = np.argmax(psds_alpha[6:13])
                    max_frequency = f_int[max_power_index]
                    power_30 = np.sum(psds_alpha[:30]) + 10e-5
                    power_alpha = np.sum(psds_alpha[6:13]) + 10e-5
                    if 6 <= max_frequency < 13 and power_alpha / power_30 > 0.45:
                        alpha_process_dict["B__core_HZ"].append(max_frequency)
                        alpha_process_dict["B__core_power_ratio"].append(power_alpha / power_30)
                        alpha_process_dict["B__pp_value"].append(pp_value)

            if lead_name in ear_keys:
                _one_lead_alpha_prob = prob_dict_with_softmax[lead_name][:, 7]
                indices = np.argwhere(_one_lead_alpha_prob > 0.70).flatten()
                for _idx in indices:
                    _alpha_signal = leads_montage_dict[lead_name][_idx * 128:(_idx + 2) * 128]
                    pp_value = get_pp_value(_alpha_signal)
                    freqs, psds_alpha = signal.welch(_alpha_signal, fs=fs, window='hann', nperseg=256)
                    f_int = np.arange(6, 13)
                    max_power_index = np.argmax(psds_alpha[6:13])
                    max_frequency = f_int[max_power_index]
                    power_30 = np.sum(psds_alpha[:30]) + 10e-5
                    power_alpha = np.sum(psds_alpha[6:13]) + 10e-5
                    if 6 <= max_frequency < 13 and power_alpha / power_30 > 0.45:
                        alpha_process_dict["E__core_HZ"].append(max_frequency)
                        alpha_process_dict["E__core_power_ratio"].append(power_alpha / power_30)
                        alpha_process_dict["E__pp_value"].append(pp_value)

        PSD_array = np.array(PSD_array)
        PSD_array_no_art = np.array(PSD_array_no_art)

        E_F3478_feature_dict = calculate_eeg_asymmetry_dict(F3478_PSD_no_ART_dict, "E")
        G_F3478_feature_dict = calculate_eeg_asymmetry_dict(F3478_PSD_dict, "G")



        for k,v in E_F3478_feature_dict.items():


            clean_arr = v[~np.isnan(v)]
            f3478_process_dict[k].extend(clean_arr)



        for k,v in G_F3478_feature_dict.items():
            f3478_process_dict[k].extend(v)



    g_out_dict = {}
    e_out_dict = {}
    f3478_out_dict = {}
    alpha_out_dict = {}
    for k,v in g_process_dict.items():
        #
        # print(k,calculate_trimmed_stats(v))
        # print("E:",k,calculate_trimmed_stats(e_process_dict[k]))

        g_out_dict[k] = calculate_trimmed_stats(v)
        e_out_dict[k] = calculate_trimmed_stats(e_process_dict[k])


    for k,v in f3478_process_dict.items():
        f3478_out_dict[k] = calculate_trimmed_stats(v)

    for k,v in alpha_process_dict.items():
        alpha_out_dict[k] = calculate_trimmed_stats(v)


    all_dict = {
        "g_info": g_out_dict,
        "e_info": e_out_dict,
        "alpha_info": alpha_out_dict,
        "f3478_info": f3478_out_dict,

    }

    with open(age_group_name+"_"+str(epoch_len_sec) + "_normal_ref_0529.pkl", "wb") as f:
        pickle.dump(all_dict, f)
    with open(age_group_name+"_"+str(epoch_len_sec) + "_normal_ref_0529.json", 'w', encoding='utf-8') as f:
        json.dump(all_dict, f, ensure_ascii=False, indent=4)



