import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from parameters import *
import numpy as np
import torch
import mne
import random
from scipy import signal



def get_lead_dict_fif(fif_path,lead_type,sig_len):

    raw = mne.io.read_raw_fif(fif_path, preload=True)
    duration_sec = raw.times[-1]
    sfreq = raw.info['sfreq']

    print(sfreq)
    cut_sec = 0

    if sig_len>duration_sec:
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

    return pp_mean_value


def get_prediction(model, x):
    torch.cuda.empty_cache()
    x = np.ascontiguousarray(x, dtype=np.float32)
    x = torch.from_numpy(x)


    if use_gpu:
        x = x.cuda()
        model = model.cuda()

    with torch.no_grad():

        prob = model(x)
        prob.detach()
        if isinstance(prob, tuple):
            prob = prob[0]
        prob = torch.nn.functional.softmax(prob, dim=1).data.cpu().numpy()

    return prob


#base feature
def get_feature_dict_by_lead(SPEC_second):
    # print(np.shape(SPEC_second))

    # _delta = np.mean(SPEC_second[:,:, 1:4], axis=2)
    _delta = np.sum(SPEC_second[:, :4], axis=1)
    _theta = np.sum(SPEC_second[:, 4:8], axis=1)
    _alpha = np.sum(SPEC_second[:, 8:13], axis=1)
    _alpha_1 = np.sum(SPEC_second[:, 8:9], axis=1)
    _alpha_2 = np.sum(SPEC_second[:, 9:11], axis=1)
    _alpha_3 = np.sum(SPEC_second[:, 11:13], axis=1)
    _beta = np.sum(SPEC_second[:, 13:30], axis=1)
    _beta_1 = np.sum(SPEC_second[:, 13:20], axis=1)
    _beta_2 = np.sum(SPEC_second[:, 20:30], axis=1)
    _gamma = np.sum(SPEC_second[:, 30:70], axis=1)
    _gamma_1 = np.sum(SPEC_second[:, 30:50], axis=1)
    _gamma_2 = np.sum(SPEC_second[:, 50:70], axis=1)

    _total = _alpha + _theta + _beta + _delta + _gamma + 1e-6

    # print(np.shape(_alpha))

    _r_delta = _delta / _total
    _r_theta = _theta / _total
    _r_alpha = _alpha / _total
    _r_alpha_1 = _alpha_1 / _total
    _r_alpha_2 = _alpha_2 / _total
    _r_beta = _beta / _total
    _r_beta_1 = _beta_1 / _total
    _r_beta_2 = _beta_2 / _total
    _r_gamma = _gamma / _total
    _r_gamma_1 = _gamma_1 / _total
    _r_gamma_2 = _gamma_2 / _total

    TBR = _theta / (_beta + 1e-6)
    DAR = _delta / (_alpha + 1e-6)
    DTR = _delta / (_theta + 1e-6)
    ABR = _alpha / (_beta + 1e-6)
    ATR = _alpha / (_theta + 1e-6)
    DT_AR = (_delta + _theta) / (_alpha + 1e-6)
    DT_total_R = (_delta + _theta) / _total

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
        "DT_total_R": DT_total_R,

        "delta_log1p": np.log1p(_delta),
        "theta_log1p": np.log1p(_theta),
        "alpha_log1p": np.log1p(_alpha),
        "alpha_1_log1p": np.log1p(_alpha_1),
        "alpha_2_log1p": np.log1p(_alpha_2),
        "alpha_3_log1p": np.log1p(_alpha_3),
        "beta_log1p": np.log1p(_beta),
        "beta_1_log1p": np.log1p(_beta_1),
        "beta_2_log1p": np.log1p(_beta_2),
        "gamma_log1p": np.log1p(_gamma),
        "gamma_1_log1p": np.log1p(_gamma_1),
        "gamma_2_log1p": np.log1p(_gamma_2),
    }

    out_dict = {}
    for k, v in spec_features_dict.items():
        clean_arr = v[~np.isnan(v)]
        out_dict[k] = clean_arr
        # print(k,np.isnan(v).sum(),np.shape(v),np.shape(clean_arr))

    return out_dict

#alpha
def process_alpha(lead_name, threshold, prefix, prob_dict, montage_dict, process_dict, fs):
    """
    处理单个导联信号并提取 Alpha 波特征。

    :param lead_name: 导联名称
    :param threshold: 概率阈值 (如 0.60 或 0.70)
    :param prefix: 字典键名前缀 (如 "A__", "B__", "E__")
    :param prob_dict: 包含概率的字典
    :param montage_dict: 包含信号数据的字典
    :param process_dict: 存储结果的字典
    :param fs: 采样率
    """
    # 1. 提取概率并筛选满足阈值的索引, index为2的时候，是alpha节律
    _one_lead_alpha_prob = prob_dict[lead_name][:, 2]
    indices = np.argwhere(_one_lead_alpha_prob > threshold).flatten()




    # 2. 循环处理每个满足条件的片段
    for _idx in indices:
        _alpha_signal = montage_dict[lead_name][_idx * 128: (_idx + 2) * 128]
        pp_value = get_pp_value(_alpha_signal)

        # 功率谱密度计算
        freqs, psds_alpha = signal.welch(_alpha_signal, fs=fs, window='hann', nperseg=256)

        f_int = np.arange(6, 13)
        max_power_index = np.argmax(psds_alpha[6:13])
        max_frequency = f_int[max_power_index]

        power_30 = np.sum(psds_alpha[:30]) + 10e-5
        power_alpha = np.sum(psds_alpha[6:13]) + 10e-5

        # 3. 条件判断与结果追加
        if 6 <= max_frequency < 13 and power_alpha / power_30 > 0.45:

            # print(max_frequency,power_alpha / power_30,pp_value)

            process_dict[f"{prefix}core_HZ"].append(max_frequency)
            process_dict[f"{prefix}core_power_ratio"].append(power_alpha / power_30)
            process_dict[f"{prefix}pp_value"].append(pp_value)


#f3478
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

def generate_age_epoch_tag(age, epoch_length):
    """
    根据输入的 age 和 epoch_length，自动划分区间并输出标准格式标签
    """
    # 1. 限制 epoch length 必须在合法范围内
    valid_lengths = [2, 5, 10, 15]
    # if epoch_length not in valid_lengths:
    #     raise ValueError(f"错误的 epoch length，必须是 {valid_lengths} 之一")

    # 2. 判断年龄所属区间
    if 0 <= age <= 6:
        start, end = 0, 6
    elif 7 <= age <= 13:
        start, end = 7, 13
    elif 14 <= age <= 18:
        start, end = 14, 18
    elif 19 <= age <= 44:
        start, end = 19, 44
    elif 45 <= age <= 59:
        start, end = 45, 59
    elif 60 <= age <= 80:
        start, end = 60, 80
    elif age > 80:
        start, end = 80, 100  # 80以上默认上限设为100
    else:
        raise ValueError("输入的年龄不能为负数")

    # 3. 拼接并返回字符串
    return f"age_{start}_{end}_{epoch_length}"

def get_psy_zscore(in_eeg_dict, algorithm_para_dict_quantify, result_verbose):
    # 校验
    # 获取montage_data
    # 获取Prob结果
    # 获取Spec结果
    # 计算zscore
    # 返回

    sensitivity = algorithm_para_dict_quantify["sensitiviy"]
    lead_type = algorithm_para_dict_quantify["lead_type"]
    raw_sample_rate = int(algorithm_para_dict_quantify["sample_rate"])
    signal_length = int(algorithm_para_dict_quantify["eeg_length"])
    epoch_len_sec = int(algorithm_para_dict_quantify["process_length"])
    cur_age = int(algorithm_para_dict_quantify["age"])

 
    age_str = generate_age_epoch_tag(cur_age,epoch_len_sec)+ "_normal_ref_0625_sum"
    # 检查参
    if signal_length < 59 * raw_sample_rate: return {}
    if lead_type not in ["EBA"]: return {}


    leads_montage_dict = get_montage_data_from_dict(in_eeg_dict, lead_type)
    leads_montage_dict_raw = get_montage_data_from_dict(in_eeg_dict, "EBA_RAW")

    nperseg_len = 1
    fs = 256
    art_th = 0.6
    signal_second_length = int(np.shape(leads_montage_dict['Fp1-A1'])[0] / fs)
    duration_sec = signal_second_length
    epoch_count = int(duration_sec / epoch_len_sec)



    g_process_dict = {}
    e_process_dict = {}
    alpha_process_dict = {k: [] for k in core_alpha_list}
    f3478_process_dict = {k: [] for k in F3478_lead_list}
    for _lead_name in Total_lead_name_list:
        for _feature in spec_features_list:
            _key = _lead_name + '__' + _feature
            g_process_dict[_key] = []
            e_process_dict[_key] = []



    F3478_PSD_dict = {}
    F3478_PSD_no_ART_dict = {}
    prob_dict_with_softmax = {}
    #获取Prob_array
    for lead_name, signal_array in leads_montage_dict.items():
        second_len = int(np.shape(signal_array)[0] / 256)
        lead_x = signal_array[:second_len * fs].reshape(second_len, fs)
        lead_x = lead_x[:, np.newaxis, ::]

        lead_x_128 = signal_array[128:second_len * fs - 128].reshape(second_len - 1, fs)
        lead_x_128 = lead_x_128[:, np.newaxis, ::]

        _x_probs = get_prediction(net_BAA_256, lead_x)
        _x_probs_128 = get_prediction(net_BAA_256, lead_x_128)

        out_prob_array = np.zeros((second_len * 2 , 3))
        out_prob_array[0::2, :] = _x_probs
        out_prob_array[1:-1:2, :] = _x_probs_128
        # rob_dict[lead_name] = out_prob_array
        prob_dict_with_softmax[lead_name] = out_prob_array





    for lead_name, one_signal in leads_montage_dict.items():

        one_signal_raw = leads_montage_dict_raw[lead_name]

        one_signal_reshape = one_signal_raw[0:epoch_count * epoch_len_sec * fs].reshape(epoch_count, epoch_len_sec * fs)
        freqs, psds = signal.welch(one_signal_reshape, fs=fs, window='hann', nperseg=fs * nperseg_len)

        _prob = prob_dict_with_softmax[lead_name]
        if epoch_count * epoch_len_sec != np.shape(_prob)[0] * 2:
            _prob = np.pad(_prob, ((0, 1), (0, 0)), mode='constant', constant_values=0)

        #index为1，为artifact


        BKG_prob_index = _prob[:, 0]
        art_prob_index = _prob[:, 1]
        ALPHA_prob_index = _prob[:, 2]


        prob_len = epoch_count * epoch_len_sec * 2
        # print(prob_len, epoch_count, epoch_len_sec, signal_second_length)
        mean_art_prob = np.max(art_prob_index[:prob_len].reshape(epoch_count, epoch_len_sec * 2), axis=1)
        mask = (mean_art_prob < art_th).reshape(-1, 1)
        psds_without_art = np.where(mask, psds, np.nan)

        if lead_name in F3478_name_list:
            F3478_PSD_dict[lead_name] = psds
            F3478_PSD_no_ART_dict[lead_name] = psds_without_art


        g_feature_dict = get_feature_dict_by_lead(psds)

        e_feature_dict = get_feature_dict_by_lead(psds_without_art)

        extended_list = [item for item in g_feature_dict["relative_alpha"] for _ in range(20)]
        extended_list.append(0)
        # if lead_name == "O2-A2":
        #     time_axis = np.arange(0, len(BKG_prob_index) * 0.5, 0.5)
        #     plt.plot(time_axis, BKG_prob_index, "b")
        #     plt.plot(time_axis, art_prob_index, "r")
        #     plt.plot(time_axis, ALPHA_prob_index, "g")
        #     plt.plot(time_axis, extended_list, "c")
        #     # plt.plot(e_process_dict["C3-P3__DAR"],"g")
        #     plt.show()
        #
        for _k, _v in g_feature_dict.items():
            _name = lead_name + "__" + _k
            g_process_dict[_name].extend(_v)

        for _k, _v in e_feature_dict.items():
            _name = lead_name + "__" + _k
            e_process_dict[_name].extend(_v)

        # 计算alpha 相关的参数
        if lead_name in avg_keys:
            process_alpha(lead_name, 0.60, "A__", prob_dict_with_softmax, leads_montage_dict,
                          alpha_process_dict, fs)

        elif lead_name in bipolar_keys:  # 如果 lead_name 互斥，建议用 elif 提高效率
            process_alpha(lead_name, 0.70, "B__", prob_dict_with_softmax, leads_montage_dict,
                          alpha_process_dict, fs)
        elif lead_name in ear_keys:
            process_alpha(lead_name, 0.70, "E__", prob_dict_with_softmax, leads_montage_dict,
                          alpha_process_dict, fs)

    # 获取F3478的参数
    E_F3478_feature_dict = calculate_eeg_asymmetry_dict(F3478_PSD_no_ART_dict, "E")
    G_F3478_feature_dict = calculate_eeg_asymmetry_dict(F3478_PSD_dict, "G")

    # if lead_name == "O2-A2":
    #     time_axis = np.arange(0, len(BKG_prob_index) * 0.5, 0.5)
    #     plt.plot(time_axis, BKG_prob_index, "b")
    #     plt.plot(time_axis, art_prob_index, "r")
    #     plt.plot(time_axis, ALPHA_prob_index, "g")
    #     # plt.plot(e_process_dict["C3-P3__DAR"],"g")
    #     plt.show()



    for k, v in E_F3478_feature_dict.items():
        clean_arr = v[~np.isnan(v)]
        f3478_process_dict[k].extend(clean_arr)

    for k, v in G_F3478_feature_dict.items():
        f3478_process_dict[k].extend(v)

    normal_g_info = normal_stat_dict[age_str]["g_info"]
    normal_e_info = normal_stat_dict[age_str]["e_info"]
    normal_alpha_info = normal_stat_dict[age_str]["alpha_info"]
    normal_f3478_info = normal_stat_dict[age_str]["f3478_info"]



    out_dict = {}
    ##处理 G,E,A,F
    for k, v in g_process_dict.items():
        out_dict[k] = [calculate_trimmed_stats(g_process_dict[k]),normal_g_info[k]]



    for k, v in e_process_dict.items():
        if len(v)<8:
            out_dict[k] = [0, normal_e_info[k]]
        else:
            out_dict[k] = [calculate_trimmed_stats(e_process_dict[k]), normal_e_info[k]]


    for k,v in E_F3478_feature_dict.items():
        clean_v = v[~np.isnan(v)].tolist()
        if len(clean_v) < 8:
            out_dict[k] = [0, normal_f3478_info[k]]
        else:
            out_dict[k] = [calculate_trimmed_stats(clean_v), normal_f3478_info[k]]

    for k, v in G_F3478_feature_dict.items():
        clean_v = v[~np.isnan(v)].tolist()
        out_dict[k] = [calculate_trimmed_stats(clean_v), normal_f3478_info[k]]


    for k, v in alpha_process_dict.items():
        if len(v)<8:
            out_dict[k] = [0, normal_alpha_info[k]]
        else:
            out_dict[k] = [calculate_trimmed_stats(v), normal_alpha_info[k]]


    for k,v in out_dict.items():
        print(k,v)


    return out_dict



if __name__ == '__main__':

    import glob

    fif_path_list = glob.glob(r"J:\NORMAL\11\eeg_save_dir\21104564\1\*raw.fif")

    last_time_list = []
    sensitivity = 90
    # 1:背景 2:发作 3.重症[BS] 4.睡眠 5.伪差

    algorithm_para_dict = {
        "sensitiviy": "80",
        "sample_rate": "256",
        "age": 20,
        "process_length": 10  # 2,5,10,15
    }
    result_verbose = [2]

    result_verbose = [1, 2, 3, 4]
    process_len = 0

    total_time = 0

    ng_count = 0
    ng_long_count = 0
    long_ng_list = []
    TP = 0.
    FP = 0.
    T = 0.

    for i, one_fif in enumerate(fif_path_list[:]):
        print(one_fif)

        line_li = one_fif.replace('[', '').replace(']', '').replace("\\\\", "/").split(',')

        ng_long_count = 0
        # try:
        seg_len = 60 * 10
        # 如果等于1，则测试SINGLE
        TEST_SINGLE_MUL = 0
        # 如果等于1，测试8通道
        TEST_EIGHT_MUL = 0


        LEAD_TYPE = '21'
        algorithm_para_dict["lead_type"] = "EBA"

        one_eeg_dict = get_lead_dict_fif(one_fif,LEAD_TYPE,seg_len)

        one_lead_eeg = random.choice(list(one_eeg_dict.values()))

        algorithm_para_dict["eeg_length"] = np.shape(one_lead_eeg)[0]
        # print("-----------", np.shape(one_lead_eeg)[0] / 256)
        one_psy_zscore_dict = get_psy_zscore(one_eeg_dict, algorithm_para_dict, result_verbose)






