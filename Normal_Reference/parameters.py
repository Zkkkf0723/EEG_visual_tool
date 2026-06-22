import json
import torch
import numpy as np



ear_keys = [
    "Fp1-A1",
    "Fp2-A2",
    "F3-A1",
    "F4-A2",
    "C3-A1",
    "C4-A2",
    "P3-A1",
    "P4-A2",
    "O1-A1",
    "O2-A2",
    "F7-A1",
    "F8-A2",
    "T3-A1",
    "T4-A2",
    "T5-A1",
    "T6-A2",
]

#
bipolar_keys = [
    "Fp1-F3",
    "Fp2-F4",
    "F3-C3",
    "F4-C4",
    "C3-P3",
    "C4-P4",
    "P3-O1",
    "P4-O2",
    "Fp1-F7",
    "Fp2-F8",
    "F7-T3",
    "F8-T4",
    "T3-T5",
    "T4-T6",
    "T5-O1",
    "T6-O2",
    "Fpz-Fz",
    "Fz-Pz",
    "Cz-Pz",
    "Pz-Oz"
]
#
#
avg_keys = [

    "Fp1-AVG",
    "Fp2-AVG",
    "F3-AVG",
    "F4-AVG",
    "C3-AVG",
    "C4-AVG",
    "P3-AVG",
    "P4-AVG",
    "O1-AVG",
    "O2-AVG",
    "F7-AVG",
    "F8-AVG",
    "T3-AVG",
    "T4-AVG",
    "T5-AVG",
    "T6-AVG",
    "Fpz-AVG",
    "Fz-AVG",
    "Cz-AVG",
    "Pz-AVG",
    "Oz-AVG",
]

Total_lead_name_list = ear_keys + bipolar_keys + avg_keys

spec_features_list = [
    "delta",
    "theta",
    "alpha",
    "alpha_1",
    "alpha_2",
    "alpha_3",
    "beta",
    "beta_1",
    "beta_2",
    "gamma",
    "gamma_1",
    "gamma_2",
    "relative_delta",
    "relative_theta",
    "relative_alpha",
    "relative_beta",
    "relative_gamma",
    "TBR",
    "DAR",
    "DTR",
    "ABR",
    "ATR",
    "DT_AR",
    "DT_total_R",
]

core_alpha_list = [
    "B__core_HZ",
    "B__core_power_ratio",
    "B__pp_value",
    "E__core_HZ",
    "E__core_power_ratio",
    "E__pp_value",
    "A__core_HZ",
    "A__core_power_ratio",
    "A__pp_value",
]

F3478_list = [
    "B_F7_F8_G",
    "B_F3_F4_G",
    "B_F7_F8_E",
    "B_F3_F4_E",

    "E_F7_F8_G",
    "E_F3_F4_G",
    "E_F7_F8_E",
    "E_F3_F4_E",

    "A_F7_F8_G",
    "A_F3_F4_G",
    "A_F7_F8_E",
    "A_F3_F4_E",
]

F3478_lead_list = [
    "E__F3_F4_E",
    "E__F7_F8_E",
    "B__F3_F4_E",
    "B__F7_F8_E",
    "A__F3_F4_E",
    "A__F7_F8_E",

    "E__F3_F4_G",
    "E__F7_F8_G",
    "B__F3_F4_G",
    "B__F7_F8_G",
    "A__F3_F4_G",
    "A__F7_F8_G",
]

F3478_name_list = [
    "F3-A1",
    "F4-A2",
    "F7-A1",
    "F8-A2",
    "F3-C3",
    "F4-C4",
    "F7-T3",
    "F8-T4",
    "F3-AVG",
    "F4-AVG",
    "F7-AVG",
    "F8-AVG",
]



#滤波器
from scipy.signal import butter, lfilter
from scipy.signal import filtfilt, iirnotch, freqz, butter
def butter_bandpass(low_cut, high_cut, fs, order=5):
    nyq = 0.5 * fs
    low = low_cut / nyq
    high = high_cut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, low_cut=0.8, high_cut=35, fs=256, order=5):
    b, a = butter_bandpass(low_cut=low_cut, high_cut= high_cut, fs= fs, order=order)
    y = lfilter(b, a, data)
    return y


def norch_50(in_signal):
    fs = 256.0  # Sample frequency (Hz)
    f0 = 50.0  # Frequency to be removed from signal (Hz)
    Q = 20.0  # Quality factor
    # Design notch filter
    w0 = f0 / (fs / 2)
    b, a = iirnotch(w0, Q)
    y_filt = filtfilt(b, a, in_signal)
    #w, h = freqz(b, a)
    #filt_freq = w * fs / (2 * np.pi)
    return y_filt


def get_montage_data_from_dict(lead_dict, lead_type):

    array_list = []
    for k,v in lead_dict.items():
        array_list.append(v)
    lead_AVG = np.mean(array_list,axis=0)

    if lead_type == 'SINGLE':
        k = list(lead_dict.keys())[0]
        out_dict = {
            "SINGLE": butter_bandpass_filter(lead_dict[k], low_cut=0.8, high_cut=35, fs=256)
        }



    elif lead_type == "EBA":
        out_dict = {
            "Fp1-A1": butter_bandpass_filter(lead_dict["Fp1"] - lead_dict["A1"], low_cut=0.8, high_cut=35, fs=256),
            "Fp2-A2": butter_bandpass_filter(lead_dict["Fp2"] - lead_dict["A2"], low_cut=0.8, high_cut=35, fs=256),
            "F3-A1": butter_bandpass_filter(lead_dict["F3"] - lead_dict["A1"], low_cut=0.8, high_cut=35, fs=256),
            "F4-A2": butter_bandpass_filter(lead_dict["F4"] - lead_dict["A2"], low_cut=0.8, high_cut=35, fs=256),
            "C3-A1": butter_bandpass_filter(lead_dict["C3"] - lead_dict["A1"], low_cut=0.8, high_cut=35, fs=256),
            "C4-A2": butter_bandpass_filter(lead_dict["C4"] - lead_dict["A2"], low_cut=0.8, high_cut=35, fs=256),
            "P3-A1": butter_bandpass_filter(lead_dict["P3"] - lead_dict["A1"], low_cut=0.8, high_cut=35, fs=256),
            "P4-A2": butter_bandpass_filter(lead_dict["P4"] - lead_dict["A2"], low_cut=0.8, high_cut=35, fs=256),
            "O1-A1": butter_bandpass_filter(lead_dict["O1"] - lead_dict["A1"], low_cut=0.8, high_cut=35, fs=256),
            "O2-A2": butter_bandpass_filter(lead_dict["O2"] - lead_dict["A2"], low_cut=0.8, high_cut=35, fs=256),
            "F7-A1": butter_bandpass_filter(lead_dict["F7"] - lead_dict["A1"], low_cut=0.8, high_cut=35, fs=256),
            "F8-A2": butter_bandpass_filter(lead_dict["F8"] - lead_dict["A2"], low_cut=0.8, high_cut=35, fs=256),
            "T3-A1": butter_bandpass_filter(lead_dict["T3"] - lead_dict["A1"], low_cut=0.8, high_cut=35, fs=256),
            "T4-A2": butter_bandpass_filter(lead_dict["T4"] - lead_dict["A2"], low_cut=0.8, high_cut=35, fs=256),
            "T5-A1": butter_bandpass_filter(lead_dict["T5"] - lead_dict["A1"], low_cut=0.8, high_cut=35, fs=256),
            "T6-A2": butter_bandpass_filter(lead_dict["T6"] - lead_dict["A2"], low_cut=0.8, high_cut=35, fs=256),

            "Fp1-F3": butter_bandpass_filter(lead_dict["Fp1"] - lead_dict["F3"], low_cut=0.8, high_cut=35, fs=256),
            "Fp2-F4": butter_bandpass_filter(lead_dict["Fp2"] - lead_dict["F4"], low_cut=0.8, high_cut=35, fs=256),
            "F3-C3": butter_bandpass_filter(lead_dict["F3"] - lead_dict["C3"], low_cut=0.8, high_cut=35, fs=256),
            "F4-C4": butter_bandpass_filter(lead_dict["F4"] - lead_dict["C4"], low_cut=0.8, high_cut=35, fs=256),
            "C3-P3": butter_bandpass_filter(lead_dict["C3"] - lead_dict["P3"], low_cut=0.8, high_cut=35, fs=256),
            "C4-P4": butter_bandpass_filter(lead_dict["C4"] - lead_dict["P4"], low_cut=0.8, high_cut=35, fs=256),
            "P3-O1": butter_bandpass_filter(lead_dict["P3"] - lead_dict["O1"], low_cut=0.8, high_cut=35, fs=256),
            "P4-O2": butter_bandpass_filter(lead_dict["P4"] - lead_dict["O2"], low_cut=0.8, high_cut=35, fs=256),
            "Fp1-F7": butter_bandpass_filter(lead_dict["Fp1"] - lead_dict["F7"], low_cut=0.8, high_cut=35, fs=256),
            "Fp2-F8": butter_bandpass_filter(lead_dict["Fp2"] - lead_dict["F8"], low_cut=0.8, high_cut=35, fs=256),
            "F7-T3": butter_bandpass_filter(lead_dict["F7"] - lead_dict["T3"], low_cut=0.8, high_cut=35, fs=256),
            "F8-T4": butter_bandpass_filter(lead_dict["F8"] - lead_dict["T4"], low_cut=0.8, high_cut=35, fs=256),
            "T3-T5": butter_bandpass_filter(lead_dict["T3"] - lead_dict["T5"], low_cut=0.8, high_cut=35, fs=256),
            "T4-T6": butter_bandpass_filter(lead_dict["T4"] - lead_dict["T6"], low_cut=0.8, high_cut=35, fs=256),
            "T5-O1": butter_bandpass_filter(lead_dict["T5"] - lead_dict["O1"], low_cut=0.8, high_cut=35, fs=256),
            "T6-O2": butter_bandpass_filter(lead_dict["T6"] - lead_dict["O2"], low_cut=0.8, high_cut=35, fs=256),

            "Fpz-Fz": butter_bandpass_filter(lead_dict["Fpz"] - lead_dict["Fz"], low_cut=0.8, high_cut=35, fs=256),
            "Fz-Pz": butter_bandpass_filter(lead_dict["Fz"] - lead_dict["Cz"], low_cut=0.8, high_cut=35, fs=256),
            "Cz-Pz": butter_bandpass_filter(lead_dict["Cz"] - lead_dict["Pz"], low_cut=0.8, high_cut=35, fs=256),
            "Pz-Oz": butter_bandpass_filter(lead_dict["Pz"] - lead_dict["Oz"], low_cut=0.8, high_cut=35, fs=256),

            "Fp1-AVG": butter_bandpass_filter(lead_dict["Fp1"] - lead_AVG, low_cut=0.8, high_cut=35, fs=256),
            "Fp2-AVG": butter_bandpass_filter(lead_dict["Fp2"] - lead_AVG, low_cut=0.8, high_cut=35, fs=256),
            "F3-AVG": butter_bandpass_filter(lead_dict["F3"] - lead_AVG, low_cut=0.8, high_cut=35, fs=256),
            "F4-AVG": butter_bandpass_filter(lead_dict["F4"] - lead_AVG, low_cut=0.8, high_cut=35, fs=256),
            "C3-AVG": butter_bandpass_filter(lead_dict["C3"] - lead_AVG, low_cut=0.8, high_cut=35, fs=256),
            "C4-AVG": butter_bandpass_filter(lead_dict["C4"] - lead_AVG, low_cut=0.8, high_cut=35, fs=256),
            "P3-AVG": butter_bandpass_filter(lead_dict["P3"] - lead_AVG, low_cut=0.8, high_cut=35, fs=256),
            "P4-AVG": butter_bandpass_filter(lead_dict["P4"] - lead_AVG, low_cut=0.8, high_cut=35, fs=256),
            "O1-AVG": butter_bandpass_filter(lead_dict["O1"] - lead_AVG, low_cut=0.8, high_cut=35, fs=256),
            "O2-AVG": butter_bandpass_filter(lead_dict["O2"] - lead_AVG, low_cut=0.8, high_cut=35, fs=256),
            "F7-AVG": butter_bandpass_filter(lead_dict["F7"] - lead_AVG, low_cut=0.8, high_cut=35, fs=256),
            "F8-AVG": butter_bandpass_filter(lead_dict["F8"] - lead_AVG, low_cut=0.8, high_cut=35, fs=256),
            "T3-AVG": butter_bandpass_filter(lead_dict["T3"] - lead_AVG, low_cut=0.8, high_cut=35, fs=256),
            "T4-AVG": butter_bandpass_filter(lead_dict["T4"] - lead_AVG, low_cut=0.8, high_cut=35, fs=256),
            "T5-AVG": butter_bandpass_filter(lead_dict["T5"] - lead_AVG, low_cut=0.8, high_cut=35, fs=256),
            "T6-AVG": butter_bandpass_filter(lead_dict["T6"] - lead_AVG, low_cut=0.8, high_cut=35, fs=256),

            "Fpz-AVG": butter_bandpass_filter(lead_dict["Fpz"] - lead_AVG, low_cut=0.8, high_cut=35, fs=256),
            "Fz-AVG": butter_bandpass_filter(lead_dict["Fz"] - lead_AVG, low_cut=0.8, high_cut=35, fs=256),
            "Cz-AVG": butter_bandpass_filter(lead_dict["Cz"] - lead_AVG, low_cut=0.8, high_cut=35, fs=256),
            "Pz-AVG": butter_bandpass_filter(lead_dict["Pz"] - lead_AVG, low_cut=0.8, high_cut=35, fs=256),
            "Oz-AVG": butter_bandpass_filter(lead_dict["Oz"] - lead_AVG, low_cut=0.8, high_cut=35, fs=256),

        }
    elif lead_type == "EBA_RAW":

        low_c = 0.5
        high_c = 70
        out_dict = {
            "Fp1-A1": butter_bandpass_filter(lead_dict["Fp1"] - lead_dict["A1"], low_c, high_c, fs=256),
            "Fp2-A2": butter_bandpass_filter(lead_dict["Fp2"] - lead_dict["A2"], low_c, high_c, fs=256),
            "F3-A1": butter_bandpass_filter(lead_dict["F3"] - lead_dict["A1"], low_c, high_c, fs=256),
            "F4-A2": butter_bandpass_filter(lead_dict["F4"] - lead_dict["A2"], low_c, high_c, fs=256),
            "C3-A1": butter_bandpass_filter(lead_dict["C3"] - lead_dict["A1"], low_c, high_c, fs=256),
            "C4-A2": butter_bandpass_filter(lead_dict["C4"] - lead_dict["A2"], low_c, high_c, fs=256),
            "P3-A1": butter_bandpass_filter(lead_dict["P3"] - lead_dict["A1"], low_c, high_c, fs=256),
            "P4-A2": butter_bandpass_filter(lead_dict["P4"] - lead_dict["A2"], low_c, high_c, fs=256),
            "O1-A1": butter_bandpass_filter(lead_dict["O1"] - lead_dict["A1"], low_c, high_c, fs=256),
            "O2-A2": butter_bandpass_filter(lead_dict["O2"] - lead_dict["A2"], low_c, high_c, fs=256),
            "F7-A1": butter_bandpass_filter(lead_dict["F7"] - lead_dict["A1"], low_c, high_c, fs=256),
            "F8-A2": butter_bandpass_filter(lead_dict["F8"] - lead_dict["A2"], low_c, high_c, fs=256),
            "T3-A1": butter_bandpass_filter(lead_dict["T3"] - lead_dict["A1"], low_c, high_c, fs=256),
            "T4-A2": butter_bandpass_filter(lead_dict["T4"] - lead_dict["A2"], low_c, high_c, fs=256),
            "T5-A1": butter_bandpass_filter(lead_dict["T5"] - lead_dict["A1"], low_c, high_c, fs=256),
            "T6-A2": butter_bandpass_filter(lead_dict["T6"] - lead_dict["A2"], low_c, high_c, fs=256),

            "Fp1-F3": butter_bandpass_filter(lead_dict["Fp1"] - lead_dict["F3"], low_c, high_c, fs=256),
            "Fp2-F4": butter_bandpass_filter(lead_dict["Fp2"] - lead_dict["F4"], low_c, high_c, fs=256),
            "F3-C3": butter_bandpass_filter(lead_dict["F3"] - lead_dict["C3"], low_c, high_c, fs=256),
            "F4-C4": butter_bandpass_filter(lead_dict["F4"] - lead_dict["C4"], low_c, high_c, fs=256),
            "C3-P3": butter_bandpass_filter(lead_dict["C3"] - lead_dict["P3"], low_c, high_c, fs=256),
            "C4-P4": butter_bandpass_filter(lead_dict["C4"] - lead_dict["P4"], low_c, high_c, fs=256),
            "P3-O1": butter_bandpass_filter(lead_dict["P3"] - lead_dict["O1"], low_c, high_c, fs=256),
            "P4-O2": butter_bandpass_filter(lead_dict["P4"] - lead_dict["O2"], low_c, high_c, fs=256),
            "Fp1-F7": butter_bandpass_filter(lead_dict["Fp1"] - lead_dict["F7"], low_c, high_c, fs=256),
            "Fp2-F8": butter_bandpass_filter(lead_dict["Fp2"] - lead_dict["F8"], low_c, high_c, fs=256),
            "F7-T3": butter_bandpass_filter(lead_dict["F7"] - lead_dict["T3"], low_c, high_c, fs=256),
            "F8-T4": butter_bandpass_filter(lead_dict["F8"] - lead_dict["T4"], low_c, high_c, fs=256),
            "T3-T5": butter_bandpass_filter(lead_dict["T3"] - lead_dict["T5"], low_c, high_c, fs=256),
            "T4-T6": butter_bandpass_filter(lead_dict["T4"] - lead_dict["T6"], low_c, high_c, fs=256),
            "T5-O1": butter_bandpass_filter(lead_dict["T5"] - lead_dict["O1"], low_c, high_c, fs=256),
            "T6-O2": butter_bandpass_filter(lead_dict["T6"] - lead_dict["O2"], low_c, high_c, fs=256),

            "Fpz-Fz": butter_bandpass_filter(lead_dict["Fpz"] - lead_dict["Fz"], low_c, high_c, fs=256),
            "Fz-Pz": butter_bandpass_filter(lead_dict["Fz"] - lead_dict["Cz"], low_c, high_c, fs=256),
            "Cz-Pz": butter_bandpass_filter(lead_dict["Cz"] - lead_dict["Pz"], low_c, high_c, fs=256),
            "Pz-Oz": butter_bandpass_filter(lead_dict["Pz"] - lead_dict["Oz"], low_c, high_c, fs=256),

            "Fp1-AVG": butter_bandpass_filter(lead_dict["Fp1"] - lead_AVG, low_c, high_c, fs=256),
            "Fp2-AVG": butter_bandpass_filter(lead_dict["Fp2"] - lead_AVG, low_c, high_c, fs=256),
            "F3-AVG": butter_bandpass_filter(lead_dict["F3"] - lead_AVG, low_c, high_c, fs=256),
            "F4-AVG": butter_bandpass_filter(lead_dict["F4"] - lead_AVG, low_c, high_c, fs=256),
            "C3-AVG": butter_bandpass_filter(lead_dict["C3"] - lead_AVG, low_c, high_c, fs=256),
            "C4-AVG": butter_bandpass_filter(lead_dict["C4"] - lead_AVG, low_c, high_c, fs=256),
            "P3-AVG": butter_bandpass_filter(lead_dict["P3"] - lead_AVG, low_c, high_c, fs=256),
            "P4-AVG": butter_bandpass_filter(lead_dict["P4"] - lead_AVG, low_c, high_c, fs=256),
            "O1-AVG": butter_bandpass_filter(lead_dict["O1"] - lead_AVG, low_c, high_c, fs=256),
            "O2-AVG": butter_bandpass_filter(lead_dict["O2"] - lead_AVG, low_c, high_c, fs=256),
            "F7-AVG": butter_bandpass_filter(lead_dict["F7"] - lead_AVG, low_c, high_c, fs=256),
            "F8-AVG": butter_bandpass_filter(lead_dict["F8"] - lead_AVG, low_c, high_c, fs=256),
            "T3-AVG": butter_bandpass_filter(lead_dict["T3"] - lead_AVG, low_c, high_c, fs=256),
            "T4-AVG": butter_bandpass_filter(lead_dict["T4"] - lead_AVG, low_c, high_c, fs=256),
            "T5-AVG": butter_bandpass_filter(lead_dict["T5"] - lead_AVG, low_c, high_c, fs=256),
            "T6-AVG": butter_bandpass_filter(lead_dict["T6"] - lead_AVG, low_c, high_c, fs=256),

            "Fpz-AVG": butter_bandpass_filter(lead_dict["Fpz"] - lead_AVG, low_c, high_c, fs=256),
            "Fz-AVG": butter_bandpass_filter(lead_dict["Fz"] - lead_AVG, low_c, high_c, fs=256),
            "Cz-AVG": butter_bandpass_filter(lead_dict["Cz"] - lead_AVG, low_c, high_c, fs=256),
            "Pz-AVG": butter_bandpass_filter(lead_dict["Pz"] - lead_AVG, low_c, high_c, fs=256),
            "Oz-AVG": butter_bandpass_filter(lead_dict["Oz"] - lead_AVG, low_c, high_c, fs=256),

        }

    else:
        return {}

    # entry的序
    for k in out_dict.keys():
        temp_a = norch_50(np.array(out_dict[k]))
        out_dict[k] = norch_50(temp_a)

    return out_dict



#正常值的载入
#若优化，用数据库
normal_file_path = ".\\normal\\combined_result_0611.json"
with open(normal_file_path, 'r', encoding='utf-8') as f:
    normal_stat_dict = json.load(f)


#神经网络模型
use_gpu = 1

#轻量化伪差检测模型
def get_BAA_model():
    print("using BAA model")
    from base_nn import TCN

    model = TCN(1, 3, layers=5 * [16], ks=7, conv_dropout=0.3, fc_dropout=0.3)
    state_dict_byte = './arch/tcn_state_20260529_BAA'
    if use_gpu:

        state_dict = torch.load(state_dict_byte)
        model.load_state_dict(state_dict)
        model = model.cuda()
    else:
        state_dict = torch.load(state_dict_byte, map_location='cpu')
        model.load_state_dict(state_dict)
        #model = model.cuda()

    model.eval()
    return model



net_BAA_256 = get_BAA_model()






