#读取EDF
#解析EDF
#获取prob
#计算相关参数

import glob
import mne
import pickle
from a_montage_tools import *

from a_psd_stat_tool import *

import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

edf_path_list = glob.glob(".\EDF_dir\\*.edf")


for one_edf_path in edf_path_list:
    print(one_edf_path)

    #获取prob

    one_prob_path = one_edf_path.replace(".edf","_prob_230516_with_softmax_half_second_EBA.pkl")
    print(one_prob_path)

    with open(one_prob_path, 'rb') as file:
        prob_dict_with_softmax = pickle.load(file)

    PROB_array = []
    for _k in prob_dict_with_softmax.keys():
        _prob = prob_dict_with_softmax[_k]
        PROB_array.append(_prob)


    if os.path.exists(one_prob_path):
        with open(one_prob_path, 'rb') as f:
            prob_dict = pickle.load(f)

            #每0.5秒一个prob value
            for k,v in prob_dict.items():
                print(k,np.shape(prob_dict[k]))





    raw = mne.io.read_raw_edf(one_edf_path, preload=True)
    duration_sec = raw.times[-1]
    print(int(duration_sec))


    #V转成正常uV
    #调整采样率
    raw._data *= 1e6
    raw.resample(256)


    all_data = {}
    # 遍历导联名称和索引

    signal_second_length = 0

    print(raw.ch_names)


    #读取所有数据
    for i, ch_name in enumerate(raw.ch_names):
        # get_data 可以指定 picks（索引或名称）
        # 返回的是二维数组，取 [0] 变成一维
        _signal = raw.get_data(picks=ch_name)[0]
        all_data[ch_name] = _signal


    #leads_montage_dict = get_bipolar_data_caueeg(all_data,0.5,70)


    leads_montage_dict = get_montage_data_from_dict(all_data, "EBA")
    leads_montage_dict_raw = get_montage_data_from_dict(all_data, "EBA_RAW")


    print(leads_montage_dict.keys())
    print(leads_montage_dict_raw.keys())

    #设置epoch的长度

    epoch_len_sec = 5

    fs = 256
    epoch_len = epoch_len_sec*fs
    epoch_count = int(duration_sec/epoch_len_sec)

    print(epoch_count)


    PSD_dict = {}

    PSD_array = []
    PSD_array_no_art =[]

    for lead_name, one_signal in leads_montage_dict.items():

        print(lead_name)

        one_signal_reshape = one_signal[0:epoch_count*epoch_len_sec*fs].reshape(epoch_count,epoch_len_sec*fs)
        #这些参数都是可以调整
        nperseg_len = 2
        freqs, psds = signal.welch(one_signal_reshape,fs=fs,window='hann', nperseg=fs * nperseg_len)

        _prob = prob_dict_with_softmax[lead_name]

        if epoch_count * epoch_len_sec != np.shape(_prob)[0] * 2:
            _prob = np.pad(_prob, ((0, 1), (0, 0)), mode='constant', constant_values=0)
        art_prob_index = _prob[:, 2] + _prob[:, 1]

        prob_len = epoch_count * epoch_len_sec * 2
        print(prob_len, epoch_count, epoch_len_sec, signal_second_length)
        mean_art_prob = np.max(art_prob_index[:prob_len].reshape(epoch_count, epoch_len_sec * 2), axis=1)

        # print(np.shape(psds))
        # fig = plt.figure()
        # fig_spectrogram_l = fig.add_subplot(2, 1, 1)
        # fig_spectrogram_l.set_ylim(0, 31)
        # fig_spectrogram_l.imshow(trans_raw_2_show(psds.T), cmap='seismic', interpolation='gaussian')
        # plt.show()
        art_th = 0.4


        psds_without_art = psds[mean_art_prob < art_th, :]

        print(np.shape(psds_without_art))

        PSD_array_no_art.append(psds_without_art)

        PSD_array.append(psds)




        # 如果要把power换算成db，这里要用10 * np.log10(psds)
    # spec_dict = get_spec_stat_info(np.array(PSD_array))
    # for k,v in spec_dict.items():
    #     print(k,np.shape(v))

    spec_dict = get_spec_stat_info(PSD_array_no_art)
    for k,v in spec_dict.items():

        for one_v in v:
            print(np.shape(one_v))

        exit()








