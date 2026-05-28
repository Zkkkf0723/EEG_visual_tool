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

edf_path_list = glob.glob("D:\EDF_PSD_test\\*.edf")


for one_edf_path in edf_path_list:
    print(one_edf_path)

    #获取prob

    one_prob_path = one_edf_path.replace(".edf","_prob_230516_with_softmax_half_second_edf.pkl")
    print(one_prob_path)

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

    leads_montage_dict = get_bipolar_data_caueeg(all_data,0.5,70)

    print(leads_montage_dict.keys())

    #设置epoch的长度

    epoch_len_sec = 5

    fs = 256
    epoch_len = epoch_len_sec*fs
    epoch_count = int(duration_sec/epoch_len_sec)

    print(epoch_count)


    PSD_dict = {}

    for lead_name, one_signal in leads_montage_dict.items():

        print(lead_name)

        one_signal_reshape = one_signal[0:epoch_count*epoch_len_sec*fs].reshape(epoch_count,epoch_len_sec*fs)
        #这些参数都是可以调整
        nperseg_len = 2
        freqs, psds = signal.welch(one_signal_reshape,fs=fs,window='hann', nperseg=fs * nperseg_len)

        print(np.shape(psds))
        fig = plt.figure()
        fig_spectrogram_l = fig.add_subplot(2, 1, 1)
        fig_spectrogram_l.set_ylim(0, 31)
        fig_spectrogram_l.imshow(trans_raw_2_show(psds.T), cmap='seismic', interpolation='gaussian')
        plt.show()



        PSD_dict[lead_name] = 10 * np.log10(psds)

        # 如果要把power换算成db，这里要用10 * np.log10(psds)
        spec_dict = get_spec_stat_info(psds,nperseg_len)

        art_prob_index = prob_dict[lead_name][:epoch_len_sec*epoch_count*2,2]
        alpha_prob_index = prob_dict[lead_name][ :epoch_len_sec*epoch_count*2, 2]

        #这里做了简单的过滤,用TBR作为例子,阈值定得比较随意
        art_prob_array = np.mean(art_prob_index.reshape(epoch_count,epoch_len_sec*2),axis=1)

        TBR_without_art = spec_dict["TBR"][art_prob_array<0.5]

        TBR_without_mean = np.mean(TBR_without_art)

        TBR_mean = np.mean(spec_dict["TBR"])


        plt.plot(spec_dict["TBR"])
        plt.show()

        print(np.shape(TBR_without_art))
        print(np.shape(art_prob_array))

        print(TBR_without_mean,TBR_mean)




        no_art_mask = []
        alpha_mask = []


        #
        # #这里每5秒一个，一共是epoch_count个
        # for k,v in spec_dict.items():
        #     print(k,np.shape(v))
        # #
        # #
        # #
        # #
        # # print(freqs)
        #
        # exit()
        #
        #
        #
        #
        # plt.semilogy(freqs, psds[1,:], color='r', lw=1.5)
        #
        # plt.show()
        #
        # print(lead_name,np.shape(one_signal_reshape))
        # print(np.shape(psds))
        #
        # continue


    #处理F3_F4 和 F7_F8
    #这里都是先计算PSD,后提取F3,F4,F7,F8的alpha区域相加后进行比较

    F3478_dict = get_F3478(PSD_dict,nperseg_len)


    F3_F4 = F3478_dict["F3_F4"]
    F7_F8 = F3478_dict["F7_F8"]

    plt.plot(F3_F4)
    plt.plot(F7_F8)

    plt.show()




