# coding:utf8
# test
import os
import struct

import numpy as np
from scipy.signal import butter, lfilter
from scipy import signal

#ceshi
def butter_bandpass(low_cut, high_cut, fs, order=5):
    nyq = 0.5 * fs
    low = low_cut / nyq
    high = high_cut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter_leads(data, low_cut, high_cut, fs, order=5):
    channel_num = data.shape[0]
    out_leads = []

    for i in range(channel_num):
        one_lead = butter_bandpass_filter_(data[i, :], low_cut, high_cut, fs, order=order)
        out_leads.append(one_lead)

    return np.array(out_leads)


def butter_bandpass_filter_(data, low_cut=0.8, high_cut=35, fs=256, order=5):
    b, a = butter_bandpass(low_cut=low_cut, high_cut= high_cut, fs= fs, order=order)
    y = lfilter(b, a, data)
    return y


def butter_bandpass_filter(data):
    b, a = butter_bandpass(low_cut=0.8, high_cut=35, fs=256, order=5)
    y = lfilter(b, a, data)
    y = y.astype("np.float32")
    return y


# 获取一个sensor的信息
def get_one_sensor_info(in_byte):
    out_info = []

    sensor_name = bytes.decode(in_byte[0:20], encoding='utf-8', errors='ignore').split('\x00')[0]
    sensor_type = bytes.decode(in_byte[20:40], encoding='utf-8', errors='ignore')
    color_value = bytes.decode(in_byte[40:44], encoding='utf-8', errors='ignore')
    color_name = bytes.decode(in_byte[44:54], encoding='utf-8', errors='ignore').split('\x00')[0]
    transducer = bytes.decode(in_byte[54:74], encoding='utf-8', errors='ignore').split('\x00')[0]
    dimension = bytes.decode(in_byte[74:94], encoding='utf-8', errors='ignore').split('\x00')[0]
    fxpos = struct.unpack('f', in_byte[94:98])[0]
    fypos = struct.unpack('f', in_byte[98:102])[0]

    out_info.append(sensor_name)
    out_info.append(sensor_type)
    out_info.append(color_value)
    out_info.append(color_name)
    out_info.append(transducer)
    out_info.append(dimension)
    out_info.append(fxpos)
    out_info.append(fypos)

    return out_info


# 获取一个amp相关信息
def get_one_ampch_info(in_byte):
    input_name = bytes.decode(in_byte[0:20], encoding='utf-8', errors='ignore').split('\x00')[0]
    b_on = struct.unpack('i', in_byte[20:24])[0]
    b_unipiolar = struct.unpack('i', in_byte[24:28])[0]
    sensor_name = bytes.decode(in_byte[28:48], encoding='utf-8', errors='ignore').split('\x00')[0]
    sample_rate = struct.unpack('i', in_byte[48:52])[0]
    dimension = bytes.decode(in_byte[52:72], encoding='utf-8', errors='ignore').split('\x00')[0]
    impedance = struct.unpack('f', in_byte[72:])[0]

    # print(input_name,sensor_name)

    return [input_name, b_on, b_unipiolar, sensor_name, sample_rate, dimension, impedance]


# montage相关结构暂时没有完全明确
def get_one_montage_info(in_byte):
    print('-------------get_more_info')

    str_active = bytes.decode(in_byte[0:20], encoding='utf-8', errors='ignore')
    print(str_active)

    str_ref = bytes.decode(in_byte[20:40], encoding='utf-8', errors='ignore')
    print(str_ref)

    str_Label = bytes.decode(in_byte[40:70], encoding='utf-8', errors='ignore')
    print(str_Label)

    color = struct.unpack('i', in_byte[70:74])[0]

    color_name = bytes.decode(in_byte[74:84], encoding='utf-8', errors='ignore')
    print(color_name)

    display_type = struct.unpack('i', in_byte[84:88])[0]
    print('display_type', display_type)

    print('-------------get_more_info_end')

    # print(str_active,str_ref,str_Label,color,color_name,display_type)
    b_item_on = struct.unpack('i', in_byte[88:92])[0]
    b_down_polarity = struct.unpack('i', in_byte[92:96])[0]
    b_special = struct.unpack('i', in_byte[96:100])[0]
    b_audio = struct.unpack('i', in_byte[100:104])[0]
    i_sensitivity = struct.unpack('i', in_byte[104:108])[0]
    blocked = struct.unpack('i', in_byte[108:112])[0]
    low_cut = struct.unpack('f', in_byte[112:116])[0]
    high_cut = struct.unpack('i', in_byte[116:120])[0]
    b_re = struct.unpack('i', in_byte[120:124])[0]
    n_re = struct.unpack('i', in_byte[124:128])[0]

    print(b_item_on, b_down_polarity, b_special, b_audio, i_sensitivity, blocked, low_cut, high_cut, b_re, n_re)

    return


# 获取整个EEG的长度
def pre_eeg_pack_len(input_len):
    a = 256 * 3600
    e_pack_len = 100
    if 0 != input_len % 100:
        e_pack_len = 98

    pre_len = int(input_len / e_pack_len)
    if pre_len >= a:
        pre_len = a

    return int(pre_len), int(e_pack_len)


def trans_round(in_array):
    tep = list(in_array)
    tep = [round(x * 1000000, 1) * -1 for x in tep]
    return tep


def get_mark(eeg_path):
    patient_number = eeg_path.split(os.sep)[-3]
    number = os.path.basename(eeg_path)[:-4]
    mark = '{}_{}'.format(patient_number, number)
    return mark


# 获取所有的原始数据
def get_all_lead_old(file_path):
    with open(file_path, 'rb') as f:
        # head_length
        file_size = os.path.getsize(file_path)
        buf = f.read()
        f.seek(4)

        bi = buf[:4]
        head_len = struct.unpack('i', bi)[0]

        pack_num, e_pack_len = pre_eeg_pack_len(file_size - head_len)

        eeg_data_b = buf[head_len:]
        # 减去3为了不越界
        p = []


        for i in range(pack_num):
            one_pack = eeg_data_b[i * e_pack_len:(i + 1) * e_pack_len]
            # 直接将相关数据映射到内存
            out_data = np.frombuffer(one_pack[30:], dtype=np.dtype('h'), count=34)

            p.append(out_data)

        #p_array_32 = np.array(p,dtype=np.float32).T
        p_array = np.array(p).T


        # exit()


        # np.save(npy_path, -p_array)
        # 取负的原因是与设备商软件保持一致
        return -p_array


# 获取整个EEG的长度
def get_eeg_pack_len(input_len):
    total_len = 256 * 3600
    one_pack_binary_len = 100
    if 0 != input_len % 100:
        one_pack_binary_len = 98

    pre_len = int(input_len / one_pack_binary_len)
    if pre_len >= total_len:
        pre_len = total_len

    return int(pre_len), int(one_pack_binary_len)



def get_all_lead(file_path):

    if not os.path.exists(file_path):
        return np.array([])

    with open(file_path, 'rb') as f:
        # head_length
        file_binary_size = os.path.getsize(file_path)
        buf = f.read()
        head_binary = buf[:4]
        head_size = struct.unpack('i', head_binary)[0]
        pack_num, e_pack_len = get_eeg_pack_len(file_binary_size - head_size)
        #print(pack_num, e_pack_len)
        eeg_data_binary = buf[head_size:]
        epoch_list = []
        for i in range(pack_num):
            one_pack = eeg_data_binary[i * e_pack_len:(i + 1) * e_pack_len]
            out_data = np.frombuffer(one_pack[30:], dtype=np.dtype('h'), count=34)
            epoch_list.append(out_data)

        p_array = np.array(epoch_list).T

        return -p_array

def get_lead_dict(eeg_path,lead_type):
    lead_34_data = get_all_lead(eeg_path)
    data_len = np.shape(lead_34_data[22,:])
    fake_zero = np.zeros(data_len)
    #lead_34_data = lead_34_data[:,:]
    if lead_type == '1':
        out_dict = {
           'SIGNAL':lead_34_data[10, :]- lead_34_data[22, :],
        }


    if lead_type == '8':
        out_dict = {
            "A1": lead_34_data[22, :],
            "A2": lead_34_data[23, :],
            "Fp1": lead_34_data[1, :],
            "T3": lead_34_data[9, :],
            "C3": lead_34_data[10, :],
            "O1": lead_34_data[19, :],
            "Fp2": lead_34_data[3, :],
            "C4": lead_34_data[12, :],
            "T4": lead_34_data[13, :],
            "O2": lead_34_data[21, :],
        }
    if lead_type == '10':
        out_dict = {
            "A1": lead_34_data[22, :],
            "A2": lead_34_data[23, :],
            "Fp1": lead_34_data[1, :],
            "T3": lead_34_data[9, :],
            "C3": lead_34_data[10, :],
            "O1": lead_34_data[19, :],
            "Fp2": lead_34_data[3, :],
            "C4": lead_34_data[12, :],
            "T4": lead_34_data[13, :],
            "O2": lead_34_data[21, :],
        }
    if lead_type == '16':
        out_dict = {
            "A1": lead_34_data[22,:],
            "A2": lead_34_data[23,:],
            "Fp1": lead_34_data[1,:],
            "F3": lead_34_data[5,:],
            "F7": lead_34_data[4, :],
            "T3": lead_34_data[9, :],
            "C3": lead_34_data[10,:],
            "T5": lead_34_data[14, :],
            "P3": lead_34_data[15,:],
            "O1": lead_34_data[19,:],
            "Fp2": lead_34_data[3,:],
            "F4": lead_34_data[7,:],
            "F8": lead_34_data[8, :],
            "C4": lead_34_data[12,:],
            "T4": lead_34_data[13, :],
            "P4": lead_34_data[17,:],
            "T6": lead_34_data[18, :],
            "O2": lead_34_data[21,:],
        }
    if lead_type == '18':
        out_dict = {
            "A1": lead_34_data[22,:],
            "A2": lead_34_data[23,:],
            "Fp1": lead_34_data[1,:],
            "F3": lead_34_data[5,:],
            "F7": lead_34_data[4, :],
            "T3": lead_34_data[9, :],
            "C3": lead_34_data[10,:],
            "T5": lead_34_data[14, :],
            "P3": lead_34_data[15,:],
            "O1": lead_34_data[19,:],
            "Fp2": lead_34_data[3,:],
            "F4": lead_34_data[7,:],
            "F8": lead_34_data[8, :],
            "C4": lead_34_data[12,:],
            "T4": lead_34_data[13, :],
            "P4": lead_34_data[17,:],
            "T6": lead_34_data[18, :],
            "O2": lead_34_data[21,:],
        }

    if lead_type == '21':
        out_dict = {
            "A1": lead_34_data[22, :],
            "A2": lead_34_data[23, :],
            "Fp1": lead_34_data[1, :],
            "F3": lead_34_data[5, :],
            "F7": lead_34_data[4, :],
            "T3": lead_34_data[9, :],
            "C3": lead_34_data[10, :],
            "T5": lead_34_data[14, :],
            "P3": lead_34_data[15, :],
            "O1": lead_34_data[19, :],
            "Fp2": lead_34_data[3, :],
            "F4": lead_34_data[7, :],
            "F8": lead_34_data[8, :],
            "C4": lead_34_data[12, :],
            "T4": lead_34_data[13, :],
            "P4": lead_34_data[17, :],
            "T6": lead_34_data[18, :],
            "O2": lead_34_data[21, :],
            "Fpz": lead_34_data[2, :],
            "Fz": lead_34_data[6, :],
            "Cz": lead_34_data[11, :],
            "Pz": lead_34_data[16, :],
            "Oz": lead_34_data[20, :],
            "EMG": lead_34_data[30, :],
            "EOG": lead_34_data[31, :],

        }


    if lead_type == '23':
        out_dict = {
            "A1": lead_34_data[22, :],
            "A2": lead_34_data[23, :],
            "Fp1": lead_34_data[1, :],
            "F3": lead_34_data[5, :],
            "F7": lead_34_data[4, :],
            "T3": lead_34_data[9, :],
            "C3": lead_34_data[10, :],
            "T5": lead_34_data[14, :],
            "P3": lead_34_data[15, :],
            "O1": lead_34_data[19, :],
            "Fp2": lead_34_data[3, :],
            "F4": lead_34_data[7, :],
            "F8": lead_34_data[8, :],
            "C4": lead_34_data[12, :],
            "T4": lead_34_data[13, :],
            "P4": lead_34_data[17, :],
            "T6": lead_34_data[18, :],
            "O2": lead_34_data[21, :],
            "Fpz":lead_34_data[2,:],
            "Fz":lead_34_data[6,:],
            "Cz":lead_34_data[11,:],
            "Pz":lead_34_data[16,:],
            "Oz":lead_34_data[30,:]
        }

    if lead_type == '25':
        out_dict = {
            "A1": lead_34_data[22, :],
            "A2": lead_34_data[23, :],
            "Fp1": lead_34_data[1, :],
            "F3": lead_34_data[5, :],
            "F7": lead_34_data[4, :],
            "T3": lead_34_data[9, :],
            "C3": lead_34_data[10, :],
            "T5": lead_34_data[14, :],
            "P3": lead_34_data[15, :],
            "O1": lead_34_data[19, :],
            "Fp2": lead_34_data[3, :],
            "F4": lead_34_data[7, :],
            "F8": lead_34_data[8, :],
            "C4": lead_34_data[12, :],
            "T4": lead_34_data[13, :],
            "P4": lead_34_data[17, :],
            "T6": lead_34_data[18, :],
            "O2": lead_34_data[21, :],
            "Fpz": lead_34_data[2, :],
            "Fz": lead_34_data[6, :],
            "Cz": lead_34_data[11, :],
            "Pz": lead_34_data[16, :],
            "Oz": lead_34_data[30, :],
            "SPH-L":lead_34_data[25,:],
            "SPH-R":lead_34_data[26,:],
        }


    return out_dict



def get_lead_list(eeg_path,lead_type, time_scale):


    lead_34_data = get_all_lead(eeg_path)
    data_len = np.shape(lead_34_data[22,:])[0]
    out_list = []
    out_index_list = []
    for i in range(0,data_len,time_scale*256):
        i_begin = i
        i_end = i+time_scale*256

       # print(i_end/256 - i_begin/256)

        if i_begin<=0:i_begin = 0
        if i_end >= data_len: i_end = data_len

        out_index_list.append([i_begin/256,i_end/256])


        if lead_type == '1':
            out_dict = {
               'SIGNAL':lead_34_data[10, i_begin:i_end]- lead_34_data[22, i_begin:i_end],
            }

        if lead_type == '8':
            out_dict = {
                "A1": lead_34_data[22, i_begin:i_end],
                "A2": lead_34_data[23, i_begin:i_end],
                "Fp1": lead_34_data[1, i_begin:i_end],
                "T3": lead_34_data[9, i_begin:i_end],
                "C3": lead_34_data[10, i_begin:i_end],
                "O1": lead_34_data[19, i_begin:i_end],
                "Fp2": lead_34_data[3, i_begin:i_end],
                "C4": lead_34_data[12, i_begin:i_end],
                "T4": lead_34_data[13, i_begin:i_end],
                "O2": lead_34_data[21, i_begin:i_end],
            }
        if lead_type == '10':
            out_dict = {
                "A1": lead_34_data[22, i_begin:i_end],
                "A2": lead_34_data[23, i_begin:i_end],
                "Fp1": lead_34_data[1, i_begin:i_end],
                "T3": lead_34_data[9, i_begin:i_end],
                "C3": lead_34_data[10, i_begin:i_end],
                "O1": lead_34_data[19, i_begin:i_end],
                "Fp2": lead_34_data[3, i_begin:i_end],
                "C4": lead_34_data[12, i_begin:i_end],
                "T4": lead_34_data[13, i_begin:i_end],
                "O2": lead_34_data[21, i_begin:i_end],
            }
        if lead_type == '16':
            out_dict = {
                "A1": lead_34_data[22,i_begin:i_end],
                "A2": lead_34_data[23,i_begin:i_end],
                "Fp1": lead_34_data[1,i_begin:i_end],
                "F3": lead_34_data[5,i_begin:i_end],
                "F7": lead_34_data[4, i_begin:i_end],
                "T3": lead_34_data[9, i_begin:i_end],
                "C3": lead_34_data[10,i_begin:i_end],
                "T5": lead_34_data[14, i_begin:i_end],
                "P3": lead_34_data[15,i_begin:i_end],
                "O1": lead_34_data[19,i_begin:i_end],
                "Fp2": lead_34_data[3,i_begin:i_end],
                "F4": lead_34_data[7,i_begin:i_end],
                "F8": lead_34_data[8, i_begin:i_end],
                "C4": lead_34_data[12,i_begin:i_end],
                "T4": lead_34_data[13, i_begin:i_end],
                "P4": lead_34_data[17,i_begin:i_end],
                "T6": lead_34_data[18, i_begin:i_end],
                "O2": lead_34_data[21,i_begin:i_end],
            }




        if lead_type == '18':
            out_dict = {
                "A1": lead_34_data[22,i_begin:i_end],
                "A2": lead_34_data[23,i_begin:i_end],
                "Fp1": lead_34_data[1,i_begin:i_end],
                "F3": lead_34_data[5,i_begin:i_end],
                "F7": lead_34_data[4, i_begin:i_end],
                "T3": lead_34_data[9, i_begin:i_end],
                "C3": lead_34_data[10,i_begin:i_end],
                "T5": lead_34_data[14, i_begin:i_end],
                "P3": lead_34_data[15,i_begin:i_end],
                "O1": lead_34_data[19,i_begin:i_end],
                "Fp2": lead_34_data[3,i_begin:i_end],
                "F4": lead_34_data[7,i_begin:i_end],
                "F8": lead_34_data[8, i_begin:i_end],
                "C4": lead_34_data[12,i_begin:i_end],
                "T4": lead_34_data[13, i_begin:i_end],
                "P4": lead_34_data[17,i_begin:i_end],
                "T6": lead_34_data[18, i_begin:i_end],
                "O2": lead_34_data[21,i_begin:i_end],
            }

        if lead_type == '21':
            out_dict = {
                "A1": lead_34_data[22, i_begin:i_end],
                "A2": lead_34_data[23, i_begin:i_end],
                "Fp1": lead_34_data[1, i_begin:i_end],
                "F3": lead_34_data[5, i_begin:i_end],
                "F7": lead_34_data[4, i_begin:i_end],
                "T3": lead_34_data[9, i_begin:i_end],
                "C3": lead_34_data[10, i_begin:i_end],
                "T5": lead_34_data[14, i_begin:i_end],
                "P3": lead_34_data[15, i_begin:i_end],
                "O1": lead_34_data[19, i_begin:i_end],
                "Fp2": lead_34_data[3, i_begin:i_end],
                "F4": lead_34_data[7, i_begin:i_end],
                "F8": lead_34_data[8, i_begin:i_end],
                "C4": lead_34_data[12, i_begin:i_end],
                "T4": lead_34_data[13, i_begin:i_end],
                "P4": lead_34_data[17, i_begin:i_end],
                "T6": lead_34_data[18, i_begin:i_end],
                "O2": lead_34_data[21, i_begin:i_end],
                "Fpz":lead_34_data[2,i_begin:i_end],
                "Fz":lead_34_data[6,i_begin:i_end],
                "Cz":lead_34_data[11,i_begin:i_end],
                "Pz":lead_34_data[16,i_begin:i_end],
                "Oz":lead_34_data[30,i_begin:i_end]
            }


        if lead_type == '23':
            out_dict = {
                "A1": lead_34_data[22, i_begin:i_end],
                "A2": lead_34_data[23, i_begin:i_end],
                "Fp1": lead_34_data[1, i_begin:i_end],
                "F3": lead_34_data[5, i_begin:i_end],
                "F7": lead_34_data[4, i_begin:i_end],
                "T3": lead_34_data[9, i_begin:i_end],
                "C3": lead_34_data[10, i_begin:i_end],
                "T5": lead_34_data[14, i_begin:i_end],
                "P3": lead_34_data[15, i_begin:i_end],
                "O1": lead_34_data[19, i_begin:i_end],
                "Fp2": lead_34_data[3, i_begin:i_end],
                "F4": lead_34_data[7, i_begin:i_end],
                "F8": lead_34_data[8, i_begin:i_end],
                "C4": lead_34_data[12, i_begin:i_end],
                "T4": lead_34_data[13, i_begin:i_end],
                "P4": lead_34_data[17, i_begin:i_end],
                "T6": lead_34_data[18, i_begin:i_end],
                "O2": lead_34_data[21, i_begin:i_end],
                "Fpz":lead_34_data[2,i_begin:i_end],
                "Fz":lead_34_data[6,i_begin:i_end],
                "Cz":lead_34_data[11,i_begin:i_end],
                "Pz":lead_34_data[16,i_begin:i_end],
                "Oz":lead_34_data[30,i_begin:i_end]
            }

        if lead_type == '25':
            out_dict = {
                "A1": lead_34_data[22, i_begin:i_end],
                "A2": lead_34_data[23, i_begin:i_end],
                "Fp1": lead_34_data[1, i_begin:i_end],
                "F3": lead_34_data[5, i_begin:i_end],
                "F7": lead_34_data[4, i_begin:i_end],
                "T3": lead_34_data[9, i_begin:i_end],
                "C3": lead_34_data[10, i_begin:i_end],
                "T5": lead_34_data[14, i_begin:i_end],
                "P3": lead_34_data[15, i_begin:i_end],
                "O1": lead_34_data[19, i_begin:i_end],
                "Fp2": lead_34_data[3, i_begin:i_end],
                "F4": lead_34_data[7, i_begin:i_end],
                "F8": lead_34_data[8, i_begin:i_end],
                "C4": lead_34_data[12, i_begin:i_end],
                "T4": lead_34_data[13, i_begin:i_end],
                "P4": lead_34_data[17, i_begin:i_end],
                "T6": lead_34_data[18, i_begin:i_end],
                "O2": lead_34_data[21, i_begin:i_end],
                "Fpz": lead_34_data[2, i_begin:i_end],
                "Fz": lead_34_data[6, i_begin:i_end],
                "Cz": lead_34_data[11, i_begin:i_end],
                "Pz": lead_34_data[16, i_begin:i_end],
                "Oz": lead_34_data[30, i_begin:i_end],
                "SPH-L":lead_34_data[25,i_begin:i_end],
                "SPH-R":lead_34_data[26,i_begin:i_end],
            }

        out_list.append(out_dict)

    return out_list,out_index_list





def get_lead_list_original_len(eeg_path,lead_type, time_scale):


    lead_34_data = get_all_lead(eeg_path)
    data_len = np.shape(lead_34_data[22,:])[0]
    out_list = []

    for i in range(0,data_len,time_scale*256):
        i_begin = i
        i_end = i+time_scale*256

       # print(i_end/256 - i_begin/256)

        if i_begin<=0:i_begin = 0
        if i_end >= data_len: i_end = data_len


        if lead_type == '1':
            out_dict = {
               'SIGNAL':lead_34_data[4, i_begin:i_end]- lead_34_data[22, i_begin:i_end],
            }

        if lead_type == '8':
            out_dict = {
                "A1": lead_34_data[22, i_begin:i_end],
                "A2": lead_34_data[23, i_begin:i_end],
                "Fp1": lead_34_data[1, i_begin:i_end],
                "T3": lead_34_data[9, i_begin:i_end],
                "C3": lead_34_data[10, i_begin:i_end],
                "O1": lead_34_data[19, i_begin:i_end],
                "Fp2": lead_34_data[3, i_begin:i_end],
                "C4": lead_34_data[12, i_begin:i_end],
                "T4": lead_34_data[13, i_begin:i_end],
                "O2": lead_34_data[21, i_begin:i_end],
            }
        if lead_type == '10':
            out_dict = {
                "A1": lead_34_data[22, i_begin:i_end],
                "A2": lead_34_data[23, i_begin:i_end],
                "Fp1": lead_34_data[1, i_begin:i_end],
                "T3": lead_34_data[9, i_begin:i_end],
                "C3": lead_34_data[10, i_begin:i_end],
                "O1": lead_34_data[19, i_begin:i_end],
                "Fp2": lead_34_data[3, i_begin:i_end],
                "C4": lead_34_data[12, i_begin:i_end],
                "T4": lead_34_data[13, i_begin:i_end],
                "O2": lead_34_data[21, i_begin:i_end],
            }
        if lead_type == '16':
            out_dict = {
                "A1": lead_34_data[22,i_begin:i_end],
                "A2": lead_34_data[23,i_begin:i_end],
                "Fp1": lead_34_data[1,i_begin:i_end],
                "F3": lead_34_data[5,i_begin:i_end],
                "F7": lead_34_data[4, i_begin:i_end],
                "T3": lead_34_data[9, i_begin:i_end],
                "C3": lead_34_data[10,i_begin:i_end],
                "T5": lead_34_data[14, i_begin:i_end],
                "P3": lead_34_data[15,i_begin:i_end],
                "O1": lead_34_data[19,i_begin:i_end],
                "Fp2": lead_34_data[3,i_begin:i_end],
                "F4": lead_34_data[7,i_begin:i_end],
                "F8": lead_34_data[8, i_begin:i_end],
                "C4": lead_34_data[12,i_begin:i_end],
                "T4": lead_34_data[13, i_begin:i_end],
                "P4": lead_34_data[17,i_begin:i_end],
                "T6": lead_34_data[18, i_begin:i_end],
                "O2": lead_34_data[21,i_begin:i_end],
            }




        if lead_type == '18':
            out_dict = {
                "A1": lead_34_data[22,i_begin:i_end],
                "A2": lead_34_data[23,i_begin:i_end],
                "Fp1": lead_34_data[1,i_begin:i_end],
                "F3": lead_34_data[5,i_begin:i_end],
                "F7": lead_34_data[4, i_begin:i_end],
                "T3": lead_34_data[9, i_begin:i_end],
                "C3": lead_34_data[10,i_begin:i_end],
                "T5": lead_34_data[14, i_begin:i_end],
                "P3": lead_34_data[15,i_begin:i_end],
                "O1": lead_34_data[19,i_begin:i_end],
                "Fp2": lead_34_data[3,i_begin:i_end],
                "F4": lead_34_data[7,i_begin:i_end],
                "F8": lead_34_data[8, i_begin:i_end],
                "C4": lead_34_data[12,i_begin:i_end],
                "T4": lead_34_data[13, i_begin:i_end],
                "P4": lead_34_data[17,i_begin:i_end],
                "T6": lead_34_data[18, i_begin:i_end],
                "O2": lead_34_data[21,i_begin:i_end],
            }

        if lead_type == '21':
            out_dict = {
                "A1": lead_34_data[22, i_begin:i_end],
                "A2": lead_34_data[23, i_begin:i_end],
                "Fp1": lead_34_data[1, i_begin:i_end],
                "F3": lead_34_data[5, i_begin:i_end],
                "F7": lead_34_data[4, i_begin:i_end],
                "T3": lead_34_data[9, i_begin:i_end],
                "C3": lead_34_data[10, i_begin:i_end],
                "T5": lead_34_data[14, i_begin:i_end],
                "P3": lead_34_data[15, i_begin:i_end],
                "O1": lead_34_data[19, i_begin:i_end],
                "Fp2": lead_34_data[3, i_begin:i_end],
                "F4": lead_34_data[7, i_begin:i_end],
                "F8": lead_34_data[8, i_begin:i_end],
                "C4": lead_34_data[12, i_begin:i_end],
                "T4": lead_34_data[13, i_begin:i_end],
                "P4": lead_34_data[17, i_begin:i_end],
                "T6": lead_34_data[18, i_begin:i_end],
                "O2": lead_34_data[21, i_begin:i_end],
                "Fpz":lead_34_data[2,i_begin:i_end],
                "Fz":lead_34_data[6,i_begin:i_end],
                "Cz":lead_34_data[11,i_begin:i_end],
                "Pz":lead_34_data[16,i_begin:i_end],
                "Oz":lead_34_data[30,i_begin:i_end]
            }


        if lead_type == '23':
            out_dict = {
                "A1": lead_34_data[22, i_begin:i_end],
                "A2": lead_34_data[23, i_begin:i_end],
                "Fp1": lead_34_data[1, i_begin:i_end],
                "F3": lead_34_data[5, i_begin:i_end],
                "F7": lead_34_data[4, i_begin:i_end],
                "T3": lead_34_data[9, i_begin:i_end],
                "C3": lead_34_data[10, i_begin:i_end],
                "T5": lead_34_data[14, i_begin:i_end],
                "P3": lead_34_data[15, i_begin:i_end],
                "O1": lead_34_data[19, i_begin:i_end],
                "Fp2": lead_34_data[3, i_begin:i_end],
                "F4": lead_34_data[7, i_begin:i_end],
                "F8": lead_34_data[8, i_begin:i_end],
                "C4": lead_34_data[12, i_begin:i_end],
                "T4": lead_34_data[13, i_begin:i_end],
                "P4": lead_34_data[17, i_begin:i_end],
                "T6": lead_34_data[18, i_begin:i_end],
                "O2": lead_34_data[21, i_begin:i_end],
                "Fpz":lead_34_data[2,i_begin:i_end],
                "Fz":lead_34_data[6,i_begin:i_end],
                "Cz":lead_34_data[11,i_begin:i_end],
                "Pz":lead_34_data[16,i_begin:i_end],
                "Oz":lead_34_data[30,i_begin:i_end]
            }

        if lead_type == '25':
            out_dict = {
                "A1": lead_34_data[22, i_begin:i_end],
                "A2": lead_34_data[23, i_begin:i_end],
                "Fp1": lead_34_data[1, i_begin:i_end],
                "F3": lead_34_data[5, i_begin:i_end],
                "F7": lead_34_data[4, i_begin:i_end],
                "T3": lead_34_data[9, i_begin:i_end],
                "C3": lead_34_data[10, i_begin:i_end],
                "T5": lead_34_data[14, i_begin:i_end],
                "P3": lead_34_data[15, i_begin:i_end],
                "O1": lead_34_data[19, i_begin:i_end],
                "Fp2": lead_34_data[3, i_begin:i_end],
                "F4": lead_34_data[7, i_begin:i_end],
                "F8": lead_34_data[8, i_begin:i_end],
                "C4": lead_34_data[12, i_begin:i_end],
                "T4": lead_34_data[13, i_begin:i_end],
                "P4": lead_34_data[17, i_begin:i_end],
                "T6": lead_34_data[18, i_begin:i_end],
                "O2": lead_34_data[21, i_begin:i_end],
                "Fpz": lead_34_data[2, i_begin:i_end],
                "Fz": lead_34_data[6, i_begin:i_end],
                "Cz": lead_34_data[11, i_begin:i_end],
                "Pz": lead_34_data[16, i_begin:i_end],
                "Oz": lead_34_data[30, i_begin:i_end],
                "SPH-L":lead_34_data[25,i_begin:i_end],
                "SPH-R":lead_34_data[26,i_begin:i_end],
            }

        out_list.append(out_dict)

    return out_list



# 获取耳极16导联的数据
def load_eeg_file(eeg_path, low_cut=[1], high_cut=[25]):
    leads = get_all_lead(eeg_path)
    leads_16 = get_A1A2_with_filter(leads, low_cut, high_cut)
    return leads_16


# 获取34导联的数据
def get_34_with_filter(lead_34_data, low_cut=[0.8], high_cut=[35]):
    for l, h in zip(low_cut, high_cut):
        lead_34_data = butter_bandpass_filter_leads(lead_34_data, low_cut=l, high_cut=h, fs=256, order=6)

    return lead_34_data


# 获取耳极，获取双极导联数据
def get_unipolar_bipolar_with_filter(lead_34_data, low_cut=[0.8], high_cut=[35]):
    '''
    input
        lead_34_data 34 导联原始数据
        low_cut   高通截止
        high_cut   低通截止
    '''
    for l, h in zip(low_cut, high_cut):
       lead_34_data = butter_bandpass_filter_leads(lead_34_data, low_cut=l, high_cut=h, fs=256,order=6)

    A1 = lead_34_data[22, :]
    A2 = lead_34_data[23, :]

    FP1 = lead_34_data[1, :] - A1
    F7 = lead_34_data[4, :] - A1
    F3 = lead_34_data[5, :] - A1
    T3 = lead_34_data[9, :] - A1
    C3 = lead_34_data[10, :] - A1
    T5 = lead_34_data[14, :] - A1
    P3 = lead_34_data[15, :] - A1
    O1 = lead_34_data[19, :] - A1

    FP2 = lead_34_data[3, :] - A2
    F4 = lead_34_data[7, :] - A2
    F8 = lead_34_data[8, :] - A2
    C4 = lead_34_data[12, :] - A2
    T4 = lead_34_data[13, :] - A2
    P4 = lead_34_data[17, :] - A2
    T6 = lead_34_data[18, :] - A2
    O2 = lead_34_data[21, :] - A2

    out_data = [FP1, FP2, F3, F4, C3, C4, P3, P4, O1, O2, F7, F8, T3, T4, T5, T6,
                FP1 - F3, FP2 - F4, F3 - C3, F4 - C4, C3 - P3, C4 - P4, P3 - O1, P4 - O2,
                FP1 - F7, FP2 - F8, F7 - T3, F8 - T4, T3 - T5, T4 - T6, T5 - O1, T6 - O2]

    return np.array(out_data)


# 耳极16导联的数据，用于尖棘慢的检测
def get_A1A2_with_filter(lead_34_data, low_cut=[0.8], high_cut=[35]):
    # lead_34_data = notch_filter.butter_bandpass_filter_leads(lead_34_data, low_cut=1, high_cut=25, fs=256, order=6)
    for l, h in zip(low_cut, high_cut):
        lead_34_data = butter_bandpass_filter_leads(lead_34_data, low_cut=l, high_cut=h, fs=256, order=6)

    A1 = lead_34_data[22, :]
    A2 = lead_34_data[23, :]

    Fp1 = lead_34_data[1, :]
    F7 = lead_34_data[4, :]
    F3 = lead_34_data[5, :]
    T3 = lead_34_data[9, :]
    C3 = lead_34_data[10, :]
    T5 = lead_34_data[14, :]
    P3 = lead_34_data[15, :]
    O1 = lead_34_data[19, :]

    Fp2 = lead_34_data[3, :]
    F4 = lead_34_data[7, :]
    F8 = lead_34_data[8, :]
    C4 = lead_34_data[12, :]
    T4 = lead_34_data[13, :]
    P4 = lead_34_data[17, :]
    T6 = lead_34_data[18, :]
    O2 = lead_34_data[21, :]

    out_data = []
    out_data.append(Fp1)
    out_data.append(Fp2)
    out_data.append(F3)
    out_data.append(F4)
    out_data.append(C3)
    out_data.append(C4)
    out_data.append(P3)
    out_data.append(P4)
    out_data.append(O1)
    out_data.append(O2)
    out_data.append(F7)
    out_data.append(F8)
    out_data.append(T3)
    out_data.append(T4)
    out_data.append(T5)
    out_data.append(T6)
    out_data.append(A1)
    out_data.append(A2)

    # left_data = []
    # left_data.append(Fp1)
    # left_data.append(F3)
    # left_data.append(C3)
    # left_data.append(P3)
    # left_data.append(O1)
    # left_data.append(F7)
    # left_data.append(T3)
    # left_data.append(T5)



    # right_data = []
    # right_data.append(FP2)
    # right_data.append(F4)
    # right_data.append(C4)
    # right_data.append(P4)
    # right_data.append(O2)
    # right_data.append(F8)
    # right_data.append(T4)
    # right_data.append(T6)


    return out_data



  # out_dict = {
  #           'Fp1':[], 'Fp2':[], 'F3':[], 'F4':[],
  #           'C3':[], 'C4':[], 'P3':[], 'P4':[],
  #           'O1':[], 'O2':[], 'F7':[], 'F8':[],
  #           'T3':[], 'T4':[], 'T5':[], 'T6':[]
  #       }


# g_LEAD_NEED = ['Fp1', 'Fp2', 'F3', 'F4',
#                'C3', 'C4', 'P3', 'P4',
#                'O1', 'O2', 'F7', 'F8',
#                'T3', 'T4', 'T5', 'T6']

def get_bipolar_with_filter(lead_34_data, low_cut=[0.8], high_cut=[35]):

    for l, h in zip(low_cut, high_cut):
        lead_34_data = butter_bandpass_filter_leads(lead_34_data, low_cut=l, high_cut=h, fs=256, order=6)

    out_data = []

    FP1 = lead_34_data[1, :]
    F7 = lead_34_data[4, :]
    F3 = lead_34_data[5, :]
    T3 = lead_34_data[9, :]
    C3 = lead_34_data[10, :]
    T5 = lead_34_data[14, :]
    P3 = lead_34_data[15, :]
    O1 = lead_34_data[19, :]

    FP2 = lead_34_data[3, :]
    F4 = lead_34_data[7, :]
    F8 = lead_34_data[8, :]
    C4 = lead_34_data[12, :]
    T4 = lead_34_data[13, :]
    P4 = lead_34_data[17, :]
    T6 = lead_34_data[18, :]
    O2 = lead_34_data[21, :]

    out_data.append(FP1-F3)
    out_data.append(FP2-F4)
    out_data.append(F3-C3)
    out_data.append(F4-C4)
    out_data.append(C3-P3)
    out_data.append(C4-P4)
    out_data.append(P3-O1)
    out_data.append(P4-O2)
    out_data.append(FP1-F7)
    out_data.append(FP2-F8)
    out_data.append(F7-T3)
    out_data.append(F8-T4)
    out_data.append(T3-T5)
    out_data.append(T4-T6)
    out_data.append(T5-O1)
    out_data.append(T6-O2)



    return np.array(out_data)

# 获取左右+耳极
def get_A1A2_SPH_with_filter(lead_34_data, low_cut=[1], high_cut=[25]):
    for l, h in zip(low_cut, high_cut):
        lead_34_data = butter_bandpass_filter_leads(lead_34_data, low_cut=l, high_cut=h, fs=256, order=6)

    A1 = lead_34_data[22, :]
    A2 = lead_34_data[23, :]
    SHP_L = lead_34_data[24, :]
    SHP_R = lead_34_data[25, :]

    FP1 = lead_34_data[1, :] - A1
    F3 = lead_34_data[5, :] - A1
    F7 = lead_34_data[4, :] - A1
    T3 = lead_34_data[9, :] - A1
    C3 = lead_34_data[10, :] - A1
    T5 = lead_34_data[14, :] - A1
    P3 = lead_34_data[15, :] - A1
    O1 = lead_34_data[19, :] - A1

    left_data = []
    left_data.append(FP1)
    left_data.append(F3)
    left_data.append(C3)
    left_data.append(P3)
    left_data.append(O1)
    left_data.append(F7)
    left_data.append(T3)
    left_data.append(T5)

    FP2 = lead_34_data[3, :] - A2
    F4 = lead_34_data[7, :] - A2
    F8 = lead_34_data[8, :] - A2
    C4 = lead_34_data[12, :] - A2
    T4 = lead_34_data[13, :] - A2
    P4 = lead_34_data[17, :] - A2
    T6 = lead_34_data[18, :] - A2
    O2 = lead_34_data[21, :] - A2

    right_data = []
    right_data.append(FP2)
    right_data.append(F4)
    right_data.append(C4)
    right_data.append(P4)
    right_data.append(O2)
    right_data.append(F8)
    right_data.append(T4)
    right_data.append(T6)

    FP1_SPHL = FP1 + A1 - SHP_L
    FP2_SPHR = FP2 + A2 - SHP_R
    SPHL_C3 = SHP_L - C3 - A1
    SPHR_C4 = SHP_R - C4 - A2
    SPHL_A1 = SHP_L - A1
    SPHR_A2 = SHP_R - A2

    sph_data = []
    sph_data.append(FP1_SPHL)
    sph_data.append(FP2_SPHR)
    sph_data.append(SPHL_C3)
    sph_data.append(SPHR_C4)
    sph_data.append(SPHL_A1)
    sph_data.append(SPHR_A2)

    return np.array(left_data + right_data + sph_data)


# 单边十二导联
def get_A1A2_SPHED_with_filter(lead_34_data, low_cut=[0.8], high_cut=[25], order=3):
    #
    # 单边十二导联
    for l, h in zip(low_cut, high_cut):
        # lead_34_data = butter_bandpass_filter_leads(lead_34_data, low_cut=l, high_cut=h, fs=256, order=6)
        lead_34_data = butter_bandpass_filter_leads(lead_34_data, low_cut=l, high_cut=h, fs=256, order=order)

    A1 = lead_34_data[22, :]
    A2 = lead_34_data[23, :]
    SHP_L = lead_34_data[24, :]
    SHP_R = lead_34_data[25, :]

    FP1 = lead_34_data[1, :] - A1
    F3 = lead_34_data[5, :] - A1
    F7 = lead_34_data[4, :] - A1
    T3 = lead_34_data[9, :] - A1
    C3 = lead_34_data[10, :] - A1
    T5 = lead_34_data[14, :] - A1
    P3 = lead_34_data[15, :] - A1
    O1 = lead_34_data[19, :] - A1

    left_data = []
    left_data.append(FP1)
    left_data.append(F3)
    left_data.append(C3)
    left_data.append(P3)
    left_data.append(O1)
    left_data.append(F7)
    left_data.append(T3)
    left_data.append(T5)

    FP2 = lead_34_data[3, :] - A2
    F4 = lead_34_data[7, :] - A2
    F8 = lead_34_data[8, :] - A2
    C4 = lead_34_data[12, :] - A2
    T4 = lead_34_data[13, :] - A2
    P4 = lead_34_data[17, :] - A2
    T6 = lead_34_data[18, :] - A2
    O2 = lead_34_data[21, :] - A2

    right_data = []
    right_data.append(FP2)
    right_data.append(F4)
    right_data.append(C4)
    right_data.append(P4)
    right_data.append(O2)
    right_data.append(F8)
    right_data.append(T4)
    right_data.append(T6)

    FP1_SPHL = FP1 + A1 - SHP_L
    FP2_SPHR = FP2 + A2 - SHP_R
    SPHL_C3 = SHP_L - C3 - A1
    SPHR_C4 = SHP_R - C4 - A2
    SPHL_A1 = A1
    SPHR_A2 = A2

    SPHL_A1_PLUS = SHP_L
    SPHR_A2_PLUS = SHP_R

    sph_data = []
    sph_data.append(FP1_SPHL)
    sph_data.append(FP2_SPHR)
    sph_data.append(SPHL_C3)
    sph_data.append(SPHR_C4)
    sph_data.append(SPHL_A1)
    sph_data.append(SPHR_A2)

    sph_data.append(SPHL_A1_PLUS)
    sph_data.append(SPHR_A2_PLUS)

    return np.array(left_data + right_data + sph_data)


# 获取完整sph,用于获取SPH的process
def get_sph_with_filter(lead_34_data, low_cut=[0.8], high_cut=[25], order=3):
    for l, h in zip(low_cut, high_cut):
        lead_34_data = butter_bandpass_filter_leads(lead_34_data, low_cut=l, high_cut=h, fs=256, order=order)

    left_data = []
    right_data = []
    A1 = lead_34_data[22, :]
    A2 = lead_34_data[23, :]
    SHP_L = lead_34_data[24, :]
    SHP_R = lead_34_data[25, :]
    ECG_1 = lead_34_data[26, :]
    ECG_2 = lead_34_data[27, :]
    OZ = lead_34_data[20, :]

    FP1 = lead_34_data[1, :]
    F3 = lead_34_data[5, :]
    F7 = lead_34_data[4, :]
    T3 = lead_34_data[9, :]
    C3 = lead_34_data[10, :]
    T5 = lead_34_data[14, :]
    P3 = lead_34_data[15, :]
    O1 = lead_34_data[19, :]

    left_data.append(FP1 - SHP_L)
    left_data.append(SHP_L - C3)
    left_data.append(O1)
    left_data.append(OZ)
    left_data.append(SHP_L - A1)
    left_data.append(SHP_L)
    left_data.append(SHP_L)
    left_data.append(A1)
    left_data.append((F3 + F7 + T3 + C3 + T5 + P3 - 6 * A1) / 6)
    left_data.append(SHP_L)

    FP2 = lead_34_data[3, :]
    F4 = lead_34_data[7, :]
    F8 = lead_34_data[8, :]
    C4 = lead_34_data[12, :]
    T4 = lead_34_data[13, :]
    P4 = lead_34_data[17, :]
    T6 = lead_34_data[18, :]
    O2 = lead_34_data[21, :]

    right_data.append(FP2 - SHP_R)
    right_data.append(SHP_R - C4)
    right_data.append(O2)
    right_data.append(OZ)
    right_data.append(SHP_R - A2)
    right_data.append(SHP_R)
    right_data.append(SHP_R)
    right_data.append(A2)
    right_data.append((F4 + F8 + C4 + T4 + P4 + T6 - 6 * A2) / 6)
    right_data.append(SHP_R)

    return np.array(left_data + right_data)


#--------------------------------------ng reader-----------------------------------------
def millisecond_to_loc(s):
    loc = int(s * 256 / 1000)
    return loc

def load_ng_info_to_list(path):
    """读取ng文件，将结果存到list中
    argv:
        path:ng文件路径
    return:
        loc_list=[list1, list2, ...], list1为：标注类别，起始位置（对应eeg数据点），终止位置（对应eeg数据点），导联名称，左右

    """
    loc_list = []

    if not os.path.exists(path):
        return loc_list

    with open(path, 'rb') as f:
        bi = f.read(4)
        eeg_seg_num = struct.unpack('i', bi)[0]
        #print('seg_num:', eeg_seg_num)
        if eeg_seg_num != 0:
            f.read(12)
        bi = f.read(4)

        event_num = struct.unpack('i', bi)[0]


        for one_event in range(event_num):
            tep = []
            # try:
            bi = f.read(30)
            # m_name
            m_strObjectName = bytes.decode(bi, encoding='gbk', errors='ignore').replace('\x00', '')
            tep.append(m_strObjectName)  # 0

            # m_bDefault
            bi = f.read(4)
            tep.append(struct.unpack('i', bi)[0])  #1

            # m_strNotes[200]
            bi = f.read(200)
            m_strNotes = bytes.decode(bi, encoding='gbk', errors='ignore').replace('\x00', '')
            tep.append(m_strNotes)  #2

            # m_bPredefined
            bi = f.read(4)
            tep.append(struct.unpack('i', bi)[0])  #3

            # m_bDuration
            bi = f.read(4)
            tep.append(struct.unpack('i', bi)[0])  #4

            # m_bEnableDelete
            bi = f.read(4)
            tep.append(struct.unpack('i', bi)[0])  #5

            # m_bAutoSet
            bi = f.read(4)
            tep.append(struct.unpack('i', bi)[0])  #6

            # m_clrColor
            bi = f.read(4)
            tep.append(struct.unpack('L', bi)[0])  #7

            # m_uShortCutKey
            bi = f.read(4)
            tep.append(struct.unpack('I', bi)[0])  #8

            # **************************************m_iEventStartTimePos *
            bi = f.read(4)
            struct.unpack('i', bi)[0]
            tep.append(struct.unpack('i', bi)[0])  #9

            # **************************************m_iEventEndTimePos ***
            bi = f.read(4)
            struct.unpack('i', bi)[0]
            tep.append(struct.unpack('i', bi)[0])  #10

            # m_iActiveChannel
            bi = f.read(4)
            tep.append(struct.unpack('i', bi)[0])  #11

            # m_iRefChannel
            bi = f.read(4)
            tep.append(struct.unpack('i', bi)[0])  #12

            # m_emEventType
            bi = f.read(4)
            tep.append(struct.unpack('i', bi)[0])

            # m_bReserved1
            bi = f.read(4)
            tep.append(struct.unpack('i', bi)[0])

            # m_bReserved2
            bi = f.read(4)
            tep.append(struct.unpack('i', bi)[0])

            loc_begin = millisecond_to_loc(tep[9])
            loc_end = millisecond_to_loc(tep[10])

            # name,begin,end,activate,ref
            loc_list.append([tep[0], loc_begin, loc_end, tep[11], tep[12]])
        # except:
        #
        #     continue

    loc_list = sorted(loc_list, key=lambda x: x[1])
    return loc_list

def load_ng_info_by_annotation(ng_path, accepted_annotation_names):
    """读取指定类别的标注
    """
    loc_list = load_ng_info_to_list(ng_path)
    loc_list_accepted = []
    for loc in loc_list:
        if loc[0] in accepted_annotation_names:
            loc_list_accepted.append(loc)
    return loc_list_accepted

def load_loc_info(ng_path):

    info_list = load_ng_info_to_list(ng_path)
    locs = [l[1:3] for l in info_list]
    return locs

def load_loc_info_by_annotation(ng_path, accepted_annotation_names):
    """读取指定类别的标注位置
    """
    loc_list = load_ng_info_to_list(ng_path)
    loc_list_accepted = []

    for loc in loc_list:
        if loc[0] in accepted_annotation_names:
            loc_list_accepted.append(loc[1:3])

    return loc_list_accepted

def get_event_num(c_path):
    if not os.path.exists(c_path):
        return 0,0

    with open(c_path, 'rb') as f:
        bi = f.read(4)
        eeg_seg_num = struct.unpack('i', bi)[0]
        if eeg_seg_num != 0:
            f.read(12)
        bi = f.read(4)

        event_num = struct.unpack('i', bi)[0]

        return  eeg_seg_num,event_num


#converter
def senconds_to_str(seconds):
    seconds = np.int(seconds)
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    string = "{:2d}h{:02d}m{:02d}s".format(h, m, s)
    return string

def loc_to_str(loc):
    second = loc/256.
    string = senconds_to_str(second)
    return string

def get_start_end_pair_in_label(label):
    # label = label.astype(np.int)
    label1 = np.concatenate([label[1:], np.array([0])])
    diff = label-label1

    start = np.where(diff==-1)[0]
    end =  np.where(diff==1)[0]
    if len(end)-len(start)==1:
       start= np.concatenate([np.array([0]), start])
    return start, end

def get_locs_from_labels(label):
    start, end = get_start_end_pair_in_label(label)
    locs = [[s,e,1,0] for (s,e) in zip(start, end)]
    return locs

def get_combined_loc_from_label(label):
    start, end = get_start_end_pair_in_label(label)
    label_merged = np.zeros_like(label)
    for s, e in zip(start, end):
        s = max(0, s-256)
        e = min(s+256, 921600)
        label_merged[s:e] = 1
    locs = get_locs_from_labels(label_merged)
    return locs


def get_label_from_locs(locs):
    label = np.zeros((60*60*256,))
    for loc in locs:
        label[loc[0]:loc[1]+1]=1
    return label

def merge_adjacent_locs(locs, margin=256*1.5):

    label = np.zeros((60*60*256,))
    for loc in locs:
        s = int(loc[0]-margin)
        s = max(0,s)
        e = int(loc[1]+margin)
        e = min(60*60*256, e)
        label[s:e]=1
    locs = get_locs_from_labels(label)
    label = np.zeros((60*60*256,))
    for loc in locs:
        if loc[1]-loc[0]>=margin*2+256:
            s = int(max(loc[0]+margin, 0))
            e = int(min(loc[1]-margin, 256*64*64))
        label[s:e]=1

    return get_locs_from_labels(label)


def merge_adjacent_locs_with(locs, margin=256*1.5):
    label = np.zeros((60*60*256,))
    for loc in locs:
        s = int(loc[0]-margin)
        s = max(0,s)
        e = int(loc[1]+margin)
        e = min(60*60*256, e)
        label[s:e]=1
    locs = get_locs_from_labels(label)
    label = np.zeros((60*60*256,))
    for loc in locs:
        if loc[1]-loc[0]>=margin*2+256:
            s = int(max(loc[0]+margin, 0))
            e = int(min(loc[1]-margin, 256*64*64))
        label[s:e]=1

    return get_locs_from_labels(label)






if __name__ == "__main__":

    file_list = [r'D:\SPH_ECG\\16111018\\2(5)\\2(5).eeg']
    print(len(file_list))

    for v in file_list[:]:
        print('------------------------:', v)
        leads = get_all_lead(v)
        print(np.shape(leads))

        # from profile import Profile
        # p = Profile()
        # p.runcall(get_sph_with_filter, leads)
        # p.print_stats()

        leads_sph = get_sph_with_filter(leads)
        import plot_eeg_tools as plot_eeg

        plot_eeg.show_leads(leads_sph)
