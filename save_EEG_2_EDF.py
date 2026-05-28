import os

import matplotlib.pyplot as plt
import numpy as np
import mne
from a_eeg_tool import *
import matplotlib
matplotlib.use('Qt5Agg')



def save_dict_to_edf(eeg_dict, file_name, sfreq=250):
    """
    将包含脑电数据的字典保存为标准的 EDF 文件。

    参数:
    eeg_dict (dict): key为导联名称(str)，value为1D numpy数组(采样数据)
    file_name (str): 要保存的EDF文件名(例如 'subject_01.edf')
    sfreq (int/float): 采样率，默认为 250 Hz
    """
    # 1. 确保文件名以 .edf 结尾
    if not file_name.endswith('.edf'):
        file_name += '.edf'

    # 2. 提取导联名称和通道类型
    ch_names = list(eeg_dict.keys())
    ch_types = ['eeg'] * len(ch_names)

    # 3. 提取数据并转换为 MNE 的标准单位（伏特 V）
    # 假设原始字典数据单位是微伏 (uV)
    data = np.array([eeg_dict[ch] for ch in ch_names]) * 1e-6

    # 4. 创建 MNE Info 对象
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

    # 5. 创建 RawArray
    raw = mne.io.RawArray(data, info)

    # 6. 导出为 EDF 格式
    print(f"正在将数据导出至: {file_name} ...")
    mne.export.export_raw(file_name, raw, fmt='edf', overwrite=True)
    print("保存成功！")


# ---------------------------------------------------------
# 使用示例：

from tqdm import  tqdm
import glob

eeg_path = "H:\右安门的病例\99\eeg_save_dir\\22015031\\4\\*.eeg"


eeg_path_list_0 = glob.glob(eeg_path)



if __name__ == "__main__":
    # 模拟你的字典数据
    my_data = {
        'Fp1': np.random.randn(2560),
        'Fp2': np.random.randn(2560),
        'O1': np.random.randn(2560),
        'O2': np.random.randn(2560)
    }




    for eeg_path in tqdm(eeg_path_list_0):

        eeg_dict = get_lead_dict(eeg_path, '21')

        # plt.plot(eeg_dict["Fp1"])
        #
        # plt.show()


    # 调用函数（比如保存 10 秒的数据，采样率为 250Hz）
        save_dict_to_edf(eeg_dict=eeg_dict, file_name='1_test.edf', sfreq=256)