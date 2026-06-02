import mne
raw = mne.io.read_raw_edf(r"D:\work\EDF_dir\4.edf", preload=True)
print(raw.ch_names)
import mne
import os

# 输入输出路径
edf_file = r"D:\work\EDF_dir\4.edf"
fif_file = r"D:\work\EDF_dir\4.fif"

# 读取 EDF 并保存为 FIF
raw = mne.io.read_raw_edf(edf_file, preload=True)
raw.save(fif_file, overwrite=True)

print(f"转换完成: {fif_file}")