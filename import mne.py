import mne
raw = mne.io.read_raw_edf(r"D:\work\EDF_dir\4.edf", preload=True)
print(raw.ch_names)