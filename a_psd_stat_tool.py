import  numpy as np



def get_spec_stat_info_old(SPEC,f):

    _delta = np.mean(SPEC[:, 0*f:4*f], axis=1)
    _theta = np.mean(SPEC[:, 4*f:8*f], axis=1)
    _alpha = np.mean(SPEC[:, 8*f:13*f], axis=1)
    _beta = np.mean(SPEC[:, 13*f:30*f], axis=1)
    _gamma_l = np.mean(SPEC[:, 30*f:48*f], axis=1)
    _gamma_h = np.mean(SPEC[:, 52*f:70*f], axis=1)

    _total = _alpha + _theta + _beta + _delta + _gamma_l + _gamma_h

    _r_delta = _delta / _total
    _r_theta = _theta / _total
    _r_alpha = _alpha / _total
    _r_beta = _beta / _total
    _r_gamma_l = _gamma_l / _total
    _r_gamma_h = _gamma_h / _total

    TBR = _theta / _beta
    DAR = _delta / _alpha
    TDR = _theta / _delta
    DTR = _delta / _theta
    ABR = _alpha / _beta
    ATR = _alpha / _theta
    DTAR = (_delta + _theta) / (_alpha + 1e-6)
    DTPWR = (_delta + _theta) / _total

    data_dict = {
        "D": _delta, "T": _theta, "A": _alpha, "B": _beta,
        "DR": _r_delta, "TR": _r_theta, "AR": _r_alpha, "BR": _r_beta,
        "TBR": TBR, "DAR": DAR, "TDR": TDR, "DTR": DTR, "ABR": ABR,
        "ATR": ATR, "DTAR": DTAR, "DTPWR": DTPWR,
        "GL": _gamma_l, "GH": _gamma_h, "GLR": _r_gamma_l, "GHR": _r_gamma_h
    }

    # 使用字典推导式一次性计算所有均值
    # means = {k: np.mean(v) for k, v in data_dict.items()}

    return data_dict

# def get_spec_stat_info(SPEC_second):
#
#
#     _delta = np.mean(SPEC_second[:,:, 1:4], axis=2)
#     _theta = np.mean(SPEC_second[:,:, 4:8], axis=2)
#
#     _alpha = np.mean(SPEC_second[:,:, 8:13], axis=2)
#     _alpha_1 = np.mean(SPEC_second[:,:, 8:9], axis=2)
#     _alpha_2 = np.mean(SPEC_second[:,:, 9:11], axis=2)
#     _alpha_3 = np.mean(SPEC_second[:,:, 11:13], axis=2)
#
#     _beta = np.mean(SPEC_second[:,:, 13:30], axis=2)
#     _beta_1 = np.mean(SPEC_second[:,:, 13:20], axis=2)
#     _beta_2 = np.mean(SPEC_second[:,:, 20:30], axis=2)
#
#     _gamma = np.mean(SPEC_second[:,:, 30:70], axis=2)
#     _gamma_1 = np.mean(SPEC_second[:,:, 30:50], axis=2)
#     _gamma_2 = np.mean(SPEC_second[:,:, 50:70], axis=2)
#
#
#     _total = _alpha + _theta + _beta + _delta + _gamma + 1e-6
#
#     #print(np.shape(_alpha))
#
#     _r_delta = _delta / _total
#     _r_theta = _theta / _total
#     _r_alpha = _alpha / _total
#     _r_alpha_1 = _alpha_1 / _total
#     _r_alpha_2 = _alpha_2 / _total
#     _r_beta = _beta / _total
#     _r_beta_1 = _beta_1 / _total
#     _r_beta_2 = _beta_2 / _total
#     _r_gamma = _gamma / _total
#     _r_gamma_1 = _gamma_1/_total
#     _r_gamma_2 = _gamma_2/_total
#
#     TBR = _theta  / (_beta+1e-6)
#     DAR = _delta  / (_alpha+1e-6)
#     DTR =  _delta / (_theta+1e-6)
#     ABR = _alpha  / (_beta+1e-6)
#     ATR = _alpha  /(_theta+1e-6)
#     DT_AR = (_delta+_theta)/(_alpha+1e-6)
#     DT_total_R = (_delta+_theta)/_total
#
#     print("ikl:",np.shape(_delta))
#
#     spec_features = {
#         "delta": _delta,
#         "theta": _theta,
#         "alpha": _alpha,
#         "alpha_1": _alpha_1,
#         "alpha_2": _alpha_2,
#         "alpha_3": _alpha_3,
#         "beta": _beta,
#         "beta_1": _beta_1,
#         "beta_2": _beta_2,
#         "gamma": _gamma,
#         "gamma_1": _gamma_1,
#         "gamma_2": _gamma_2,
#         "relative_delta": _r_delta,
#         "relative_theta": _r_theta,
#         "relative_alpha": _r_alpha,
#         "relative_beta": _r_beta,
#         "relative_gamma": _r_gamma,
#         "TBR": TBR,
#         "DAR": DAR,
#         "DTR": DTR,
#         "ABR": ABR,
#         "ATR": ATR,
#         "DT_AR": DT_AR,
#         "DT_total_R": DT_total_R
#     }
#
#     return spec_features


import numpy as np


def get_spec_stat_info(SPEC_second):
    """
    支持输入的 SPEC_second 是一个长度为 57 的 list。
    其中每个元素的形状是 (N_i, 128)，N_i 可以各不相同。
    """
    # 初始化一个字典，用来存储 57 个样本各自计算出的特征列表
    # 我们先用列表把它们存起来
    compiled_features = {
        "delta": [], "theta": [], "alpha": [],
        "alpha_1": [], "alpha_2": [], "alpha_3": [],
        "beta": [], "beta_1": [], "beta_2": [],
        "gamma": [], "gamma_1": [], "gamma_2": [],
        "relative_delta": [], "relative_theta": [],
        "relative_alpha": [], "relative_beta": [], "relative_gamma": [],
        "TBR": [], "DAR": [], "DTR": [], "ABR": [], "ATR": [],
        "DT_AR": [], "DT_total_R": []
    }

    # 循环遍历这 57 个样本，每个样本的形状是 (N, 128)
    for sample in SPEC_second:
        # 确保当前的样本是标准 NumPy 数组 (N, 128)
        sample = np.asarray(sample)

        # --- 以下是单张矩阵的特征计算逻辑 (针对当前这个样本的 N) ---
        # 因为现在 sample 是 2D 数组 (N, 128)，所以压缩频率轴要用 axis=1
        _total = np.mean(sample[:, 1:70], axis=1) + 1e-6

        _delta = np.mean(sample[:, 1:4], axis=1)
        _theta = np.mean(sample[:, 4:8], axis=1)
        _alpha = np.mean(sample[:, 8:13], axis=1)
        _alpha_1 = np.mean(sample[:, 8:9], axis=1)
        _alpha_2 = np.mean(sample[:, 9:11], axis=1)
        _alpha_3 = np.mean(sample[:, 11:13], axis=1)
        _beta = np.mean(sample[:, 13:30], axis=1)
        _beta_1 = np.mean(sample[:, 13:20], axis=1)
        _beta_2 = np.mean(sample[:, 20:30], axis=1)
        _gamma = np.mean(sample[:, 30:70], axis=1)
        _gamma_1 = np.mean(sample[:, 30:50], axis=1)
        _gamma_2 = np.mean(sample[:, 50:70], axis=1)

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

        # 将当前样本计算出的特征（形状为 (N_i,)）存入对应的列表中
        current_loop_features = {
            "delta": _delta, "theta": _theta, "alpha": _alpha,
            "alpha_1": _alpha_1, "alpha_2": _alpha_2, "alpha_3": _alpha_3,
            "beta": _beta, "beta_1": _beta_1, "beta_2": _beta_2,
            "gamma": _gamma, "gamma_1": _gamma_1, "gamma_2": _gamma_2,
            "relative_delta": _r_delta, "relative_theta": _r_theta,
            "relative_alpha": _r_alpha, "relative_beta": _r_beta, "relative_gamma": _r_gamma,
            "TBR": TBR, "DAR": DAR, "DTR": DTR, "ABR": ABR, "ATR": ATR,
            "DT_AR": DT_AR, "DT_total_R": DT_total_R
        }

        for key in compiled_features.keys():
            compiled_features[key].append(current_loop_features[key])

    # --- 关键一步：后处理 ---
    # 如果你的 57 个元素的 N 最终其实是【完全一样】的，我们可以直接包装成标准 NumPy 数组输出 (57, N)
    # 如果 57 个元素的 N 【确实各不相同】，我们就保持它们为包含 57 个 Array 的 List
    # final_features = {}
    # try:
    #     for key, val_list in compiled_features.items():
    #         final_features[key] = np.array(val_list)  # 尝试转成 (57, N) 的规整矩阵
    #     print("所有样本的 N 相同，输出特征矩阵形状为: (57, N)")
    # except ValueError:
    #     # 如果报错了，说明 N 真的长短不一，那就保留为 list 结构
    #     final_features = compiled_features
    #     print("样本的 N 长短不一，输出特征为包含 57 个数组的 List")

    return compiled_features


def get_F3478(psd_dict,f):

    print(psd_dict.keys())


    F3 = np.sum(psd_dict["F3-AVG"][:,8*f:13*f],axis=1)
    F4 = np.sum(psd_dict["F4-AVG"][:,8*f:13*f],axis=1)
    F7 = np.sum(psd_dict["F7-AVG"][:,8*f:13*f],axis=1)
    F8 = np.sum(psd_dict["F8-AVG"][:,8*f:13*f],axis=1)



    F3_F4 = (F3-F4)/(F3+F4+1e-4)
    F7_F8 = (F7-F8)/(F7+F8+1e-4)

    print(np.shape(F3_F4))
    print(np.shape(F7_F8))


    F_dict = {
        "F3_F4":F3_F4,
        "F7_F8":F7_F8
    }



    return F_dict



def trans_raw_2_show(in_array):
    high_th = 1.5
    in_array = np.log10(in_array + 1.002)
    # c_psd_max = np.max(in_array)

    # print(c_psd_max,np.percentile(in_array,95),np.min(in_array))

    # if c_psd_max <= high_th:  c_psd_max = high_th

    gp = np.ma.where(in_array > high_th, high_th + np.log2(in_array), in_array)
    # print(c_psd_max, np.percentile(gp, 95), np.min(gp))
    # gp = preprocessing.minmax_scale(gp, axis=0)
    # gp = np.ma.where(one_seg_psd>high_th,high_th , one_seg_psd)

    return gp