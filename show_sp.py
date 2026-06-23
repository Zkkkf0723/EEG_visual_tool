import numpy as np
import json
import os

np.set_printoptions(linewidth=120)

SPT = ["SP","SPAV"]
SPY = ["2","4","5","6"]
names = ["Fp1","Fp2","F3","F4",
         "C3","C4","P3","P4",
         "O1","O2","F7","F8",
         "T3","T4","T5","T6"]

# 将单电极名映射为常模中的耳参考导联名
ear_lead_names = ["Fp1-A1","Fp2-A2","F3-A1","F4-A2",
                  "C3-A1","C4-A2","P3-A1","P4-A2",
                  "O1-A1","O2-A2","F7-A1","F8-A2",
                  "T3-A1","T4-A2","T5-A1","T6-A2"]

bands = ["δ","θ","α1","α2","α3","β"]
nbands = ["δ","θ","α","β"]
outs = ["θ/β","(δ+θ)/(α+β)","δ/α","θ/δ","δ/θ","α/β"]

dict_data = {}
for j in range(len(SPY)):    
    for i in range(len(SPT)):
        filename = "{}{}".format(SPT[i],SPY[j])

        data = np.fromfile("D:/work/SP/{}".format(filename),dtype=np.float32,count=-1).reshape((-1,6))

        # print(filename)
        # print(data[-16:,:])
        # print()
        dict_data[filename] = data[-16:,:]


for j in range(len(SPY)):
    with open("n{}.csv".format(SPY[j]),"w",encoding="utf-8-sig") as wf:
        wf.write(",{}\n".format(",".join(names)))

        data = np.zeros((2,16,6)) # "δ","θ","α1","α2","α3","β"

        data[0,:,:] = dict_data["SP{}".format(SPY[j])]
        data[1,:,:] = dict_data["SPAV{}".format(SPY[j])]

        for i in range(len(bands)):
            wf.write("{}均值,{}\n".format(bands[i],",".join("{:.2f}".format(x) for x in data[0,:,i])))
            wf.write("{}标准差,{}\n".format(bands[i],",".join("{:.2f}".format(x) for x in data[1,:,i])))

        ndata = np.zeros((2,16,4)) # "δ","θ","α","β"

        ndata[:,:,:2] = data[:,:,:2] # "δ","θ"
        ndata[:,:,2] = data[:,:,2] + data[:,:,3] + data[:,:,4] # "α"
        ndata[:,:,3] = data[:,:,5] # "β"

        for i in range(len(nbands)):
            wf.write("{}均值,{}\n".format(nbands[i],",".join("{:.2f}".format(x) for x in ndata[0,:,i])))
            wf.write("{}标准差,{}\n".format(nbands[i],",".join("{:.2f}".format(x) for x in ndata[1,:,i])))

        odata = np.zeros((2,16,6)) # "θ/β","(δ+θ)/(α+β)","δ/α","θ/δ","δ/θ","α/β"
        odata[:,:,0] = ndata[:,:,1]/ndata[:,:,3]     
        odata[:,:,1] = (ndata[:,:,0]+ndata[:,:,1])/(ndata[:,:,2]+ndata[:,:,3])
        odata[:,:,2] = ndata[:,:,0]/ndata[:,:,2]
        odata[:,:,3] = ndata[:,:,1]/ndata[:,:,0]
        odata[:,:,4] = ndata[:,:,0]/ndata[:,:,1] 
        odata[:,:,5] = ndata[:,:,2]/ndata[:,:,3]     

        for i in range(len(outs)):
            xdata = odata[0,:,i]
            ydata = odata[1,:,i]
            wf.write("{}均值,{}\n".format(outs[i],",".join("{:.2f}".format(x) for x in xdata)))     
            wf.write("{}标准差,{}\n".format(outs[i],",".join("{:.2f}".format(x) for x in ydata)))     
            ldata = xdata - 3.*ydata
            ldata[ldata<0] = 0
            rdata = xdata + 3.*ydata

            wf.write("下限(-3σ),{}\n".format(",".join("{:.2f}".format(x) for x in ldata)))     
            wf.write("上限(+3σ),{}\n".format(",".join("{:.2f}".format(x) for x in rdata)))     

# ========== 输出 JSON 参考值文件（格式与常模 combined_result_0611.json 对齐） ==========
# 定义与常模 parameters.py 中 spec_features_list 一致的参数列表
gamma_bands = ["gamma", "gamma_1", "gamma_2", "relative_gamma"]

# 为每个 SPY 编号分别生成 JSON
for j in range(len(SPY)):
    data = np.zeros((2,16,6)) # [mean/std, channel, band]  bands: δ,θ,α1,α2,α3,β
    data[0,:,:] = dict_data["SP{}".format(SPY[j])]
    data[1,:,:] = dict_data["SPAV{}".format(SPY[j])]

    # 计算合并 α 波段
    ndata = np.zeros((2,16,4)) # δ,θ,α,β
    ndata[:,:,:2] = data[:,:,:2]
    ndata[:,:,2] = data[:,:,2] + data[:,:,3] + data[:,:,4]  # α
    ndata[:,:,3] = data[:,:,5]  # β

    # 计算总功率（用于 relative 比值）
    total_power_mean = ndata[0,:,:].sum(axis=1)  # δ+θ+α+β

    # 计算比率
    odata = np.zeros((2,16,6))
    odata[:,:,0] = ndata[:,:,1]/ndata[:,:,3]       # θ/β
    odata[:,:,1] = (ndata[:,:,0]+ndata[:,:,1])/(ndata[:,:,2]+ndata[:,:,3])  # (δ+θ)/(α+β)
    odata[:,:,2] = ndata[:,:,0]/ndata[:,:,2]       # δ/α
    odata[:,:,3] = ndata[:,:,1]/ndata[:,:,0]       # θ/δ
    odata[:,:,4] = ndata[:,:,0]/ndata[:,:,1]       # δ/θ
    odata[:,:,5] = ndata[:,:,2]/ndata[:,:,3]       # α/β

    g_info = {}

    for ch_idx in range(16):
        lead = ear_lead_names[ch_idx]  # 使用耳参考导联名

        # 1. 绝对功率: delta, theta, alpha, alpha_1, alpha_2, alpha_3, beta
        band_map = {
            "delta": (0, 0),    # δ, data[0,ch,0]=mean, data[1,ch,0]=std
            "theta": (0, 1),    # θ
            "alpha_1": (0, 2),  # α1
            "alpha_2": (0, 3),  # α2
            "alpha_3": (0, 4),  # α3
            "beta": (0, 5),    # β
        }
        for bname, (mean_idx, band_idx) in band_map.items():
            key = "{}__{}".format(lead, bname)
            g_info[key] = {
                "mean": float(data[0, ch_idx, band_idx]),
                "std": float(data[1, ch_idx, band_idx]),
                "trimmed_data_length": 0
            }

        # alpha（合并）
        key = "{}__alpha".format(lead)
        g_info[key] = {
            "mean": float(ndata[0, ch_idx, 2]),
            "std": float(ndata[1, ch_idx, 2]),
            "trimmed_data_length": 0
        }

        # 2. 缺少的频段: beta_1, beta_2 设为 0
        for bname in ["beta_1", "beta_2"]:
            key = "{}__{}".format(lead, bname)
            g_info[key] = {
                "mean": 0.0,
                "std": 0.0,
                "trimmed_data_length": 0
            }

        # 3. γ 频段: 全部设为 0
        for bname in ["gamma", "gamma_1", "gamma_2"]:
            key = "{}__{}".format(lead, bname)
            g_info[key] = {
                "mean": 0.0,
                "std": 0.0,
                "trimmed_data_length": 0
            }

        # 4. 相对功率
        total = total_power_mean[ch_idx]
        rel_map = {
            "relative_delta": (0, 0),
            "relative_theta": (0, 1),
            "relative_alpha": (0, 2),  # 使用 ndata 的 α
            "relative_beta": (0, 3),   # 使用 ndata 的 β
        }
        for bname, (mean_idx, nband_idx) in rel_map.items():
            mean_val = ndata[0, ch_idx, nband_idx] / total if total > 0 else 0.0
            # 相对功率的 std 近似用 cv 传递: std_rel ≈ std_abs / total
            std_val = ndata[1, ch_idx, nband_idx] / total if total > 0 else 0.0
            key = "{}__{}".format(lead, bname)
            g_info[key] = {
                "mean": float(mean_val),
                "std": float(std_val),
                "trimmed_data_length": 0
            }

        # relative_gamma = 0
        key = "{}__relative_gamma".format(lead)
        g_info[key] = {
            "mean": 0.0,
            "std": 0.0,
            "trimmed_data_length": 0
        }

        # 5. 比率
        # TBR = θ/β,  DAR = δ/α,  DTR = δ/θ,  ABR = α/β,  ATR = α/θ,  DT_AR = (δ+θ)/(α+β)
        ratio_map = {
            "TBR": (0, 0),           # θ/β = odata[:,:,0]
            "DAR": (2, 2),           # δ/α = odata[:,:,2]
            "DTR": (4, 4),           # δ/θ = odata[:,:,4]
            "ABR": (5, 5),           # α/β = odata[:,:,5]
            "ATR": (3, 3),           # θ/δ = odata[:,:,3]  (但常模中 ATR 是 α/θ)
            "DT_AR": (1, 1),         # (δ+θ)/(α+β) = odata[:,:,1]
        }

        # 计算 ATR = α/θ  (常模中 ATR 是 α/θ 比率)
        theta_mean = data[0, ch_idx, 1]
        atr_mean = ndata[0, ch_idx, 2] / theta_mean if theta_mean > 0 else 0.0
        theta_std = data[1, ch_idx, 1]
        atr_std = ndata[1, ch_idx, 2] / theta_mean if theta_mean > 0 else 0.0
        key = "{}__ATR".format(lead)
        g_info[key] = {
            "mean": float(atr_mean),
            "std": float(atr_std),
            "trimmed_data_length": 0
        }

        for bname, (mean_idx, oidx) in ratio_map.items():
            key = "{}__{}".format(lead, bname)
            g_info[key] = {
                "mean": float(odata[0, ch_idx, oidx]),
                "std": float(odata[1, ch_idx, oidx]),
                "trimmed_data_length": 0
            }

        # DT_total_R = (δ+θ)/(δ+θ+α+β)
        dt = ndata[0, ch_idx, 0] + ndata[0, ch_idx, 1]
        total_all = dt + ndata[0, ch_idx, 2] + ndata[0, ch_idx, 3]
        dtt_mean = dt / total_all if total_all > 0 else 0.0
        dt_std = ndata[1, ch_idx, 0] + ndata[1, ch_idx, 1]
        dtt_std_val = dt_std / total_all if total_all > 0 else 0.0
        key = "{}__DT_total_R".format(lead)
        g_info[key] = {
            "mean": float(dtt_mean),
            "std": float(dtt_std_val),
            "trimmed_data_length": 0
        }

    ref_data = {
        "sp_reference_{}".format(SPY[j]): {
            "g_info": g_info
        }
    }

    output_dir = os.path.dirname(os.path.abspath(__file__))
    json_filename = os.path.join(output_dir, "sp_reference_{}.json".format(SPY[j]))
    with open(json_filename, "w", encoding="utf-8") as jf:
        json.dump(ref_data, jf, ensure_ascii=False, indent=2)

    print("✅ 已输出参考值 JSON 文件: {}".format(json_filename))
