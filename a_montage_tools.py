# 添加具体导联位置信息
from a_eeg_tool import *
from a_filter_tool import *

single_lead_dict = {0: ['FP1', 'A1'], 1: ['FP2', 'A2'], 2: ['F3', 'A1'], 3: ['F4', 'A2'],
                    4: ['C3', 'A1'], 5: ['C4', 'A2'], 6: ['P3', 'A1'], 7: ['P4', 'A2'],
                    8: ['O1', 'A1'], 9: ['O2', 'A2'], 10: ['F7', 'A1'], 11: ['F8', 'A2'],
                    12: ['T3', 'A1'], 13: ['T4', 'A2'], 14: ['T5', 'A1'], 15: ['T6', 'A2']}

bipolar_lead_dict = {0: ['FP1', 'F3'], 1: ['FP2', 'F4'], 2: ['F3', 'C3'], 3: ['F4', 'C4'],
                     4: ['C3', 'P3'], 5: ['C4', 'P4'], 6: ['P3', 'O1'], 7: ['P4', 'O2'],
                     8: ['FP1', 'F7'], 9: ['FP2', 'F8'], 10: ['F7', 'T3'], 11: ['F8', 'T4'],
                    12: ['T3', 'T5'], 13: ['T4', 'T6'], 14: ['T5', 'O1'], 15: ['T6', 'O2']}

LEFT_NAME = [
    "Fp1",
    "F3",
    "C3",
    "P3",
    "Fp1",
    "F7",
    "T3",
    "T5",
    "Fp1",
    "C3",
    "Fp1",
    "T3", ]

RIGHT_NAME = [
    "Fp2",
    "F4",
    "C4",
    "P4",
    "Fp2",
    "F8",
    "T4",
    "T6",
    "Fp2",
    "C4",
    "Fp2",
    "T4",
]


bipolar_list = [
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
"Fp1-C3",
"Fp2-C4",
"C3-O1",
"C4-O2",
"Fp1-T3",
"Fp2-T4",
"T3-O1",
"T4-O2",
]

bkg_name_dict = {1: 'B.alpha', 2: 'B.K-complex', 3: 'B.vertex', 4: 'B.spindle'}

base_th = 0.95

EAR_2_BIO_MAP_DICT_8 = {'Fp1-A1':"Fp1-C3",
                        'C3-A1':"C3-O1",
                        'T3-A1':"Fp1-T3",
                        'O1-A1':"T3-O1",
                        'Fp2-A2':"Fp2-C4",
                        'C4-A2':"C4-O2",
                        'T4-A2':"Fp2-T4",
                        'O2-A2':"T4-O2",
                        'Fp1-C3':'Fp1-C3', 'Fp2-C4':'Fp2-C4', 'C3-O1':'C3-O1', 'C4-O2':'C4-O2',
                        'Fp1-T3':'Fp1-T3', 'Fp2-T4':'Fp2-T4', 'T3-O1':'T3-O1', 'T4-O2':'T4-O2'}

A1A2_2_BIO_MAP_DICT_8 ={'Fp1-A1A2':"Fp1-C3",
                        'C3-A1A2':"C3-O1",
                        'T3-A1A2':"Fp1-T3",
                        'O1-A1A2':"T3-O1",
                        'Fp2-A1A2':"Fp2-C4",
                        'C4-A1A2':"C4-O2",
                        'T4-A1A2':"Fp2-T4",
                        'O2-A1A2':"T4-O2",
                        'Fp1-C3': 'Fp1-C3', 'Fp2-C4': 'Fp2-C4', 'C3-O1': 'C3-O1', 'C4-O2': 'C4-O2',
                        'Fp1-T3': 'Fp1-T3', 'Fp2-T4': 'Fp2-T4', 'T3-O1': 'T3-O1', 'T4-O2': 'T4-O2'
                        }





AV_2_BIO_MAP_DICT_8 = {'Fp1-AV':"Fp1-C3",
                        'C3-AV':"C3-O1",
                        'T3-AV':"Fp1-T3",
                        'O1-AV':"T3-O1",
                        'Fp2-AV':"Fp2-C4",
                        'C4-AV':"C4-O2",
                        'T4-AV':"Fp2-T4",
                        'O2-AV':"T4-O2",
                        'Fp1-C3': 'Fp1-C3', 'Fp2-C4': 'Fp2-C4', 'C3-O1': 'C3-O1', 'C4-O2': 'C4-O2',
                        'Fp1-T3': 'Fp1-T3', 'Fp2-T4': 'Fp2-T4', 'T3-O1': 'T3-O1', 'T4-O2': 'T4-O2'
                        }


EAR_2_BIO_MAP_DICT_16 = {'Fp1-A1':"Fp1-F3", 'F3-A1':"F3-C3", 'C3-A1':"C3-P3", 'P3-A1':"P3-O1",
                        'O1-A1':"P3-O1", 'F7-A1':"F7-T3", 'T3-A1':"T3-T5", 'T5-A1':"T5-O1",
                        'Fp2-A2':"Fp2-F4", 'F4-A2':"F4-C4", 'C4-A2':"C4-P4", 'P4-A2':"P4-O2",
                        'O2-A2':"P4-O2",  'F8-A2':"F8-T4", 'T4-A2':"T4-T6", 'T6-A2':"T6-O2",
                         'Fpz-A1':'Fpz-Fz', 'Fz-A1':'Fz-Pz', 'Pz-A1':'Cz-Pz', 'Oz-A1':'Pz-Oz', 'Cz-A1':'Cz-Pz',

                        'Fp1-F3':'Fp1-F3', 'F3-C3':'F3-C3', 'C3-P3':'C3-P3', 'P3-O1':'P3-O1',
                        'Fp1-F7':'Fp1-F7', 'F7-T3':'F7-T3', 'T3-T5':'T3-T5', 'T5-O1':'T5-O1',
                        'Fp2-F4':'Fp2-F4', 'F4-C4':'F4-C4', 'C4-P4':'C4-P4', 'P4-O2': 'P4-O2',
                        'Fp2-F8':'Fp2-F8', 'F8-T4':'F8-T4', 'T4-T6':'T4-T6', 'T6-O2':'T6-O2',
                        'Fpz-Fz':'Fpz-Fz','Fz-Pz':'Fz-Pz', 'Cz-Pz':'Cz-Pz','Pz-Oz':'Pz-Oz',
                         }

A1A2_2_BIO_MAP_DICT_16 ={'Fp1-A1A2':"Fp1-F3", 'F3-A1A2':"F3-C3", 'C3-A1A2':"C3-P3", 'P3-A1A2':"P3-O1",
                        'O1-A1A2':"P3-O1", 'F7-A1A2':"F7-T3", 'T3-A1A2':"T3-T5", 'T5-A1A2':"T5-O1",
                        'Fp2-A1A2':"Fp2-F4", 'F4-A1A2':"F4-C4", 'C4-A1A2':"C4-P4", 'P4-A1A2':"P4-O2",
                        'O2-A1A2':"P4-O2",  'F8-A1A2':"F8-T4", 'T4-A1A2':"T4-T6", 'T6-A1A2':"T6-O2",
                        'Fpz-A1A2':'Fpz-Fz', 'Fz-A1A2':'Fz-Pz', 'Pz-A1A2':'Cz-Pz', 'Oz-A1A2':'Pz-Oz', 'Cz-A1A2':'Cz-Pz',

                        'Fp1-F3': 'Fp1-F3', 'F3-C3': 'F3-C3', 'C3-P3': 'C3-P3', 'P3-O1': 'P3-O1',
                        'Fp1-F7': 'P3-O1', 'F7-T3': 'F7-T3', 'T3-T5': 'T3-T5', 'T5-O1': 'T5-O1',
                        'Fp2-F4': 'Fp2-F4', 'F4-C4': 'F4-C4', 'C4-P4': 'C4-P4', 'P4-O2': 'P4-O2',
                        'Fp2-F8': 'Fp2-F8', 'F8-T4': 'F8-T4', 'T4-T6': 'T4-T6', 'T6-O2': 'T6-O2',
                        'Fpz-Fz':'Fpz-Fz','Fz-Pz':'Fz-Pz','Cz-Pz':'Cz-Pz','Pz-Oz':'Pz-Oz',
                         }


AV_2_BIO_MAP_DICT_16 = {'Fp1-AV':"Fp1-F3", 'F3-AV':"F3-C3", 'C3-AV':"C3-P3", 'P3-AV':"P3-O1",
                        'O1-AV':"P3-O1", 'F7-AV':"F7-T3",'T3-AV':"T3-T5", 'T5-AV':"T5-O1",
                        'Fp2-AV':"Fp2-F4", 'F4-AV':"F4-C4", 'C4-AV':"C4-P4", 'P4-AV':"P4-O2",
                        'O2-AV':"P4-O2",  'F8-AV':"F8-T4", 'T4-AV':"T4-T6", 'T6-AV':"T6-O2",
                        'Fpz-AV':'Fpz-Fz', 'Fz-AV':'Fz-Pz', 'Pz-AV':'Cz-Pz', 'Oz-AV':'Pz-Oz', 'Cz-AV':'Cz-Pz',

                        'Fp1-F3': 'Fp1-F3', 'F3-C3': 'F3-C3', 'C3-P3': 'C3-P3', 'P3-O1': 'P3-O1',
                        'Fp1-F7': 'P3-O1', 'F7-T3': 'F7-T3', 'T3-T5': 'T3-T5', 'T5-O1': 'T5-O1',
                        'Fp2-F4': 'Fp2-F4', 'F4-C4': 'F4-C4', 'C4-P4': 'C4-P4', 'P4-O2': 'P4-O2',
                        'Fp2-F8': 'Fp2-F8', 'F8-T4': 'F8-T4', 'T4-T6': 'T4-T6', 'T6-O2': 'T6-O2',
                        'Fpz-Fz':'Fpz-Fz','Fz-Pz':'Fz-Pz','Cz-Pz':'Cz-Pz','Pz-Oz':'Pz-Oz',
                        }





LEFT_LEADS_LIST_A_8 = ['Fp1-A1', 'C3-A1', 'T3-A1', 'O1-A1']
LEFT_LEADS_LIST_B_8 = ['Fp1-C3', 'C3-O1', 'T3-O1', 'Fp1-T3']
LEFT_LEADS_LIST_A1A2_8 = ['Fp1-A1A2', 'C3-A1A2', 'T3-A1A2', 'O1-A1A2']
LEFT_LEADS_LIST_AVG_8 = ['Fp1-AV', 'C3-AV', 'T3-AV', 'O2-AV']

RIGHT_LEADS_LIST_A_8 = ['Fp2-A2', 'C4-A2', 'T4-A2', 'O2-A2']
RIGHT_LEADS_LIST_B_8 = ['Fp2-C4', 'C4-O2', 'T4-O2', 'Fp2-T4']
RIGHT_LEADS_LIST_A1A2_8 = ['Fp2-A1A2', 'C4-A1A2', 'T4-A1A2', 'O2-A1A2']
RIGHT_LEADS_LIST_AVG_8 = ['Fp2-AV', 'C4-AV', 'T4-AV', 'O2-AV']

F_LEADS_LIST_B_8 = ['Fp1-C3', 'Fp2-C4', ]
C_LEADS_LIST_B_8 = ['C3-O1', 'C4-O2']
T_LEADS_LIST_B_8 = ['T3-O1', 'Fp2-T4']
O_LEADS_LIST_B_8 = ['C4-O2', 'C3-O1', 'T3-O1', 'T4-O2']
F_LEADS_LIST_8 = ['Fp1-A1', 'Fp2-A2']
C_LEADS_LIST_8 = ['C3-A1', 'C4-A2']
T_LEADS_LIST_8 = ['T3-A1', 'T4-A2']
O_LEADS_LIST_8 = ['O1-A1', 'O2-A2']



LEFT_LEADS_LIST_A_16 = ['Fp1-A1', 'F3-A1', 'C3-A1', 'P3-A1', 'O1-A1', 'F7-A1', 'T3-A1', 'T5-A1']
LEFT_LEADS_LIST_B_16 = ['Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1']
LEFT_LEADS_LIST_A1A2_16 = ['Fp1-A1A2', 'F3-A1A2', 'C3-A1A2', 'P3-A1A2', 'O1-A1A2', 'F7-A1A2', 'T3-A1A2', 'T5-A1A2']
LEFT_LEADS_LIST_AVG_16 = ['Fp1-AV', 'F3-AV', 'C3-AV', 'P3-AV', 'O1-AV', 'F7-AV', 'T3-AV', 'T5-AV']

RIGHT_LEADS_LIST_A_16 = ['Fp2-A2', 'F4-A2', 'C4-A2', 'P4-A2', 'O2-A2',  'F8-A2', 'T4-A2', 'T6-A2']
RIGHT_LEADS_LIST_B_16 = ['Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'Fp2-F8',  'F8-T4', 'T4-T6', 'T6-O2']
RIGHT_LEADS_LIST_A1A2_16 = ['Fp2-A1A2', 'F4-A1A2', 'C4-A1A2', 'P4-A1A2', 'O2-A1A2',  'F8-A1A2', 'T4-A1A2', 'T6-A1A2']
RIGHT_LEADS_LIST_AVG_16 = ['Fp2-AV', 'F4-AV', 'C4-AV', 'P4-AV', 'O2-AV',  'F8-AV', 'T4-AV', 'T6-AV']




LEFT_LEADS_LIST = ['Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1']
RIGHT_LEADS_LIST = ['Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'Fp2-F8',  'F8-T4', 'T4-T6', 'T6-O2']
Z_AREA_LIST = ["Fpz-Fz","Fz-Pz","Cz-Pz","Pz-Oz"]

# F_LEADS_LIST = ['Fp1-F3','Fp2-F4','Fp1-F7', 'Fp2-F8']
# C_LEADS_LIST = ['C3-P3', 'F3-C3','C4-P4', 'F4-C4']
# T_LEADS_LIST = ['F7-T3', 'T3-T5', 'T5-O1', 'F8-T4', 'T4-T6', 'T6-O2']
# O_LEADS_LIST = ['P3-O1', 'T5-O1','P4-O2','T6-O2']
F_LEADS_LIST_A = ['Fp1-A1', 'Fp2-A2', 'F7-A1', 'F8-A2', 'F3-A1', 'F4-A2',
                  'Fp1-AV', 'Fp2-AV', 'F7-AV', 'F8-AV', 'F3-AV', 'F4-AV',
                  'Fp1-A1A2', 'Fp2-A1A2', 'F7-A1A2', 'F8-A1A2', 'F3-A1A2', 'F4-A1A2']
C_LEADS_LIST_A = ['C3-A1', 'C4-A2', 'P3-A1', 'P4-A2','Cz-Pz',
                  'C3-AV', 'C4-AV', 'P3-AV', 'P4-AV','Cz-AV',
                  'C3-A1A2', 'C4-A1A2', 'P3-A1A2', 'P4-A1A2','Cz-A1A2']
T_LEADS_LIST_A = ['T3-A1', 'T4-A2', 'T5-A1', 'T6-A2',
                  'T3-AV', 'T4-AV', 'T5-AV', 'T6-AV',
                  'T3-A1A2', 'T4-A1A2', 'T5-A1A2', 'T6-A1A2',
                  ]
O_LEADS_LIST_A = ['O1-A1', 'O2-A2',
                  'O1-AV', 'O2-AV',
                  'O1-A1A2', 'O2-A1A2']

F_LEADS_LIST_B = ['Fp1-F3', 'F3-C3', 'Fp1-F7', 'F7-T3','Fp2-F4', 'F4-C4','Fp2-F8', 'F8-T4']
C_LEADS_LIST_B = ['F3-C3', 'C3-P3','F4-C4', 'C4-P4']
T_LEADS_LIST_B = ['T3-T5', 'T4-T6', 'T5-O1', 'T6-O2']
O_LEADS_LIST_B = ['P3-O1', 'P4-O2','T5-O1','T6-O2']



AEEG_TYPE = ["aEEG_max","aEEG_min"]
AEEG_A_DICT = {"aEEG_max":["L","R"],"aEEG_min":["L","R"]}


QEEG_TYPE = ["spectrogram",
             "ADR",
             "ATR",
             "ABR",
             "ASI",
             "ALV",
             "RASI",
             "entropy",
             "RBP_A",
             "RBP_B",
             "RBP_T",
             "RBP_D",
             "RBP_G",
             "spectrogram_display",
             "DAR",
             "TAR",
             "BAR",
             "TBR",
             "DTABR",
             "A_DT_R",
             "A_ASI",
             "DT_ASI",
             "LZC",
             "Arousal"
             ]

QEEG_A_DICT = {"spectrogram":["L","R","LR","F","C","O","T","FL","FR","CL","CR","OL","OR","TL","TR"],
               "ADR":["L","R","LR"],
               "ATR":["L","R","LR"],
               "ABR":["L","R","LR"],
               "ALV":["L","R","LR"],
               "ASI":["LR"],
               "RASI":["LR"],
               "entropy":["L","R","LR"],
               "RBP_A":["L","R","LR"],
               "RBP_B":["L","R","LR"],
               "RBP_T":["L","R","LR"],
               "RBP_D":["L","R","LR"],
               "RBP_G":["L","R","LR"],
               "spectrogram_display":["L","R","LR","F","C","O","T","FL","FR","CL","CR","OL","OR","TL","TR"],
               "DAR":["L","R","LR"],
               "TAR":["L","R","LR"],
               "BAR":["L","R","LR"],
               "TBR":["L","R","LR"],
               "DTABR":["L","R","LR"],
               "A_DT_R":["L","R","LR"],
               "A_ASI":["LR"],
               "DT_ASI":["LR"],
               "LZC":["LR"],
               "Arousal":["LR"]
}

AI_TYPE = ["IED_index","ESZ_index","ART_index","SLOW_index","KVS_index","ALPHA_index","LOWV_index"]

AI_A_DICT = {"IED_index":["L","R","LR","F","C","O","T","Z"],
             "ESZ_index":["L","R","LR"],
             "ART_index":["L","R","LR"],
             "SLOW_index":["LR"],
             "KVS_index":["LR"],
             "LOWV_index":["LR"],
             "ALPHA_index":["LR"],
            }

#BS_TYPE = ["BS_index","SLOW_index","KVS_index","ALPHA_index"]
BS_TYPE = ["BS_index", "BS_list","BS_rate","normal_rate","burst_rate","suppression_rate"]


BS_A_DICT = {
            "BS_index":["L","R","LR","Z"],
            "BS_list":["LR"],

             "BS_rate" :["LR"],
             "normal_rate":["LR"],
             "burst_rate":["LR"],
             "suppression_rate":["LR"],
             "SLOW_index": ["LR"],
             "KVS_index": ["LR"],
             "ALPHA_index": ["LR"]
             }

type_8_keys = ['A1', 'A2', 'Fp1', 'T3', 'C3', 'O1', 'Fp2', 'C4', 'T4', 'O2']
type_10_keys = ['A1', 'A2', 'Fp1', 'T3', 'C3', 'O1', 'Fp2', 'C4', 'T4', 'O2']

type_16_keys = ["A1","A2",
                "Fp1","F3","F7","T3","C3","T5","P3","O1",
                "Fp2","F4","F8","C4","T4","P4","T6","O2"]

type_18_keys = ["A1","A2",
                "Fp1","F3","F7","T3","C3","T5","P3","O1",
                "Fp2","F4","F8","C4","T4","P4","T6","O2"]

type_21_keys = ["A1","A2",
                "Fp1","F3","F7","T3","C3","T5","P3","O1",
                "Fp2","F4","F8","C4","T4","P4","T6","O2",
                "Fpz","Fz","Cz",'Pz',"Oz"]
type_23_keys = ["A1","A2",
                "Fp1","F3","F7","T3","C3","T5","P3","O1",
                "Fp2","F4","F8","C4","T4","P4","T6","O2",
                "Fpz","Fz","Cz",'Pz',"Oz"]

type_22_keys = ["A1","A2",
                "Fp1","F3","F7","T3","C3","T5","P3","O1",
                "Fp2","F4","F8","C4","T4","P4","T6","O2",
                "Fpz","Fz","Pz","Cz","SPHL","SPHR"]

type_25_keys = ["A1","A2",
                "Fp1","F3","F7","T3","C3","T5","P3","O1",
                "Fp2","F4","F8","C4","T4","P4","T6","O2",
                "Fpz","Fz","Pz","Cz","SPHL","SPHR"]




def lead_type_check(lead_type,lead_keys):


    if lead_type not in ["8","10","16","18","20","23","25"]:
        return  False

    if lead_type == "8" and set(type_8_keys) == set(lead_keys):
        return True
    if lead_type == "10" and set(type_10_keys) == set(lead_keys):
        return True

    if lead_type == "16" and set(type_16_keys) == set(lead_keys):
        return True

    if lead_type == "18" and set(type_18_keys) == set(lead_keys):
        return True

    if lead_type == "21" and set(type_21_keys) == set(lead_keys):
        return True
    if lead_type == "23" and set(type_23_keys) == set(lead_keys):
        return True

    if lead_type == "25" and set(type_25_keys) == set(lead_keys):
        return True

    return  False





sensitivity_threshold_dict = {
    "S1": [0.50, 0.45, 3],
    "S2": [0.55, 0.5, 3],
    "S3": [0.60, 0.60, 4],
    "S4": [0.70, 0.70, 4],
    "S5": [0.80, 0.80, 4],
    "S6": [0.85, 0.80, 4],
    "S7": [0.90, 0.85, 3],
    "S8": [0.95, 0.95, 3],
    "S10": [0.99, 0.98, 5],
    "SD": [base_th, base_th, 3],
    "TEST": [base_th, base_th-0.05, 2]}


def get_threshold_by_sens(sens):
    try:
        sens = int(float(sens))
    except:
        threshold = 'SD'
    if sens < 40:
        threshold = 'S10'
    elif 40 <= sens <= 60:
        threshold = 'S8'
    elif 60 <= sens < 70:
        threshold = 'S7'
    elif 70 <= sens < 80:
        threshold = 'S6'
    elif 80 <= sens < 90:
        threshold = 'S5'
    elif 90 <= sens < 95:
        threshold = 'S4'
    elif 95 <= sens < 100:
        threshold = 'S3'
    elif 100 == sens:
        threshold = 'TEST'
    else:
        threshold = 'SD'

    return sensitivity_threshold_dict[threshold]

def para_check(algorithm_para_dict, lead_keys):
    sensitivity = algorithm_para_dict["sensitiviy"]
    lead_type = algorithm_para_dict["lead_type"]
    raw_sample_rate = int(algorithm_para_dict["sample_rate"])
    signal_length = int(algorithm_para_dict["eeg_length"])

    #print(algorithm_para_dict)

    # 小于5s
    if signal_length <= 5 * raw_sample_rate:

        return False
    # lead type 不匹配
    if lead_type not in ["8", "10","16","18","20","22","23", "25","32", "64"]:

        return False

    if not lead_type_check(lead_type, lead_keys):

        return False

    return True

def sigmoid(x):
    return  1/(1+np.exp(-x))



def output_adjust_AI_OFF(a_dict,q_dict,q_type,a_type,lead_type):


    out_type_dict = {}
    if lead_type == 'SINGLE':
        for k in a_dict["LR"].keys():
            if  k in q_type:
                out_type_dict[k] = {}
                out_type_dict[k]["LR"] = a_dict["LR"][k]
        for k in q_dict["LR"].keys():
            if k in q_type:
                out_type_dict[k] = {}
                out_type_dict[k]["LR"] = q_dict["LR"][k]


    else:
        if a_type == "LR":
            area_type_list = ["L","R","LR"]
        elif a_type == "FCOT":
            area_type_list = ["F","C","O","T","LR"]
        elif a_type == "FCOTLR":
            area_type_list = ["FL","FR","CL","CR","OL","OR","TL","TR","LR"]
        elif a_type == "ALL":
            area_type_list = ["L","R","LR","F","C","O","T","FL","FR","CL","CR","OL","OR","TL","TR"]

        out_type_dict = {}

        for one in q_type:
            out_type_dict[one] = {}

        for one_a_type in area_type_list:
            for one_q_type in q_type:

                if one_q_type in AEEG_TYPE and one_a_type in AEEG_A_DICT[one_q_type]:
                    out_type_dict[one_q_type][one_a_type] = a_dict[one_a_type][one_q_type]
                    #print(one_a_type,one_q_type)

                if one_q_type in QEEG_TYPE and one_a_type in QEEG_A_DICT[one_q_type]:
                    out_type_dict[one_q_type][one_a_type] = q_dict[one_a_type][one_q_type]

    return out_type_dict




def output_adjust(a_dict,q_dict,ai_dict,bs_dict,q_type,a_type,lead_type):


    out_type_dict = {}
    if lead_type == 'SINGLE':
        for k in a_dict["LR"].keys():
            if  k in q_type:
                out_type_dict[k] = {}
                out_type_dict[k]["LR"] = a_dict["LR"][k]
        for k in q_dict["LR"].keys():
            if k in q_type:
                out_type_dict[k] = {}
                out_type_dict[k]["LR"] = q_dict["LR"][k]
        for k in ai_dict["LR"].keys():
            if k in q_type:
                out_type_dict[k] = {}
                out_type_dict[k]["LR"] = ai_dict["LR"][k]
        for k in bs_dict["LR"].keys():
            if k in q_type:
                out_type_dict[k] = {}
                out_type_dict[k]["LR"] = bs_dict["LR"][k]

    else:
        if a_type == "LR":
            area_type_list = ["L","R","LR"]
        elif a_type == "FCOT":
            area_type_list = ["F","C","O","T","LR"]
        elif a_type == "FCOTLR":
            area_type_list = ["FL","FR","CL","CR","OL","OR","TL","TR","LR"]
        elif a_type == "ALL":
            area_type_list = ["L","R","LR","F","C","O","T","FL","FR","CL","CR","OL","OR","TL","TR"]

        out_type_dict = {}

        for one in q_type:
            out_type_dict[one] = {}

        for one_a_type in area_type_list:
            for one_q_type in q_type:


                if one_q_type in AEEG_TYPE and one_a_type in AEEG_A_DICT[one_q_type]:
                    out_type_dict[one_q_type][one_a_type] = a_dict[one_a_type][one_q_type]
                    #print(one_a_type,one_q_type)

                if one_q_type in QEEG_TYPE and one_a_type in QEEG_A_DICT[one_q_type]:

                    out_type_dict[one_q_type][one_a_type] = q_dict[one_a_type][one_q_type]

                if one_q_type in AI_TYPE and one_a_type in AI_A_DICT[one_q_type]:
                    #print(one_a_type, one_q_type)
                    out_type_dict[one_q_type][one_a_type] = ai_dict[one_a_type][one_q_type]


                if one_q_type in BS_TYPE and one_a_type in BS_A_DICT[one_q_type]:

                    out_type_dict[one_q_type][one_a_type] = bs_dict[one_a_type][one_q_type]


    return out_type_dict




from scipy.signal import butter, lfilter
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



def get_bipolar_data_caueeg(in_eeg_dict,l_cut= 0.5,h_cut=70):

    has_avg_suffix = "Fp1-AVG" in in_eeg_dict
    
    def get_ch(name):
        if has_avg_suffix:
            key = name + "-AVG"
            if key in in_eeg_dict:
                return in_eeg_dict[key]
            key_lower = name.lower() + "-avg"
            for k in in_eeg_dict:
                if k.lower() == key_lower:
                    return in_eeg_dict[k]
            raise KeyError(f"{key} not found")
        else:
            if name in in_eeg_dict:
                return in_eeg_dict[name]
            name_lower = name.lower()
            for k in in_eeg_dict:
                if k.lower() == name_lower:
                    return in_eeg_dict[k]
            raise KeyError(f"{name} not found")
    
    Fp1 = get_ch("Fp1")
    F3 = get_ch("F3")
    C3 = get_ch("C3")
    P3 = get_ch("P3")
    O1 = get_ch("O1")
    Fp2 = get_ch("Fp2")
    F4 = get_ch("F4")
    C4 = get_ch("C4")
    P4 = get_ch("P4")
    O2 = get_ch("O2")
    F7 = get_ch("F7")
    T3 = get_ch("T3")
    T5 = get_ch("T5")
    F8 = get_ch("F8")
    T4 = get_ch("T4")
    T6 = get_ch("T6")
    _midline_available = {}
    for _mch in ["FZ", "CZ", "PZ", "OZ"]:
        try:
            _midline_available[_mch] = get_ch(_mch)
        except KeyError:
            _midline_available[_mch] = None
    FZ = _midline_available.get("FZ")
    CZ = _midline_available.get("CZ")
    PZ = _midline_available.get("PZ")
    OZ = _midline_available.get("OZ")
    
    if has_avg_suffix:
        A1 = (Fp1+F3+C3+P3+O1+F7+T3+T5)/8
        A2 = (Fp2+F4+C4+P4+O2+F8+T4+T6)/8
    else:
        A1 = get_ch("A1")
        A2 = get_ch("A2")


    # l_cut = 0.5
    # h_cut = 70

    out_dict = {
        "Fp1-A1": butter_bandpass_filter(get_ch("Fp1") - A1, l_cut, h_cut, fs=256),
        "Fp2-A2": butter_bandpass_filter(get_ch("Fp2") - A2, l_cut, h_cut, fs=256),
        "F3-A1": butter_bandpass_filter(get_ch("F3") - A1, l_cut, h_cut, fs=256),
        "F4-A2": butter_bandpass_filter(get_ch("F4") - A2, l_cut, h_cut, fs=256),
        "C3-A1": butter_bandpass_filter(get_ch("C3") - A1, l_cut, h_cut, fs=256),
        "C4-A2": butter_bandpass_filter(get_ch("C4") - A2, l_cut, h_cut, fs=256),
        "P3-A1": butter_bandpass_filter(get_ch("P3") - A1, l_cut, h_cut, fs=256),
        "P4-A2": butter_bandpass_filter(get_ch("P4") - A2, l_cut, h_cut, fs=256),
        "O1-A1": butter_bandpass_filter(get_ch("O1") - A1, l_cut, h_cut, fs=256),
        "O2-A2": butter_bandpass_filter(get_ch("O2") - A2, l_cut, h_cut, fs=256),
        "F7-A1": butter_bandpass_filter(get_ch("F7") - A1, l_cut, h_cut, fs=256),
        "F8-A2": butter_bandpass_filter(get_ch("F8") - A2, l_cut, h_cut, fs=256),
        "T3-A1": butter_bandpass_filter(get_ch("T3") - A1, l_cut, h_cut, fs=256),
        "T4-A2": butter_bandpass_filter(get_ch("T4") - A2, l_cut, h_cut, fs=256),
        "T5-A1": butter_bandpass_filter(get_ch("T5") - A1, l_cut, h_cut, fs=256),
        "T6-A2": butter_bandpass_filter(get_ch("T6") - A2, l_cut, h_cut, fs=256),




        "Fp1-F3": butter_bandpass_filter(Fp1 - F3, low_cut=l_cut, high_cut=h_cut, fs=256),
        "Fp2-F4": butter_bandpass_filter(Fp2 - F4, low_cut=l_cut, high_cut=h_cut, fs=256),
        "F3-C3": butter_bandpass_filter(F3 - C3, low_cut=l_cut, high_cut=h_cut, fs=256),
        "F4-C4": butter_bandpass_filter(F4 - C4, low_cut= l_cut, high_cut=h_cut, fs=256),
        "C3-P3": butter_bandpass_filter(C3 - P3, low_cut= l_cut, high_cut=h_cut, fs=256),
        "C4-P4": butter_bandpass_filter(C4 - P4, low_cut= l_cut, high_cut=h_cut, fs=256),
        "P3-O1": butter_bandpass_filter(P3 - O1, low_cut= l_cut, high_cut=h_cut, fs=256),
        "P4-O2": butter_bandpass_filter(P4 - O2, low_cut= l_cut, high_cut=h_cut, fs=256),
        "Fp1-F7": butter_bandpass_filter(Fp1 - F7, low_cut= l_cut, high_cut=h_cut, fs=256),
        "Fp2-F8": butter_bandpass_filter(Fp2 - F8, low_cut= l_cut, high_cut=h_cut, fs=256),
        "F7-T3": butter_bandpass_filter(F7 - T3, low_cut= l_cut, high_cut=h_cut, fs=256),
        "F8-T4": butter_bandpass_filter(F8 - T4, low_cut= l_cut, high_cut=h_cut, fs=256),
        "T3-T5": butter_bandpass_filter(T3 - T5, low_cut= l_cut, high_cut=h_cut, fs=256),
        "T4-T6": butter_bandpass_filter(T4 - T6, low_cut= l_cut, high_cut=h_cut, fs=256),
        "T5-O1": butter_bandpass_filter(T5 - O1, low_cut= l_cut, high_cut=h_cut, fs=256),
        "T6-O2": butter_bandpass_filter(T6 - O2, low_cut= l_cut, high_cut=h_cut, fs=256),


        "Fp1-AVG": butter_bandpass_filter(Fp1, low_cut= l_cut, high_cut=h_cut, fs=256),
        "Fp2-AVG": butter_bandpass_filter(Fp2, low_cut= l_cut, high_cut=h_cut, fs=256),
        "F3-AVG": butter_bandpass_filter(F3, low_cut= l_cut, high_cut=h_cut, fs=256),
        "F4-AVG": butter_bandpass_filter(F4, low_cut= l_cut, high_cut=h_cut, fs=256),
        "C3-AVG": butter_bandpass_filter(C3, low_cut= l_cut, high_cut=h_cut, fs=256),
        "C4-AVG": butter_bandpass_filter(C4, low_cut= l_cut, high_cut=h_cut, fs=256),
        "P3-AVG": butter_bandpass_filter(P3, low_cut= l_cut, high_cut=h_cut, fs=256),
        "P4-AVG": butter_bandpass_filter(P4, low_cut= l_cut, high_cut=h_cut, fs=256),
        "O1-AVG": butter_bandpass_filter(O1, low_cut= l_cut, high_cut=h_cut, fs=256),
        "O2-AVG": butter_bandpass_filter(O2, low_cut= l_cut, high_cut=h_cut, fs=256),
        "F7-AVG": butter_bandpass_filter(F7, low_cut= l_cut, high_cut=h_cut, fs=256),
        "F8-AVG": butter_bandpass_filter(F8, low_cut= l_cut, high_cut=h_cut, fs=256),
        "T3-AVG": butter_bandpass_filter(T3, low_cut= l_cut, high_cut=h_cut, fs=256),
        "T4-AVG": butter_bandpass_filter(T4, low_cut= l_cut, high_cut=h_cut, fs=256),
        "T5-AVG": butter_bandpass_filter(T5, low_cut= l_cut, high_cut=h_cut, fs=256),
        "T6-AVG": butter_bandpass_filter(T6, low_cut= l_cut, high_cut=h_cut, fs=256),
    }
    # 中线电极 AV 参考（AV = (A1 + A2) / 2），仅在中线电极存在时计算
    _AV = (A1 + A2) / 2
    if FZ is not None:
        out_dict["Fz-AV"] = butter_bandpass_filter(FZ - _AV, l_cut, h_cut, fs=256)
    if CZ is not None:
        out_dict["Cz-AV"] = butter_bandpass_filter(CZ - _AV, l_cut, h_cut, fs=256)
    if PZ is not None:
        out_dict["Pz-AV"] = butter_bandpass_filter(PZ - _AV, l_cut, h_cut, fs=256)
    # 中线电极导联（可选，如果电极不存在则跳过）
    if FZ is not None and PZ is not None:
        out_dict["Fz-Pz"] = butter_bandpass_filter(FZ - PZ, low_cut=l_cut, high_cut=h_cut, fs=256)
    if CZ is not None and PZ is not None:
        out_dict["Cz-Pz"] = butter_bandpass_filter(CZ - PZ, low_cut=l_cut, high_cut=h_cut, fs=256)
    if PZ is not None and OZ is not None:
        out_dict["Pz-Oz"] = butter_bandpass_filter(PZ - OZ, low_cut=l_cut, high_cut=h_cut, fs=256)
    if FZ is not None:
        out_dict["Fz-AVG"] = butter_bandpass_filter(FZ, low_cut=l_cut, high_cut=h_cut, fs=256)
    if CZ is not None:
        out_dict["Cz-AVG"] = butter_bandpass_filter(CZ, low_cut=l_cut, high_cut=h_cut, fs=256)
    if PZ is not None:
        out_dict["Pz-AVG"] = butter_bandpass_filter(PZ, low_cut=l_cut, high_cut=h_cut, fs=256)
    
    # 可选电极（10-10系统扩展电极，可能不存在）
    _optional_electrodes = ["Fpz", "Oz"]
    for opt_ch in _optional_electrodes:
        try:
            out_dict[f"{opt_ch}-AVG"] = butter_bandpass_filter(get_ch(opt_ch), low_cut=l_cut, high_cut=h_cut, fs=256)
        except KeyError:
            pass  # 电极不存在则跳过


    for k in out_dict.keys():
        temp_a = norch_50(np.array(out_dict[k]))
        out_dict[k] = norch_50(temp_a)


    return out_dict














def get_specific_list(in_locs, sp_name):
    out_list = []
    for v in in_locs:
        if v[3] == sp_name:
            out_list.append(v)
    return out_list

def get_label_list_by_range(in_label_list,time_range):

    if len(in_label_list)<10:return in_label_list


    label_list_t = merge_adjacent_locs(in_label_list, margin=2 * 256)

    label_list_t_limited = label_list_t[:1]
    out_time = label_list_t[0][1]
    for one_label in label_list_t:
        if one_label[0] - out_time > 1*256: label_list_t_limited.append(one_label)
        out_time = one_label[1]

    out_list = []

    short_begin_index = 0

    for one_long_label in label_list_t_limited:

        one_long_begin = one_long_label[0]
        one_long_end = one_long_label[1]

        one_range_label_list_A = []
        one_range_label_list_B = []

        for one_short_label in in_label_list :
            short_begin = one_short_label[0]
            short_end = one_short_label[1]
            short_begin_index += 1

            if one_long_begin<=short_begin <= one_long_end:
                if one_short_label[-1] in ["A1", "A2"]:
                    one_range_label_list_A.append(one_short_label)
                else:
                    one_range_label_list_B.append(one_short_label)

        if len(one_range_label_list_A) > 0:
            one_range_label_list_A = sorted(one_range_label_list_A, key=lambda x: x[2])
            out_list.append(one_range_label_list_A[-1])
            #print(one_range_label_list_A)

        if len(one_range_label_list_B) > 0:
            one_range_label_list_B = sorted(one_range_label_list_B, key=lambda x: x[2])
            out_list.append(one_range_label_list_B[-1])
            #print(one_range_label_list_B)

    #print(len(in_label_list),len(out_list))

    return out_list

def get_out_label_by_range(in_label_list,time_range):
    # for v in in_label_list:
    #     print(v)

    time_range = 5

    if len(in_label_list)<10:return in_label_list

    last_time_stamp = int(in_label_list[-1][1]/128)
    z_array = np.zeros(last_time_stamp+1)

    for v in in_label_list:
        label_begin = int(v[0]/128)
        label_end = int(v[1]/128)
        z_array[label_begin:label_end] = 1


    out_list = []
    begin_index = 0
    for i in range(1,last_time_stamp+1,time_range*2):
        c_mark = np.sum(z_array[i:i+time_range*2])
        #print(i, i + time_range * 2,c_mark,len(in_label_list),begin_index)
        if c_mark > 0:
            one_range_label_list_A = []
            one_range_label_list_B = []
            #print("-:",begin_index,)
            for one_label in in_label_list[begin_index:]:
                #print("--:",one_label[1],(i+time_range*2)*128,begin_index)

                if one_label[1] <= (i+time_range*2)*128:
                    if one_label[-1] in ["A1","A2"]:
                        one_range_label_list_A.append(one_label)
                    else:
                        one_range_label_list_B.append(one_label)
                    begin_index+= 1
                else:

                    break

            if len(one_range_label_list_A) > 0:
                one_range_label_list_A = sorted(one_range_label_list_A,key=lambda x:x[2])
                out_list.append(one_range_label_list_A[-1])

            if len(one_range_label_list_B) > 0:
                one_range_label_list_B = sorted(one_range_label_list_B,key=lambda x:x[2])
                out_list.append(one_range_label_list_B[-1])

    # print("----------",len(out_list))
    # for v in out_list:
    #     #print(v[0]/128,v[1]/128,v)
    return out_list


# 调整检测结果输出形式，调整长度，调整输出类型
def get_modified_out_result(result_dict):
    loc_EDs_T = result_dict["EDs"]
    loc_ICU_T = result_dict["ICU"]

    loc_ED_L = result_dict["ED"]
    loc_BKG_L = result_dict["BKG"]
    loc_SLEEP_T = result_dict["SLEEP"]
    loc_ART_L = result_dict["ART"]
    loc_TRI_L = result_dict["ITRI"]
    loc_BLOW_L = result_dict["BLOW"]
    loc_ALPHA_L = result_dict["ALPHA"]
    loc_SLOW_L = result_dict["SLOW"]
    loc_KVS_L = result_dict["KVS"]
    loc_EYE_L = result_dict["EYE"]
    loc_ICTAL_TIME = result_dict["ICTAL"]


    loc_ICU_PD_T = get_specific_list(loc_ICU_T, 'C.periodic-discharge')
    #loc_ICU_BS_T = get_specific_list(loc_ICU_T, 'C.burst')
    #loc_SLOW_R_L= get_specific_list(loc_SLOW_L, 'C.find_slow-rhythmic')
    #
    # loc_TRI_L = get_label_list_by_range(loc_TRI_L,5)
    # loc_BLOW_L = get_label_list_by_range(loc_BLOW_L,5)
    # loc_ALPHA_L = get_label_list_by_range(loc_ALPHA_L,5)
    # loc_ART_L = get_label_list_by_range(loc_ART_L,5)
    # loc_ED_L = get_label_list_by_range(loc_ED_L,5)
    # loc_SLOW_L = get_label_list_by_range(loc_SLOW_L, 5)
    # loc_KVS_L = get_label_list_by_range(loc_KVS_L, 5)

    #print(len(loc_SLOW_L),len(loc_KVS_L))

    loc_EDs_time = merge_adjacent_locs(loc_EDs_T, margin=2 * 256)
    loc_ICU_PD_time = merge_adjacent_locs(loc_ICU_PD_T, margin=1.5 * 256)
    loc_ICTAL_TIME = merge_adjacent_locs(loc_ICTAL_TIME, margin=3 * 256)



    loc_out = []
    for one_loc in loc_EDs_time:
        loc_out.append(['EDs', one_loc[0], one_loc[1], one_loc[2], 'TIME', 'TIME'])

    for one_loc in loc_TRI_L:
        loc_out.append(['I.triple', one_loc[0], one_loc[1], one_loc[2], one_loc[4], one_loc[5]])
    # for one_loc in loc_BLOW_L:
    #     loc_out.append(['B.lowV', one_loc[0], one_loc[1], one_loc[2], one_loc[4], one_loc[5]])
    for one_loc in loc_ALPHA_L:
        loc_out.append(['B.alpha', one_loc[0], one_loc[1], one_loc[2], one_loc[4], one_loc[5]])
    for one_loc in loc_ED_L:
        loc_out.append(['ED', one_loc[0], one_loc[1], one_loc[2], one_loc[4], one_loc[5]])
    # for one_loc in loc_BKG_L:
    #     loc_out.append([one_loc[3], one_loc[0], one_loc[1], one_loc[2], one_loc[4], one_loc[5]])
    for one_loc in loc_ART_L:
        loc_out.append(['A.artifact', one_loc[0], one_loc[1], one_loc[2], one_loc[4], one_loc[5]])
    for one_loc in loc_SLOW_L:
        #print('C.find_slow', one_loc[0], one_loc[1], one_loc[2], one_loc[4], one_loc[5])
        loc_out.append(['C.find_slow', one_loc[0], one_loc[1], one_loc[2], one_loc[4], one_loc[5]])
    for one_loc in loc_KVS_L:
        loc_out.append(['B.KVS', one_loc[0], one_loc[1], one_loc[2], one_loc[4], one_loc[5]])

    for one_loc in loc_EYE_L:
        loc_out.append(['A.EYE', one_loc[0], one_loc[1], one_loc[2], one_loc[4], one_loc[5]])

    for one_loc in loc_ICTAL_TIME:

        loc_out.append(['I.ictal', one_loc[0], one_loc[1], one_loc[2], 'TIME', 'TIME'])

    loc_out = sorted(loc_out, key=lambda x: x[2])

    #print(loc_ICTAL_TIME)


    return loc_out


def get_signal_seg_array(in_signal):
    out_signal_array = []
    for i in range(0, len(in_signal), 128):
        begin_index = i
        end_index = i + 3 * 256
        if end_index <= len(in_signal):
            out_signal_array.append(in_signal[begin_index:end_index])

    return np.array(out_signal_array)


def estimate_left_right(in_name):
    if "-" in in_name:

        in_name = in_name.split("-")[0]

        _mark = 0
        if in_name in LEFT_NAME:
            _mark = "LEFT"
        if in_name in RIGHT_NAME:
            _mark = "RIGHT"
    else:
        _mark = 0
    return _mark




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

    elif lead_type == '8' or lead_type == '10':


        out_dict = {
            "Fp1-A1": butter_bandpass_filter(lead_dict["Fp1"] - lead_dict["A1"], low_cut=0.8, high_cut=35, fs=256),
            "Fp2-A2": butter_bandpass_filter(lead_dict["Fp2"] - lead_dict["A2"], low_cut=0.8, high_cut=35, fs=256),
            "C3-A1": butter_bandpass_filter(lead_dict["C3"] - lead_dict["A1"], low_cut=0.8, high_cut=35, fs=256),
            "C4-A2": butter_bandpass_filter(lead_dict["C4"] - lead_dict["A2"], low_cut=0.8, high_cut=35, fs=256),
            "O1-A1": butter_bandpass_filter(lead_dict["O1"] - lead_dict["A1"], low_cut=0.8, high_cut=35, fs=256),
            "O2-A2": butter_bandpass_filter(lead_dict["O2"] - lead_dict["A2"], low_cut=0.8, high_cut=35, fs=256),
            "T3-A1": butter_bandpass_filter(lead_dict["T3"] - lead_dict["A1"], low_cut=0.8, high_cut=35, fs=256),
            "T4-A2": butter_bandpass_filter(lead_dict["T4"] - lead_dict["A2"], low_cut=0.8, high_cut=35, fs=256),
            "Fp1-C3": butter_bandpass_filter(lead_dict["Fp1"] - lead_dict["C3"], low_cut=0.8, high_cut=35, fs=256),
            "Fp2-C4": butter_bandpass_filter(lead_dict["Fp2"] - lead_dict["C4"], low_cut=0.8, high_cut=35, fs=256),
            "C3-O1": butter_bandpass_filter(lead_dict["C3"] - lead_dict["O1"], low_cut=0.8, high_cut=35, fs=256),
            "C4-O2": butter_bandpass_filter(lead_dict["C4"] - lead_dict["O2"], low_cut=0.8, high_cut=35, fs=256),
            "Fp1-T3": butter_bandpass_filter(lead_dict["Fp1"] - lead_dict["T3"], low_cut=0.8, high_cut=35, fs=256),
            "Fp2-T4": butter_bandpass_filter(lead_dict["Fp2"] - lead_dict["T4"], low_cut=0.8, high_cut=35, fs=256),
            "T3-O1": butter_bandpass_filter(lead_dict["T3"] - lead_dict["O1"], low_cut=0.8, high_cut=35, fs=256),
            "T4-O2": butter_bandpass_filter(lead_dict["T4"] - lead_dict["O2"], low_cut=0.8, high_cut=35, fs=256),
        }


    elif lead_type == '16' or lead_type == '18':
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
        }

    elif lead_type == '21' or lead_type == '23':
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
            "Fz-Pz": butter_bandpass_filter(lead_dict["Fz"] - lead_dict["Pz"], low_cut=0.8, high_cut=35, fs=256),
            "Cz-Pz": butter_bandpass_filter(lead_dict["Cz"] - lead_dict["Pz"], low_cut=0.8, high_cut=35, fs=256),
            "Pz-Oz": butter_bandpass_filter(lead_dict["Pz"] - lead_dict["Oz"], low_cut=0.8, high_cut=35, fs=256),
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
            "Fz-Pz": butter_bandpass_filter(lead_dict["Fz"] - lead_dict["Pz"], low_cut=0.8, high_cut=35, fs=256),
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

        low_c = 1.5
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
            "Fz-Pz": butter_bandpass_filter(lead_dict["Fz"] - lead_dict["Pz"], low_c, high_c, fs=256),
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

