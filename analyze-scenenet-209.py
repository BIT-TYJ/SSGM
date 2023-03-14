# -*- coding:UTF-8 -*-
import re

def get_score_from_file_GM_new():   #read P1 P2 P3 for calculate  the final score
    scores = []
    scores_num = []
    f_data = open("result.txt", "r")
    try:
        all_the_text = f_data.read()  
        every_frame_information = all_the_text.split('query')
        for string in every_frame_information[1:]:
            P1_first_index = string.index('P1: ')   #first time it exist
            if string[P1_first_index+7].isdigit() == True:
                P1_score = float(string[P1_first_index+4:P1_first_index+8])
            else:
                P1_score = float(string[P1_first_index+4:P1_first_index+7])
            if P1_score < 0.2: 
                scores_num.append(0)
            else:
                P2_first_index = string.index('P2: ')
                if string[P2_first_index+5].isdigit() == True:
                    P2_score = float(string[P2_first_index+4:P2_first_index+6])
                else:
                    P2_score = float(string[P2_first_index+4:P2_first_index+5])
                
                P3_first_index = string.index('max_same_angle_count is : ')
                if string[P3_first_index+27].isdigit() == True:       #
                    if string[P3_first_index+28] == '.':
                        P3_score = float(string[P3_first_index+26:P3_first_index+30])  #like: 10.5
                    else:
                        P3_score = int(string[P3_first_index+26:P3_first_index+28])  #like: 10
                elif string[P3_first_index+27] == '.':      
                    P3_score = float(string[P3_first_index+26:P3_first_index+29])  #like: 8.5
                else:                                                              
                    P3_score = int(string[P3_first_index+26:P3_first_index+27])   #like: 8

                tem_final_score = P1_score + 5*P2_score + 2*P3_score
                scores_num.append(tem_final_score)
    finally:
        f_data.close()
    return scores_num

score_GM = get_score_from_file_GM_new()

def R_P_calculate(frames_num, scores, score_threshold, true_positive, flag = ' '):
    FP_list = [];  TP_list = []
    TP=0 ; FP =0 ; TN=0 ; FN=0
    for i in range(frames_num):
        if scores[i] ==-1: 
            continue
        elif  scores[i] >=score_threshold:   #positive
            if true_positive[i] == 1:
                TP = TP+ 1
                TP_list.append(i)
            elif true_positive[i] == 0:
                FP = FP + 1
                FP_list.append(i)
        elif scores[i] < score_threshold:  #negetive
            if true_positive[i] == 1:
                FN = FN+ 1
            elif true_positive[i] == 0:
                TN = TN + 1
                
    if TP +FN == 0:
        recall_once = 0
    else:
        recall_once = float(TP) / (TP +FN) 
    
    if TP +FP == 0:
        precision_once = 0
    else:
        precision_once =  float(TP) / (TP +FP) 
        
    # print(FP_list)
    return recall_once, precision_once, FP_list, TP_list

#true positive
frames_num = 300
true_positive = [0 for i in range(frames_num)]
for i in range(frames_num):                #scenenet-209-1675frame
    if (i>=0 and i<=74) or (i>=167 and i<=213) or (i>=293 and i<=299) :
        true_positive[i] = 1  #positive
    else:             #negetive
        true_positive[i] = 0

#P-R curve
recall_GM = [] ; precision_GM = []  
precision_GM.append(1) ; recall_GM.append(0)
for score_threshold in range(1000, -1, -1):
    score_threshold = float(score_threshold)/10
    
    recall_GM_once, precision_GM_once, FP_list, TP_list = R_P_calculate(frames_num, score_GM, score_threshold, true_positive, 'ours')
    if recall_GM_once != 0 or precision_GM_once != 0:
        recall_GM.append( recall_GM_once )
        precision_GM.append( precision_GM_once )
        # print(recall_GM_once, precision_GM_once)


def R_P_area_calculate(precision_list, recall_list): #P-R curve area
    area = 0
    for i in range(len(recall_list)-1):
        area += float(precision_list[i] + precision_list[i+1]) * (recall_list[i+1] - recall_list[i]) /2
    print( "The area of the P-R curve is: "+str(area))
    return area

R_P_area_calculate(precision_GM,recall_GM)


import matplotlib.pyplot as plt
plt.plot(recall_GM, precision_GM, 'r', label='area:0.986')
plt.legend()

plt.show()
