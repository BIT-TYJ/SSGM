# -*- coding:UTF-8 -*-
import copy
import cv2
import math
import random
import time, sys, os
from ros import rosbag
import roslib
roslib.load_manifest('sensor_msgs')
from sensor_msgs.msg import Image#, PointCloud2, PointField
import sensor_msgs.point_cloud2 as pcl2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter

from mpl_toolkits.mplot3d import Axes3D
from math import *
from datetime import datetime


dataset_is_scenenet_209 = 1
src_frame = 67  #set the source frame

'''write the graph matching results to the file'''

dateTime_p =  datetime.now() # get the current time
str_p =  datetime.strftime(dateTime_p,'%Y-%m-%d-%h-%m-%s')
    
if dataset_is_scenenet_209 == 1:
    f_GM_information = open("result.txt", "a+")
    
query_frame_origin = 0  #query the frame from this number   ：）
if_search_specially_query_frame = 1 # whether select the start  query frame 

'''flag variable'''
show_ = 0 #the switch,  whether draw some graph results
show_vertex_graph = 1     #whether show the vertex graph
show_pointcloud =0 #show point cloud
points_src =[]               #point cloud of the src frame
points_query =[]        #point cloud of the query frame
only_show_once = 0   #show once and stop
show_common_semantic = 1  #whether the ratio of same semantics
#GM_first_parameter
frames = []
common_RGB_perceptions = []
GM_switch = 1 #the switch for graph matching


#SSGM_thresholds
first_threhold = 0.2  #the threshold of : the ratio of same semantics
spatial_consistency_threhold = 0.15     #m
theta_threshold = 20      #degree
fai_threshold_1 = 15        #degree
fai_threshold_2 = 20      #degree
distance_threshold = 2    #the distance threshold for measure the local features of selected vertex pairs
distance_threshold_1 = 1  # threshold of the local distance features
angle_interval =3 # angle sample interval for selecting axes 

vector_z_axis_src = [] ;  vector_x_axis_src = [] ;  vector_y_axis_src = []     #three axes
vector_z_axis_query = [] ;  vector_x_axis_query = [] ;  vector_y_axis_query = []
P1 = 0  #the ratio of same semantics
P2 = 0  #max two order spatial consistency value
P3 = 0   #similar Spherical coordinate vertex pair

#scannet_10_01 random RGB
RGB_wall = [187,47,155]
RGB_desk = [243,233,81]
RGB_windows = [37,174,197]
RGB_floor = [7,4,196]

'''Visualization initialization'''
if show_:
    #ax  3D
    if not only_show_once :
        plt.ion()
    fig=plt.figure()    
    ax=Axes3D(fig)    
    
    #ax_query  3D
    fig_query=plt.figure()    
    ax_query=Axes3D(fig_query)    
    
    #ax1  2D
    fig_1=plt.figure()  
    #draw 2D curve：the ratio of same semantics
    left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
    ax1 = fig_1.add_axes([left, bottom, width, height])
    

# get the fourth element of elem
def takeFourth(elem):
    return elem[3]

# get the seventh element of elem
def takeSeven(elem):
    return elem[6]

#remove the repeat element in list  according to the value in specific dimension
def DelRepeat(data,key):   #data: list ,  key:dimension 
    new_data = [] #   new list after duplicate removal
    key_values = []  #  store current values
    for d in data:
        if d[key] not in key_values:
            new_data.append(d)
            key_values.append(d[key])
    return new_data, key_values

#Average, maximum and minimum values of specific dimensions of elements within a certain range in the list
def Mean_Max_Min(data,key,start_index, end_index):   #data: list ,  key:dimensions  start_index, end_index: 
    sum_value = 0
    #search maximum and minimum values
    max_coordinate = 0
    min_coordinate = 0
    num_x0y0z0 =0 
    for d in data[start_index:end_index]:
        if float(d[0]) == 0 and float(d[1]) == 0 and float(d[2]) == 0:
            num_x0y0z0 = num_x0y0z0 + 1
            continue
        if max_coordinate < d[key]:
            max_coordinate = d[key]
        if min_coordinate > d[key]:
            min_coordinate = d[key]
        sum_value =sum_value + d[key]
    if end_index - start_index-num_x0y0z0 != 0:
        mean_value = sum_value / (end_index - start_index-num_x0y0z0) 
        continue_flag = 0       #  Whether to skip  generating the vertex, the flag variable = 0 means not to skip
    else:
        mean_value = 0
        continue_flag = 1     
        # print("mean_value, max_coordinate, min_coordinate", mean_value, max_coordinate, min_coordinate)
    return mean_value, max_coordinate, min_coordinate, continue_flag


#Downsampling a frame of point cloud (reducing the display time) and visualization
def down_sample_and_show(data,down_rate, ax_num):   #data: list,  key:dimensions
    down_sample_data_x = [] 
    down_sample_data_y = [] 
    down_sample_data_z = [] 
    down_sample_data_RGB = [] 
    for d in data[0::down_rate]:
        down_sample_data_x.append(d[0])
        down_sample_data_y.append(d[1])
        down_sample_data_z.append(d[2])
        down_sample_data_RGB.append( (float(d[3]/255) ,float(d[4]/255) ,float(d[5]/255) ,1) )
    down_sample_data_x= np.array(down_sample_data_x)
    down_sample_data_y= np.array(down_sample_data_y)
    down_sample_data_z= np.array(down_sample_data_z)
    #visualize the point cloud
    ax_num.scatter(down_sample_data_x, down_sample_data_z, -down_sample_data_y, c = down_sample_data_RGB, s = 1)

#Find the RGB value with the most points of the same RGB within a certain range in the list
def SearchRGB(data,start_index, end_index):   #data: list ,  key:dimensions 
    RGB_list = []  # store existing RGB
    for d in data[start_index:end_index]:
        # RGB = d[3:6]
        RGB_list.append(d[3:6])
    number = Counter(RGB_list) #Sort the quantity from large to small
    result = number.most_common()   # result like: [((0,0,0), 44340), ((10,100,23),2340), ...]
    # print("RGB_sort", result)
    if result[0][0] == (0,0,0): #If the most semantic points are unknown semantic points, it depends on what the second most semantic points are and the number of the second most semantic points
        if len(result) >1 :
            if result[0][1] >= 0.5*( result[1][1] + result[0][1] ) : 
                return result[0][0], result[0][1]
        else:  #If there is only unknown point cloud, return directly
            return result[0][0], result[0][1]
    else:
        return result[0][0], result[0][1]

#for each value in instance_list, find where it first appeared in data, with the dimension of key
def SearchIndex(data,instance_list, key):     #data: list , instance_list  ,key:dimensions
    ins_index_list = []  # store existing value
    instance_num = 0   #
    for Index in range(len(data)):
        if instance_num == len(instance_list):
            ins_index_list.append(len(data)-1)       #The index of the last point of the last instance
            break
        if data[Index][key] == instance_list[instance_num]:
            ins_index_list.append(Index)
            instance_num = instance_num + 1
    return  ins_index_list

def  get_RGBlist_from_vertexslist(vertexs):        #Vextex = [x_mean, y_mean, z_mean, R, G, B, instance,x_max, y_max, z_max, x_min, y_min, z_min]
    RGB_list = []
    for vertex in vertexs:
        R, G, B= vertex[3:6]
        RGB_list.append((R,G,B))
    return RGB_list

def print_vertexs_information(Vertexs):
    print("vertexs_information of the current frame: ")
    print("x_mean, y_mean, z_mean, R, G, B, instance,x_max, y_max, z_max, x_min, y_min, z_min")
    for v in Vertexs:
        print(v)

def show_vertexs_graph(vertexs_x_, vertexs_y_,vertexs_z_, color_vertexs_, src_flag, query_flag):
    vertexs_x_ = np.array(vertexs_x_)
    vertexs_y_ = np.array(vertexs_y_)
    vertexs_z_ = np.array(vertexs_z_)
    if src_flag ==1 :
        ax.scatter(vertexs_x_ , vertexs_z_, -vertexs_y_ , c = color_vertexs_ , s=100, alpha = 0.5, norm = 0.5)
    elif query_flag == 1:
        ax_query.scatter(vertexs_x_ , vertexs_z_, -vertexs_y_ , c = color_vertexs_ , s=100, alpha = 0.5, norm = 0.5)

def distance(x,y,z):    
    return math.sqrt(pow(x,2)+pow(y,2)+pow(z,2))

def plot_ax_parameter(only_show_once, ax_num):
    ax_num.set_xlabel('X')  
    ax_num.set_ylabel('Y')  
    ax_num.set_zlabel('Z')  
    ax_num.view_init(elev=0, azim=5)                # ax_num.view_init(elev=20, azim=0 ）          
    if not only_show_once :      
        # plt.savefig(str(frame)+".png")
        fig.canvas.draw() #update
        # time.sleep(2)
        ax_num.cla()   #remove old data
    else:   #
        plt.show()
        time.sleep(50)

def plot_ax_parameter_GM_Spherical(only_show_once, ax_num, ax_num1):
    ax_num.set_xlabel('X')  
    ax_num.set_ylabel('Y')  
    ax_num.set_zlabel('Z')  
    ax_num1.set_xlabel('X')  
    ax_num1.set_ylabel('Y')  
    ax_num1.set_zlabel('Z')  
    ax_num.view_init(elev=0, azim=5)                # ax_num.view_init(elev=20, azim=0 ） 
    ax_num1.view_init(elev=0, azim=5)                # ax_num.view_init(elev=20, azim=0 ）   

    if not only_show_once :      
        # plt.savefig(str(frame)+".png")
        fig.canvas.draw() #update
        # time.sleep(2)
        ax_num.cla()  #remove old data
        ax_num1.cla()  #remove old data
    else:  
        plt.show()
        time.sleep(5)


def  GM_first(vertexs_src, vertexs_query, frame_src, frame_query):        #V.append(Vextex)   Vextex = [x_mean, y_mean, z_mean, R, G, B, instance,x_max, y_max, z_max, x_min, y_min, z_min, density]
    first_remained_flag = 0   
    #obtain RGB values
    global RGBlist_src, RGBlist_query
    RGBlist_src = get_RGBlist_from_vertexslist(vertexs_src)
    RGBlist_query = get_RGBlist_from_vertexslist(vertexs_query)
    '''calculate common semantics / semantics in src or query (IOU)'''
    src_RGB_sort = set(RGBlist_src) 
    query_RGB_sort = set(RGBlist_query) 
    global common_RGB_sort
    common_RGB_sort = src_RGB_sort & query_RGB_sort

    tem_denominator = float( len( src_RGB_sort ) ) + float( len( query_RGB_sort ) ) - float( len( common_RGB_sort ) ) 
    common_RGB_perception =float( len(  common_RGB_sort) ) /   tem_denominator
    global P1 
    P1 = common_RGB_perception
    f_GM_information.write("P1: "+str(common_RGB_perception)+" common_RGB_perception: "+str(len(  common_RGB_sort))+" all semantic sorts: "+str(  tem_denominator)+"\n")
    # print("The ",frame_src, "frame and the ", frame_query,"frame has the common RGB perception of :", len( common_RGB_sort ), " / ",len( src_RGB_sort )," : ",common_RGB_perception)
    frames.append(frame_query)
    common_RGB_perceptions.append(common_RGB_perception)

    if show_ and show_common_semantic:
        ax1.plot(frames, common_RGB_perceptions)
    #compare with the threshold
    if  common_RGB_perception >= first_threhold:
        first_remained_flag = 1
    
    return first_remained_flag, common_RGB_perceptions

#find the vertexs close to point ，return their semantics and average distances
def most_near_class(point, vertexs, near_number):
    distance_list = []
    vertex_num_in_range = 0
    for i in range(len(vertexs)):
        tem_point_i_src = np.array( vertexs[i] )
        tem_distance = np.linalg.norm(point[0:3] - tem_point_i_src[0:3])

        if tem_distance < distance_threshold and tem_distance != 0:
            vertex_num_in_range += 1
        
        # print(point - tem_point_i_src)
        distance_list.append(tem_distance)
    # print(distance_list)
    sorted_distance = sorted(distance_list)   #Sort in ascending order of distance. The smallest is 0, which is itself
    class_list = []
    top_near_dis_average = 0

    if len(vertexs) < near_number + 1:
        near_number = len(vertexs) -1
    for i in range(1,1+vertex_num_in_range):
        index = distance_list.index(sorted_distance[i]) 
        RGB_class = vertexs[index][3:6]
        # print(sorted_distance[i], index)
        class_list.append(RGB_class)
        top_near_dis_average = top_near_dis_average + sorted_distance[i] / vertex_num_in_range
    
    return class_list , top_near_dis_average

def matrix(m,n,initial):  
    return [[initial for j in range(n)] for i in range(m)]

def max_in_matrix(matrix): #find max in a matrix and the position
    max_in_two_stage_spatial_consistency_list = [] ;  max_belonging_row = [] ; max_belonging_col = []
    max_in_two_stage_spatial_consistency_list .append( matrix[0][0] );   max_belonging_row .append( 0 ) ;  max_belonging_col .append( 0 )
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[i][j] > max_in_two_stage_spatial_consistency_list[0]:
                max_in_two_stage_spatial_consistency_list = [] ;  max_belonging_row = [] ; max_belonging_col = []   
                max_in_two_stage_spatial_consistency_list .append( matrix[i][j] );   max_belonging_row .append( i ) ;  max_belonging_col .append( j )
            elif matrix[i][j] == max_in_two_stage_spatial_consistency_list[0]:
                max_in_two_stage_spatial_consistency_list .append( matrix[i][j] );   max_belonging_row .append( i ) ;  max_belonging_col .append( j )
    return max_in_two_stage_spatial_consistency_list, max_belonging_row, max_belonging_col

def angle(v1, v2): #the angle of two vectors
    cos = np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2))
    return np.arccos(cos)

def count(list):           #Return the different elements in the list and their occurrence times
    # count = 0 
    no_repeat_list = []  #A new list composed of non-repeating elements in list
    corresponding_num = [0 for i in range(len(list))] 
    if len(list) == 0 :
        return no_repeat_list, corresponding_num
    for i in range(len(list)):
        if list[i] not in no_repeat_list:
            no_repeat_list.append(list[i])
            corresponding_num[len(no_repeat_list)-1] += 1 
        else:
            Index = no_repeat_list.index(list[i])
            corresponding_num[Index] += 1 
    return no_repeat_list, corresponding_num

#cosine similarity
def similarity_dot_compare(no_repeat_class_src, corresponding_num_src, no_repeat_class_query, corresponding_num_query):
    import operator
    normalized_dot_product = 0 
    normalized_dot_product_numerator = 0  #th numerator of the dot
    normalized_dot_product_denominator = 0  #denominator of the dot
    normalized_dot_product_denominator += np.linalg.norm( np.array(corresponding_num_src) ) * np.linalg.norm( np.array(corresponding_num_query) )
    for i in no_repeat_class_src:
        for j in no_repeat_class_query:
            if operator.eq(i,j) == True:
                normalized_dot_product_numerator += corresponding_num_src[no_repeat_class_src.index(i)] * corresponding_num_query[no_repeat_class_query.index(j)]
    if normalized_dot_product_denominator != 0 :
        normalized_dot_product = float(normalized_dot_product_numerator) / float(normalized_dot_product_denominator)
    else:
        normalized_dot_product = 0
    return normalized_dot_product

#Build a spherical coordinate system, measure the spatial distribution of two spatial semantic graph
def GM_Spherical_plane(vertexs_src, vertexs_query, frame_src, frame_query,first_remained_flag, common_RGB_perception):     #V.append(Vextex)   Vextex =  [x_mean, y_mean, z_mean, R, G, B, instance,x_max, y_max, z_max, x_min, y_min, z_min, instance_points, volume, density]
    if first_remained_flag == 1:
        
        # f_GM_information.write("第"+str(frame_query)+"帧query\n")
        
        vertex_pairs = []        # vertex_pairs =  [[i1,j1],[i2,j2],...]
        
        #obtain RGB list of source and query，common_RGB_sort
        global RGBlist_src, RGBlist_query, common_RGB_sort
        
        i_volumn_difference_min = 0
        j_volumn_difference_min = 0
        
        '''Construct vertex pairs with the same semantics in two vertex graphs'''  
        
        #many-to-many 
        most_num_true_vertex_pair = 0  #Maximum number of vertex pairs that may be correct
        # print("index_vertex of common RGB src and query are:")
        # f_GM_information.write("index_vertex of common RGB src and query are:\n")
        for common_RGB in common_RGB_sort:        
            index_vertexs_src = [index  for index, RGB_value in enumerate(RGBlist_src) if RGB_value == common_RGB]   
            index_vertexs_query = [index  for index, RGB_value in enumerate(RGBlist_query) if RGB_value == common_RGB]
            
            most_num_true_vertex_pair = most_num_true_vertex_pair + min( len(index_vertexs_src), len(index_vertexs_query) )
            
            # print(index_vertexs_src,' and ', index_vertexs_query)

            # f_GM_information.write(str(index_vertexs_src)+' and '+ str(index_vertexs_query)+"\n")
            
            for i in index_vertexs_src:
                for j in index_vertexs_query:
                    vertex_pairs.append([i,j])
        # if vertex_pairs != []:
        #     print(vertex_pairs)
        # print('most_num_true_vertex_pair: ', most_num_true_vertex_pair)
        '''second order spatial_consistency'''

        # spatial_consistency_sum = [0]*len(vertex_pairs)
        # one_stage_spatial_consistency_matrix = matrix( len(vertex_pairs), len(vertex_pairs), 0)
        consistent_list_row = []   
        consistent_list_col = []   #      corresponding to consistent_list_row，record the consistent element in one_order_spatial_consistency matrix
        two_stage_spatial_consistency_list = [] # record the second order spatial_consistency value corresponding to consistent_list_row， consistent_list_col


        global spatial_consistency_threhold
        # two_stage_spatial_consistency_matrix = matrix( len(vertex_pairs), len(vertex_pairs), 0)
        
        #first order spatial_consistency
        for i  in range( len(vertex_pairs) ):                              #pair_1_index_src , pair_1_index_query ;   pair_2_index_src ；  pair_2_index_query 
            for j in range((i+1),len(vertex_pairs)):
                pair_1_index_src = vertex_pairs[i][0]
                pair_1_index_query = vertex_pairs[i][1]
                pair_2_index_src = vertex_pairs[j ][0]
                pair_2_index_query = vertex_pairs[j ][1]      
                line_src = np.array(vertexs_src[pair_1_index_src][0:3]) - np.array(vertexs_src[pair_2_index_src][0:3])  
                line_query = np.array(vertexs_query[pair_1_index_query][0:3]) - np.array(vertexs_query[pair_2_index_query][0:3])  
                if abs( np.linalg.norm(line_src) - np.linalg.norm(line_query) ) < spatial_consistency_threhold:
                    # one_stage_spatial_consistency_matrix[i][j] = 1
                    # one_stage_spatial_consistency_matrix[j][i] = 1
                    consistent_list_row.append(i)   
                    consistent_list_col.append(j)

        # print(consistent_list_row, consistent_list_col)
        if len(consistent_list_row) == 0:
            f_GM_information.write("P2: 0.000"+"\n")
            f_GM_information.write("P3:   max_same_angle_count is : 0.0000" +  "\n")
            f_GM_information.write("final score: 0.0000"+", P1: "+str(0)+", P2: "+str(0)+", P3: "+str(0)+"\n")
            print('consistent_list_row= 0')
            return -1

        #constructing sysmetric matrix
        consistent_list_row_col = consistent_list_row + consistent_list_col
        consistent_list_col_row = consistent_list_col + consistent_list_row

        #caluculate two_order_spatial_consistency_matrix values
        for index_ in range(len(consistent_list_row)):          #( consistent_list_row[index_], consistent_list_col[index_] ) record the position of element =1  in one_stage_spatial_consistency matrix

            current_row_1 = consistent_list_row[index_]  ;   current_row_1_corresponding_consistent_col = []
            current_row_2 = consistent_list_col[index_] ;  current_row_2_corresponding_consistent_col = []
            for index_1 in [i for i, x in enumerate(consistent_list_row_col) if x == current_row_1]:
                current_row_1_corresponding_consistent_col.append(consistent_list_col_row[index_1])
            for index_2 in [i for i, x in enumerate(consistent_list_row_col) if x == current_row_2]:
                current_row_2_corresponding_consistent_col.append(consistent_list_col_row[index_2])

            tem_two_stage_value = 0
            for i in current_row_1_corresponding_consistent_col:
                if i in current_row_2_corresponding_consistent_col:
                    tem_two_stage_value +=1
            two_stage_spatial_consistency_list.append(tem_two_stage_value)

        #find the max one in two_order_spatial_consistency_matrix values
        max_in_two_stage_spatial_consistency_list = 0 ; row =[] ; col =[]
        for index_ in range(len(two_stage_spatial_consistency_list)):
            if max_in_two_stage_spatial_consistency_list < two_stage_spatial_consistency_list[index_]:
                row = []
                col = []
                max_in_two_stage_spatial_consistency_list = two_stage_spatial_consistency_list[index_]
                row.append(consistent_list_row[index_])
                col.append(consistent_list_col[index_])
            elif max_in_two_stage_spatial_consistency_list == two_stage_spatial_consistency_list[index_]:
                row.append(consistent_list_row[index_])
                col.append(consistent_list_col[index_])
        # print(max_in_two_stage_spatial_consistency_list, row, col)

        # print(len(vertex_pairs), max_in_two_stage_spatial_consistency_list, one_stage_spatial_consistency_matrix[row][col])
        
        '''Judge the similarity of local features of selected vertice pairs, which is used to judge the correctness of the origin and z-axis'''
        similarity_score = 0

        most_SC_index =row[0]
        second_most_SC_index = col[0]
        max_similarity_score = 0
        
        if max_in_two_stage_spatial_consistency_list ==0 :   
            f_GM_information.write("P2: 0.000"+"\n")
            f_GM_information.write("P3:   max_same_angle_count is : 0.0000" +  "\n")
            f_GM_information.write("final score: 0.0000"+", P1: "+str(0)+", P2: "+str(0)+", P3: "+str(0)+"\n")
            print("max_in_two_stage_spatial_consistency_list = 0")
            return -1

        #search the most likely correct vertex pairs 
        for i in range(len(row)):
            point_origin_src = np.array(vertexs_src[vertex_pairs[row[i]][0] ])
            point_z_src = np.array(vertexs_src[vertex_pairs[col[i]][0] ])
            point_origin_query = np.array(vertexs_query[vertex_pairs[row[i]][1] ])
            point_z_query = np.array(vertexs_query[vertex_pairs[col[i]][1] ])

            origin_src_class_list  , dis1      =  most_near_class(point_origin_src, vertexs_src, 3)   # Returns the semantic class and average distance of the nearest vertices
            z_src_class_list  , dis2               =  most_near_class(point_z_src, vertexs_src, 3)
            origin_query_class_list , dis3 =  most_near_class(point_origin_query, vertexs_query, 3)
            z_query_class_list ,  dis4          =  most_near_class(point_z_query, vertexs_query, 3)

            no_repeat_class_origin_src, corresponding_num_origin_src = count(origin_src_class_list)

            # print(origin_src_class_list, no_repeat_class_origin_src, corresponding_num_origin_src)

            no_repeat_class_z_src, corresponding_num_z_src = count(z_src_class_list)
            no_repeat_class_origin_query, corresponding_num_origin_query = count(origin_query_class_list)
            no_repeat_class_z_query, corresponding_num_z_query = count(z_query_class_list)


            
            origin_similarity = similarity_dot_compare(no_repeat_class_origin_src, corresponding_num_origin_src, no_repeat_class_origin_query, corresponding_num_origin_query)
            z_similarity = similarity_dot_compare(no_repeat_class_z_src, corresponding_num_z_src, no_repeat_class_z_query, corresponding_num_z_query)
            # print(origin_similarity, z_similarity)

            #local feature of distances: comparison
            if abs(dis1 - dis3) >distance_threshold_1 or abs(dis2 - dis4) >distance_threshold_1:
                f_GM_information.write("P2: 0.000"+"\n")
                f_GM_information.write("P3:   max_same_angle_count is : 0.0000" +  "\n")
                f_GM_information.write("final score: 0.0000"+", P1: "+str(0)+", P2: "+str(0)+", P3: "+str(0)+"\n")
                print(abs(dis1 - dis3), abs(dis2 - dis4))
                return -1

           #local features of semantics:  cosine similarity
            if origin_similarity != 0 and z_similarity != 0:
                similarity_score = min(origin_similarity, z_similarity)
            if origin_similarity != 0 and z_similarity == 0:
                if z_src_class_list ==[] and z_query_class_list == []:
                    similarity_score = origin_similarity
                else:
                    similarity_score = 0
            if origin_similarity == 0 and z_similarity != 0:
                if origin_src_class_list ==[] and origin_query_class_list == []:
                    similarity_score = z_similarity
                else:
                    similarity_score = 0

            if max_similarity_score < similarity_score:
                max_similarity_score = similarity_score
                most_SC_index =row[i]
                second_most_SC_index = col[i]
        print(max_similarity_score)

        # # print(origin_src_class_list, z_src_class_list, origin_query_class_list, z_query_class_list)

        #P2
        great_vertex_pair_by_consistency = 0  #
        # sorted_nums = one_stage_spatial_consistency_matrix[0]
        sorted_nums = 0
        global P2
        P2 = max_in_two_stage_spatial_consistency_list       
        # print("P2: ",P2, "great_vertex_pair_by_consistency: ", great_vertex_pair_by_consistency, "len(sorted_nums): ",len(sorted_nums))
        f_GM_information.write("P2: "+str(P2)+"\n")
                
        
        '''origin, x, y ,z axis'''
        #origin  : most_SC_index，z axis ×  xaxis  = y axis
        vertex_for_original_src = vertexs_src[vertex_pairs[most_SC_index][0] ] ;     vertex_for_original_query = vertexs_query[vertex_pairs[most_SC_index][1]]             #origin

        point_original_src = [ vertex_for_original_src[0], vertex_for_original_src[1], vertex_for_original_src[2]]     #src  origin
        point_original_query = [vertex_for_original_query[0], vertex_for_original_query[1], vertex_for_original_query[2] ]  #query  origin

        global vector_z_axis_src, vector_x_axis_src, vector_y_axis_src
        global vector_z_axis_query, vector_x_axis_query, vector_y_axis_query
        #z axis ： second_most_SC_index
        vertex_for_z_axis_src = vertexs_src[vertex_pairs[second_most_SC_index][0] ] ;     vertex_for_z_axis_query = vertexs_query[vertex_pairs[second_most_SC_index][1]]     
        vector_z_axis_src = np.array([vertex_for_z_axis_src[0] - vertex_for_original_src[0] , vertex_for_z_axis_src[1] - vertex_for_original_src[1] , vertex_for_z_axis_src[2] - vertex_for_original_src[2]])        #src z axis
        vector_z_axis_src = vector_z_axis_src / np.linalg.norm(vector_z_axis_src)     #z axis : Unitization
        vector_z_axis_query = [vertex_for_z_axis_query[0] - vertex_for_original_query[0], vertex_for_z_axis_query[1] - vertex_for_original_query[1], vertex_for_z_axis_query[2] - vertex_for_original_query[2]]  #query z axis
        vector_z_axis_query = vector_z_axis_query / np.linalg.norm(vector_z_axis_query)     #z axis : Unitization
        array_z_axis_src_unit = np.array(vector_z_axis_src  )                             #src unit z axis
        array_z_axis_query_unit = np.array(vector_z_axis_query)      #query unit z axis
        
            #x axis : Perpendicular to z-axis
        if array_z_axis_src_unit[0] != 0 :
            vector_x_axis_src = [- ( array_z_axis_src_unit[1] + array_z_axis_src_unit[2] ) / array_z_axis_src_unit[0],1,1]
        elif array_z_axis_src_unit[1] != 0 :
            vector_x_axis_src = [1, - ( array_z_axis_src_unit[0] + array_z_axis_src_unit[2] ) / array_z_axis_src_unit[1],1]
        elif array_z_axis_src_unit[2] != 0 :
            vector_x_axis_src = [1,1, - ( array_z_axis_src_unit[0] + array_z_axis_src_unit[1] ) / array_z_axis_src_unit[2]]
        vector_x_axis_src = vector_x_axis_src / np.linalg.norm(vector_x_axis_src)     #x axis : Unitization
        
        array_x_axis_src_unit  = np.array(vector_x_axis_src)                                                #src unit x axis
        
        if array_z_axis_query_unit[0] != 0 :
            vector_x_axis_query = [- ( array_z_axis_query_unit[1] + array_z_axis_query_unit[2] ) / array_z_axis_query_unit[0],1,1]
        elif array_z_axis_query_unit[1] != 0 :
            vector_x_axis_query = [1, - ( array_z_axis_query_unit[0] + array_z_axis_query_unit[2] ) / array_z_axis_query_unit[1],1]
        elif array_z_axis_query_unit[2] != 0 :
            vector_x_axis_query = [1,1, - ( array_z_axis_query_unit[0] + array_z_axis_query_unit[1] ) / array_z_axis_query_unit[2]]
        vector_x_axis_query = vector_x_axis_query / np.linalg.norm(vector_x_axis_query)     #x axis : Unitization
        
        array_x_axis_query_unit  = np.array(vector_x_axis_query )                 #query unit x axis
           
            #y axis   y axis(unit)  = z axis (unit)  ×  x axis (unit)  
        vector_y_axis_src = np.cross(array_z_axis_src_unit , array_x_axis_src_unit)
        vector_y_axis_query = np.cross(array_z_axis_query_unit , array_x_axis_query_unit)
        
        array_y_axis_src_unit = np.array(vector_y_axis_src )                  #src unit y axis
        array_y_axis_query_unit = np.array(vector_y_axis_query)  #query unit y axis

        # print( angle(array_z_axis_src_unit, array_x_axis_src_unit) , angle(array_z_axis_src_unit, array_y_axis_src_unit) ,angle(array_y_axis_src_unit, array_x_axis_src_unit))
        # print( angle(array_z_axis_query_unit, array_x_axis_query_unit) , angle(array_z_axis_query_unit, array_y_axis_query_unit) ,angle(array_y_axis_query_unit, array_x_axis_query_unit))

         #    r^2 = x^2 +y^2 +z^2 ; z = r*cos(theta) ; x=rsinθcosφ ；  y=rsinθsinφ
        spherical_parameters_src = [ ] ;    spherical_parameters_query = [ ]        
        
        array_point_original_src = np.array(point_original_src)              
        array_point_original_query = np.array(point_original_query)  
        
        #calculate theta  fai  r
        for index in range(len(vertex_pairs)):
            # if index == row or index == col :
            #     continue
            vertex_n_src = vertexs_src[vertex_pairs[index][0] ] ;     vertex_n_query = vertexs_query[vertex_pairs[index][1]]                                                                           
            point_n_src = [ vertex_n_src[0], vertex_n_src[1], vertex_n_src[2]]  ;        point_n_query = [ vertex_n_query[0], vertex_n_query[1], vertex_n_query[2]]         
            array_point_n_src =np.array(point_n_src);        array_point_n_query =np.array(point_n_query)           

            # print(vector_x_axis_query, array_point_n_src , array_point_original_src)
            point_n_src_new_x = np.dot(array_x_axis_src_unit, array_point_n_src - array_point_original_src)   ;        point_n_query_new_x = np.dot(array_x_axis_query_unit, array_point_n_query - array_point_original_query)
            point_n_src_new_y = np.dot(array_y_axis_src_unit, array_point_n_src - array_point_original_src)   ;        point_n_query_new_y = np.dot(array_y_axis_query_unit, array_point_n_query - array_point_original_query)
            point_n_src_new_z = np.dot(array_z_axis_src_unit, array_point_n_src - array_point_original_src)   ;        point_n_query_new_z = np.dot(array_z_axis_query_unit, array_point_n_query - array_point_original_query)
            #r
            r_point_n_src = np.linalg.norm(array_point_n_src - array_point_original_src)                                             ;        r_point_n_query = np.linalg.norm(array_point_n_query - array_point_original_query)        
            
            #theta:0-180°，  fai:-180-180°
            if r_point_n_src == 0:
                theta_point_n_src = 0
            else:
                theta_point_n_src = np.arccos( min(1, point_n_src_new_z / r_point_n_src) )                                                                     
            if r_point_n_query == 0:
                theta_point_n_query =0
            else:
                theta_point_n_query = np.arccos( min(1, point_n_query_new_z / r_point_n_query) )        #theta = arccos( z/r )

            fai_point_n_src = np.arctan2(point_n_src_new_y , point_n_src_new_x)                    ;        fai_point_n_query = np.arctan2(point_n_query_new_y , point_n_query_new_x)        #fai = arctan2( y/x )    
            if r_point_n_src == 0:
                theta_point_n_src = 0
                
            if r_point_n_query == 0:
                theta_point_n_query = 0
            spherical_parameters_src.append([theta_point_n_src/math.pi*180, fai_point_n_src/math.pi*180, r_point_n_src, vertex_pairs[index][0], vertex_pairs[index][1], index])    
            spherical_parameters_query.append([theta_point_n_query/math.pi*180, fai_point_n_query/math.pi*180, r_point_n_query, vertex_pairs[index][0], vertex_pairs[index][1], index])
            # print([theta_point_n_src/math.pi*180, fai_point_n_src/math.pi*180])   ;           print([theta_point_n_query/math.pi*180, fai_point_n_query/math.pi*180])   ;  print("\n")
            
            # f_GM_information.write(str(theta_point_n_src/math.pi*180)+" " + str(fai_point_n_src/math.pi*180)+"   "+str(theta_point_n_query/math.pi*180)+" " + str(fai_point_n_query/math.pi*180)+"\n")   

        '''theta and fai'''
        #1. loop ：fai_angle : sample
        max_same_angle_count = 0  
        max_incre_fai_angle = 0    
        
        # vertex_pairs_with_same_sphercial_parameters = []
        
        # global spatial_consistency_threhold
        global theta_threshold
        global fai_threshold_1
        global fai_threshold_2

        for incre_fai_angle in range(0,360,angle_interval):
            same_angle_count = 0   
            #2.loop：compare theta and fai
            for i in range(len(spherical_parameters_src)):
                theta_i_src = spherical_parameters_src[i][0]   ;  fai_i_src = spherical_parameters_src[i][1]    ;   r_i_src = spherical_parameters_src[i][2]           
                theta_i_query = spherical_parameters_query[i][0]   ;  fai_i_query = spherical_parameters_query[i][1] ;r_i_query = spherical_parameters_query[i][2]    
                fai_i_src = fai_i_src + incre_fai_angle
                if abs( theta_i_src - theta_i_query ) < theta_threshold and abs(r_i_src - r_i_query) < spatial_consistency_threhold:
                    
                    if fai_i_src > 180:  
                        fai_i_src = fai_i_src -360
                    
                    abs_delta_fai = abs( fai_i_src - fai_i_query )   
                    
                    if abs_delta_fai > 180:     
                        abs_delta_fai = 360 - abs_delta_fai
                    
                    if abs_delta_fai < fai_threshold_1:         
                        same_angle_count = same_angle_count + 1
        
                    elif abs_delta_fai < fai_threshold_2:        
                        same_angle_count = same_angle_count + 0.5
                
            if same_angle_count > max_same_angle_count :
                max_same_angle_count = same_angle_count
                max_incre_fai_angle = incre_fai_angle
            if max_same_angle_count > most_num_true_vertex_pair:   
                max_same_angle_count = most_num_true_vertex_pair
                break
            # if float(max_same_angle_count) / most_num_true_vertex_pair > 0.6:
            #     break
        # print("P3: max_same_angle_count is : " + str(max_same_angle_count) + ",  most_num_true_vertex_pair is : "+ str(most_num_true_vertex_pair)+" ," + str(float(max_same_angle_count) / most_num_true_vertex_pair) + " ," + str(max_incre_fai_angle) + "angle\n")
        #P3 record
        f_GM_information.write("P3:   max_same_angle_count is : " + str(max_same_angle_count) + "\n")
        
        # print(vertex_pairs_with_same_sphercial_parameters)
        
        # visualization
        if show_ == 1:
            for index in range(len(vertex_pairs)):
                vertex_n_src = vertexs_src[vertex_pairs[index][0] ] ;     vertex_n_query = vertexs_query[vertex_pairs[index][1]]    
                color_vertex_show_src =  ( float(vertex_n_src[3]/255) ,float(vertex_n_src[4]/255) ,float(vertex_n_src[5]/255) ,1)
                color_vertex_show_query =  ( float(vertex_n_query[3]/255) ,float(vertex_n_query[4]/255) ,float(vertex_n_query[5]/255) ,1)
                if index == most_SC_index:        #origin
                    # print(float(vertex_n_query[3]) ,float(vertex_n_query[4]) ,float(vertex_n_query[5]))
                    ax.scatter(vertex_n_src[0] , vertex_n_src[2], -vertex_n_src[1] , c = color_vertex_show_src , s=500, alpha = 0.5, norm = 0.5)
                    ax_query.scatter(vertex_n_query[0] , vertex_n_query[2], -vertex_n_query[1] , c = color_vertex_show_query , s=500, alpha = 0.5, norm = 0.5)
                    #src
                    line_z_src = [];  line_x_src = [];      line_y_src = []
                    line_z_src.append( [ vertex_n_src[0], vertex_n_src[0]+array_z_axis_src_unit[0] ] );            line_z_src.append([ vertex_n_src[1], vertex_n_src[1]+array_z_axis_src_unit[1]]);   line_z_src.append([ vertex_n_src[2], vertex_n_src[2]+array_z_axis_src_unit[2]])
                    line_x_src.append( [ vertex_n_src[0], vertex_n_src[0]+array_x_axis_src_unit[0] ] );            line_x_src.append([ vertex_n_src[1], vertex_n_src[1]+array_x_axis_src_unit[1]]);   line_x_src.append([ vertex_n_src[2], vertex_n_src[2]+array_x_axis_src_unit[2]])
                    line_y_src.append( [ vertex_n_src[0], vertex_n_src[0]+array_y_axis_src_unit[0] ] );            line_y_src.append([ vertex_n_src[1], vertex_n_src[1]+array_y_axis_src_unit[1]]);   line_y_src.append([ vertex_n_src[2], vertex_n_src[2]+array_y_axis_src_unit[2]])
                    
                    ax.plot(np.array(line_z_src[0]), np.array(line_z_src[2]), -np.array(line_z_src[1]), c='r')
                    ax.plot( np.array(line_x_src[0]), np.array(line_x_src[2]), -np.array(line_x_src[1]), c='g')
                    ax.plot(np.array(line_y_src[0]), np.array(line_y_src[2]), -np.array(line_y_src[1]), c='b')
                    #query
                    line_z_query = [];  line_x_query = [];      line_y_query = []
                    line_z_query.append( [ vertex_n_query[0], vertex_n_query[0]+array_z_axis_query_unit[0] ] );            line_z_query.append([ vertex_n_query[1], vertex_n_query[1]+array_z_axis_query_unit[1]]);   line_z_query.append([ vertex_n_query[2], vertex_n_query[2]+array_z_axis_query_unit[2]])
                    line_x_query.append( [ vertex_n_query[0], vertex_n_query[0]+array_x_axis_query_unit[0] ] );            line_x_query.append([ vertex_n_query[1], vertex_n_query[1]+array_x_axis_query_unit[1]]);   line_x_query.append([ vertex_n_query[2], vertex_n_query[2]+array_x_axis_query_unit[2]])
                    line_y_query.append( [ vertex_n_query[0], vertex_n_query[0]+array_y_axis_query_unit[0] ] );            line_y_query.append([ vertex_n_query[1], vertex_n_query[1]+array_y_axis_query_unit[1]]);   line_y_query.append([ vertex_n_query[2], vertex_n_query[2]+array_y_axis_query_unit[2]])
                    
                    ax_query.plot(np.array(line_z_query[0]), np.array(line_z_query[2]), -np.array(line_z_query[1]), c='r')
                    ax_query.plot(np.array(line_x_query[0]), np.array(line_x_query[2]), -np.array(line_x_query[1]), c='g')
                    ax_query.plot(np.array(line_y_query[0]), np.array(line_y_query[2]), -np.array(line_y_query[1]), c='b')
                # elif index in vertex_pairs_with_same_sphercial_parameters:
                elif index == second_most_SC_index:
                    ax.scatter(vertex_n_src[0] , vertex_n_src[2], -vertex_n_src[1] , c = color_vertex_show_src , s=300, alpha = 0.5, norm = 0.5)
                    ax_query.scatter(vertex_n_query[0] , vertex_n_query[2], -vertex_n_query[1] , c = color_vertex_show_query , s=300, alpha = 0.5, norm = 0.5)
                else:
                    ax.scatter(vertex_n_src[0] , vertex_n_src[2], -vertex_n_src[1] , c = color_vertex_show_src , s=50, alpha = 0.5, norm = 0.5)
                    ax_query.scatter(vertex_n_query[0] , vertex_n_query[2], -vertex_n_query[1] , c = color_vertex_show_query , s=50, alpha = 0.5, norm = 0.5)
    
    if first_remained_flag == 0 :       
        P = 0
        f_GM_information.write("final score: "+str(P)+"\n")
    else:
        global P3
        P3 = max_same_angle_count
        f_GM_information.write("final score: 0.0000"+", P1: "+str(P1)+", P2: "+str(P2)+", P3: "+str(P3)+"\n")



def get_vertex_pair(points_frame,frame,point_num_threhold):                

    '''Point clouds are sorted by instance'''
    instance_list = []                                                  #store instance categories
    points_frame.sort(key=takeSeven)            #point cloud sorted by instance
    ins_index_in_pc = []                                          
    a, instance_list = DelRepeat(points_frame,6)                                       
    ins_index_in_pc = SearchIndex(points_frame,instance_list, 6)   #The first subscript of each instance in a frame of instance sorting point cloud
    V= []  #  V=[ [x, y, z, RGB, instance, number_of_points], [], ... ]
    
    vertexs_x_show = []
    vertexs_y_show = []
    vertexs_z_show = []
    color_vertexs_show = []
    
    '''generating vertex for each instance'''
    all_points_num = len(points_frame)
    for i in range(len(ins_index_in_pc)):
        # the first and last subscripts in the frame of each  instance
        if i == len(ins_index_in_pc) - 1:
            instance_start_index = ins_index_in_pc[i]
            instance_end_index = all_points_num
            instance_points = instance_end_index- instance_start_index                   #Number of points for an instance
        else:
            instance_start_index = ins_index_in_pc[i]
            instance_end_index = ins_index_in_pc[i+1]
            instance_points = instance_end_index- instance_start_index                   #Number of points for an instance

        # print(instance_points)
        if instance_points < point_num_threhold:                                                                                           
            continue
        
        (R, G, B), num_RGB = SearchRGB(points_frame, instance_start_index, instance_end_index)     #num_ RGB is the number of points corresponding to the RGB value

        x_mean, x_max, x_min, continue_flag = Mean_Max_Min(points_frame, 0, instance_start_index, instance_end_index) 
        if continue_flag == 1:  #continue_flag = 1 ，Skip generating the vertex, because its points are the origin
            continue
        y_mean, y_max, y_min, continue_flag = Mean_Max_Min(points_frame, 1, instance_start_index, instance_end_index)
        z_mean, z_max, z_min, continue_flag = Mean_Max_Min(points_frame, 2, instance_start_index, instance_end_index)

        Vertex = [x_mean, y_mean, z_mean, R, G, B]

        global dataset_is_scenenet_209
        if dataset_is_scenenet_209 == 1:
            if R == 0 and G == 217 and B == 0 :  #ceiling
                continue
            if R == 0 and G == 139 and B == 249 :  #wall
                continue
            if R == 23 and G == 241 and B == 222 :  #floor
                continue
            if R == 194 and G == 228 and B == 225 :  #wall
                continue

        vertexs_z_show.append(Vertex[2])
        '''visualization'''
        if show_vertex_graph:
            vertexs_x_show.append(Vertex[0])
            vertexs_y_show.append(Vertex[1])
            
            color_vertex_show =  ( float(Vertex[3]/255) ,float(Vertex[4]/255) ,float(Vertex[5]/255) ,1)
            color_vertexs_show.append( color_vertex_show )     
        

        V.append(Vertex)

    global spatial_consistency_threhold, theta_threshold, fai_threshold_1, fai_threshold_2


    spatial_consistency_threhold = 0.15
    theta_threshold = 20
    fai_threshold_1 = 15
    fai_threshold_2 = 20

    # print(color_vertexs_show)
    return  V, vertexs_x_show , vertexs_y_show, vertexs_z_show, color_vertexs_show

def main():
    ###################################
    if dataset_is_scenenet_209 == 1:
        vertexs_src = []
        bag_name = "/media/tang/yujie2/dataset/train_0/train/scenenet0_209.bag"
        bag = rosbag.Bag(bag_name, 'r')

        point_num_threhold = 10     
        
    ####################################
    try:    
        #src 

        bag_data = bag.read_messages('/camera_point')
        frame = -1
        for topic, msg, t in bag_data:
            frame = frame + 1
            if src_frame == frame:
                #read point cloud
                lidar = pcl2.read_points(msg)
                points_frame = list(lidar)
                points_src = points_frame
                print("src: the "+str(frame)+"th frame")
                vertexs_src, vertexs_src_x_show , vertexs_src_y_show, vertexs_src_z_show, color_src_vertexs_show = get_vertex_pair(points_frame,frame,point_num_threhold)
                # print(vertexs_src)
                break
        #query
        bag_data = bag.read_messages('/camera_point')
        frame = -1
        for topic, msg, t in bag_data:
            frame = frame + 1
            print("the "+str(frame)+"th frame")
            if if_search_specially_query_frame and  frame < query_frame_origin:  
                continue 
            
            #read point cloud
            lidar = pcl2.read_points(msg)
            points_frame = list(lidar)
            
            V, vertexs_query_x_show , vertexs_query_y_show, vertexs_query_z_show, color_query_vertexs_show = get_vertex_pair(points_frame,frame,point_num_threhold)
        

        #----------------------------------------------------inner 'for' loop end--------------------------------------------------------#
            '''visualization'''
            if show_:
                if show_pointcloud:       
                        points_query = points_frame
                        down_sample_and_show(points_query,5,ax_query)
                        down_sample_and_show(points_src,5,ax)
                if show_vertex_graph:
                    show_vertexs_graph(vertexs_query_x_show, vertexs_query_y_show,vertexs_query_z_show, color_query_vertexs_show, src_flag =0, query_flag =1)
                    show_vertexs_graph(vertexs_src_x_show , vertexs_src_y_show, vertexs_src_z_show, color_src_vertexs_show, src_flag =1, query_flag =0)
                    
             
            '''grapg matching'''
            if GM_switch == 1 and vertexs_src != [] and frame>=query_frame_origin:
                f_GM_information.write("the "+str(frame)+" frame query\n")
                
                first_remained_flag, common_RGB_perception = GM_first( vertexs_src, V, src_frame, frame)   
                if first_remained_flag == 0:
                    print("fail in the first filter part!  OUT!")
                GM_Spherical_plane( vertexs_src, V, src_frame, frame,first_remained_flag, common_RGB_perception) 
                if show_:
                    plot_ax_parameter_GM_Spherical(only_show_once, ax, ax_query)      

    #---------------------------------------------------------------------------external 'for'  loop end--------------------------------------------------------------------------------------#
    finally:
        bag.close()

if __name__ == "__main__":
    main()

f_GM_information.close()
