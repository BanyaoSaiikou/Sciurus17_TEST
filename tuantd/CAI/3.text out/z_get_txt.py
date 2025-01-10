#! /usr/bin/env python
# -*- coding: utf-8 -*-
from multiprocessing.connection import answer_challenge
import os
import re
from sre_constants import AT_BEGINNING_LINE
from tokenize import Double
from unittest import result
import math
import rospy
from std_msgs.msg import Float32MultiArray

def Read_Txt(x1,y1,z1,path=None, a_list=None):
        #file_name = os.listdir(path)  # 读取路径path下的所有文件
        s2=[]
    
        #cu_path = path + file_name[k].split(".")[0] + ".txt"
        cu_path = "/home/sciurus/Desktop/tuantd/1.txt"
  

        txt = open(cu_path)
        line = txt.readline()
        while line is not None and line != '':
            rs = line.rstrip('\n') 
            inf = rs.split(" ")#shan chu \n
            a_list.append(inf)
            line = txt.readline()    # 读取下一行

        # a_list.append(line)
        
        #print(type(a_list[0][0]))
        #print(len(a_list))
        #print(len(a_list[0]))
        # print((a_list))
        
        result=100
        in2=0
        for t in range(len(a_list)):
            if ([a_list[in2][6]])==['nan']:
                continue
            r=(float(a_list[t][0])-x1)*(float(a_list[t][0])-x1)+(float(a_list[t][2])-y1)*(float(a_list[t][2])-y1)+(float(a_list[t][4])-z1)*(float(a_list[t][4])-z1)

            if r<result:
                result=r
                in2=t
        #print("distance:",result)
        #print("No. :",in2+1)
        #print((a_list[in2]))
        s2.append([a_list[in2][0]])
        s2.append([a_list[in2][2]])
        s2.append([a_list[in2][4]])
  #      s2.append([a_list[in2][6]])
 #       s2.append([a_list[in2][8]])
#        s2.append([a_list[in2][10]])

        #print("Normal:",s2)# 输出法线向量
        t1=float(a_list[in2][6])
        t3=float(a_list[in2][10])
        TTT=(t1/t3)
        #a=-0.181901/-0.220675#x/z OR x/y  -----》从这里找方向
        b=math.atan(TTT)
        # print("a:",a)
        #print("-------------")
        #print("RadianforZ:",b)
        angle1=b*180/3.14159268
        #print("angle:",b*180/3.14159268)
        return s2,b
def Read_Txt2(x1,y1,z1,path=None, a_list=None):
        #file_name = os.listdir(path)  # 读取路径path下的所有文件
        s2=[]
    
        #cu_path = path + file_name[k].split(".")[0] + ".txt"
        cu_path = "/home/sciurus/Desktop/tuantd/1.txt"
  

        txt = open(cu_path)
        line = txt.readline()
        while line is not None and line != '':
            rs = line.rstrip('\n') 
            inf = rs.split(" ")#shan chu \n
            a_list.append(inf)
            line = txt.readline()    # 读取下一行

        # a_list.append(line)
        
        #print(type(a_list[0][0]))
        #print(len(a_list))
        #print(len(a_list[0]))
        # print((a_list))
        
        result=100
        in2=0
        for t in range(len(a_list)):
            if ([a_list[in2][6]])==['nan']:
                continue
            r=(float(a_list[t][0])-x1)*(float(a_list[t][0])-x1)+(float(a_list[t][2])-y1)*(float(a_list[t][2])-y1)+(float(a_list[t][4])-z1)*(float(a_list[t][4])-z1)

            if r<result:
                result=r
                in2=t
        #print("distance:",result)
        #print("No. :",in2+1)
        #print((a_list[in2]))
        s2.append([a_list[in2][0]])
        s2.append([a_list[in2][2]])
        s2.append([a_list[in2][4]])
  #      s2.append([a_list[in2][6]])
 #       s2.append([a_list[in2][8]])
#        s2.append([a_list[in2][10]])

        #print("Normal:",s2)# 输出法线向量
        t1=float(a_list[in2][6])
        t2=float(a_list[in2][8])
        TTT=(t2/t1)
        #a=-0.181901/-0.220675#x/z OR x/y  -----》从这里找方向
        b=math.atan(TTT)
        # print("a:",a)
        #print("-------------")
        #print("RadianforZ:",b)
        angle1=b*180/3.14159268
        print("angle:",b*180/3.14159268)
        return s2,b
               
def callback(data):
         #rospy.loginfo(data.data)
         txt1_path = "/home/sciurus/Desktop/tuantd"
         s1 = []
         S1,A1 = Read_Txt(data.data[0], data.data[1], data.data[2],path=txt1_path, a_list=s1)
         S2,A2 = Read_Txt2(data.data[3], data.data[4], data.data[5],path=txt1_path, a_list=s1)
         #S3,A3 = Read_Txt(data.data[6], data.data[7], data.data[8],path=txt1_path, a_list=s1)
         #S4,A4 = Read_Txt2(data.data[9], data.data[10], data.data[11],path=txt1_path, a_list=s1)
         #S5,A5 = Read_Txt(data.data[12], data.data[13], data.data[14],path=txt1_path, a_list=s1)
         #S6,A6 = Read_Txt2(data.data[15], data.data[16], data.data[17],path=txt1_path, a_list=s1)
         print(S1,A1,S2,A2)
	 #print(S1,A1,S2,A2,S3,A3,S4,A4,S5,A5,S6,A6)#输入法线向量
         #print("S1[]type",type(S1[0][0]))#str
         #print("S1[]",float(S1[0][0]))#float
         #print("type.data.data",type(S2[0][0])) --> S2[0][0] S2[1][0] S2[2][0]
         pub = rospy.Publisher('chattertorviz', Float32MultiArray, queue_size=10)
         array=[]
        



         array.append(float(S1[0][0]) ) 
         array.append(float(S1[1][0]) )
         array.append(float(S1[2][0]) )
         array.append(A1)

         array.append(float(S2[0][0]) ) 
         array.append(float(S2[1][0]) )
         array.append(float(S2[2][0]) )
         array.append(A2)

         #array.append(float(S3[0][0]) ) 
         #array.append(float(S3[1][0]) )
         #array.append(float(S3[2][0]) )
         #array.append(A3)

         #array.append(float(S4[0][0]) ) 
         #array.append(float(S4[1][0]) )
         #array.append(float(S4[2][0]) )
         #array.append(A4)

         #array.append(float(S5[0][0]) ) 
         #array.append(float(S5[1][0]) )
         #array.append(float(S5[2][0]) )
         #array.append(A5)
        
         #array.append(float(S6[0][0]) ) 
         #array.append(float(S6[1][0]) )
         #array.append(float(S6[2][0]) )
         #array.append(A6)
         array_forPublish = Float32MultiArray(data=array)
         #print(array_forPublish)

         pub.publish(array_forPublish)



         

def listener():
    rospy.init_node('listenertxt', anonymous=True)
    rospy.Subscriber("chatter", Float32MultiArray, callback)
    rospy.spin()

if __name__ == '__main__':

    #Read_Txt(0.46853769081381913, 0.09434759563341566, 0.025326812169754243,path=txt1_path, a_list=s1)#输入法线向量
    listener()
    
