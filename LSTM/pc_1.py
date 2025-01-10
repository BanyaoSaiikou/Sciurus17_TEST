# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 21:24:27 2020

@author: tuan
"""

import socket

TCP_IP = "10.40.0.144"#CAI Weihao
#TCP_IP = "127.0.0.1"
TCP_PORT = 5432             #设置端口号
BUFFER_SIZE = 1024     #要接受的最大数据量


############TCP 服务器 1、创建套接字，绑定套接字到本地IP与端口
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)# 创建 socket 对象 	基于TCP的流式socket通信

s.bind((TCP_IP, TCP_PORT))        #将套接字绑定到地址，在AF_INET下，以tuple(host, port)的方式传入，如s.bind((host, port))


############ 2、开始监听链接
s.listen(1)                                         #开始监听TCP传入连接，backlog指定在拒绝链接前，
                                                            #操作系统可以挂起的最大连接数，该值最少为1，大部分应用程序设为5就够用了

##########3、进入循环，不断接受客户端的链接请求
conn, addr = s.accept()                  #接受TCP链接并返回（conn, address），其中conn是新的套接字对象，可以用来接收和发送数据，
                                                                                                                                            #address是链接客户端的地址。
print("Connection address:", addr)
while 1:
###############4、接收客户端传来的数据，并且发送给对方发送数据
    data = conn.recv(BUFFER_SIZE)#接受TCP套接字的数据，数据以字符串形式返回，
                                                                #buffsize指定要接受的最大数据量
    if not data:
        break
    print("Received data:", data)
    conn.send(data)     #发送TCP数据，将字符串中的数据发送到链接的套接字，返回值是要发送的字节数量，该数量可能小于string的字节大小


##########5、传输完毕后，关闭套接字
conn.close()
