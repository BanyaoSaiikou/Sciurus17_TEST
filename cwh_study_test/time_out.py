# -*- coding:utf-8 -*-

# import time
# import eventlet  #导入eventlet这个模块
# eventlet.monkey_patch()   #必须加这条代码
# with eventlet.Timeout(0.2,False):   #设置超时时间为2秒
#    print ('这条语句正常执行')
#    time.sleep(0.1)
#    print ('没有跳过这条输出')
#=====================================================-
# coding=utf-8
import signal
import time
def set_timeout(num, callback):
  def wrap(func):
    def handle(signum, frame): # 收到信号 SIGALRM 后的回调函数，第一个参数是信号的数字，第二个参数是the interrupted stack frame.
      raise RuntimeError
    def to_do(*args, **kwargs):
      try:
        signal.signal(signal.SIGALRM, handle) # 设置信号和回调函数
        signal.alarm(num) # 设置 num 秒的闹钟
        print('start alarm signal.')
        r = func(*args, **kwargs)
        print('close alarm signal.')
        signal.alarm(0) # 关闭闹钟
        return r
      except RuntimeError as e:
        callback()
    return to_do
  return wrap
def after_timeout(): # 超时后的处理函数
  print("Time out!")



@set_timeout(2, after_timeout) # 限时 2 秒超时
def connect(): # 要执行的函数
  time.sleep(3) # 函数执行时间，写大于2的值，可测试超时
  print('Finished without timeout.')
if __name__ == '__main__':
  connect()