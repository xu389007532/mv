# 單個編號OCR, 再過濾邊界的黑點.
import configparser
import os.path
import tkinter
from tkinter import ttk
#import tkinter.messagebox
import cv2
# from cv2 import cv2     #要用這種方式才可以.
from os import getcwd,mkdir,path,system,listdir
import re
import pytesseract
import threading
import datetime
import numpy as np
import keyboard
from itertools import product
import mvsdk
import platform
from shutil import copyfile
import time
from shutil import move
import _winapi



def scan_area(img):
    """
    邊緣檢測

    :param img: 每一幀圖像
    :return:
    """
    config = configparser.ConfigParser()
    config.read("config.ini")
    template_pixel = tuple(eval(config['scan_area']['標準邊界底色']))
    error_range= int(config['scan_area']['顏色允許誤差范圍'])
    cr1 = tuple(eval(config['scan_area']['邊界底色范圍']))
    cr2 = tuple(eval(config['scan_area']['打印邊緣范圍']))
    #template_pixel = (21, 32, 13)  # 标准边界颜色 (Test_area.mov)   要在python Opencv read() Frame 看Array 數組, 小畫家裏看是不准確的. 不能用.
    #error_range = 20  # 颜色允许误差范围

    #column1, row1 = (467, 885)  # 开始 列行位置, 用于取这个位置的像素
    #column2, row2 = (493, 897)  # 结束 列行位置, 用于取这个位置的像素
    #for pos1 in product(range(row1, row2), range(column1, column2)):  # product(range(20, width), range(10, height))  表示在20-33(width) 到 10-20(height) 这个区域
    for pos1 in product(range(cr1[1], cr1[3]), range(cr1[0], cr1[2])):  # product(range(20, width), range(10, height))  表示在20-33(width) 到 10-20(height) 这个区域
        gp1 = img[pos1]
        # print(pos1,gp1)
        if gp1[0] in range(template_pixel[0] - error_range, template_pixel[0] + error_range) and gp1[1] in range(template_pixel[1] - error_range, template_pixel[1] + error_range) and gp1[2] in range(template_pixel[2] - error_range, template_pixel[2] + error_range):

            check_border = True
        else:
            check_border = False
            break
    check_crop = None
    if check_border:
        #column3, row3 = (463, 845)  # 开始 列行位置, 用于取这个位置的像素
        #column4, row4 = (495, 857)  # 结束 列行位置, 用于取这个位置的像素
        #for pos2 in product(range(row3, row4), range(column3, column4)):
        for pos2 in product(range(cr2[1], cr2[3]), range(cr2[0], cr2[2])):
            gp2 = img[pos2]
            # print(pos2, gp2)
            if gp2[0] in range(template_pixel[0] - error_range, template_pixel[0] + error_range) and gp2[1] in range(template_pixel[1] - error_range, template_pixel[1] + error_range) and gp2[2] in range(template_pixel[2] - error_range, template_pixel[2] + error_range):
                check_crop = False
                break
            else:
                check_crop = True
    # print(check_crop)
    return check_crop


def check_data(jobver, identifier, data_type, composing, box):
    """
    不再使用
    將OCR列表編號與標準的編號對比輸出漏印的編號. 標準編號分為兩種情況: 1. 一棟落打印; 2. 順序打印

    :param jobver: 處理工單版本
    :param identifier: 識別碼
    :param data_type: 1. 一棟落打印; 2. 順序打印
    :param composing: 排版個數
    :param box: 一箱數量
    :return: 補數編號
    """
    ic = re.compile("(-*\d+\.*\d*)", re.S)  # 找數字, 不管是小數, 負數, 整數.
    # 打开文件
    fo = open("./jobver_list.txt", "a+")
    # print("文件名为: ", fo.name)
    fo.seek(0, 0)
    line = fo.read()
    str1 = re.findall(jobver + '-' + '[0-9]{4}', line)
    jobver_id = jobver + "-" + str(str1.__len__()).rjust(4, '0')
    print(jobver_id)
    # fo.writelines(jobver_id + "\n")
    fo.close()
    # 关闭文件

    # list1=[5,9,13,17,21,25,29,33,37,41,49,53,57,61,65,69,73,77,81,85,89,93,97,105]  #順序打印數據
    # list1 = [12, 13, 14, 15, 16, 17, 18, 19, 20, 51, 52, 53, 54, 55, 56, 57, 59, 60, 91, 92, 93]  # 一棟落打印數據

    sn = open("./job/" + jobver_id + "/seqnumber.txt", 'r')
    text = sn.read()
    #print(text)
    str1 = re.findall('[0-9]{6}', text)  # 指定变量, 用于前面识别码, 更加准确
    sn.close()
    #print(str1)
    org_int=set()
    for i in str1:
        org_int.add(int(i))
    print("列表轉集合: ",org_int)
    first_seq, end_seq = int(min(str1)), int(max(str1))
    star = first_seq - (first_seq % int(box)) + 1
    print(first_seq, end_seq)
    standard_seq = []
    # end=max(list2)+(box-(max(list2) % box))+1  # 不需要檢測end
    # print(star, end)
    if data_type == 1:
        # first_seq, end_seq=min(list1),max(list1)+1
        standard_seq = list(range(first_seq, end_seq + 1, composing))
    else:
        list2a = star
        flat = True
        while flat:
            for i in range(list2a, list2a + int(box)):
                if i <= end_seq:
                    if i >= first_seq:
                        standard_seq.append(i)
                else:
                    flat = False
                    break

            list2a = list2a + int(box) * int(composing)

    print(first_seq, end_seq)
    print(standard_seq)
    print("打印編號: ",str1)
    #diff = set(standard_seq).difference(set(str1))
    diff = set(standard_seq).difference(org_int)
    list_difference = list(diff)
    list_difference.sort()
    bs=[]
    for k in list_difference:
        bs.append(str(k).rjust(6, '0'))
    print("補數編號: ", bs)
    return bs


def ocr_thread(jobver_id, filename,filename_OCR, identifier):
    """
    OCR 處理線程

    :param jobver_id: 工單版本id
    :param filename: OCR 識別的txt檔案
    :param identifier: 識別碼
    :return:
    """
    #re1=re.compile("([A-Z,a-z]-?\d{1,7})", re.S)  # 找第一個字符是[A-Z,a-z] 第二個字符是[-] (可以沒有) , 後面是1-7位數字.  (通用)
    seq1 = '[A|a]{1}-?([0-9]{1,7})'  # 過濾條件( 指定識別碼 )
    re1 = re.compile(seq1, re.S)  # 找數字, 不管是小數, 負數, 整數.

    seq2 = '[^\d]{1}?-?([0-9]{1,7})'  # 過濾條件 (識別碼不指定, 但要是非數字, 實際上因為有空格, 所以都是非數字的.)
    re2 = re.compile(seq2, re.S)  # 找數字, 不管是小數, 負數, 整數.

    # seq3 = '\s*(.*?)\n+'  # 不能檢測過白紙, 但可以多個編號一張圖片.
    # seq3 = '(.*?)\n?\f'  # 檢測過白紙, 但只能一個編號一張圖片.

    # seq3 = '.*('+identifier+'\d{1,7}).*\n'
    # seq3 = '.*(' + identifier + '\d{1,7}).*'

    seq3 = '.*(' + identifier + '\d{1,7}[-]{0,1}\d{0,2}).*'
    re3 = re.compile(seq3, re.S)  # 不做過濾條件
    # print("OCR 線程開始處理: ", filename, "OCR 編號", filename_OCR)
    global end_filename
    text = pytesseract.image_to_string(filename,lang="Domino", config="--psm 6")
    # text = pytesseract.image_to_string(filename, lang="eng", config="--psm 1")
    # text = pytesseract.image_to_string(filename,lang="W130+eng", config="--psm 1")
    #print(text)
    #str1 = re.findall(identifier + '[a-z,A-Z]{6}', text)  # 指定变量, 用于前面识别码, 更加准确
    # str1 = re.findall(identifier + '[0-9]{5}', text)  # 指定变量, 用于前面识别码, 更加准确
    # str1 = re3.findall(text.replace(' ','').replace('-',''))  # 把空格, 橫線去掉
    str1 = re3.findall(text.replace(' ', ''))  # 把空格, 橫線去掉
    if len(str1)==0:   # 如果找不到編號, 就用模糊查找.
        # seq4 = '([A-Z,a-z]{0,2}\d{1,7}).*'  # 找前面1~2 個字母, 後面1~7個數字的數據
        seq4 = '([A-Z,a-z]{0,2}\d{1,7}[-]{0,1}\d{0,2}).*'  # 找前面1~2 個字母, 後面1~7個數字的數據
        re4 = re.compile(seq4, re.S)  # 不做過濾條件
        str1 = re4.findall(text)  # 指定变量, 用于前面识别码, 更加准确
    # sn = open(filename[:-4]+ "-OCR.txt", 'a')
    sn = open(filename_OCR, 'a')
    # sn.write('\n'.join(str1).replace(' ',''))
    sn.write('\n'.join(str1))
    sn.close()
    # print("OCR 線程結束處理: ", filename)
    end_filename=filename
    """
    l3 = list(l2)  # 集合转列表.
    l3.sort(reverse=False)  # 列表排序.  reverse=False 是默认的.
    print(l3)
    print("OCR 線程結束: ", filename)
    """


def read_sample_bak(x1_tk, y1_tk):
    """
    讀取坐標樣板

    :param x1_tk: x軸坐標
    :param y1_tk: y軸坐標
    :return:
    """
    x1_tk, y1_tk = int(x1_tk), int(y1_tk)

    config = configparser.ConfigParser()
    config.read("config.ini")
    rtsp = config['DEFAULT']['rtsp']
    rtsp = "rtsp://admin:123456789@192.168.0.128:554"
    width1 = int(config['DEFAULT']['裁剪寬度'])
    height1 = int(config['DEFAULT']['裁剪高度'])
    #videoCapture = cv2.VideoCapture(rtsp)
    #videoCapture = cv2.VideoCapture(0)
    videoCapture = cv2.VideoCapture(r"D:\Xu\python\Visual Inspection\SourceFile\Test_area.mov")  # 讀本地視頻
    videoCapture.set(cv2.CAP_PROP_POS_FRAMES, 1)  # 跳到当前帧+多少
    success, frame = videoCapture.read()
    tkinter.messagebox("test")

    cv2.rectangle(frame, (x1_tk, y1_tk), (x1_tk + width1, y1_tk + height1), (0, 255, 0), 3)
    cv2.imwrite("Sample_Test.png", frame)
    cv2.imshow("Images sampel", frame)


def read_sample2(rotate):
    """
     讀視頻幀

     :param jobver: 工單版本
     :param x1_tk: x軸坐標
     :param y1_tk: y軸坐標
     :param identifier: 識別碼
     :param data_type: 1. 一棟落打印; 2. 順序打印
     :param composing: 排版個數
     :param box: 一箱數量
     :return:
     """
    fo = open(r"C:\Program Files (x86)\MindVision\Camera\Configs\MV-SUA134GM-Group0.config", "r")
    config_txt = fo.read()
    Gamma = int(''.join(re.compile('gamma = (.*?);', re.S).findall(config_txt)))  # 咖嗎值
    Contrast = int(''.join(re.compile('contrast = (.*?);', re.S).findall(config_txt)))  # 對比度
    ag = float(''.join(re.compile('analog_gain = (.*?);', re.S).findall(config_txt)[0]))/2  # 模擬增益(要除2才是界面數)
    et = float(''.join(re.compile('exp_time = (.*?);', re.S).findall(config_txt)))  # 曝光時間(已經乘以1000)
    # iRot = int(''.join(re.compile('rotate_dir = (.*?);', re.S).findall(config_txt)))  # 旋轉角度(0, 禁用, 1=90, 2=180, 3=270)
    print("读取的字符串是gamma : ", Gamma)
    print("读取的字符串是contrast : ", Contrast)
    print("模擬增益 : ", ag)
    print("曝光時間 : ", et)
    # print("旋轉角度 : ", iRot)

    # 关闭打开的文件
    fo.close()


    config = configparser.ConfigParser()
    config.read("config.ini")
    rtsp = config['DEFAULT']['rtsp']
    # ag = float(config['DEFAULT']['模擬增益(位數)'])
    # et = float(config['DEFAULT']['曝光時間(毫秒)'])
    # Gamma = int(config['DEFAULT']['伽馬值'])
    # Contrast = int(config['DEFAULT']['對比度值'])
    space = int(config['DEFAULT']['每張圖片的間隔'])  # 每张图片的间隔
    row_count = int(config['DEFAULT']['字符水平最大寬度'])  # 有多少行图片
    column_count = int(config['DEFAULT']['字符垂直最大高度'])  # 有多少列图片
    S_target = float(config['DEFAULT']['最大的小目標'])  # 最大的小目標
    print(row_count,column_count)
    file_count = int(config['DEFAULT']['一個檔案放多少個圖像檔'])  # 一个txt文件列出多少个images.
    process_department = config['DEFAULT']['部門']


    print(datetime.datetime.now())

    # 開始:  相機初始化處理

    # print('p:',pImageResolution)
    # 枚举相机
    DevList = mvsdk.CameraEnumerateDevice()
    nDev = len(DevList)
    if nDev < 1:
        print("找不到相机!")
        return
    DevInfo = DevList[0]
    # 打开相机
    hCamera = 0
    try:
        hCamera = mvsdk.CameraInit(DevInfo, -1, -1)
    except mvsdk.CameraException as e:
        print("CameraInit Failed({}): {}".format(e.error_code, e.message))
        return
    # 获取相机特性描述
    cap = mvsdk.CameraGetCapability(hCamera)
    # 判断是黑白相机还是彩色相机
    monoCamera = (cap.sIspCapacity.bMonoSensor != 0)
    # 黑白相机让ISP直接输出MONO数据，而不是扩展成R=G=B的24位灰度
    if monoCamera:
        mvsdk.CameraSetIspOutFormat(hCamera, mvsdk.CAMERA_MEDIA_TYPE_MONO8)
    else:
        mvsdk.CameraSetIspOutFormat(hCamera, mvsdk.CAMERA_MEDIA_TYPE_BGR8)
    # 相机模式切换成连续采集
    mvsdk.CameraSetTriggerMode(hCamera, 2)  # 0表示连续采集模式；1表示软件触发模式；2表示硬件触发模式。

    mvsdk.CameraSetExtTrigSignalType(hCamera, mvsdk.EXT_TRIG_LEADING_EDGE)

    mvsdk.CameraSetLutMode(hCamera, 'LUTMODE_PARAM_GEN')  ## 设置相机的查表变换模式LUT模式= 通过调节参数动态生成LUT表。
    # 手动曝光，曝光时间30ms
    mvsdk.CameraSetAeState(hCamera, 0)
    mvsdk.CameraSetAnalogGainX(hCamera, ag)  # 设置相机的模拟增益放大倍数。
    # mvsdk.CameraSetExposureTime(hCamera, et * 1000)  # 设置曝光时间。单位为微秒。
    mvsdk.CameraSetExposureTime(hCamera, et)  # 设置曝光时间。单位为微秒。
    mvsdk.CameraSetGamma(hCamera, Gamma)  # 设定伽馬值
    mvsdk.CameraSetContrast(hCamera, Contrast)  # 设定对比度值

    mvsdk.CameraSetMirror(hCamera, 1, True)  # 设置图像镜像操作。镜像操作分为水平和垂直两个方向。  之後不需要每個圖像翻转
    # iRot = int(rotate[0])
    # mvsdk.CameraSetRotate(hCamera, iRot)  # 设置图像旋转操作 （0：不旋转 1:90度 2:180度 3:270度）
    # test = mvsdk.CameraGetRotate(hCamera)
    # print(test)
    # 手动Frame
    mvsdk.CameraSetFrameSpeed(hCamera, 1)

    # 让SDK内部取图线程开始工作 (图像采集模式)
    mvsdk.CameraPlay(hCamera)
    # 计算RGB buffer所需的大小，这里直接按照相机的最大分辨率来分配
    # FrameBufferSize = cap.sResolutionRange.iWidthMax * cap.sResolutionRange.iHeightMax * (1 if monoCamera else 3)
    # FrameBufferSize = cap.sResolutionRange.iWidthMin * cap.sResolutionRange.iHeightMin * (1 if monoCamera else 3)
    FrameBufferSize = 1280 * 1024
    # 分配RGB buffer，用来存放ISP输出的图像
    # 备注：从相机传输到PC端的是RAW数据，在PC端通过软件ISP转为RGB数据（如果是黑白相机就不需要转换格式，但是ISP还有其它处理，所以也需要分配这个buffer）
    pFrameBuffer = mvsdk.CameraAlignMalloc(FrameBufferSize, 16)
    # mvsdk.CameraSetImageResolution(hCamera, pImageResolution)     ##设置预览的分辨率
    # SetCameraResolution(hCamera, 0, 0, 1280, 1024)  ##设置预览的分辨率
    mvsdk.CameraSetImageResolutionEx(hCamera, 0xff, 0, 0, 0, 0, 1280, 1024, 0, 0)
    # if iRot == 0 or iRot == 2:  # 如果圖像旋转了, width1,height1 也要跟進調換.
    #     width1 = pImageResolution.iWidth
    #     height1 = pImageResolution.iHeight
    # else:
    #     height1 = pImageResolution.iWidth
    #     width1 = pImageResolution.iHeight

    print("讀視頻線程開始處理")
    # system(r"D:\Xu\vfp\Check\QC.exe")
    while (cv2.waitKey(1) & 0xFF) != 27:  # ord('q')
        # if i % file_count == 1:  # 第一个images就建立txt文件   2021-11-03
        #     txt_file = open(filename, mode='w+', newline='')

        # 从相机取一帧图片
        try:
            # CameraGetImageBuffer函数成功调用后，必须调用CameraReleaseImageBuffer释放缓冲区,以便让内核继续使用该缓冲区。
            pRawData, FrameHead = mvsdk.CameraGetImageBuffer(hCamera,
                                                             200)  # FrameHead=图像的帧头信息指针; pRawData=返回图像数据的缓冲区指针; 200=抓取图像的超时时间，单位毫秒。在wTimes时间内还未获得图像，则该函数会返回超时错误。
            mvsdk.CameraImageProcess(hCamera, pRawData, pFrameBuffer, FrameHead)  # 将相机原始输出图像转换为RGB 格式图像.
            mvsdk.CameraReleaseImageBuffer(hCamera, pRawData)  # 释放缓冲区
            # ima1 = "D:/Xu/python/MV/test/test" + str(ia) + ".bmp"
            # mvsdk.CameraSaveImage(hCamera, ima1, pFrameBuffer, FrameHead, mvsdk.FILE_BMP, 100)
            # print(ima1)
            # windows下取到的图像数据是上下颠倒的，以BMP格式存放。转换成opencv则需要上下翻转成正的
            # linux下直接输出正的，不需要上下翻转
            # if platform.system() == "Windows":
            #     mvsdk.CameraFlipFrameBuffer(pFrameBuffer, FrameHead, 1)

            # 此时图片已经存储在pFrameBuffer中，对于彩色相机pFrameBuffer=RGB数据，黑白相机pFrameBuffer=8位灰度数据
            # 把pFrameBuffer转换成opencv的图像格式以进行后续算法处理
            frame_data = (mvsdk.c_ubyte * FrameHead.uBytes).from_address(pFrameBuffer)
            frame = np.frombuffer(frame_data, dtype=np.uint8)  # 将缓冲区Frame_data解释为一维数组
            # frame = frame.reshape((FrameHead.iHeight, FrameHead.iWidth, 1 if FrameHead.uiMediaType == mvsdk.CAMERA_MEDIA_TYPE_MONO8 else 3))
            frame = frame.reshape(FrameHead.iHeight, FrameHead.iWidth, 1)
            # print(frame.shape)
            # frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)
            cv2.imshow("Press <ESC> to Exit Program!", frame)
            cv2.imwrite("Sample.bmp", frame)

        except mvsdk.CameraException as e:
            if e.error_code != mvsdk.CAMERA_STATUS_TIME_OUT:
                print("CameraGetImageBuffer failed({}): {}".format(e.error_code, e.message))

        # if keyboard.is_pressed("esc"):
        #     cv2.destroyAllWindows()
        #     # 关闭相机
        #     mvsdk.CameraUnInit(hCamera)
        #     # 释放帧缓存
        #     mvsdk.CameraAlignFree(pFrameBuffer)
        #     print("key:esc")
        #     esc = False

    cv2.waitKey()
    cv2.destroyAllWindows()
    # 关闭相机
    mvsdk.CameraUnInit(hCamera)
    # 释放帧缓存
    mvsdk.CameraAlignFree(pFrameBuffer)


    print("讀視頻線程結束")
    print(datetime.datetime.now())

def read_sample():
    global pImageResolution
    """
    讀取坐標樣板

    :param x1_tk: x軸坐標
    :param y1_tk: y軸坐標
    :return:
    """
    # x1_tk, y1_tk = int(x1_tk), int(y1_tk)

    fo = open(r"C:\Program Files (x86)\MindVision\Camera\Configs\MV-SUA134GM-Group0.config", "r")
    config_txt = fo.read()
    Gamma = int(''.join(re.compile('gamma = (.*?);', re.S).findall(config_txt)))  # 咖嗎值
    Contrast = int(''.join(re.compile('contrast = (.*?);', re.S).findall(config_txt)))  # 對比度
    ag = float(''.join(re.compile('analog_gain = (.*?);', re.S).findall(config_txt)[0]))/2  # 模擬增益(要除2才是界面數)
    et = float(''.join(re.compile('exp_time = (.*?);', re.S).findall(config_txt)))  # 曝光時間(已經乘以1000)
    # iRot = int(''.join(re.compile('rotate_dir = (.*?);', re.S).findall(config_txt)))  # 旋轉角度(0, 禁用, 1=90, 2=180, 3=270)
    print("读取的字符串是gamma : ", Gamma)
    print("读取的字符串是contrast : ", Contrast)
    print("模擬增益 : ", ag)
    print("曝光時間 : ", et)
    # print("旋轉角度 : ", iRot)

    # 关闭打开的文件
    fo.close()

    config = configparser.ConfigParser()
    config.read("config.ini")
    rtsp = config['DEFAULT']['rtsp']
    rtsp = "rtsp://admin:123456789@192.168.0.128:554"
    # ag = float(config['DEFAULT']['模擬增益(位數)'])
    # et = float(config['DEFAULT']['曝光時間(毫秒)'])
    # Gamma = int(config['DEFAULT']['伽馬值'])
    # Contrast = int(config['DEFAULT']['對比度值'])
    #videoCapture = cv2.VideoCapture(rtsp)
    #videoCapture = cv2.VideoCapture(0)
    # videoCapture = cv2.VideoCapture(r"D:\Xu\python\Visual Inspection\SourceFile\Test_area.mov")  # 讀本地視頻
    # videoCapture.set(cv2.CAP_PROP_POS_FRAMES, 1)  # 跳到当前帧+多少
    # success, frame = videoCapture.read()
    # 枚举相机
    DevList = mvsdk.CameraEnumerateDevice()
    nDev = len(DevList)
    if nDev < 1:
        print("找不到相机!")
        return
    DevInfo = DevList[0]
    # 打开相机
    hCamera = 0
    try:
        hCamera = mvsdk.CameraInit(DevInfo, -1, -1)
    except mvsdk.CameraException as e:
        print("CameraInit Failed({}): {}".format(e.error_code, e.message))
        return


    # 获取相机特性描述
    cap = mvsdk.CameraGetCapability(hCamera)
    # 判断是黑白相机还是彩色相机
    monoCamera = (cap.sIspCapacity.bMonoSensor != 0)
    # 黑白相机让ISP直接输出MONO数据，而不是扩展成R=G=B的24位灰度
    if monoCamera:
        mvsdk.CameraSetIspOutFormat(hCamera, mvsdk.CAMERA_MEDIA_TYPE_MONO8)
    else:
        mvsdk.CameraSetIspOutFormat(hCamera, mvsdk.CAMERA_MEDIA_TYPE_BGR8)
    # 相机模式切换成连续采集
    mvsdk.CameraSetTriggerMode(hCamera, 0)  # 0表示连续采集模式；1表示软件触发模式；2表示硬件触发模式。  (之前是0)
    # 手动曝光，曝光时间30ms

    mvsdk.CameraSetLutMode(hCamera, 'LUTMODE_PARAM_GEN')  ## 设置相机的查表变换模式LUT模式= 通过调节参数动态生成LUT表。
    mvsdk.CameraSetAeState(hCamera, 0)      #设置相机曝光的模式。自动或者手动。 TRUE:自动曝光；FALSE:手动曝光。
    mvsdk.CameraSetAnalogGainX(hCamera, ag)  # 设置相机的模拟增益放大倍数。
    mvsdk.CameraSetExposureTime(hCamera, et * 1000)  # 设置曝光时间。单位为微秒。

    mvsdk.CameraSetGamma(hCamera, Gamma)  # 设定伽馬值
    mvsdk.CameraSetContrast(hCamera, Contrast)  # 设定对比度值
    mvsdk.CameraSetMirror(hCamera,1,True)   #设置图像镜像操作。镜像操作分为水平和垂直两个方向。  之後不需要每個圖像翻转. 0，表示水平方向；1，表示垂直方向。 TRUE，使能镜像;FALSE，禁止镜像
    iRot = int(rotate.get()[0])
    mvsdk.CameraSetRotate(hCamera, iRot)       #设置图像旋转操作 （0：不旋转 1:90度 2:180度 3:270度）
    # mvsdk.CameraSetRotate(hCamera, 0)  # 设置图像旋转操作 （0：不旋转 1:90度 2:180度 3:270度）
    # test=mvsdk.CameraGetRotate(hCamera)
    # print(test)
    # 让SDK内部取图线程开始工作 (图像采集模式)


    mvsdk.CameraPlay(hCamera)



    # 计算RGB buffer所需的大小，这里直接按照相机的最大分辨率来分配
    # FrameBufferSize = cap.sResolutionRange.iWidthMax * cap.sResolutionRange.iHeightMax * (1 if monoCamera else 3)
    FrameBufferSize = 1280 * 1024
    # 分配RGB buffer，用来存放ISP输出的图像
    # 备注：从相机传输到PC端的是RAW数据，在PC端通过软件ISP转为RGB数据（如果是黑白相机就不需要转换格式，但是ISP还有其它处理，所以也需要分配这个buffer）
    pFrameBuffer = mvsdk.CameraAlignMalloc(FrameBufferSize, 16)
    pImageResolution = mvsdk.CameraCustomizeResolution(hCamera)     #打开分辨率自定义面板，并通过可视化的方式来配置一个自定义分辨率。
    mvsdk.CameraSetImageResolution(hCamera, pImageResolution)
    x_offset.set(pImageResolution.iHOffsetFOV)
    y_offset.set(pImageResolution.iVOffsetFOV)
    x_width.set(pImageResolution.iWidth)
    y_hight.set(pImageResolution.iHeight)
    # 从相机取一帧图片
    try:
        # CameraGetImageBuffer函数成功调用后，必须调用CameraReleaseImageBuffer释放缓冲区,以便让内核继续使用该缓冲区。
        pRawData, FrameHead = mvsdk.CameraGetImageBuffer(hCamera, 200)  # FrameHead=图像的帧头信息指针; pRawData=返回图像数据的缓冲区指针; 200=抓取图像的超时时间，单位毫秒。在wTimes时间内还未获得图像，则该函数会返回超时错误。
        mvsdk.CameraImageProcess(hCamera, pRawData, pFrameBuffer, FrameHead)  # 将相机原始输出图像转换为RGB 格式图像.
        mvsdk.CameraReleaseImageBuffer(hCamera, pRawData)  # 释放缓冲区

        # windows下取到的图像数据是上下颠倒的，以BMP格式存放。转换成opencv则需要上下翻转成正的
        # linux下直接输出正的，不需要上下翻转
        if platform.system() == "Windows":
            a =1
            # mvsdk.CameraFlipFrameBuffer(pFrameBuffer, FrameHead, 1)


        # 此时图片已经存储在pFrameBuffer中，对于彩色相机pFrameBuffer=RGB数据，黑白相机pFrameBuffer=8位灰度数据
        # 把pFrameBuffer转换成opencv的图像格式以进行后续算法处理
        frame_data = (mvsdk.c_ubyte * FrameHead.uBytes).from_address(pFrameBuffer)
        frame = np.frombuffer(frame_data, dtype=np.uint8)  # 将缓冲区Frame_data解释为一维数组
        frame = frame.reshape((FrameHead.iHeight, FrameHead.iWidth, 1 if FrameHead.uiMediaType == mvsdk.CAMERA_MEDIA_TYPE_MONO8 else 3))

        # frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)
        # cv2.imshow("Press q to end", frame)
        # cv2.rectangle(frame, (x1_tk, y1_tk), (x1_tk + width1, y1_tk + height1), (0, 255, 0), 3)
        cv2.imwrite("Sample_Test.jpg", frame)
        # cv2.imshow("Images sampel", frame)
    except mvsdk.CameraException as e:
        if e.error_code != mvsdk.CAMERA_STATUS_TIME_OUT:
            print("CameraGetImageBuffer failed({}): {}".format(e.error_code, e.message))

    # cv2.destroyAllWindows()
    # videocapture.release()  # 释放资源
    # 关闭相机
    mvsdk.CameraUnInit(hCamera)
    # 释放帧缓存
    mvsdk.CameraAlignFree(pFrameBuffer)


def get_pixel(x1_tk, y1_tk):
    """
    讀取坐標樣板Opencv 顏色, 注意與小畫家裏的顏色是不同的.

    :param x1_tk: x軸坐標
    :param y1_tk: y軸坐標
    :return:
    """
    videoCapture = cv2.VideoCapture(r"D:\Xu\python\Visual Inspection\SourceFile\Test_area.mov")  # 讀本地視頻
    videoCapture.set(cv2.CAP_PROP_POS_FRAMES, 1)  # 跳到当前帧+多少
    success, frame = videoCapture.read()
    #xy=tuple(frame[int(y1_tk), int(x1_tk)])
    xy = frame[int(y1_tk), int(x1_tk)]
    entry_template_backcolor.set("("+str(xy[0])+", "+str(xy[1])+", "+str(xy[2])+")")


def process_img(img1,S_target,row_count,column_count,jobver_id,i,identifier,frame):
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # ret, binary = cv2.threshold(frame, 150, 255, cv2.THRESH_BINARY)      #将图像信息二值化，处理过后的图片只有二种色值(黑0,白255)  ; 大於100的值變為白色(255), 小於等於100的變為黑色(0)
    ret, binary = cv2.threshold(img1, 100, 255,cv2.THRESH_BINARY)  # 将图像信息二值化，处理过后的图片只有二种色值(黑0,白255)  ; 大於100的值變為白色(255), 小於等於100的變為黑色(0)
    # ret, binary = cv2.threshold(img1, 50, 255,cv2.THRESH_BINARY)  # 将图像信息二值化，处理过后的图片只有二种色值(黑0,白255)  ; 大於100的值變為白色(255), 小於等於100的變為黑色(0)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 找轮廓
    ni = 0
    for cnt in contours:
        max_xy = cnt.max(axis=0)
        min_xy = cnt.min(axis=0)
        ni = ni + 1

        # if cv2.contourArea(cnt) > 200:  # 計算轮廓面積, 大於100的不處理. 小於100的用才處理.
        if ((cv2.contourArea(cnt) > S_target) and (max_xy[0, 0] - min_xy[0, 0]) < row_count and (
                max_xy[0, 1] - min_xy[0, 1]) < column_count) or ni == 1:  # 計算轮廓面積, 大於100的不處理. 小於100的用才處理.
            continue
        cv2.drawContours(binary, contours, ni - 1, (255, 255, 255), -1)  # 小於100的用白(255)填充 (用二值黑白(0,255))
        # cv2.drawContours(frame, cnt, -1, (255, 255, 255), -1)  # 小於100的用白(255)填充 (用源圖片)+
    # x1, y1 = 4, 4
    # x2, y2 = width1+4, height1+4
    # img1 = np.zeros(((height1 + 8), (width1 + 8)), np.uint8) + 255  # 建立空白图片.
    # # img1[y1:y2, x1:x2] = binary
    # img1[y1+2:y2, x1+2:x2] = binary[2:height1,2:width1]
    img_file="/img" + str(i).rjust(7, '0')
    filename = "./images/" + jobver_id + img_file + ".png"
    # filename_org = "./images_org/" + jobver_id + img_file + ".png"      ##折书部不用
    filename_OCR = "./OCR/" + jobver_id + img_file + ".txt"

    # cv2.imwrite("./images/" + jobver_id + "/img" + str(i).rjust(7, '0') + ".png", binary)  # 保存图像
    # cv2.imwrite(filename_org, frame)  # 保存图像
    cv2.imwrite(filename, binary)  # 保存图像
    thread_ocr = threading.Thread(target=ocr_thread, args=(jobver_id, filename, filename_OCR, identifier))
    thread_ocr.start()


def readv(jobver, rotate, identifier, data_type, composing, box):
    """
    讀視頻幀

    :param jobver: 工單版本
    :param x1_tk: x軸坐標
    :param y1_tk: y軸坐標
    :param identifier: 識別碼
    :param data_type: 1. 一棟落打印; 2. 順序打印
    :param composing: 排版個數
    :param box: 一箱數量
    :return:
    """
    global pImageResolution
    # 注意, OCR 图像识别如果不是正方向, 识别文字速度会很慢, 很慢. 所以相机要正方向, 不能90度角, 180度角, 等等.
    # 全局设置

    # re3 = re.compile('gamma = (.*?);', re.S)  # 不做過濾條件

    # 打开一个文件
    update_std_char()

    config = configparser.ConfigParser()
    config.read("config.ini")
    rtsp = config['DEFAULT']['rtsp']
    # ag = float(config['DEFAULT']['模擬增益(位數)'])
    # et = float(config['DEFAULT']['曝光時間(毫秒)'])
    # Gamma = int(config['DEFAULT']['伽馬值'])
    # Contrast = int(config['DEFAULT']['對比度值'])
    space = int(config['DEFAULT']['每張圖片的間隔'])  # 每张图片的间隔
    row_count = int(config['DEFAULT']['字符水平最大寬度'])  # 有多少行图片
    column_count = int(config['DEFAULT']['字符垂直最大高度'])  # 有多少列图片
    S_target = float(config['DEFAULT']['最大的小目標'])  # 最大的小目標
    print(row_count,column_count)
    file_count = int(config['DEFAULT']['一個檔案放多少個圖像檔'])  # 一个txt文件列出多少个images.
    process_department = config['DEFAULT']['部門']
    MV_config = config['DEFAULT']['相機配置文件']
    # ocr_list=set()

    fo = open(MV_config, "r")
    config_txt = fo.read()
    Gamma = int(''.join(re.compile('gamma = (.*?);', re.S).findall(config_txt)))  # 咖嗎值
    Contrast = int(''.join(re.compile('contrast = (.*?);', re.S).findall(config_txt)))  # 對比度
    ag = float(''.join(re.compile('analog_gain = (.*?);', re.S).findall(config_txt)[0]))/2  # 模擬增益(要除2才是界面數)
    et = float(''.join(re.compile('exp_time = (.*?);', re.S).findall(config_txt)))  # 曝光時間(已經乘以1000)
    # iRot = int(''.join(re.compile('rotate_dir = (.*?);', re.S).findall(config_txt)))  # 旋轉角度(0, 禁用, 1=90, 2=180, 3=270)
    print("读取的字符串是gamma : ", Gamma)
    print("读取的字符串是contrast : ", Contrast)
    print("模擬增益 : ", ag)
    print("曝光時間 : ", et)
    # print("旋轉角度 : ", iRot)

    # 关闭打开的文件
    fo.close()


    # 打开文件
    fo = open("./jobver_list.txt", "a+")
    # print("文件名为: ", fo.name)
    fo.seek(0, 0)
    line = fo.read()
    str1 = re.findall(jobver + '-' + '[0-9]{4}', line)
    jobver_id = jobver + "-" + str(str1.__len__() + 1).rjust(4, '0')
    print(jobver_id)
    fo.writelines(jobver_id + "\n")
    fo.close()
    # 关闭文件
    # print("test",pImageResolution.iHOffsetFOV)
    fo2 = open("./working.txt", "w")


    # pImageResolution.iWidth

    # pImageResolution.iHeight

    # pointsize=";\n水平偏移=" + str(pImageResolution.iHOffsetFOV) +";\n垂直偏移=" + str(pImageResolution.iVOffsetFOV)+";\n寬=" + str(pImageResolution.iWidthFOV)+";\n高=" + str(pImageResolution.iHeightFOV)+";"
    pointsize = ";\nx_offset=" + x_offset.get() + ";\ny_offset=" + y_offset.get() + ";\nx_width=" + x_width.get() + ";\ny_hight=" + y_hight.get() + ";"
    t1=print_type.get()[0]
    t2=layout_count.get()
    t3=start_seq.get()
    t4=repeat_print.get()
    t5=box_qty.get()
    t6=dbf_file.get()
    info1="jobver_id=" + jobver_id + ";\nidentifier=" + identifier + ";\nprint_type=" + t1 + ";\nlayout_count=" + t2 + ";\nstart_seq=" + t3 + ";\nrepeat_print=" + t4 + ";\nbox_qty=" + t5 + ";\norg_data=" + t6 + ";\ndepartment=" + process_department + ";\nrotate=" + rotate
    fo2.write(info1 + pointsize)
    fo2.close()
    if not os.path.exists("./job/" + jobver):
        mkdir("./job/" + jobver)
    mkdir("./OCR/" + jobver_id)
    mkdir("./images/" + jobver_id)
    mkdir("./images_org/" + jobver_id)
    copyfile('D:/Xu/python/MV/working.txt', "D:/Xu/python/MV/job/" + jobver + "/setting.txt")
    # str=re.findall(jobver+'-'+'\d{3}', jvr)

    print(datetime.datetime.now())

    # 開始:  相機初始化處理

    # print('p:',pImageResolution)
    # 枚举相机
    DevList = mvsdk.CameraEnumerateDevice()
    nDev = len(DevList)
    if nDev < 1:
        print("找不到相机!")
        return
    DevInfo = DevList[0]
    # 打开相机
    hCamera = 0
    try:
        hCamera = mvsdk.CameraInit(DevInfo, -1, -1)
    except mvsdk.CameraException as e:
        print("CameraInit Failed({}): {}".format(e.error_code, e.message))
        return
    # 获取相机特性描述
    cap = mvsdk.CameraGetCapability(hCamera)
    # 判断是黑白相机还是彩色相机
    monoCamera = (cap.sIspCapacity.bMonoSensor != 0)
    # 黑白相机让ISP直接输出MONO数据，而不是扩展成R=G=B的24位灰度
    if monoCamera:
        mvsdk.CameraSetIspOutFormat(hCamera, mvsdk.CAMERA_MEDIA_TYPE_MONO8)
    else:
        mvsdk.CameraSetIspOutFormat(hCamera, mvsdk.CAMERA_MEDIA_TYPE_BGR8)
    # 相机模式切换成连续采集
    mvsdk.CameraSetTriggerMode(hCamera, 2)  # 0表示连续采集模式；1表示软件触发模式；2表示硬件触发模式。

    mvsdk.CameraSetExtTrigSignalType(hCamera, mvsdk.EXT_TRIG_LEADING_EDGE)

    mvsdk.CameraSetLutMode(hCamera, 'LUTMODE_PARAM_GEN')  ## 设置相机的查表变换模式LUT模式= 通过调节参数动态生成LUT表。
    # 手动曝光，曝光时间30ms
    mvsdk.CameraSetAeState(hCamera, 0)
    mvsdk.CameraSetAnalogGainX(hCamera, ag)  # 设置相机的模拟增益放大倍数。
    # mvsdk.CameraSetExposureTime(hCamera, et * 1000)  # 设置曝光时间。单位为微秒。
    mvsdk.CameraSetExposureTime(hCamera, et)  # 设置曝光时间。单位为微秒。
    mvsdk.CameraSetGamma(hCamera,Gamma)       #设定伽馬值
    mvsdk.CameraSetContrast(hCamera,Contrast)    #设定对比度值

    mvsdk.CameraSetMirror(hCamera,1,True)   #设置图像镜像操作。镜像操作分为水平和垂直两个方向。  之後不需要每個圖像翻转
    iRot=int(rotate[0])
    mvsdk.CameraSetRotate(hCamera, iRot)       #设置图像旋转操作 （0：不旋转 1:90度 2:180度 3:270度）
    test=mvsdk.CameraGetRotate(hCamera)
    # print(test)
    #手动Frame
    mvsdk.CameraSetFrameSpeed(hCamera, 1)

    # 让SDK内部取图线程开始工作 (图像采集模式)
    mvsdk.CameraPlay(hCamera)
    # 计算RGB buffer所需的大小，这里直接按照相机的最大分辨率来分配
    # FrameBufferSize = cap.sResolutionRange.iWidthMax * cap.sResolutionRange.iHeightMax * (1 if monoCamera else 3)
    # FrameBufferSize = cap.sResolutionRange.iWidthMin * cap.sResolutionRange.iHeightMin * (1 if monoCamera else 3)
    FrameBufferSize = 1280 * 1024
    # 分配RGB buffer，用来存放ISP输出的图像
    # 备注：从相机传输到PC端的是RAW数据，在PC端通过软件ISP转为RGB数据（如果是黑白相机就不需要转换格式，但是ISP还有其它处理，所以也需要分配这个buffer）
    pFrameBuffer = mvsdk.CameraAlignMalloc(FrameBufferSize, 16)
    mvsdk.CameraSetImageResolution(hCamera, pImageResolution)     ##设置预览的分辨率

    # mvsdk.CameraSetImageResolutionEx(hCamera, 0xff, 0, 0, int(x_offset.get()), int(y_offset.get()), int(x_width.get()), int(y_hight.get()), 0, 0)
    if iRot==0 or iRot==2:      #如果圖像旋转了, width1,height1 也要跟進調換.
        width1 = pImageResolution.iWidth
        height1 =pImageResolution.iHeight
        # width1 = int(x_width.get())
        # height1 =int(y_hight.get())
    else:
        height1 = pImageResolution.iWidth
        width1=pImageResolution.iHeight
        # height1 = int(x_width.get())
        # width1=int(y_hight.get())

    # 結束:  相機初始化處理

    # x1c, y1c = int(x1_tk), int(y1_tk)
    # x2c, y2c = x1c+width1, y1c+height1

    success = True
    esc = True
    wk = ""
    i = 1
    filename = "./job/" + jobver_id + "/" + jobver_id + "-" + str(i).rjust(6, '0') + ".txt"  # 后期要变量

    filename_OCR = "./OCR/" + jobver_id + "/" + jobver_id + "-" + str(i).rjust(6, '0') + ".txt"  # 后期要变量
    x1, y1 = 0, 0
    # img1 = np.zeros(((height1 + space) * row_count, (width1 + space) * column_count, 3), np.uint8) + 255  # 建立空白图片.
    img1 = np.zeros(((height1 + 8), (width1 + 8), 1), np.uint8) + 255  # 建立空白图片.
    x_pixel = (width1 + space) * column_count
    y_pixel = (height1 + space) * row_count
    #with open(filename, mode='w+', newline='') as txt_file:
    if int(repeat_print.get()) > 1:
        thread_checkrepeat = threading.Thread(target=checkrepeat1)
        thread_checkrepeat.start()
    else:
        thread_qc = threading.Thread(target=qc)
        thread_qc.start()
    print("讀視頻線程開始處理")
    # system(r"D:\Xu\vfp\Check\QC.exe")
    while (cv2.waitKey(1) & 0xFF) != 27:      #ord('q')
        # if i % file_count == 1:  # 第一个images就建立txt文件   2021-11-03
        #     txt_file = open(filename, mode='w+', newline='')

        # 从相机取一帧图片
        try:
            # CameraGetImageBuffer函数成功调用后，必须调用CameraReleaseImageBuffer释放缓冲区,以便让内核继续使用该缓冲区。
            pRawData, FrameHead = mvsdk.CameraGetImageBuffer(hCamera, 200)  # FrameHead=图像的帧头信息指针; pRawData=返回图像数据的缓冲区指针; 200=抓取图像的超时时间，单位毫秒。在wTimes时间内还未获得图像，则该函数会返回超时错误。
            mvsdk.CameraImageProcess(hCamera, pRawData, pFrameBuffer, FrameHead)  # 将相机原始输出图像转换为RGB 格式图像.
            mvsdk.CameraReleaseImageBuffer(hCamera, pRawData)  # 释放缓冲区
            # ima1 = "D:/Xu/python/MV/test/test" + str(ia) + ".bmp"
            # mvsdk.CameraSaveImage(hCamera, ima1, pFrameBuffer, FrameHead, mvsdk.FILE_BMP, 100)
            # print(ima1)
            # windows下取到的图像数据是上下颠倒的，以BMP格式存放。转换成opencv则需要上下翻转成正的
            # linux下直接输出正的，不需要上下翻转
            # if platform.system() == "Windows":
            #     mvsdk.CameraFlipFrameBuffer(pFrameBuffer, FrameHead, 1)

            # 此时图片已经存储在pFrameBuffer中，对于彩色相机pFrameBuffer=RGB数据，黑白相机pFrameBuffer=8位灰度数据
            # 把pFrameBuffer转换成opencv的图像格式以进行后续算法处理
            frame_data = (mvsdk.c_ubyte * FrameHead.uBytes).from_address(pFrameBuffer)
            frame = np.frombuffer(frame_data, dtype=np.uint8)  # 将缓冲区Frame_data解释为一维数组
            # frame = frame.reshape((FrameHead.iHeight, FrameHead.iWidth, 1 if FrameHead.uiMediaType == mvsdk.CAMERA_MEDIA_TYPE_MONO8 else 3))
            frame = frame.reshape(FrameHead.iHeight, FrameHead.iWidth,1)
            # print(frame.shape)
            # frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)

            cv2.imshow("Press <ESC> to Exit Program!", frame)

            #####加入圖像處理去黑點
            x1, y1 = 4, 4
            x2, y2 = width1 + 4, height1 + 4
            img1 = np.zeros(((height1 + 8), (width1 + 8),1), np.uint8) + 255  # 建立空白图片.
            img1[y1 + 2:y2, x1 + 2:x2] = frame[2:height1, 2:width1]


            thread_process_img = threading.Thread(target=process_img, args=(img1,S_target,row_count,column_count,jobver_id,i,identifier,frame))
            thread_process_img.start()
            # txt_file.write(getcwd() + "/images/" + jobver_id + "/img" + str(i) + ".png" + "\n")  # 保存图像文件路径; getcwd() #返回当前工作目录
            # if i % file_count == 0:  # 最后一个images就Close txt文件   2021-11-03
            #     txt_file.close()
            #     thread = threading.Thread(target=ocr_thread, args=(jobver_id, filename, filename_OCR, identifier))
            #     thread.start()
            #     filename = "./job/" + jobver_id + "/" + jobver_id + "-" + str(int(i / file_count) + 1).rjust(6, '0') + ".txt"
            #     filename_OCR = "./OCR/" + jobver_id + "/" + jobver_id + "-" + str(int(i / file_count) + 1).rjust(6, '0') + ".txt"
            i += 1
            #####
        except mvsdk.CameraException as e:
            if e.error_code != mvsdk.CAMERA_STATUS_TIME_OUT:
                print("CameraGetImageBuffer failed({}): {}".format(e.error_code, e.message))

        # if keyboard.is_pressed("esc"):
        #     cv2.destroyAllWindows()
        #     # 关闭相机
        #     mvsdk.CameraUnInit(hCamera)
        #     # 释放帧缓存
        #     mvsdk.CameraAlignFree(pFrameBuffer)
        #     print("key:esc")
        #     esc = False

    cv2.waitKey()
    cv2.destroyAllWindows()
    # 关闭相机
    mvsdk.CameraUnInit(hCamera)
    # 释放帧缓存
    mvsdk.CameraAlignFree(pFrameBuffer)
    condition_std_char("./images/" + jobver_id + "/img0000001.png")
    thread = threading.Thread(target=ocr_thread, args=(jobver_id, filename,filename_OCR, identifier,))
    thread.start()
    # print("結束OCR", filename)
    # # 合並txt 檔案
    # copyfile('D:/Xu/python/MV/working.txt', "D:/Xu/python/MV/job/" + jobver_id + "/setting.txt")
    # OCR_txt = "./OCR/" + jobver_id
    # file = listdir(OCR_txt)
    # file.sort()
    # sn1 = open(OCR_txt+"/"+jobver_id+"_OCR_seq.txt", 'a+')
    # for f in file:
    #     sn2 = open(OCR_txt+"/" + f, 'r')
    #     txt = sn2.read()
    #     sn2.close()
    #     sn1.write(txt)
    #     sn1.write("\n")
    # sn1.close()
    #
    # # 合並txt 檔案
    print("讀視頻線程結束")
    print(datetime.datetime.now())


def new_config():
    """
    重置Config.ini 參數

    :return:
    """
    config=configparser.ConfigParser()
    config['DEFAULT'] = {"每張圖片的間隔": 5, "字符水平最大寬度": 3, "字符垂直最大高度": 3, "模擬增益(位數)": 3.5, "曝光時間(毫秒)": 3.5,"一個檔案放多少個圖像檔":3, "備份檔案路徑": "./", "輸出資料路徑": "./", "rtsp":"rtsp://admin:zxt12345@10.7.86.115:554/ch1-s1?tcp"}
    config['scan_area'] = {"標準邊界底色":(21,32,13), "顏色允許誤差范圍":20, "邊界底色范圍":(467, 885, 493, 897), "打印邊緣范圍":(463, 845, 495, 857)}
    with open("Config.ini", 'w') as config_file:
        config.write(config_file)


def save_config():
    """
    保存Config.ini 參數

    :return:
    """
    config = configparser.ConfigParser()
    config.read("config.ini")
    #print(entry_space.get())

    config.set("DEFAULT", "每張圖片的間隔", entry_space.get())
    config.set("DEFAULT", "字符水平最大寬度", entry_row_count.get())
    config.set("DEFAULT", "字符垂直最大高度", entry_column_count.get())
    config.set("DEFAULT", "模擬增益(位數)", entry_AnalogGainX.get())
    config.set("DEFAULT", "曝光時間(毫秒)", entry_ExposureTime.get())
    config.set("DEFAULT", "伽馬值", entry_Gamma.get())
    config.set("DEFAULT", "對比度值", entry_Contrast.get())

    config.set("DEFAULT", "一個檔案放多少個圖像檔", entry_images_count.get())

    config.set("scan_area", "標準邊界底色", entry_template_backcolor.get())
    config.set("scan_area", "顏色允許誤差范圍", entry_tolerance_scope.get())
    config.set("scan_area", "邊界底色范圍", entry_back_scope.get())
    config.set("scan_area", "打印邊緣范圍", entry_edge_scope.get())

    config.set("DEFAULT", "備份檔案路徑", entry_images_path.get())
    config.set("DEFAULT", "輸出資料路徑", entry_data_path.get())
    config.set("DEFAULT", "rtsp", entry_rtsp.get())

    config.write(open('config.ini', 'w'))


def show_config():
    """
    顯示配置界面, 載入config.ini 數據
    :return:
    """
    config = configparser.ConfigParser()
    config.read("config.ini")
    frame1.grid_forget()
    frame2.grid_forget()
    # frame2B.grid_forget()
    frame2A.grid_forget()
    frame3.grid(row=0, column=0)
    frame3b.grid(row=1, column=0)
    frame4.grid(row=2, column=0)
    entry_space.set(config['DEFAULT']['每張圖片的間隔'])
    entry_row_count.set(config['DEFAULT']['字符水平最大寬度'])
    entry_column_count.set(config['DEFAULT']['字符垂直最大高度'])
    entry_AnalogGainX.set(config['DEFAULT']['模擬增益(位數)'])
    entry_ExposureTime.set(config['DEFAULT']['曝光時間(毫秒)'])
    entry_Gamma.set(config['DEFAULT']['伽馬值'])
    entry_Contrast.set(config['DEFAULT']['對比度值'])
    entry_images_count.set(config['DEFAULT']['一個檔案放多少個圖像檔'])
    entry_template_backcolor.set(config['scan_area']['標準邊界底色'])
    entry_tolerance_scope.set(config['scan_area']['顏色允許誤差范圍'])
    entry_back_scope.set(config['scan_area']['邊界底色范圍'])
    entry_edge_scope.set(config['scan_area']['打印邊緣范圍'])
    entry_images_path.set(config['DEFAULT']['備份檔案路徑'])
    entry_data_path.set(config['DEFAULT']['輸出資料路徑'])
    entry_rtsp.set(config['DEFAULT']['rtsp'])


def show_main():
    """
    顯示主界面
    :return:
    """
    frame3.grid_forget()
    frame3b.grid_forget()
    frame4.grid_forget()
    frame1.grid(row=0, column=0)
    frame2.grid(row=1, column=0)
    # frame2B.grid(row=2, column=0)
    frame2A.grid(row=3, column=0)



def load_check():
    """
    載入tk 時, 檢測環境.

    :return:
    """
    if not path.exists("./job"):
        mkdir("./job")
    if not path.exists("./images_org"):
        mkdir("./images_org")
    if not path.exists("./images"):
        mkdir("./images")
    if not path.exists("./OCR"):
        mkdir("./OCR")


def MoveFile():
    config = configparser.ConfigParser()
    config.read("config.ini")
    day_data = int(config['DEFAULT']['每張圖片的間隔'])  # MV 文件夾保留多少天的數據
    bakup_data_path = config['DEFAULT']['備份檔案路徑']  # MV 備份檔案路徑
    movefold = ['D:/Xu/python/MV/images/', 'D:/Xu/python/MV/job/', 'D:/Xu/python/MV/OCR/']

    for k in movefold:
        print(k)
        if k.endswith("images/"):
            # print('images')
            fold2 = bakup_data_path+"/images/"  ##新目錄
        if k.endswith("job/"):
            # print('job')
            fold2 = bakup_data_path+"/job/"  ##新目錄
        if k.endswith("OCR/"):
            # print('OCR')
            fold2 =bakup_data_path+"/OCR/"  ##新目錄
        z=listdir(k)
        date1=datetime.date.today()

        for i in z:
            fold1=k + i   ##需要移動的目錄
            ctime = time.localtime(path.getctime(fold1))
            date2 = datetime.date(ctime.tm_year, ctime.tm_mon, ctime.tm_mday)
            date3 = date1 - date2
            if date3.days>day_data:
                move(fold1, fold2)
                a1=1
            # print("目錄:", fold1, fold2, "日期相差", date3.days)


def condition_std_char(frame):
    # frame = cv2.imread(r'D:\Xu\python\MV\std_char\img0000001.png')
    # frame2 = cv2.imread(r'D:\Xu\python\MV\std_char\Small_target.png')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 150, 255,
                                cv2.THRESH_BINARY)  # 将图像信息二值化，处理过后的图片只有二种色值(黑0,白255)  ; 大於100的值變為白色(255), 小於等於100的變為黑色(0)
    # ret2, binary2 = cv2.threshold(gray2, 150, 255,
    #                               cv2.THRESH_BINARY)  # 将图像信息二值化，处理过后的图片只有二种色值(黑0,白255)  ; 大於100的值變為白色(255), 小於等於100的變為黑色(0)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 找轮廓
    # contours2, hierarchy2 = cv2.findContours(binary2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 找轮廓
    # RETR_TREE
    # RETR_LIST
    # RETR_CCOMP
    ni = 0
    list1 = []
    list2 = []
    for cnt in contours:
        max_xy = cnt.max(axis=0)
        min_xy = cnt.min(axis=0)
        ni = ni + 1
        if ni != 1:
            list1.append(max_xy[0, 0] - min_xy[0, 0])
            list2.append(max_xy[0, 1] - min_xy[0, 1])
    ni2 = 0
    list2a = []
    # for cnt2 in contours2:
    #     ni2 = ni2 + 1
    #     if ni2 != 1:
    #         list2a.append(cv2.contourArea(cnt2))

    # Small_target = float(max(list2a))
    char_width = int(max(list1) * 1.5)
    char_hight = int(max(list2) * 1.35)
    print("字符寬度 ", char_width)
    print("字符高度: ", char_hight)
    # print("最大的小目標: ", Small_target)
    if char_width>37 and char_hight>52:
        copyfile(r"D:\xu\Python\mv\std_char\7pt-img0000001.png", r"D:\xu\Python\mv\std_char\img0000001.png")
        copyfile(r"D:\xu\Python\mv\std_char\7pt-Small_target.png", r"D:\xu\Python\mv\std_char\Small_target.png")
    else:
        copyfile(r"D:\xu\Python\mv\std_char\5pt-img0000001.png", r"D:\xu\Python\mv\std_char\img0000001.png")
        copyfile(r"D:\xu\Python\mv\std_char\5pt-Small_target.png", r"D:\xu\Python\mv\std_char\Small_target.png")
    update_std_char()


def update_std_char():
    frame = cv2.imread(r'D:\Xu\python\MV\std_char\img0000001.png')
    frame2 = cv2.imread(r'D:\Xu\python\MV\std_char\Small_target.png')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 150, 255,
                                cv2.THRESH_BINARY)  # 将图像信息二值化，处理过后的图片只有二种色值(黑0,白255)  ; 大於100的值變為白色(255), 小於等於100的變為黑色(0)
    ret2, binary2 = cv2.threshold(gray2, 150, 255,
                                  cv2.THRESH_BINARY)  # 将图像信息二值化，处理过后的图片只有二种色值(黑0,白255)  ; 大於100的值變為白色(255), 小於等於100的變為黑色(0)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 找轮廓
    contours2, hierarchy2 = cv2.findContours(binary2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 找轮廓
    # RETR_TREE
    # RETR_LIST
    # RETR_CCOMP
    ni = 0
    list1 = []
    list2 = []
    for cnt in contours:
        max_xy = cnt.max(axis=0)
        min_xy = cnt.min(axis=0)
        ni = ni + 1
        if ni != 1:
            list1.append(max_xy[0, 0] - min_xy[0, 0])
            list2.append(max_xy[0, 1] - min_xy[0, 1])
    ni2 = 0
    list2a = []
    for cnt2 in contours2:
        ni2 = ni2 + 1
        if ni2 != 1:
            list2a.append(cv2.contourArea(cnt2))

    Small_target = float(max(list2a))
    char_width = int(max(list1) * 1.5)
    char_hight = int(max(list2) * 1.35)
    print("字符寬度 ", char_width)
    print("字符高度: ", char_hight)
    print("最大的小目標: ", Small_target)
    config = configparser.ConfigParser()
    config.read("config.ini")
    config.set("DEFAULT", "字符水平最大寬度", char_width.__str__())
    config.set("DEFAULT", "字符垂直最大高度", char_hight.__str__())
    config.set("DEFAULT", "最大的小目標", Small_target.__str__())
    config.write(open('config.ini', 'w'))


def Get_Config():
    re3 = re.compile('gamma = (.*?);', re.S)  # 不做過濾條件

    # 打开一个文件
    fo = open("MV-SUA134GM-Group0.config", "r")
    str = fo.read()
    gamma = re.compile('gamma = (.*?);', re.S).findall(str)  # 咖嗎值
    contrast = re.compile('contrast = (.*?);', re.S).findall(str)  # 對比度
    analog_gain = re.compile('analog_gain = (.*?);', re.S).findall(str)  # 模擬增益(要除2才是界面數)
    exp_time = re.compile('exp_time = (.*?);', re.S).findall(str)  # 曝光時間(已經乘以1000)

    rotate_dir = re.compile('rotate_dir = (.*?);', re.S).findall(str)  # 旋轉角度(0, 禁用, 1=90, 2=180, 3=270)
    # print("读取的字符串是gamma : ", gamma)
    # print("读取的字符串是contrast : ", contrast)

    # 关闭打开的文件
    fo.close()


def MVDCP():
    # hp, ht, pid, tid = _winapi.CreateProcess(None, r'"C:\Program Files (x86)\MindVision\MVDCP_X64.exe"', None, None, False, 0,None, None, None)
    system(r'"C:\Program Files (x86)\MindVision\MVDCP_X64.exe"')
    # win32api.ShellExecute(0, 'open', r'"C:\Program Files (x86)\MindVision\MVDCP_X64.exe"', '', '', 0)


def qc():
    hp, ht, pid, tid = _winapi.CreateProcess(None, r"D:\Xu\vfp\Check\qc.exe", None, None, False, 0, None, None, None)
    # if int(repeat_print.get())>1:
    #     print("大於1")
    #     hp, ht, pid, tid = _winapi.CreateProcess(None, r"D:\Xu\vfp\CheckRepeat\checkrepeat.exe", None, None, False, 0, None, None, None)
    # else:
    #     print("不大於1")
    #     hp, ht, pid, tid = _winapi.CreateProcess(None, r"D:\Xu\vfp\Check\qc.exe", None, None, False, 0, None, None, None)


    # system(r"D:\Xu\vfp\Check\qc.exe")
    # win32api.ShellExecute(0, 'open', r"D:\Xu\vfp\Check\qc.exe", '', '', 0)

def checkrepeat1():
    hp, ht, pid, tid = _winapi.CreateProcess(None, r"D:\Xu\vfp\CheckRepeat\checkrepeat.exe", None, None, False, 0, None, None, None)
    # if int(repeat_print.get())>1:
    #     print("大於1")
    #     hp, ht, pid, tid = _winapi.CreateProcess(None, r"D:\Xu\vfp\CheckRepeat\checkrepeat.exe", None, None, False, 0, None, None, None)
    # else:
    #     print("不大於1")
    #     hp, ht, pid, tid = _winapi.CreateProcess(None, r"D:\Xu\vfp\Check\qc.exe", None, None, False, 0, None, None, None)


    # system(r"D:\Xu\vfp\Check\qc.exe")
    # win32api.ShellExecute(0, 'open', r"D:\Xu\vfp\Check\qc.exe", '', '', 0)+

def load_jobver(load_jobver):
    job_setting="D:/Xu/python/MV/job/" + load_jobver + "/setting.txt"
    if os.path.isfile(job_setting):
        fo = open(job_setting, "r")
        # print("文件名为: ", fo.name)
        jobinfo = fo.read()
        load_identifier = "".join(re.findall("identifier=(.*?);", jobinfo,re.S))
        load_print_type = "".join(re.findall("print_type=(.*?);", jobinfo, re.S))
        load_layout_count = "".join(re.findall("layout_count=(.*?);", jobinfo, re.S))
        load_box_qty = "".join(re.findall("box_qty=(.*?);", jobinfo, re.S))
        load_start_seq = "".join(re.findall("start_seq=(.*?);", jobinfo, re.S))
        load_rotate = "".join(re.findall("rotate=(.*?);", jobinfo, re.S))
        load_a = "".join(re.findall("x_offset=(.*?);", jobinfo, re.S))
        load_b = "".join(re.findall("y_offset=(.*?);", jobinfo, re.S))
        load_c = "".join(re.findall("x_width=(.*?);", jobinfo, re.S))
        load_d = "".join(re.findall("y_hight=(.*?);", jobinfo, re.S))
        identifier.set(load_identifier)
        if load_print_type=="1":
            print_type.set("1 - 一棟落順序打印")
        elif load_print_type=="2":
            print_type.set("2-一棟落倒序打印")
        elif load_print_type=="3":
            print_type.set("3-兜圈順序打印")

        layout_count.set(load_layout_count)
        box_qty.set(load_box_qty)
        start_seq.set(load_start_seq)
        rotate.set(load_rotate)
        x_offset.set(load_a)
        y_offset.set(load_b)
        x_width.set(load_c)
        y_hight.set(load_d)
        # remark.set("水平偏移:"+load_a.__str__()+"; 垂直偏移:"+load_b.__str__()+"; 寬:"+load_c.__str__()+"; 高:"+load_d.__str__())


def SetCameraResolution(hCamera, offsetx, offsety, width, height):

    # sRoiResolution=mvsdk.tSdkImageResolution
    #
    #
    # sRoiResolution.iIndex = 0xff
    # sRoiResolution.iWidth=width
    # sRoiResolution.iWidthFOV = width
    # sRoiResolution.iHeight = height
    # sRoiResolution.iHeightFOV = height
    # #视场偏移
    # sRoiResolution.iHOffsetFOV = offsetx
    # sRoiResolution.iVOffsetFOV = offsety
    #
    # sRoiResolution.iWidthZoomSw = 0
    # sRoiResolution.iHeightZoomSw = 0
    # sRoiResolution.uBinAverageMode = 0
    # sRoiResolution.uBinSumMode = 0
    # sRoiResolution.uResampleMask = 0
    # sRoiResolution.uSkipMode = 0
    # mvsdk.CameraSetImageResolution(hCamera,sRoiResolution)
    mvsdk.CameraSetImageResolutionEx(hCamera, 0xff, 0, 0, 404, 189, 288, 521, 0, 0)



end_filename=""
load_check()
update_std_char()
# thread = threading.Thread(target=MoveFile)        ##獨立處理
# thread.start()
myWindow = tkinter.Tk()
entry_space =tkinter.StringVar(myWindow)
entry_row_count=tkinter.StringVar(myWindow)
entry_column_count=tkinter.StringVar(myWindow)
entry_AnalogGainX=tkinter.StringVar(myWindow)
entry_ExposureTime=tkinter.StringVar(myWindow)
entry_Gamma=tkinter.StringVar(myWindow)
entry_Contrast=tkinter.StringVar(myWindow)
entry_images_count=tkinter.StringVar(myWindow)
entry_template_backcolor=tkinter.StringVar(myWindow)    #標準邊界底色
entry_tolerance_scope=tkinter.StringVar(myWindow)    #顏色允許誤差范圍
entry_back_scope=tkinter.StringVar(myWindow)    #邊界底色范圍
entry_edge_scope=tkinter.StringVar(myWindow)    #打印邊緣范圍

entry_images_path=tkinter.StringVar(myWindow)
entry_data_path=tkinter.StringVar(myWindow)
entry_rtsp=tkinter.StringVar(myWindow)

entry_jobver=tkinter.StringVar(myWindow)
coordinate_position_x=tkinter.StringVar(myWindow)
coordinate_position_y=tkinter.StringVar(myWindow)
entry_status=tkinter.StringVar(myWindow)


layout_count=tkinter.StringVar(myWindow)
start_seq=tkinter.StringVar(myWindow)
repeat_print=tkinter.StringVar(myWindow)
box_qty=tkinter.StringVar(myWindow)
dbf_file=tkinter.StringVar(myWindow)
identifier=tkinter.StringVar(myWindow)
# remark=tkinter.StringVar(myWindow)
x_offset=tkinter.StringVar(myWindow)
y_offset=tkinter.StringVar(myWindow)
x_width=tkinter.StringVar(myWindow)
y_hight=tkinter.StringVar(myWindow)
# print_type=[("一棟落打印 ", "1"),("順序打印 ", "2")]
# rb1=tkinter.StringVar(myWindow)
# rb1.set("1")

print_type = tkinter.StringVar()
print_type.set('1-一棟落順序打印')
print_types = ['1-一棟落順序打印', '2-一棟落倒序打印', '3-兜圈順序打印']

rotate = tkinter.StringVar()
rotate.set('0-0')
# rotates = ['0-不旋转', '1-90度', '2-180度', '3-270度']
rotates = ['0-0', '1-90', '2-180', '3-270']

type1 = tkinter.StringVar()
type1.set('1-指定識別碼')
type1s = ['1-指定識別碼', '2-不指定識別碼']

#主界面
#logo1 = tkinter.PhotoImage(file="./SourceFile/ima1.png")  # 格式： PGM, PPM, GIF, PNG format
#logo2 = tkinter.PhotoImage(file="./SourceFile/ima2.JPEG")  # 格式： PGM, PPM, GIF, PNG format

frame1=tkinter.Frame(myWindow, bd=5, height=200, width=100)
# label1=tkinter.Label(frame1, text="  工單版本 ", justify="left").grid(row=0, column=0)
label1=tkinter.Button(frame1, text=" 載入工單版本 ", command=lambda: load_jobver(entry_jobver.get())).grid(row=0, column=0)
entry1=tkinter.Entry(frame1, textvariable=entry_jobver, justify="left", width=20).grid(row=0, column=1)
label_ocr6=tkinter.Label(frame1, text="  識別碼", justify="left").grid(row=0, column=2)
entry_ocr7=tkinter.Entry(frame1, text="", textvariable=identifier, justify="left", width=4).grid(row=0, column=3)

label1c1=tkinter.Label(frame1, text="  識別類型", justify="left").grid(row=0, column=4)
entry1x1=ttk.Combobox(frame1, textvariable=type1,values=type1s, justify="left", width=12, state="disabled").grid(row=0, column=5)
label1c=tkinter.Label(frame1, text="  旋转角度", justify="left").grid(row=0, column=6)
entry1x=ttk.Combobox(frame1, textvariable=rotate,values=rotates, justify="left", width=10).grid(row=0, column=7)
label1ca=tkinter.Label(frame1, text="  ", justify="left").grid(row=1, column=0)
frame1.grid(row=0, column=0)

frame2 = tkinter.Frame(myWindow, bd=5, height=200, width=100)
label_ocr1=tkinter.Label(frame2, text="排版方式", justify="left").grid(row=0, column=0)
# cl=1
# for text1, text2 in print_type:
#     rb1a=tkinter.Radiobutton(frame2, variable=rb1, text=text1, value=text2, command="print_type_select").grid(row=0, column=cl)
#     cl+=1
Combobox3=ttk.Combobox(frame2, textvariable=print_type,values=print_types, justify="left", width=15).grid(row=0, column=1)
# state="disabled"

label_ocr2=tkinter.Label(frame2, text="  排版個數", justify="left").grid(row=0, column=2)
entry1_ocr3=tkinter.Entry(frame2, text="", textvariable=layout_count, justify="left", width=5).grid(row=0, column=3)
layout_count.set('1')
label_ocr2a=tkinter.Label(frame2, text="  編號位置", justify="left").grid(row=0, column=4)
entry1_ocr3a=tkinter.Entry(frame2, text="", textvariable=start_seq, justify="left", width=8).grid(row=0, column=5)
start_seq.set('1')
label_ocr2b=tkinter.Label(frame2, text="  重複打印", justify="left").grid(row=0, column=6)
# entry1_ocr3b=tkinter.Entry(frame2, text="", textvariable=repeat_print, justify="left", width=8, state="readonly").grid(row=0, column=7)
entry1_ocr3b=tkinter.Entry(frame2, text="", textvariable=repeat_print, justify="left", width=8).grid(row=0, column=7)
repeat_print.set('1')
# layout.set("1")
# label_ocr4=tkinter.Label(frame2, text="   一箱數量", justify="left").grid(row=0, column=7)
label_ocr4=tkinter.Label(frame2, text="     一箱數量", justify="left").grid(row=0, column=8)
entry_ocr5=tkinter.Entry(frame2, text="", textvariable=box_qty, justify="left", width=6).grid(row=0, column=9)
label0a=tkinter.Label(frame2, text="           ", justify="left").grid(row=1, column=0)
label2a=tkinter.Label(frame2, text=" 水平偏移 ", justify="left").grid(row=2, column=0)
entry2a=tkinter.Entry(frame2, text="", textvariable=x_offset, justify="left", width=8).grid(row=2, column=1)
label2a=tkinter.Label(frame2, text=" 垂直偏移 ", justify="left").grid(row=2, column=2)
entry2a=tkinter.Entry(frame2, text="", textvariable=y_offset, justify="left", width=8).grid(row=2, column=3)
label2a=tkinter.Label(frame2, text=" 寬 ", justify="left").grid(row=2, column=4)
entry2a=tkinter.Entry(frame2, text="", textvariable=x_width, justify="left", width=8).grid(row=2, column=5)
label2a=tkinter.Label(frame2, text=" 高 ", justify="left").grid(row=2, column=6)
entry2a=tkinter.Entry(frame2, text="", textvariable=y_hight, justify="left", width=8).grid(row=2, column=7)

frame2.grid(row=1, column=0)

# frame2B = tkinter.Frame(myWindow, bd=5, height=200, width=100)
# label_ocr4a=tkinter.Label(frame2B, text="    打印Data(dbf檔)", justify="left").grid(row=0, column=0)
# entry_ocr5b=tkinter.Entry(frame2B, text="", textvariable=dbf_file, justify="left", width=70, state="readonly").grid(row=0, column=1)
# label_ocr4a=tkinter.Label(frame2B, text="                  ", justify="left").grid(row=1, column=0)
# frame2B.grid(row=2, column=0)

frame2A = tkinter.Frame(myWindow, bd=5, height=200, width=100)
label2A1=tkinter.Label(frame2A, text="    ", justify="left").grid(row=0, column=1)
label2A1=tkinter.Label(frame2A, text="    ", justify="left").grid(row=0, column=2)
label2A1=tkinter.Label(frame2A, text="    ", justify="left").grid(row=0, column=3)
# label2A1=tkinter.Entry(frame2A, text="   ", textvariable=remark,justify="left", width=38).grid(row=0, column=4)
label2A1=tkinter.Label(frame2A, text="        ", justify="left").grid(row=1, column=1)
button0=tkinter.Button(frame2A, text=" 打開調試程序 ",font=('Arial',28,'bold'), command=lambda: MVDCP()).grid(row=1, column=1)
# button0=tkinter.Button(frame2A, text=" 打開調試程序 ",font=('Arial',28,'bold'), command=lambda: read_sample2(rotate.get())).grid(row=1, column=1)
label2A2=tkinter.Label(frame2A, text="                ", justify="left").grid(row=1, column=2)
# label2A2=tkinter.Label(frame2A, text="                ", justify="left").grid(row=0, column=3)
button1=tkinter.Button(frame2A, text="   取樣板坐標  ",font=('Arial',28,'bold'), command=lambda: read_sample()).grid(row=1, column=4)

label2A2=tkinter.Label(frame2A, text="    ", justify="left").grid(row=1, column=5)

label2A2=tkinter.Label(frame2A, text=" ", justify="left").grid(row=1, column=6)
label2A2=tkinter.Label(frame2A, text="   ", justify="left").grid(row=1, column=7)
label2A2=tkinter.Label(frame2A, text="              ", justify="left").grid(row=2, column=1)
label2A2=tkinter.Label(frame2A, text="              ", justify="left").grid(row=3, column=1)
button2=tkinter.Button(frame2A, text=" 開始編號讀取 ",font=('Arial',28,'bold'), justify="left", command=lambda: readv(entry_jobver.get(),rotate.get(), identifier.get(), print_type.get(),layout_count.get(),box_qty.get())).grid(row=4, column=1)
label2A2=tkinter.Label(frame2A, text=" ", justify="left").grid(row=4, column=2)
# button3=tkinter.Button(frame2A, text=" 運行編號檢測 ",font=('Arial',28,'bold'), command=lambda: qc()).grid(row=3, column=4)
frame2A.grid(row=4, column=0)

# label2a=tkinter.Label(frame1, text="     ", justify="left").grid(row=0, column=8)
# label2b=tkinter.Label(frame1, text="        ", justify="left").grid(row=0, column=9)
# label2c=tkinter.Label(frame1, text="                      ", justify="left").grid(row=0, column=10)
# label_blank1=tkinter.Label(frame1, text=" ", justify="left").grid(row=1, column=0)
#

# label2=tkinter.Label(frame1, text="      ", justify="left").grid(row=2, column=2)
# button2=tkinter.Button(frame1, text="   暫停(按Esc鍵)   ", justify="left").grid(row=2, column=4)
# label2a1=tkinter.Label(frame1, text="OCR完成狀態", justify="left").grid(row=2, column=5)
# entry2c=tkinter.Entry(frame1, textvariable=entry_status, justify="left", width=8).grid(row=2, column=6)
#
# label2a2=tkinter.Label(frame1, text="      ", justify="left").grid(row=3, column=2)
# label2b3=tkinter.Label(frame1, text="      ", justify="left").grid(row=4, column=2)
# label2c4=tkinter.Label(frame1, text="      ", justify="left").grid(row=5, column=2)
# button3=tkinter.Button(frame1, text="   處理數據   ", justify="left",command=lambda: check_data(entry_jobver.get(), identifier.get(), rb1.get(),layout_count.get(),box_qty.get())).grid(row=8, column=1)





# 配置
frame3 = tkinter.Frame(myWindow, bd=5, height=800, width=200)
label_space = tkinter.Label(frame3, text="檔案保存天數 ").grid(row=0, column=0)
entry_space1 = tkinter.Entry(frame3, textvariable=entry_space, width=10).grid(row=0, column=1)
space1=tkinter.Label(frame3, text="      ", justify="left").grid(row=0, column=2)
label_row = tkinter.Label(frame3, text="字符水平最大寬度").grid(row=0, column=3)
entry_row = tkinter.Entry(frame3, textvariable=entry_row_count, width=10).grid(row=0, column=4)
space2=tkinter.Label(frame3, text="      ", justify="left").grid(row=0, column=5)
label_column = tkinter.Label(frame3, text="字符垂直最大高度").grid(row=0, column=6)
entry_column = tkinter.Entry(frame3, textvariable=entry_column_count, width=10).grid(row=0, column=7)
space3=tkinter.Label(frame3, text="      ", justify="left").grid(row=1, column=8)

label_width = tkinter.Label(frame3, text="    模擬增益(位數) ", justify="left").grid(row=1, column=0)
entry_width1 = tkinter.Entry(frame3, textvariable=entry_AnalogGainX, width=10, state="disabled").grid(row=1, column=1)
space4=tkinter.Label(frame3, text="      ", justify="left").grid(row=1, column=2)

label_height = tkinter.Label(frame3, text="    曝光時間(毫秒) ", justify="left").grid(row=1, column=3)
entry_height1 = tkinter.Entry(frame3, textvariable=entry_ExposureTime, width=10, state="disabled").grid(row=1, column=4)

label_width2 = tkinter.Label(frame3, text="    伽馬值 ", justify="left").grid(row=2, column=0)
entry_width2 = tkinter.Entry(frame3, textvariable=entry_Gamma, width=10, state="disabled").grid(row=2, column=1)
space5=tkinter.Label(frame3, text="      ", justify="left").grid(row=2, column=2)
label_height2 = tkinter.Label(frame3, text="    對比度值 ", justify="left").grid(row=2, column=3)
entry_height2 = tkinter.Entry(frame3, textvariable=entry_Contrast, width=10, state="disabled").grid(row=2, column=4)


label_images_count1 = tkinter.Label(frame3, text="一個檔案放多少個圖像檔 ", justify="left").grid(row=3, column=0)
entry_images_count1 = tkinter.Entry(frame3, textvariable=entry_images_count, width=10).grid(row=3, column=1)

frame3b = tkinter.Frame(myWindow, bd=5, height=800, width=200)
label_edge1 = tkinter.Button(frame3b, text="獲取坐標標準邊界底色 ", justify="left", command=lambda: get_pixel(coordinate_position_x.get(), coordinate_position_y.get())).grid(row=3, column=0)
entry_edge2 = tkinter.Entry(frame3b, textvariable=entry_template_backcolor, width=25).grid(row=3, column=1)
label_edge3 = tkinter.Label(frame3b, text="顏色允許誤差范圍 ", justify="left").grid(row=3, column=2)
entry_edge4 = tkinter.Entry(frame3b, textvariable=entry_tolerance_scope, width=25).grid(row=3, column=3)
label_edge5 = tkinter.Label(frame3b, text="邊界底色坐標范圍 ", justify="left").grid(row=4, column=0)
entry_edge6 = tkinter.Entry(frame3b, textvariable=entry_back_scope, width=25).grid(row=4, column=1)
label_edge7 = tkinter.Label(frame3b, text="打印邊界坐標范圍 ", justify="left").grid(row=4, column=2)
entry_edge8 = tkinter.Entry(frame3b, textvariable=entry_edge_scope, width=25).grid(row=4, column=3)

#frame3b.grid(row=2, column=0)       #是否显示出来

frame4 = tkinter.Frame(myWindow, bd=5, height=800, width=200)
#f4space2=tkinter.Label(frame4, text="      ", justify="left").grid(row=3, column=0)
f4space3=tkinter.Label(frame4, text="      ", justify="left").grid(row=4, column=0)
f4space4=tkinter.Label(frame4, text="      ", justify="left").grid(row=5, column=0)
f4space5=tkinter.Label(frame4, text="      ", justify="left").grid(row=6, column=0)
label_images_path = tkinter.Label(frame4, text="備份檔案路徑 ").grid(row=7, column=0)
entry_images_path1 = tkinter.Entry(frame4, textvariable=entry_images_path, width=80).grid(row=7, column=1)

label_data_path = tkinter.Label(frame4, text="輸出圖片路徑 ").grid(row=8, column=0)
entry_data_path1 = tkinter.Entry(frame4, textvariable=entry_data_path, width=80).grid(row=8, column=1)

label_rtsp1 = tkinter.Label(frame4, text="rtsp ").grid(row=9, column=0)
entry_rtsp1 = tkinter.Entry(frame4, textvariable=entry_rtsp, width=80).grid(row=9, column=1)

f4space6=tkinter.Label(frame4, text="      ", justify="left").grid(row=10, column=0)
button4 = tkinter.Button(frame4, text="保存", command=save_config).grid(row=11, column=15)
#菜單
menubar = tkinter.Menu(myWindow)   # #創建橫向菜單條
myWindow.config(menu=menubar)

# 檔菜單
menu1 = tkinter.Menu(menubar, tearoff=False)   # #創建功能表選項
for item in ["讀視頻", "退出"]:
    if item == "退出":
        menu1.add_separator()           # #添加分割線(可選)
        menu1.add_command(label=item, command=myWindow.quit)     # #退出表單程式
    else:
        menu1.add_command(label=item, command=show_main)           # #給功能表加具體的功能表內容
menubar.add_cascade(label="功能", menu=menu1)    # #往功能表條上增加功能表叫【檔】

# 系統菜單
menu2 = tkinter.Menu(menubar, tearoff=False)   # #創建第二個菜單
menu2.add_command(label="配置", command=show_config)
menu2.add_command(label="選項", command="show_main")
menu2.add_command(label="重置配置", command='new_config')
menubar.add_cascade(label="系統", menu=menu2)     # #將第二個子功能表清單綁定到菜單2上

#explanation = "abc"
#logo = tkinter.PhotoImage(file=r"SourceFile/ima.gif")    # 格式： PGM, PPM, GIF, PNG format
'''
Label 用法： 
注意Pack / Place 
pack 只可以放在TOP or BOTTOM or LEFT or RIGHT
Place 可以指定位置
grid 網格
'''

# tkinter.Label(myWindow, image=logo).pack(side='left')
# tkinter.Label(myWindow,compound="center",text=explanation,image=logo).pack(side="right")
# tkinter.Label(myWindow, text="user-name",font=('Arial 12 bold'),width=20,height=3,padx=100, pady=50).pack()
# tkinter.Label(myWindow, text="password2",bg='green',width=20,height=3).pack(ipadx = 50, ipady = 120)
# tkinter.Label(myWindow, text="password2").pack(ipadx = 120, ipady = 120)
# label1=tkinter.Label(myWindow, text="user-name",font=('Arial 12 bold'),justify="left")
# label2=tkinter.Label(myWindow, text="password",font=('Arial 12 bold'),justify="left")
# label3=tkinter.Label(myWindow, text="user",font=('Arial 12 bold'))
# label1.place(x=5,y=5)
# label2.place(x=1,y=40)
# label3.place(x=150,y=5)
# label1.grid(row=1,column=1)
# label2.grid(row=2,column=1)
# logo2=tkinter.BitmapImage(file="SourceFile/ima.xbm")    #格式： XBM format
# tkinter.Label(myWindow, image=logo2).pack(side='left')

"""
# 标签控件布局
tkinter.Label(myWindow, text="定位 左上角:").grid(row=0, column=0)
entry1 = tkinter.Entry(myWindow).grid(row=0, column=1)
tkinter.Label(myWindow, text="右下角:").grid(row=0, column=2)
entry2 = tkinter.Entry(myWindow).grid(row=0, column=3)
tkinter.Button(myWindow, text='取樣板', command=images_process.read_sample).grid(row=0, column=4, padx=15, pady=15)
# Entry控件布局



# Quit按钮退出；Run按钮打印计算结果

tkinter.Button(myWindow, text='開始', command=new_config).grid(row=1, column=0, padx=15, pady=15)
tkinter.Button(myWindow, text='暫停', command=printinfo).grid(row=2, column=1, padx=15, pady=15)
"""

# 设置标题
myWindow.title('Visual Inspection')
# 设置窗口大小
myWindow.geometry('700x400')
# 设置窗口是否可变长、宽，True：可变，False：不可变
myWindow.resizable(width=True, height=True)

# 进入消息循环

myWindow.mainloop()
