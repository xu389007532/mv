"""
Tesseract语言包
https://tesseract-ocr.github.io/tessdoc/Data-Files
"""
import pytesseract
import re
filename=r"D:\xu\Python\mv\err\img0004077.png"
identifier='J'
# filename=r"D:\Xu\python\vi\editor\Test.png"
# text1 = pytesseract.image_to_string(filename)
# text2 = pytesseract.image_to_string(filename,lang="num")
text = pytesseract.image_to_string(filename)
seq3 = '.*(' + identifier + '\d{1,7}[-]{0,1}\d{0,2}).*'
re3 = re.compile(seq3, re.S)  # 不做過濾條件
str1 = re3.findall(text.replace(' ',''))  # 把空格, 橫線去掉

seq4 = '([A-Z,a-z]{0,2}\d{1,7}[-]{0,1}\d{0,2}).*'  # 找前面1~2 個字母, 後面1~7個數字的數據
re4 = re.compile(seq4, re.S)  # 不做過濾條件
str2 = re4.findall(text)  # 指定变量, 用于前面识别码, 更加准确

print("text: ", text)
print("str1: ", str1)
print("str1: ", str2)


def ocr_thread( filename, identifier):
    """
    OCR 處理線程

    :param doctimage_to_stringjobver_id: 工單版本id
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
    seq3 = '.*(' + identifier + '\d{1,7}).*'


    re3 = re.compile(seq3, re.S)  # 不做過濾條件
    # print("OCR 線程開始處理: ", filename, "OCR 編號", filename_OCR)
    global end_filename
    text = pytesseract.image_to_string(filename,lang="Domino", config="--psm 6")
    # text = pytesseract.image_to_string(filename, lang="eng", config="--psm 1")
    # text = pytesseract.image_to_string(filename,lang="W130+eng", config="--psm 1")
    #print(text)
    #str1 = re.findall(identifier + '[a-z,A-Z]{6}', text)  # 指定变量, 用于前面识别码, 更加准确
    # str1 = re.findall(identifier + '[0-9]{5}', text)  # 指定变量, 用于前面识别码, 更加准确
    str1 = re3.findall(text.replace(' ','').replace('-',''))  # 把空格, 橫線去掉
    if len(str1)==0:   # 如果找不到編號, 就用模糊查找.
        seq4 = '([A-Z,a-z]{0,2}\d{1,7}).*'  # 找前面1~2 個字母, 後面1~7個數字的數據
        re4 = re.compile(seq4, re.S)  # 不做過濾條件
        str1 = re4.findall(text)  # 指定变量, 用于前面识别码, 更加准确

    print(str1)

