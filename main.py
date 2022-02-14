import cv2  # 计算机视觉相关包
import glob  # 文件管理包
import argparse  # 命令行包
import numpy as np
from tqdm import tqdm  # 进度条工具库
from itertools import product  # 迭代器工具



# 图片文件
def parseArgs():
    parser = argparse.ArgumentParser('拼接马赛克图片')
    parser.add_argument("--targetpath", type=str, default='template/3.jpg', help="目标图像路径")  # 插入命令行可选参数，分别为参数名称、类型、缺省值、提示信息
    parser.add_argument("--outputpath", type=str, default='output.jpg', help="输出图像路径")
    parser.add_argument("--sourcepath", type=str, default='sourceimages', help="原图像路径")
    parser.add_argument("--blocksize", type=int, default=20, help="每张图片占像素的大小")
    args = parser.parse_args() # 获取命令行对象
    return args


# 读取所有原图片并计算对应颜色的平均值
def readSourceImages(sourcepath, blocksize):
    print("开始读取图像")
    # 获取合法图像
    sourceimages = []
    # 平均颜色列表
    avgcolors  = []
    for path in tqdm(glob.glob('{}/*.jpg'.format(sourcepath))): # glob.glob匹配文件名
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        '''
        cv2.IMREAD_COLOR：默认参数，读入一副彩色图片，忽略alpha通道，可用1作为实参替代
        cv2.IMREAD_GRAYSCALE：读入灰度图片，可用0作为实参替代
        cv2.IMREAD_UNCHANGED：顾名思义，读入完整图片，包括alpha通道，可用-1作为实参替代
        PS：alpha通道，又称A通道，是一个8位的灰度通道，该通道用256级灰度来记录图像中的透明度复信息，定义透明、不透明和半透明区域，其中黑表示全透明，白表示不透明，灰表示半透明
        '''
        if image.shape[-1] != 3: # 读取image数组最后一维的长度，不为3说明包含Alpha通道，为了图片格式统一，舍去
            continue
        image = cv2.resize(image, (blocksize, blocksize)) # 缩小图片大小
        avgcolor = np.sum(np.sum(image, axis=0), axis=0) / (blocksize * blocksize) # 计算图片平均RGB值
        sourceimages.append(image)
        avgcolors.append(avgcolor)
    print("结束读取图像")
    return sourceimages, np.array(avgcolors)

def main(args):
    targetimage = cv2.imread(args.targetpath)  # 获取目标图像
    outputimage = np.zeros(targetimage.shape, np.uint8)  # 建立一个和目标图像一样大的空图
    sourceimages, avgcolors = readSourceImages(args.sourcepath, args.blocksize) # 获取原图片
    print("开始制作")
    for i, j in tqdm(product(range(int(targetimage.shape[1]/args.blocksize)), range(int(targetimage.shape[0]/args.blocksize)))): # 遍历
        block = targetimage[i * args.blocksize: (i + 1) * args.blocksize, j * args.blocksize : (j + 1) * args.blocksize, : ]
        avgcolor = np.sum(np.sum(block, axis=0), axis=0) / (args.blocksize * args.blocksize) # 目标图像某个区块的平均颜色值 1*3
        distances = np.linalg.norm(avgcolors - avgcolor, axis=1)  # axis=1, 等价于求RGB向量的长度，可以用于代表这个向量的特征值
        idx = np.argmin(distances) # 找到特征值最小的，就是最接近的，替换即可
        outputimage[i*args.blocksize : (i+1) * args.blocksize, j * args.blocksize : (j+1) * args.blocksize, :] = sourceimages[idx]
    cv2.imwrite(args.outputpath, outputimage)
    cv2.imshow('result', outputimage)
    print("制作完成")


main(parseArgs())