import cv2
import numpy as np
import os
from PIL import Image
from scipy import ndimage
import shutil
from skimage import morphology
from hessian import FAZ_Preprocess
from OCTNet import OCTANetwork



def ChooseCircleCenter(gray, r):
    h, w = gray.shape
    sp = (w // 2, h // 2)
    Mask = np.zeros((h, w), np.uint8) * 255.0
    cv2.circle(Mask, sp, int(r), (255, 255, 255), -1)  # 内圆   min(h, w) // 12
    return Mask


def ChooseCircleSecond(gray, angbegin, angend):
    """
    :param gray: 灰度图像
    :param angbegin: 起始角度
    :param angend: 终止角度
    :return: 区域二值图
    """
    h, w = gray.shape
    sp = (w // 2, h // 2)
    Mask = np.zeros((h, w), np.uint8) * 255.0
    mask = np.zeros((h, w), np.uint8) * 255.0
    # 参数 1.目标图片  2.椭圆圆心  3.长短轴长度  4.偏转角度  5.圆弧起始角度  6.终止角度  7.颜色  8.是否填充
    cv2.ellipse(Mask, sp, (int(min(h, w) // 4), int(min(h, w) // 4)), 0, angbegin, angend, (255, 255, 255), -1)
    cv2.circle(mask, sp, int(min(h, w) // 12), (255, 255, 255), -1)  # 内圆
    return Mask - mask


def ChooseCircleThird(gray, angbegin, angend):
    h, w = gray.shape
    sp = (w // 2, h // 2)
    Mask = np.zeros((h, w), np.uint8) * 255.0
    mask = np.zeros((h, w), np.uint8) * 255.0
    # 参数 1.目标图片  2.椭圆圆心  3.长短轴长度  4.偏转角度  5.圆弧起始角度  6.终止角度  7.颜色  8.是否填充
    cv2.ellipse(Mask, sp, (int(min(h, w) // 2), int(min(h, w) // 2)), 0, angbegin, angend, (255, 255, 255), -1)
    cv2.circle(mask, sp, int(min(h, w) // 4), (255, 255, 255), -1)  # 内圆
    return Mask - mask


def Forward_FAZ_Make(net, img_path, typ):
    """
    :param net: 分割FAZ区域的模型网络
    :param img_path: 单张图像路径
    :return: 输入图像的FAZ分割二值图
    """
    gray = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    H, W = gray.shape
    pred = net.forword(img_path, typ)
    if typ == 6*6:
        org_pred = cv2.resize(pred, (W//2, H//2), interpolation=cv2.INTER_NEAREST)
        FAZ_Mat = cv2.copyMakeBorder(org_pred, H//4, H//4, W//4, W//4, cv2.BORDER_REPLICATE)      # 根据图像的边界的像素值，向外扩充图片，每个方向扩充还原为原图大小
    if typ == 3*3:
        FAZ_Mat = org_pred = cv2.resize(pred, (W, H), interpolation=cv2.INTER_NEAREST)
    return FAZ_Mat


def makeCircle(img_path):
    """
    :param img_path: 输入图片的路径
    :return: 返回分割好的各个区域，公9个区域
    """
    gray = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    C = ChooseCircleCenter(gray, min(gray.shape[0], gray.shape[1]) // 12)
    U1 = ChooseCircleSecond(gray, 225, 315)
    R1 = ChooseCircleSecond(gray, 0, 45) + ChooseCircleSecond(gray, 315, 360)
    D1 = ChooseCircleSecond(gray, 45, 135)
    L1 = ChooseCircleSecond(gray, 135, 225)
    U2 = ChooseCircleThird(gray, 225, 315)
    R2 = ChooseCircleThird(gray, 0, 45) + ChooseCircleThird(gray, 315, 360)
    D2 = ChooseCircleThird(gray, 45, 135)
    L2 = ChooseCircleThird(gray, 135, 225)
    return C, U1, R1, D1, L1, U2, R2, D2, L2


def ComputeAroundParam(gray, area):
    """
    :param gray: 灰度图
    :param area: 计算区域
    :return: 该区域的参数
    """
    area[area > 0] = 255
    area[area < 0] = 0
    contours, _ = cv2.findContours(area.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    M = cv2.moments(contours[0])
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    area = area / 255
    ret, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    whit_area = cv2.countNonZero(area)
    choseMat = binary * area
    size_elements = cv2.countNonZero(choseMat)
    conts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for j in range(len(conts)):  # 拆除交点后进行预支判定，小于阈值的就不要
        are = cv2.contourArea(conts[j])
        if are < 10:
            cv2.fillPoly(binary, [conts[j]], 0)
    skeleton0 = morphology.skeletonize(binary, method="lee")  # 细化提取骨架
    intMat = skeleton0.astype(np.uint8)
    chosearea = intMat * area
    Length = cv2.countNonZero(chosearea)
    VD = round((6 * Length / gray.shape[1]) / (6 * 6 * whit_area / (gray.shape[0] * gray.shape[1])), 2)  # 血管长度/区域面积
    PD = round(size_elements / whit_area, 2)
    VDI = round((6 * 6 * size_elements / (gray.shape[0] * gray.shape[1])) / (6 * Length / gray.shape[1]), 3)
    return VD, PD, VDI, (cx, cy)


def ComputeCenterParam(gray, Mask, area):
    """
    :param gray: 灰度图
    :param Mask: FAZ分割图
    :param area: 中心区域的二值图
    :return: 该区域的参数
    """
    area[area>0] = 255
    area[area<0] = 0
    contours, _ = cv2.findContours(area.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    M = cv2.moments(contours[0])
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    area = area/255
    ret, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    whit_area = cv2.countNonZero(area)
    choseMat = binary * area
    choseMat = choseMat.astype(np.uint8)
    choseMat = choseMat - cv2.bitwise_and(Mask, choseMat)

    size_elements = cv2.countNonZero(choseMat)
    bina = binary.copy()
    intersection = cv2.bitwise_and(bina, Mask)
    bina = bina - intersection
    conts, _ = cv2.findContours(bina, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for j in range(len(conts)):  # 拆除交点后进行预支判定，小于阈值的就不要
        are = cv2.contourArea(conts[j])
        if are < 10:
            cv2.fillPoly(bina, [conts[j]], 0)
    skeleton0 = morphology.skeletonize(bina, method="lee")  # 细化提取骨架
    intMat = skeleton0.astype(np.uint8)
    chosearea = intMat * area
    Length = cv2.countNonZero(chosearea)
    VD = round((6 * Length/gray.shape[1]) / (6 * 6 * whit_area/(gray.shape[0]*gray.shape[1])), 2)  # 血管长度/区域面积
    PD = round(size_elements / whit_area, 2)
    VDI = round((6 * 6 * size_elements/(gray.shape[0]*gray.shape[1])) / (6 * Length/gray.shape[1]), 3)
    return VD, PD, VDI, (cx, cy)


def ImageProcess(path):
    """
    :param path: 输入原始图像路径
    :return: 返回经过海森矩阵处理过的图像，突出血管
    """
    image = FAZ_Preprocess(path, [0.5, 1, 1.5, 2, 2.5], 0.5, 0.5)
    image = image.vesselness2d()
    image = image * 255
    image = image.astype(np.uint8)
    return image


def DrawImgLine(cv_img):
    h, w, c = cv_img.shape
    cv2.circle(cv_img, (w // 2, h // 2), int(min(w // 2, h // 2)), (255, 255, 255), 2)  # 外圆
    cv2.circle(cv_img, (w // 2, h // 2), int(min(w // 4, h // 4)), (255, 255, 255), 2)  # 中圆
    cv2.circle(cv_img, (w // 2, h // 2), int(min(w // 12, h // 12)), (255, 255, 255), 2)  # 内圆

    P1 = (int((h-1.414*0.5*h)/2), int((h-1.414*0.5*h)/2))
    p1 = (int(h/2 - h/(12*1.414)), int(h/2 - h/(12*1.414)))
    cv2.line(cv_img, P1, p1, (255,255,255), 2)

    P2 = (h - int((h - 1.414 * 0.5 * h)/2), int((h-1.414*0.5*h)/2))
    p2 = (int(h/2 + h/(12 * 1.414)), int(h/2 - h/(12*1.414)))
    cv2.line(cv_img, P2, p2, (255, 255, 255), 2)

    P3 = (h - int((h - 1.414 * 0.5 * h)/2), h -int((h - 1.414 * 0.5 * h)/2))
    p3 = (int(h/2 + h/(12 * 1.414)), int(h/2 + h/(12 * 1.414)))
    cv2.line(cv_img, P3, p3, (255, 255, 255), 2)

    P4 = (int((h-1.414*0.5*h)/2), h - int((h - 1.414 * 0.5 * h) / 2))
    p4 = (int(h/2 - h/(12*1.414)), int(h / 2 + h / (12 * 1.414)))
    cv2.line(cv_img, P4, p4, (255, 255, 255), 2)

    return cv_img