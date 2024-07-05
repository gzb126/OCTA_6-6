import os
import shutil
import csv
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from FRNet import FRNetModelLoad
from utils_param import *
from utils_vess import *


net = OCTANetwork('./models/Se_resnext50-920eef84.pth', 'Se_resnext50')
init = FRNetModelLoad('FRNet', 'OCTA500_6M')


def CsvWrite(fil, count):
    f = open(fil, 'a', newline='')
    writer = csv.writer(f)
    writer.writerow(count)
    f.close()

Y = []

def ProcessMake(img_path, type=6*6):
    """
    :param img_path: 输入单张图像路径
    :return: 返回完整的展示图像
    """
    VessPred, gray = init.forword(img_path)
    FAZPred =  Forward_FAZ_Make(net, img_path, type)

    def makeParam(img_path, image, FAZPred):
        C, U1, R1, D1, L1, U2, R2, D2, L2 = makeCircle(img_path)
        imgHS = ImageProcess(img_path)
        VD_C, PD_C, VDI_C, P_C = ComputeCenterParam(imgHS, FAZPred, C)
        VD_U1, PD_U1, VDI_U1, P_U1 = ComputeAroundParam(imgHS, U1)
        VD_R1, PD_R1, VDI_R1, P_R1 = ComputeAroundParam(imgHS, R1)
        VD_D1, PD_D1, VDI_D1, P_D1 = ComputeAroundParam(imgHS, D1)
        VD_L1, PD_L1, VDI_L1, P_L1 = ComputeAroundParam(imgHS, L1)
        VD_U2, PD_U2, VDI_U2, P_U2 = ComputeAroundParam(imgHS, U2)
        VD_R2, PD_R2, VDI_R2, P_R2 = ComputeAroundParam(imgHS, R2)
        VD_D2, PD_D2, VDI_D2, P_D2 = ComputeAroundParam(imgHS, D2)
        VD_L2, PD_L2, VDI_L2, P_L2 = ComputeAroundParam(imgHS, L2)

        itms = (VD_C, PD_C, VDI_C,
                VD_U1, PD_U1, VDI_U1,
                VD_R1, PD_R1, VDI_R1,
                VD_D1, PD_D1, VDI_D1,
                VD_L1, PD_L1, VDI_L1,
                VD_U2, PD_U2, VDI_U2,
                VD_R2, PD_R2, VDI_R2,
                VD_D2, PD_D2, VDI_D2,
                VD_L2, PD_L2, VDI_L2)
        for itm in itms:
                Y.append(itm)
        A, B, C = [], [], []
        for i in (VD_C, VD_U1, VD_R1, VD_D1, VD_L1, VD_U2, VD_R2, VD_D2, VD_L2):
            A.append(i)
        for j in (PD_C, PD_U1, PD_R1, PD_D1, PD_L1, PD_U2, PD_R2, PD_D2, PD_L2):
            B.append(j)
        for k in (VDI_C, VDI_U1, VDI_R1, VDI_D1, VDI_L1, VDI_U2, VDI_R2, VDI_D2, VDI_L2):
            C.append(k)
        VD_mean = round(sum(A)/9, 3)
        PD_mean = round(sum(B)/9, 3)
        VDI_mean = round(sum(C)/9, 3)

        Y.append(VD_mean)
        Y.append(PD_mean)
        Y.append(VDI_mean)
        drLine = DrawImgLine(image)
        cv2.putText(drLine, str(VD_C) + ' ' + str(PD_C) + ' ' + str(VDI_C), (P_C[0] - 70, P_C[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        cv2.putText(drLine, str(VD_U1) + ' ' + str(PD_U1) + ' ' + str(VDI_U1), (P_U1[0] - 100, P_U1[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        cv2.putText(drLine, str(VD_R1) + ' ' + str(PD_R1) + ' ' + str(VDI_R1), (P_R1[0] - 100, P_R1[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        cv2.putText(drLine, str(VD_D1) + ' ' + str(PD_D1) + ' ' + str(VDI_D1), (P_D1[0] - 100, P_D1[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        cv2.putText(drLine, str(VD_L1) + ' ' + str(PD_L1) + ' ' + str(VDI_L1), (P_L1[0] - 100, P_L1[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        cv2.putText(drLine, str(VD_U2) + ' ' + str(PD_U2) + ' ' + str(VDI_U2), (P_U2[0] - 100, P_U2[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        cv2.putText(drLine, str(VD_R2) + ' ' + str(PD_R2) + ' ' + str(VDI_R2), (P_R2[0] - 100, P_R2[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        cv2.putText(drLine, str(VD_D2) + ' ' + str(PD_D2) + ' ' + str(VDI_D2), (P_D2[0] - 100, P_D2[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        cv2.putText(drLine, str(VD_L2) + ' ' + str(PD_L2) + ' ' + str(VDI_L2), (P_L2[0] - 100, P_L2[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        return drLine

    def makeVess(VessPred, image):
        circle = ChooseCircleCenter(gray, min(image.shape[0], image.shape[1] / 2))
        ves_drw = VessMake(VessPred * (circle / 255), image)
        return ves_drw

    def makeFAZ(Mask, im):
        binary = Mask.astype(np.uint8)
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        areas = []
        for c in range(len(contours)):
            areas.append(cv2.contourArea(contours[c]))
        max_id = areas.index(max(areas))
        zeros = np.zeros((im.shape), dtype=np.uint8)
        points = np.array([contours[max_id]], dtype=np.int32)
        ma = cv2.fillPoly(zeros, points, color=(80, 127, 255))
        mask_img = 0.4 * ma + im

        area = cv2.contourArea(contours[max_id])
        area_mm = round(6 * 6 * area / (gray.shape[0] * gray.shape[1]), 2)
        perimeter = cv2.arcLength(contours[max_id], True)  # 轮廓周长 (perimeter)
        circular = round(4 * np.pi * area / (perimeter ** 2), 2)  # 轮廓的圆度 (circularity)
        per = round(6 * perimeter / gray.shape[1], 2)
        Y.append(area_mm)
        Y.append(per)
        Y.append(circular)
        cv2.putText(mask_img, str(area_mm) + 'mm2' + ' ' + str(per) + 'mm ' + str(circular),
                    (gray.shape[0] // 2 - 100, gray.shape[1] // 2 - 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        return mask_img

    image = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
    drw_param = makeParam(img_path, image, FAZPred)
    drw_vess = makeVess(VessPred, drw_param)
    drw_faz = makeFAZ(FAZPred, drw_vess)
    return drw_faz


def Progress_3_6(img3, img6, out):
    def FAZmake(Mask, im):
        binary = Mask.astype(np.uint8)
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        areas = []
        for c in range(len(contours)):
            areas.append(cv2.contourArea(contours[c]))
        max_id = areas.index(max(areas))
        zeros = np.zeros((im.shape), dtype=np.uint8)
        points = np.array([contours[max_id]], dtype=np.int32)
        ma = cv2.fillPoly(zeros, points, color=(80, 127, 255))
        mask_img = 0.4 * ma + im
        return mask_img
    FAZ6 = Forward_FAZ_Make(net, img6, 6*6)
    FAZ3 = Forward_FAZ_Make(net, img3, 3*3)
    image3 = cv2.imdecode(np.fromfile(img3, dtype=np.uint8), -1)
    image6 = cv2.imdecode(np.fromfile(img6, dtype=np.uint8), -1)
    im3 = FAZmake(FAZ3, image3)
    im6 = FAZmake(FAZ6, image6)
    H, W = FAZ6.shape
    FAZ_3 = cv2.copyMakeBorder(cv2.resize(FAZ3, (W//2, H//2)), H // 4, H // 4, W // 4, W // 4, cv2.BORDER_REPLICATE)
    im6_ = FAZmake(FAZ_3, image6)

    cv2.imencode('.jpg', im3)[1].tofile(r'C:\Users\GIGABYTE\Downloads\33_66_out/' + out + '_3.png')
    cv2.imencode('.jpg', im6)[1].tofile(r'C:\Users\GIGABYTE\Downloads\33_66_out/' + out + '_6.png')
    cv2.imencode('.jpg', im6_)[1].tofile(r'C:\Users\GIGABYTE\Downloads\33_66_out/' + out + '_drw.png')


if __name__ == '__main__':

    # head = ['path',
    #         'VD_C', 'PD_C', 'VDI_C',
    #         'VD_U1', 'PD_U1', 'VDI_U1',
    #         'VD_R1', 'PD_R1', 'VDI_R1',
    #         'VD_D1', 'PD_D1', 'VDI_D1',
    #         'VD_L1', 'PD_L1', 'VDI_L1',
    #         'VD_U2', 'PD_U2', 'VDI_U2',
    #         'VD_R2', 'PD_R2', 'VDI_R2',
    #         'VD_D2', 'PD_D2', 'VDI_D2',
    #         'VD_L2', 'PD_L2', 'VDI_L2',
    #         'VD_mean', 'PD_mean', 'VDI_mean',
    #         'FAZ_mm2','per_mm', 'circular'
    #         ]
    # CsvWrite(r'C:\Users\GIGABYTE\Downloads\073/' 'Merge.csv', head)
    #
    #
    # root = r'C:\Users\GIGABYTE\Downloads\2024.5.30_OCTA图'
    # for root, dirs, files in os.walk(root, topdown=True):
    #     if len(files) > 0:
    #         for img in files:
    #             if '血管造影术_表面' not in img:
    #                 continue
    #             imgpath = os.path.join(root, img)
    #             Y.append(imgpath)
    #             drw = ProcessMake(imgpath)
    #             cv2.imencode('.jpg', drw)[1].tofile(r'C:\Users\GIGABYTE\Downloads\073/' + img)
    #             CsvWrite(r'C:\Users\GIGABYTE\Downloads\073/' 'Merge.csv', Y)
    #             Y.clear()

    root = r'F:\2_PycharmWorks\OCTAReadProject\imgfiles\66/'
    for i, im in enumerate(sorted(os.listdir(root))):
        img_path = os.path.join(root, im)
        VessPred, gray = init.forword(img_path)
        FAZPred = Forward_FAZ_Make(net, img_path, 6*6)
        cv2.imencode('.jpg', VessPred)[1].tofile(r'C:\Users\GIGABYTE\Desktop\pic/' + im.replace('.', '_ves.'))
        cv2.imencode('.jpg', FAZPred)[1].tofile(r'C:\Users\GIGABYTE\Desktop\pic/' + im.replace('.', '_faz.'))
