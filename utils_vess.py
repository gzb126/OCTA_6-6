import cv2
from skimage import morphology
import numpy as np
import math

def boxcount(Z, k):
    S = np.add.reduceat(
        np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                            np.arange(0, Z.shape[1], k), axis=1)
    return len(np.where((S > 0) & (S < k*k))[0])


def fractal_dimension(Z, threshold=0.9):
    # Remove the edges of the image which contain only zeros
    Z = Z[~np.all(Z == 0, axis=1)]
    Z = Z[:, ~np.all(Z == 0, axis=0)]
    # Determine the grid size
    p = min(Z.shape)
    n = 2**np.floor(np.log(p)/np.log(2))
    n = int(np.log(n)/np.log(2))
    sizes = 2**np.arange(n, 1, -1)
    # Count the boxes for each size
    counts = []
    for size in sizes:
        counts.append(boxcount(Z, size))
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]


def find_farthest_points(points):
    max_distance = 0
    farthest_points = None
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            distance = math.sqrt((points[i][0] - points[j][0]) ** 2 + (points[i][1] - points[j][1]) ** 2)
            if distance > max_distance:
                max_distance = distance
                farthest_points = (points[i], points[j])
    return farthest_points


def VessMake(pred, image):
    """
    :param pred:预测到的血管二值图
    :param image:原始图像，三通道的
    :return:绘制好的血管图
    """
    def LineMid(binary):
        binary[binary == 255] = 1
        skeleton0 = morphology.skeletonize(binary, method="lee")  # 细化提取骨架
        skeleton = skeleton0.astype(np.uint8) * 255
        intMat = skeleton0.astype(int)
        return skeleton, intMat

    def getPoints(thinSrc, raudis=4, thresholdMax=6, thresholdMin=4):
        height, width = thinSrc.shape[0], thinSrc.shape[1]
        tmp = thinSrc.copy()
        points = []
        for i in range(height):   # prange
            for j in range(width):   # prange
                if (tmp[i][j]) == 0:
                    continue
                count = 0
                for k in range(i - raudis, i + raudis + 1):
                    for l in range(j - raudis, j + raudis + 1):
                        if k < 0 or l < 0 or k > height - 1 or l > width - 1:
                            continue
                        elif tmp[k][l] == 255:
                            count += 1
                if thresholdMin < count < thresholdMax:
                    point = (j, i)
                    points.append(point)
        return points

    mask = pred.astype(np.uint8)
    Mask = mask.copy()
    contours, hierarchy = cv2.findContours(Mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for j in range(len(contours)):
        area = contours[j]
        if cv2.contourArea(area) < 30:
            cv2.fillPoly(Mask, [area], 0)
    _, Mask = cv2.threshold(Mask, 8, 255, cv2.THRESH_BINARY)
    Limask = Mask.copy()
    skeleton, _ = LineMid(Limask)
    points = getPoints(skeleton, raudis=1, thresholdMin=3, thresholdMax=8)  # 找存在分支的交点
    for p in points:
        cv2.circle(skeleton, p, 1, 0, -1)
    conts1, _ = cv2.findContours(skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pixs, dis = [], []
    for indx in range(len(conts1)):  # 拆解区域分块操作
        Mat2 = skeleton.copy()
        for cov2 in range(len(conts1)):
            if indx != cov2:
                cv2.fillPoly(Mat2, [conts1[cov2]], 0)
        white_pix = np.sum(Mat2 > 200)
        if white_pix < 15:
            continue
        white_pixels = cv2.findNonZero(Mat2)
        plist = [(i[0][0], i[0][1]) for i in white_pixels]
        (x1, y1), (x2, y2) = find_farthest_points(plist)
        distance = max(abs(x1 - x2), abs((y1 - y2)))
        tortuosity = round(white_pix / distance, 2)

        xy = np.where(Mat2 > 200)
        for i in range(len(xy[0])):
            x = xy[0][i]
            y = xy[1][i]
            if tortuosity > 1.1:
                image[x][y] = (0, 0, 255)
            else:
                image[x][y] = (238, 178, 0)
        cv2.putText(image, str(tortuosity), (int((x1 + x2) / 2), int((y1 + y2) / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                    (0, 255, 0), 1)
        pixs.append(white_pix)
        dis.append(distance)
    mean_tortuosity = round(sum(pixs) / sum(dis), 2)
    cv2.putText(image, str(mean_tortuosity), (int(image.shape[0] / 2), int(image.shape[1] / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return image