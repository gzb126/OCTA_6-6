import cv2
import numpy as np
import os
from PIL import Image
from scipy import ndimage
import shutil

class FAZ_Preprocess:
    def __init__(self, image_name, sigma, spacing, tau):
        super(FAZ_Preprocess, self).__init__()
        image = Image.open(image_name).convert("RGB")
        image = np.array(image)
        image = 255 - image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.size = image.shape
        thr = np.percentile(image[(image > 0)], 1) * 0.9
        image[(image <= thr)] = thr
        image = image - np.min(image)
        image = image / np.max(image)

        self.image = image
        self.sigma = sigma
        self.spacing = spacing
        self.tau = tau

    def gaussian(self, image, sigma):
        siz = sigma * 6
        temp = round(siz / self.spacing / 2)
        # processing x-axis
        x = [i for i in range(-temp, temp + 1)]
        x = np.array(x)
        H = np.exp(-(x ** 2 / (2 * ((sigma / self.spacing) ** 2))))
        H = H / np.sum(H)
        Hx = H.reshape(len(H), 1)
        I = ndimage.filters.convolve(image, Hx, mode='nearest')

        # processing y-axis
        temp = round(siz / self.spacing / 2)
        x = [i for i in range(-temp, temp + 1)]
        x = np.array(x)
        H = np.exp(-(x ** 2 / (2 * ((sigma / self.spacing) ** 2))))
        H = H / np.sum(H[:])
        Hy = H.reshape(1, len(H))
        I = ndimage.filters.convolve(I, Hy, mode='nearest')
        return I

    def gradient2(self, F, option):
        k = self.size[0]
        l = self.size[1]
        D = np.zeros(F.shape)
        if option == "x":
            D[0, :] = F[1, :] - F[0, :]
            D[k - 1, :] = F[k - 1, :] - F[k - 2, :]

            # take center differences on interior points
            D[1:k - 2, :] = (F[2:k - 1, :] - F[0:k - 3, :]) / 2
        else:
            D[:, 0] = F[:, 1] - F[:, 0]
            D[:, l - 1] = F[:, l - 1] - F[:, l - 2]
            D[:, 1:l - 2] = (F[:, 2:l - 1] - F[:, 0:l - 3]) / 2
        return D

    def Hessian2d(self, image, sigma):
        image = self.gaussian(image, sigma)
        #     image = ndimage.gaussian_filter(image, sigma, mode = 'nearest')
        Dy = self.gradient2(image, "y")
        Dyy = self.gradient2(Dy, "y")

        Dx = self.gradient2(image, "x")
        Dxx = self.gradient2(Dx, "x")
        Dxy = self.gradient2(Dx, 'y')
        return Dxx, Dyy, Dxy

    def eigvalOfhessian2d(self, Dxx, Dyy, Dxy):
        tmp = np.sqrt((Dxx - Dyy) ** 2 + 4 * (Dxy ** 2))
        # compute eigenvectors of J, v1 and v2
        mu1 = 0.5 * (Dxx + Dyy + tmp)
        mu2 = 0.5 * (Dxx + Dyy - tmp)
        # Sort eigen values by absolute value abs(Lambda1) < abs(Lambda2)
        indices = (np.absolute(mu1) > np.absolute(mu2))
        Lambda1 = mu1
        Lambda1[indices] = mu2[indices]

        Lambda2 = mu2
        Lambda2[indices] = mu1[indices]
        return Lambda1, Lambda2

    def imageEigenvalues(self, I, sigma):
        hxx, hyy, hxy = self.Hessian2d(I, sigma)
        # hxx, hyy, hxy = self.Hessian2d(I, sigma)
        c = sigma ** 2
        hxx = -c * hxx
        # hxx = hxx.flatten()
        hyy = -c * hyy
        # hyy = hyy.flatten()
        hxy = -c * hxy
        # hxy = hxy.flatten()

        # # reduce computation by computing vesselness only where needed
        B1 = -(hxx + hyy)
        B2 = hxx * hyy - hxy ** 2
        T = np.ones(B1.shape)
        T[(B1 < 0)] = 0
        T[(B1 == 0) & (B2 == 0)] = 0
        T = T.flatten()
        indeces = np.where(T == 1)[0]
        hxx = hxx.flatten()
        hyy = hyy.flatten()
        hxy = hxy.flatten()
        hxx = hxx[indeces]
        hyy = hyy[indeces]
        hxy = hxy[indeces]
        #     lambda1i, lambda2i = hessian_matrix_eigvals([hxx, hyy, hxy])
        lambda1i, lambda2i = self.eigvalOfhessian2d(hxx, hyy, hxy)
        lambda1 = np.zeros(self.size[0] * self.size[1], )
        lambda2 = np.zeros(self.size[0] * self.size[1], )

        lambda1[indeces] = lambda1i
        lambda2[indeces] = lambda2i

        # removing noise
        lambda1[(np.isinf(lambda1))] = 0
        lambda2[(np.isinf(lambda2))] = 0

        lambda1[(np.absolute(lambda1) < 1e-4)] = 0
        lambda1 = lambda1.reshape(self.size)

        lambda2[(np.absolute(lambda2) < 1e-4)] = 0
        lambda2 = lambda2.reshape(self.size)
        return lambda1, lambda2

    def vesselness2d(self):
        for j in range(len(self.sigma)):
            lambda1, lambda2 = self.imageEigenvalues(self.image, self.sigma[j])
            lambda3 = lambda2.copy()
            new_tau = self.tau * np.min(lambda3)
            lambda3[(lambda3 < 0) & (lambda3 >= new_tau)] = new_tau
            different = lambda3 - lambda2
            response = ((np.absolute(lambda2) ** 2) * np.absolute(different)) * 27 / (
                        (2 * np.absolute(lambda2) + np.absolute(different)) ** 3)
            response[(lambda2 < lambda3 / 2)] = 1
            response[(lambda2 >= 0)] = 0

            response[np.where(np.isinf(response))[0]] = 0
            if j == 0:
                vesselness = response
            else:
                vesselness = np.maximum(vesselness, response)
        #     vesselness = vesselness / np.max(vesselness)
        vesselness[(vesselness < 1e-2)] = 0
        #         vesselness = vesselness.reshape(self.size)
        return vesselness


def hessian_matrix(gray):
    dx=cv2.Sobel(gray,cv2.CV_64F, 1, 0, ksize=3)
    dy =cv2.Sobel(gray,cv2.CV_64F, 0, 1,ksize=3)
    dxx=cv2.Sobel(dx,cv2.CV_64F, 1, 0, ksize=3)
    dxy=cv2.Sobel(dx, cv2.CV_64F, 0, 1, ksize=3)
    dyy =cv2.Sobel(dy, cv2.CV_64F, 0, 1,ksize=3)
    H = np.array([[dxx.mean(),dxy.mean()],[dxy.mean(),dyy.mean()]])
    eigval, eigvec = np.linalg.eig(H)
    return eigval,eigvec


def enhance_image(gray):
    eigval,eigvec =hessian_matrix(gray)
    lambda1 = eigval[0]
    lambda2 = eigval[1]
    alpha =0.5
    beta =1
    enhanced_img = np.zeros_like(gray)
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            I= gray[i, j]
            J= alpha * np.exp(-lambda1 * lambda1 /(2 * beta * beta))* eigvec[0, 0] + np.exp(-lambda2 *lambda2/(2*beta* beta))* eigvec[0,1]
            enhanced_img[i,j]=I+J
    return enhanced_img

if __name__ == '__main__':
    root = r'F:\3_Data\ImageSegmentation\OCTA\test\66'
    # rootmask = r'F:\3_Data\ImageSegmentation\OCTA\OtherDatasets\peocessed_FAZID\octamask'
    # outjpg = r'F:\3_Data\ImageSegmentation\OCTA\train\img/'
    # outpng = r'F:\3_Data\ImageSegmentation\OCTA\train\gt/'
    # num = len(outjpg)
    imgs = os.listdir(root)
    for img in imgs:
        path = os.path.join(root, img)
        cv_img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), 1)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        # enhance_img = enhance_image(gray)
        # show = cv2.hconcat([gray, enhance_img])
        # cv2.imencode('.jpg', show)[1].tofile(r'./imgfiles\out/' + img)

        image = FAZ_Preprocess(path, [0.5, 1, 1.5, 2, 2.5], 1, 2)
        image = image.vesselness2d()
        image = image*255
        image = image.astype(np.uint8)
        show = cv2.hconcat([gray, image])
        cv2.imencode('.jpg', show)[1].tofile(r'F:\1_PycharmProjects\FAZSEG-main\output/' + img)
        # shutil.copy(os.path.join(rootmask, img), os.path.join(outpng, str(num + 1) + '.png'))
        # num += 1


