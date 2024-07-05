from FRNetFile.dataset import *
from FRNetFile.utils import *
from FRNetFile.settings_benchmark import *

class FRNetModelLoad(object):
    def __init__(self, net, datname):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        root_model = os.path.join('FRNetModel', net, datname, 'model_best.pth')
        model: nn.Module = models[net]().to(self.device)
        model.load_state_dict(torch.load(root_model))
        model.eval()
        self.model = model

    def dataLoad(self, img_path):
        image = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE).astype("float32")
        H, W = image.shape
        image = image[0:H, 5:W]
        image /= 255
        # 填充到32的倍数
        pad_x = (image.shape[1] // 32 + 1) * 32 - image.shape[1]
        pad_x %= 32
        pad_y = (image.shape[0] // 32 + 1) * 32 - image.shape[0]
        pad_y %= 32
        im = cv2.copyMakeBorder(image, pad_y // 2, pad_y // 2, pad_x // 2, pad_x // 2, cv2.BORDER_CONSTANT, value=0)
        image = im.reshape((1, im.shape[0], im.shape[1]))
        return image, im

    def forword(self, imgpath):
        dat, image = self.dataLoad(imgpath)
        testLoader = DataLoader(dataset=dat)
        data = [i for i in testLoader][0]
        data = torch.unsqueeze(data, 0)
        data = data.to(self.device)
        with torch.no_grad():
            out = self.model(data)
        pred = out[0][0].detach().cpu().numpy()*255
        return pred, image