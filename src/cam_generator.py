import importlib
from constant import *
import torchvision.transforms as transforms
from PIL import Image
import cv2
import torch
from torch.autograd import Variable
import numpy as np

class CAMGenerator(object):
    
    def __init__(self, architecture='models.densenet', variant='densenet121', model_name='20180429-130928'):
        
        # init model
        network = importlib.import_module(architecture)
        print('Network')
        self.model = network.build(variant)
        model_file = '%s/%s/%s/%s/model.path.tar' % (MODEL_DIR, network.architect, variant, model_name)
        checkpoint = torch.load(model_file)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        self.weights = list(self.model.classifier.parameters())[0].cpu().data.numpy()
        
        # init transforms, use 10 crop? No
        normalize = transforms.Normalize(self.model.mean, self.model.std)
        tfs = []
        tfs.append(transforms.Resize(self.model.input_size))
        tfs.append(transforms.ToTensor())
        tfs.append(normalize)   
        self.transform = transforms.Compose(tfs)
        
        
    def cam(self, image_file):
        # return original image, cam heatmap and blended image
        
        # load image
        image_file = '/home/dattran/data/xray/%s' % image_file
        pil_image = Image.open(image_file).convert('RGB') 
        cv_image = cv2.imread(image_file)
        pil_image = self.transform(pil_image)
        pil_image = pil_image.unsqueeze_(0)
        input_ = Variable(pil_image).cuda()
        
        # extract image feature + label
        feature = self.model.extract(input_).cpu().data.numpy()
        # bz, nc, h, w = feature.shape
        # feature = feature.reshape((nc, h*w))
        probs = self.model(input_, pooling='AVG')
        probs = probs.cpu().data.numpy().squeeze()
        disease_ids = np.where(probs > 0.1)[0]
        diseases = np.array(CLASS_NAMES)[disease_ids]
        

        # no disease
        if len(disease_ids) == 0:
            return cv_image, 'NORMAL'

        # If disease, make CAM for each disease

        blendeds = []
        for disease_id in disease_ids:
            am = CAMGenerator._activation_map(feature, self.weights, disease_id)
            h, w, _ = cv_image.shape
            heatmap = cv2.applyColorMap(cv2.resize(am, (w, h)), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
            blended = heatmap * 0.3 + cv_image * 0.5
            blended = CAMGenerator._normalize_image(blended)
            blendeds.append(blended)
        
        results = zip(blendeds, diseases, probs[disease_ids])
        return cv_image, results

        
    @staticmethod
    def _normalize_image(img):
        # Normalised [0,1]
        img = img - np.min(img)
        normalized = img/np.ptp(img)
        return normalized
    
    
    @staticmethod
    def _activation_map(feature_conv, weight_softmax, disease_index):
        # generate the class activation maps upsample to 256x256
        size_upsample = (256, 256)
        bz, nc, h, w = feature_conv.shape # 1x1024xhxw

        # cam = weight_softmax[disease_index].dot(feature_conv.reshape((nc, h*w)))
        # cam = cam.reshape(h, w)
        # cam = cam - np.min(cam)
        # cam_img = cam / np.max(cam)
        # cam_img = np.uint8(255 * cam_img)
        # cam_img = cv2.resize(cam_img, size_upsample)
        
        am = weight_softmax[disease_index].dot(feature_conv.reshape((nc, h*w)))
        am = am.reshape(h, w)
        am = am - np.min(am)
        am = am / np.max(am)
        am = np.uint8(255 * am)
        am = cv2.resize(am, size_upsample)
        return am
        
        
        
        
        