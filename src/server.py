from flask import Flask, request
from flask_restful import Resource, Api
from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cv2

from chexnet import ChexNet
from unet import Unet
from heatmap import HeatmapGenerator
from constant import IMAGENET_MEAN, IMAGENET_STD, CLASS_NAMES
from utils import blend_segmentation

unet_model = '20190211-101020'
chexnet_model = '20180429-130928'
DISEASES = np.array(CLASS_NAMES)


class CXRApi(Resource):
    unet = Unet(trained=True, model_name=unet_model).cuda()
    chexnet = ChexNet(trained=True, model_name=chexnet_model).cuda()
    heatmap_generator = HeatmapGenerator(chexnet, mode='cam')
    unet.eval();
    chexnet.eval()

    def post(self):
        image_name = request.json.get('image_name')
        print(image_name)
        image = Image.open(f'frontend/public/images/uploads/{image_name}').convert('RGB')

        # run through net
        (t, l, b, r), mask = self.unet.segment(image)
        cropped_image = image.crop((l, t, r, b))
        prob = self.chexnet.predict(cropped_image)

        # save segmentation result
        blended = blend_segmentation(image, mask)
        cv2.rectangle(blended, (l, t), (r, b), (255, 0, 0), 5)
        plt.imsave(f'frontend/public/images/results/segment/{image_name}', blended)

        # save cam result
        w, h = cropped_image.size
        heatmap, _ = self.heatmap_generator.from_prob(prob, w, h)
        p_l, p_t = l, t
        p_r, p_b = 1024-r, 1024-b
        heatmap = np.pad(heatmap, ((p_t, p_b), (p_l, p_r)), mode='linear_ramp', end_values=0)
        heatmap = ((heatmap - heatmap.min()) * (1 / (heatmap.max() - heatmap.min())) * 255).astype(np.uint8)
        cam = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET) * 0.4 + np.array(image)
        cv2.imwrite(f'frontend/public/images/results/cam/{image_name}', cam)

        # top-10 disease
        idx = np.argsort(-prob)
        top_prob = prob[idx[:10]]
        top_prob = map(lambda x: f'{x:.3}', top_prob)
        top_disease = DISEASES[idx[:10]]
        prediction = dict(zip(top_disease, top_prob))

        result = {'result': prediction, 'image_name': image_name}
        print(result)
        return result
if __name__ == '__main__':
    app = Flask(__name__)
    api = Api(app)
    api.add_resource(CXRApi, '/cxr')
    app.run(port=5002, threaded=False, debug=True)
