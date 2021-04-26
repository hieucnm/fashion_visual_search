import warnings
warnings.filterwarnings('ignore')

import cv2
import os.path as osp
import pandas as pd

from retrieval.utils import restrict_bbox

# ====================================================================================================


class DemoDetector(object):
    
    def __init__(self, detectron, classes, colors, position_dict):
        self.detectron = detectron
        self.classes = classes
        self.colors = colors
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.position_dict = position_dict
    
    
    def get_color(self, idx=None):
        if idx is None:
            idx = 0
        color = tuple(c*255 for c in self.colors[int(idx)])
        color = (.7*color[2],.7*color[1],.7*color[0])
        return color
    
    
    def draw_bbox(self, detector_output, img):
        x1, y1, x2, y2, cls_conf, cls_pred = detector_output
        x1,x2,y1,y2 = restrict_bbox(x1, x2, y1, y2, max_x=img.shape[1], max_y=img.shape[0])
        cv2.rectangle(img,(x1,y1) , (x2,y2) , self.get_color(), 3)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    
    
    def cut_bbox(self, detector_output, img):
        x1, y1, x2, y2, _, _ = detector_output
        x1,x2,y1,y2 = restrict_bbox(x1, x2, y1, y2, max_x=img.shape[1], max_y=img.shape[0])
        bbox = img[y1:y2,x1:x2,:]
        bbox = cv2.cvtColor(bbox, cv2.COLOR_BGR2RGB)
        return bbox
    
    
    def filter_result(self, detections):
        detections = pd.DataFrame(detections, columns=['x1', 'y1', 'x2', 'y2', 'cls_conf', 'cls_pred'])
        detections.cls_pred = detections.cls_pred.apply(lambda x: str(self.classes[int(x)]).replace(' ', '-'))
        detections.cls_pred = detections.cls_pred.apply(lambda x: self.position_dict[x])
        detections = detections.sort_values(by='cls_conf', ascending=False)
        detections = detections.drop_duplicates('cls_pred')
        return detections.values.tolist()
    
    
    def run(self, path):
        
        assert osp.exists(path), f'Path {path} not found!'

        img = cv2.imread(path)
        detections = self.detectron.get_detections(img)

        res = []
        if len(detections) != 0:
            detections = self.filter_result(detections)
            for detector_output in detections:
                classname  = detector_output[-1]
                render_img = self.draw_bbox(detector_output, img.copy())
                bbox = self.cut_bbox(detector_output, img)
                res.append((classname, render_img, bbox))
        return res
