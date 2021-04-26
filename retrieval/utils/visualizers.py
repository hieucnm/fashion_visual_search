
import cv2
import numpy as np
import matplotlib.pyplot as plt
from . import restrict_bbox

class BoundingboxVisualizer(object):
    
    def __init__(self, n_class, scaled=True):
        assert n_class > 0, 'n_class must greater than zero'
        self.n_class = int(n_class)
        self.scaled = scaled
        cmap = plt.get_cmap("rainbow")
        self.colors = np.array([cmap(i) for i in np.linspace(0, 1, n_class)])

        
    def get_color(self, idx=None):
        if idx is None:
            idx = 0
        color = tuple(c*255 for c in self.colors[int(idx)])
        color = (.7*color[2],.7*color[1],.7*color[0])
        return color

    def draw_bbox(self, img, x1,x2,y1,y2, title='', idx=None):
        x1,x2,y1,y2 = restrict_bbox(x1, x2, y1, y2, max_x=img.shape[1], max_y=img.shape[0])
        color = self.get_color(idx)
        cv2.rectangle(img,(x1,y1) , (x2,y2) , color, 3)

        y1 = 0 if y1<0 else y1
        y1_rect = y1-25
        y1_text = y1-5

        if y1_rect<0:
            y1_rect = y1+27
            y1_text = y1+20
        cv2.rectangle(img,(x1-2,y1_rect) , (x1 + int(8.5*len(title)),y1) , color,-1)
        cv2.putText(img,title,(x1,y1_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),1,cv2.LINE_AA)
        return img    

    def visualize(self, inp, x1_col='x1', x2_col='x2', y1_col='y1', y2_col='y2', class_col='', figsize=(10,10)):
        inp = inp.reset_index(drop=True)
        img = cv2.imread(inp.iloc[0].path)
        for i,x in inp.iterrows():
            x1, x2, y1, y2 = x[x1_col], x[x2_col], x[y1_col], x[y2_col]
            if self.scaled:
                x1 = x1 * img.shape[1]
                x2 = x2 * img.shape[1]
                y1 = y1 * img.shape[0]
                y2 = y2 * img.shape[0]
            title = f'{i} : {x[class_col]}'
            img = self.draw_bbox(img, x1,x2,y1,y2, title, i)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=figsize)
        plt.imshow(img)
        plt.show()
        return
    
# ========================================================================
# ========================================================================
    
class MultiImagesVisualizer(object):
    
    def __init__(self, max_show=4, figsize=(16, 4)):
        self.max_show = int(max_show)
        self.figsize = figsize
        
    
    def _verify(self, _list):
        if isinstance(_list[0], str) or isinstance(_list[0], np.ndarray):
            return True
        return False

    def visualize(self, inputs, title=''):
        if not isinstance(inputs, list):
            inputs = [inputs]
        assert self._verify(inputs), f'Input type must be path(s) or image-like array(s), found {type(inputs[0])}'
    
        plt.figure(figsize=self.figsize).suptitle(title)
        for i,img in enumerate(inputs[:self.max_show]):
            plt.subplot(1, len(inputs), i+1), plt.xticks([]), plt.yticks([])
            if isinstance(img, str):
                img = mpimg.imread(img)
            plt.imshow(img)
        return
    