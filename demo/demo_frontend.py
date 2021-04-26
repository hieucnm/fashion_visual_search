import warnings
warnings.filterwarnings('ignore')

import io
import os
import os.path as osp
import numpy as np
from PIL import Image
import ipywidgets as widgets
from IPython.core.display import display, clear_output, Javascript

# ====================================================================================================


class DemoImageBox(widgets.VBox):
    
    def __init__(self, path=None, title=None, link=None, query_callback=None, width=150, height=180):
        
        self.width = width
        self.height = height
        self.path = path
        
        self.image = self.create_image(path)
        self.label = widgets.Label(title)
        self.link = link
        
        if self.link is None:
            self.query  = widgets.Button(description='Query', button_style='success', 
                                         layout=widgets.Layout(width=f'{self.width}px'))
            self.buttons = widgets.HBox([self.query])
        else:
            self.query  = widgets.Button(description='Query', button_style='success', 
                                         layout=widgets.Layout(width=f'{self.width // 2}px'))
            self.browse = widgets.Button(description='Browse', button_style='warning', 
                                         layout=widgets.Layout(width=f'{self.width // 2}px'))
            self.browse.on_click(self.on_browse_clicked)
            self.buttons = widgets.HBox([self.query, self.browse])
        
        self.query.on_click(query_callback)
        self.update_children()
        self.layout = widgets.Layout(width = f'{self.width + 50}px' if self.link is None else f'{self.width+20}px')
        super().__init__(children=self.children, layout=self.layout)
    
    
    def update_children(self):
        self.children = [self.label, self.image, self.buttons]
    
    def on_browse_clicked(self, b):
        # this also can open url, but only work on local :
        # import webbrowser
        # webbrowser.open(self.link)
        display(Javascript(f'window.open("{self.link}");'))    
    
    def to_bytearray(self, img):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        
        # PIL Image to byte array
        img_byte = io.BytesIO()
        img.save(img_byte, format='PNG')
        img_byte = img_byte.getvalue()
        return img_byte
    
    def create_image(self, path):
        if path is None:
            return widgets.Image(value=b'', format='PNG', width=self.width, height=self.height)
        
        img = Image.open(path)
        img = img.resize((self.width, self.height), Image.ANTIALIAS)
        
        img_byte = self.to_bytearray(img)
        return widgets.Image(value=img_byte, format='PNG', width=self.width, height=self.height)
    
    
    def fromarray(self, img_array):
        img = Image.fromarray(img_array)
        img = img.resize((self.width, self.height), Image.ANTIALIAS)
        img_byte = self.to_bytearray(img)
        self.image = widgets.Image(value=img_byte, format='PNG', width=self.width, height=self.height)
        self.buttons = widgets.HBox([])
        self.update_children()
        return self
    

# ====================================================================================================


class DemoNavigation(widgets.HBox):
    
    def __init__(self, demo_dir):
        
        self.accept_format = ['jpg', 'jpeg', 'png']
        self.demo_dir = demo_dir
        
        paths = self.get_demo_paths()
        
        self.uploader = widgets.FileUpload(accept=','.join(self.accept_format), multiple=True)
        self.uploader.observe(self.on_uploader_change)
        self.dropdown = widgets.Dropdown(options=paths, value=paths[0][1], description='File name:')
        self.clear_button = widgets.Button(description='Clear', button_style='info', icon='refresh')
        self.back_button = widgets.Button(description='Back', icon='arrow-left', layout=widgets.Layout(width='auto'))
        
        children = [self.back_button, self.uploader, self.dropdown, self.clear_button]
        super().__init__(children = children)
    

    def on_uploader_change(self, change):
        if change['type'] == 'change' and change['name'] == 'value':
            uploader_value = self.uploader.value
            if len(uploader_value) == 0:
                return
            
            self.upload(uploader_value)
            self.dropdown.options = self.get_demo_paths()
            self.dropdown.value = osp.join(self.demo_dir,list(uploader_value.keys())[-1])
        return
    
    
    def get_demo_paths(self):
        files = sorted(os.listdir(self.demo_dir))
        files = [f for f in files if f.split('.')[-1] in self.accept_format]
        full_paths = [osp.join(self.demo_dir, f) for f in files]
        full_paths = [(f,p) for f,p in zip(files, full_paths)]
        return full_paths
    
    
    def upload(self,uploader_value):
        if len(uploader_value) != 0:
            for name,data in uploader_value.items():
                with open(osp.join(self.demo_dir, name), 'wb') as f:
                    f.write(data['content'])
        return self.get_demo_paths()
    

# ====================================================================================================

class DemoFrontend(object):
    
    def __init__(self, demo_dir):
        
        self.navigation = DemoNavigation(demo_dir)
        self.navigation.clear_button.on_click(self.on_clear_clicked)
        self.navigation.dropdown.observe(self.on_dropdown_change)
        
        self.item_label = widgets.Label('Query result using whole image:')
        self.obj_label = widgets.Label('Query result using detected object bound box:')
        
        self.item_nn = []
        self.obj_nn_list = []
        
        self.input_image = None
        self.item_box = widgets.HBox([])
        self.obj_boxes = widgets.VBox([widgets.HBox([])])

        self.wrapper = None
        self.on_query_clicked = None

    
    def init_display(self):
        clear_output()
        display(self.navigation)
    
    def update_wrapper(self):
        self.wrapper = widgets.VBox([self.navigation, self.item_label, self.item_box, self.obj_label, self.obj_boxes])
    
    def display(self, update_wrapper=True):
        if update_wrapper:
            self.update_wrapper()
        clear_output()
        display(self.wrapper)
        
    def clear_old_result(self):
        self.item_nn = []
        self.obj_nn_list = []
        self.item_box = widgets.HBox([])
        self.obj_boxes = widgets.VBox([widgets.HBox([])])
    
    
    def on_clear_clicked(self, b):
        self.clear_old_result()
        self.display()
    
    def on_dropdown_change(self, change):
        if change['type'] == 'change' and change['name'] == 'value':
            path = self.navigation.dropdown.value
            if path is not None:
                self.clear_old_result()
                self.update_input_image(path)
                self.item_box = widgets.HBox([self.input_image])
                self.wrapper = widgets.VBox([self.navigation, self.item_box])
                self.display(update_wrapper=False)
        return
    
    def list_all_image_boxes(self):
        obj_boxes = []
        for vbox in self.obj_boxes.children:
            obj_boxes.extend(vbox.children)
        return list(self.item_box.children) + obj_boxes
    
    
    def find_path_of_clicked_button(self, b):
        all_image_boxes = self.list_all_image_boxes()
        for box in all_image_boxes:
            if isinstance(box, DemoImageBox) and box.query == b:
                return box.path
        raise ValueError('No ImageBox found!')


    def update_query_callback(self, callback):
        self.on_query_clicked = callback
    
    def update_input_image(self, path):
        self.input_image = DemoImageBox(path=path, title='Input image :', query_callback=self.on_query_clicked)
    
    
    def update_item_box(self, item_nn):
        item_nn['title'] = item_nn.apply(lambda x: f'Top-{x["rank"]} | {x.item_id} | {x.score : .3f}', axis=1)
        self.item_nn = item_nn.apply(lambda x: DemoImageBox(path=x.path, title=x.title, link=x.link, query_callback=self.on_query_clicked), axis=1).tolist()
        self.item_box = widgets.HBox([self.input_image] + self.item_nn)
        
    def update_obj_boxes(self, obj_nn_list):
        if len(obj_nn_list) == 0:
            self.obj_label = widgets.Label('No item found!')
            return
        
        obj_box_list = []
        for obj_classname, render_img, obj_nn in obj_nn_list:
            render_img = DemoImageBox(title=f'* Cloth position: {obj_classname}').fromarray(render_img)
            obj_nn['title'] = obj_nn.apply(lambda x: f'Top-{x["rank"]} | {x.item_id} | {x.score : .3f}', axis=1)
            self.obj_nn = obj_nn.apply(lambda x: DemoImageBox(path=x.raw_path, title=x.title, link=x.link, query_callback=self.on_query_clicked), axis=1).tolist()
            obj_box = widgets.HBox([render_img] + self.obj_nn)
            obj_box_list.append(obj_box)
        
        self.obj_boxes = widgets.VBox(obj_box_list)
        self.obj_label = widgets.Label('Query result using detected object bound box:')

