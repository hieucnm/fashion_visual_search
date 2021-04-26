
class DemoController(object):
    
    def __init__(self, frontend, backend):
        self.frontend = frontend
        self.backend = backend
        self.frontend.update_query_callback(self.on_query_clicked)
    
    def init_display(self):
        self.frontend.init_display()
        
    def on_query_clicked(self, b):
        path = self.frontend.find_path_of_clicked_button(b)
        item_nn, objects_nn = self.backend.run(path, topk=5)
        self.frontend.update_input_image(path)
        self.frontend.update_item_box(item_nn)
        self.frontend.update_obj_boxes(objects_nn)
        self.frontend.display()
