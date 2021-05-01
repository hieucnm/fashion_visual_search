<div align="center">
    <h1>
    The Future of E-Commerce: Visual Search	
    </h1>
</div>

# 1. Introduction

## Visual Search

Visual Search is the process of searching for something given images. In online shopping, imagine that we can search for a product just by capturing a photo of it and send to the system (for example, a mobile application), without typing in the description of that one, or even sometimes we cannot describe it correctly. Visual search is the future of e-commerce. Many companies applied visual search in their products, such as Shopee, Lazada, Amazon, and Pinterest, too.

![](static/introduction.jpeg)
*How Visual Search works in e-commerce ([Source](https://medium.com/@virtua/visual-search-in-e-commerce-41ecf52b66d2))*


## Deep Metric Learning
Deep Metric Learning (DML) aims to learn a function mapping images, often in the shape of 3D arrays, into a one-dimension feature space. The learnt function should output low-distance feature vectors with respect to visually similar input images, and vice versa. Classification-based losses are usually used to train a DML model from scratch before switching to time-consuming metric learning losses [1](https://arxiv.org/abs/1811.12649). Hence, for the purpose of demo, here I use a classification-based training method [in this paper](https://arxiv.org/abs/1811.12649) to train my model.

## Object Detection
Object detection is one of the most interesting task in computer vision. It not only detects the location of the objects in a given image, but also classify them into pre-defined categories. In real-life visual search, we may often deal with images containing so much background, and the results of the DML model consequencely could be bad. Taking out the main object in an image, which can be done by a good object detection, can help improve the searching results in these cases.

## This project

In this project, I demonstrated a simple visual search system in e-commerce, using deep metric learning and object detection models. For the sake of demonstration, I used datasets consisting of clothing items only, other products might be applied similarly. I trained a DML model using a classification-based method. The training codes were forked and edited from [this repo](https://github.com/azgo14/classification_metric_learning). Moreover, to handle the noise cases mentioned above, I also employ a pretrained yolo_v3 model, which were trained on fashion items from [this repo](https://github.com/simaiden/Clothing-Detection) . There are hence 2 types of searching results: image-based and item-based. You can see some screenshots below.


# 2. Dataset

In this project, I combined the 2 benchmarks: [Inshop Deep Fashion dataset](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/InShopRetrieval.html) and [Street2Shop dataset](http://www.tamaraberg.com/street2shop/). I also worked around on other datasets but they are very noisy and not compatible with each other in the class definition. The numbers of instances and classes in each dataset are given below:

|                   	| # classes	| # instances 	|
|-------------------	| --------	| --------	|
| Deep Fashion		| 7982		| 52712	|
| Street2Shop		| 6128		| 39985	|

For stability, I kept only the classes having at least 2 images and at most 30 images in each dataset. After filtering and merging, I applied a random 80-20 train-validation split on the classes. In the validation set,  each image in each class was randomly chosen as a gallery or a query one at the probability of 50%. The desired distribution is given below:

|                   	| # classes	| # instances 	|
|-------------------	| --------	| --------	|
| Training		| 11201	| 70109	|
| Gallery		| 2801		| 8922		|
| Query		| 2801		| 8780		|


- To live demo (with no labels),  my partners and I collected data from an Vietnamese e-commerce website, and got 28165 images of 8561 different clothing items. You can find them [here](https://drive.google.com/drive/folders/15aLp2AtTD6okkKgx1cMEIyx6J7PGE8aj?usp=sharing)


# 3. Result

Firstly, I evaluated the two deep metric learning models: one outputs 256-d embedding vectors and the other outputs 2048-d embedding vectors. Both models were trained in 30 epochs, starting learning rate is 0.01, batch size is 64, decay learning rate by a gamma of 0.1 after every 10 epochs. The whole system were run on a machine including 4 Intel(R) Xeon(R) CPU E5-2630 v4 @ 2.20GHz, 128GB RAM, and one GeForce GTX 1080Ti 12GB. The result is given below:

| 			| 256-d	| 2048-d 	|
|-------------		| --------	| --------	|
| R@1			| 78.76	| 89.28	|
| R@5			| 85.27	| 95.93	|
| R@10			| 88.04	| 97.76	|
| R@100		| 92.41	| 99.46	|
| Inference time 	| 22 ms	| 21 ms	|
| Search time	 	| 0.15 s	| 0.9 s 	|

How I applied YoloV3: I used YoloV3 to pre-extract bounding boxes of all images in the gallery and saved them. When searching, apart from returning top-k items having the most similar images, the system also returns items having the most similar objects (which are images too and were saved in advance) to the object in the query extracted by YoloV3. Hence, as I said before, the results have two levels: image-based and item-based. The smaller the cosine distance between the corresponding embedding vectors, the more similar the images are.

A small problem: YoloV3 or other object detectors sometimes detect multiple classes of the same object. To handle this issue, I map the classes of the YoloV3 to one of two pre-defined superclasses, which I call positions: *upper* and *lower*. For example, if the YoloV3 says that there is a *short sleeve top*, a *short sleeve outwears* and a *sling* in an image, then the system only considers the one having the highest confidence score. This method turned out another benefit that, the system will not return an upper object when the query image contains a lower object and vice versa.

The mapper is given below:
- upper: short-sleeve-top, long-sleeve-top, long-sleeve-outwear, short-sleeve-outwear, long-sleeve-dress, short-sleeve-dress, sling-dress, vest-dress, vest, sling
- lower: trousers, shorts, skirt

Here are some results of the live testset:

Example 1:
![](static/example1.png)

Example 2:
![](static/example2.png)


# 4. How to run
1. Setup environment
```sh
# Create conda environment
conda create --name fashion_visual_search python=3.7 pip ipykernel
conda activate fashion_visual_search
python -m ipykernel install --user --name=fashion_visual_search

# Install all requirements
pip install -r requirements.txt
```


2. Download weights of the pretrained YoloV3 model at this [link](https://drive.google.com/file/d/1P2BtqrIKbz2Dtp3qfPCkvp16bj9xSVIw/view?usp=sharing)  and put it into folder *detector/yolo/weights*
3. Download pre-extracted databases of the live testset at this [folder](https://drive.google.com/drive/folders/11j83EtIkTmxE2FN6pSvHPNAIytOSIGWu?usp=sharing), and put them into folder *datasets/database*
4. Download the images of the live testset at this [folder](https://drive.google.com/drive/folders/15aLp2AtTD6okkKgx1cMEIyx6J7PGE8aj?usp=sharing) and put them into folder *datasets/tiki*
5. Run the notebook Demo.ipynb, in the beginning you have to change the environment of the notebook to *fashion_visual_search*, which you init above.
6. Have fun.
