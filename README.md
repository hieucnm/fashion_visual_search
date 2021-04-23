<div align="center">
    <h1>
    The Future of E-Commerce: Visual Search	
    </h1>
</div>

# 1. Introduction

## Visual Search

Visual Search is the process of searching for something given images. In online shopping, imagine that we can search for a product just by capturing a photo of it and send to the system (for example, a mobile application), without typing in the description of that one, or even sometimes we cannot describe it correctly. Visual search is the future of e-commerce. Many companies applied visual search in their products, such as Shopee, Lazada, Amazon, and Pinterest, too.

![](static/introduction.jpeg)
How visual searching work in e-commerce ([Source](https://medium.com/@virtua/visual-search-in-e-commerce-41ecf52b66d2))

## Deep Metric Learning
DML is blah blah blah

## Object Detection
Object detection is blah blah blah.
Yolo is bla bla ...


Trong project này, tui sẽ train và demo một cái UI đơn giản về visual search, dùng deep metric learning và yolov3. Do scope của e-commerce là rất lớn, nên ở level demo, tui chỉ cover data quần áo, yolov3 tui cũng dùng pretrained model đã train trên data quần áo.

Why object detection here ? Vì nhận thấy search bằng full ảnh cho kết quả ko chính xác, nếu dùng object detection tìm object chính trong ảnh sẽ cho kq chính xác hơn. Cũng do đó, tui sẽ demo ở 2 mức: mức full ảnh và mức object detect đc từ ảnh.

Code và weight của YoloV3 tui kế thừa từ đây.
Tui cũng fork và edit code DML từ great project này.


# 2. Datasets
- Để train và đánh giá, tui dùng bộ public Deep Fashion. Tui cũng đã tìm nhiều bộ khác nhưng số class ít và có vấn đề bla bla
(Kẻ cái bảng các bộ data và mô tả ở đây)
- Tui dùng nhiêu đây class nè, tui cũng map các class output của pretrained YoloV3 sang các class của bộ Deep Fashion cho đồng nhất.
(Kẻ bảng các class và số sample mỗi class ra)
- Để demo, tui và partners collect data từ một sàn e-commerce, thu đc nhiêu đây ảnh, và nhiều đây object nè

# 3. How to run
1. Git clone
2. Install requirements
3. Tải weights yolov3, weights model dml và pre-extracted embedding
4. chạy notebook



