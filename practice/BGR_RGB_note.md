# BGR/RGB 坑笔记

1. PIL 读图的三个数排队顺序是：RGB
2. OpenCV（cv2）读图的排队顺序是：BGR
3. 同一张 grace_hopper.jpg 的第一像素，PIL 读出 [21 24 77]，
   cv2 读出：[77 24 21]（规律：哪个位置没动、哪两个互换？）
4. 把 cv2 读的图转回正常顺序，用的一句代码：
   ```python
   img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
   ```
5. 我自己的话——什么情况下会踩这个坑、踩了会看到什么现象：
   当用 OpenCV 读取图片时，如果图片路径中包含中文，就会报错。

6. 附送一个今天才知道的坑：cv2.imread 碰到中文路径会：None但不炸。
