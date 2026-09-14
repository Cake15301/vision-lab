@echo off
title YOLO 工作终端
cd /d "D:\AI\zcode files\YOLO入门学习"
echo.
echo  ============================================
echo     欢迎使用 YOLO 终端
echo     YOLO 版本: 8.4.128   显卡: RTX 3060
echo  --------------------------------------------
echo     常用命令（输入后按回车执行）:
echo.
echo     [1] 检测一张图片:
echo         yolo predict model=yolov8n.pt source=图片路径
echo.
echo     [2] 检测整个文件夹:
echo         yolo predict model=yolov8n.pt source=文件夹路径
echo.
echo     [3] 训练模型(官方示例数据):
echo         yolo detect train data=coco8.yaml model=yolov8n.pt epochs=30
echo.
echo     [4] 查看版本和帮助:
echo         yolo version
echo.
echo  --------------------------------------------
echo     检测结果自动保存在 runs\detect\predict 文件夹
echo     关闭此窗口请输 exit 或直接点右上角 X
echo  ============================================
echo.
cmd /k
