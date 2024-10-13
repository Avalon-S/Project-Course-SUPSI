import cv2
import os
# import numpy as np

# 设置图片文件夹和视频输出路径
image_folder = './output_grid/'  
video_name = 'output_video_grid.mp4'


images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
images.sort(key=lambda x: float(os.path.splitext(x)[0]))  # 按时间排序

# 获取第一张图片的尺寸
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

# 设置视频编码器及参数
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(video_name, fourcc, 24, (width, height))

# 逐帧写入视频
for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()

