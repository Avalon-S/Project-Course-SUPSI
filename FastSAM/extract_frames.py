# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 18:13:24 2024

@author: 24760
"""

import cv2
import os

def extract_frames(video_path, output_folder):
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    
    # 获取视频的帧率
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # 计算当前帧的时间戳 (秒)
        timestamp = frame_count / fps
        
        # 创建输出文件名，保留小数点后三位以毫秒为单位
        output_filename = os.path.join(output_folder, f"{timestamp:.3f}.jpg")
        
        # 保存当前帧为 JPG 图片
        cv2.imwrite(output_filename, frame)
        
        frame_count += 1
    
    # 释放视频捕获对象
    cap.release()
    print(f"视频拆分完成，总帧数: {frame_count}")

if __name__ == "__main__":
    video_path = "output_video_point.mp4"  # 输入视频路径
    output_folder = "./frames_output_point/"  # 输出图片文件夹路径
    extract_frames(video_path, output_folder)
