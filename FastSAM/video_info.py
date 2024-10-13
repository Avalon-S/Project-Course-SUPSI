# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 21:38:21 2024

@author: 24760
"""

import cv2

def get_video_info(video_path):
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    # 获取视频的相关信息
    video_info = {}
    video_info['视频路径'] = video_path
    video_info['宽度'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_info['高度'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_info['帧率'] = cap.get(cv2.CAP_PROP_FPS)
    video_info['总帧数'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_info['视频格式'] = cap.get(cv2.CAP_PROP_FORMAT)
    video_info['时长(秒)'] = video_info['总帧数'] / video_info['帧率'] if video_info['帧率'] > 0 else 0

    cap.release()
    
    # 打印视频信息
    for key, value in video_info.items():
        print(f"{key}: {value}")

# 测试视频信息获取函数
video_path = "output_video_2.mp4"  # 替换成你的视频文件路径
get_video_info(video_path)
