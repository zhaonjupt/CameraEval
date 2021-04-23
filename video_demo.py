#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2018 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : video_demo.py
#   Author      : YunYang1994
#   Created date: 2018-11-30 15:56:37
#   Description :
#
#================================================================

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import cv2
import time
import numpy as np
import core.utils as utils
import tensorflow as tf
from PIL import Image


yolo_return_elements = ["input/input_data:0", "pred_sbbox/concat_2:0", "pred_mbbox/concat_2:0", "pred_lbbox/concat_2:0"]
yolo_pb_file         = "./checkpoint/yolov3.pb"
video_path      = 0
#video_path = "rtsp://admin:12345@10.10.20.22"
#video_path = "rtsp://admin:12345@180.109.129.157:20003"
yolo_num_classes     = 8
yolo_input_size      = 416
yolo_graph           = tf.Graph()
yolo_return_tensors  = utils.read_pb_return_tensors(yolo_graph, yolo_pb_file, yolo_return_elements)



with tf.Session(graph=yolo_graph) as yolo_sess:
    vid = cv2.VideoCapture(video_path)
    while(True):
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            vid = cv2.VideoCapture(video_path)
            continue

        frame_size = frame.shape[:2]
        image_data = utils.image_preporcess(np.copy(frame), [yolo_input_size, yolo_input_size])
        image_data = image_data[np.newaxis, ...]
        prev_time = time.time()
        pred_sbbox, pred_mbbox, pred_lbbox = yolo_sess.run(
            [yolo_return_tensors[1], yolo_return_tensors[2], yolo_return_tensors[3]],
                    feed_dict={ yolo_return_tensors[0]: image_data})

        pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + yolo_num_classes)),
                                    np.reshape(pred_mbbox, (-1, 5 + yolo_num_classes)),
                                    np.reshape(pred_lbbox, (-1, 5 + yolo_num_classes))], axis=0)

        bboxes = utils.postprocess_boxes(pred_bbox, frame_size, yolo_input_size, 0.3)
        bboxes = utils.nms(bboxes, 0.45, method='nms')
        image = utils.draw_bbox(frame, bboxes)

        curr_time = time.time()
        exec_time = curr_time - prev_time
        result = np.asarray(image)
        #info = "time: %.2f ms" %(1000*exec_time)
        info = "fps: %.2f " % (1.0 / exec_time)
        cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
        result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        font = cv2.FONT_HERSHEY_SIMPLEX
        result = cv2.putText(result, info, (25, 25), font, 1.2, (255, 255, 0))

        cv2.imshow("result", result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break




