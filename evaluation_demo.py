from PIL import Image
import tensorflow as tf
import core.utils as utils
import numpy as np
import time 
import cv2

tf.reset_default_graph()
# video_path = 0
video_list = ['rtsp://admin:12345@10.10.20.22', 'rtsp://admin:12345@10.10.20.23']
eval_time = 1  # 单位分钟
yolo_pb_file = './checkpoint/yolov3.pb'
deep_pb_file = './checkpoint/frozen_inference_graph.pb'
yolo_input_size = 416
deep_input_size = 513
yolo_graph = tf.get_default_graph()
deep_graph = tf.Graph()
yolo_num_classes = 8
deep_num_classes = 11
#定义不参与计算的标签列表
mask = ['sky', 'vegetation']
# 声明会话
yolo_sess = tf.Session(graph=yolo_graph)
deep_sess = tf.Session(graph=deep_graph)

with deep_graph.as_default():
    deep_placeholder = tf.placeholder(dtype=tf.uint8, shape=[1, deep_input_size, deep_input_size, 3])
yolo_return_elements = ["input/input_data:0", "pred_sbbox/concat_2:0", "pred_mbbox/concat_2:0", "pred_lbbox/concat_2:0"]
deep_return_elements = ["ImageTensor:0", "SemanticPredictions:0"]



#获取GraphDef返回的tensor
with yolo_sess.as_default():
    yolo_return_tensors = utils.read_pb_return_tensors(yolo_graph, yolo_pb_file, yolo_return_elements)
with deep_sess.as_default():
    deep_output = utils.get_deep_tensors(deep_graph, deep_pb_file, deep_placeholder)

scores = []

# while (True):
#     if video_path == 1:  # 测试用值  实际运算时置为0
#         #################################
#         # 事实上需要尝试捕获下一个video_path
#         continue
#     else:
#         init_time = time.time()
#         vid = cv2.VideoCapture(video_path)
#         count = 0
#         while (time.time() - init_time < eval_time * 60):
#             return_value, frame = vid.read()
#             if return_value:
#                 frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 image = Image.fromarray(frame)
#             else:
#                 vid = cv2.VideoCapture(video_path)
#                 continue
#             # 对同一个视频地址，我们只针对第####   1   ####个有效帧做语义分割，即有效区域检测
#             # 可根据需求进行调整，例如取多帧求均值
#             if count == 0:
#                 with deep_sess.as_default():
#                     with deep_graph.as_default():
#                         count += 1
#                         start = time.time()
#                         resized_image = utils.deep_image_process(image, deep_input_size)
#                         batch_seg_map = deep_sess.run(deep_output, feed_dict={deep_placeholder: resized_image})
#                         seg_img = utils.label_to_color_image(batch_seg_map[0][0]).astype(np.uint8)
#                         ratio = utils.cal_ratio(seg_img, mask)
#                         print("该场景的有效区域占比为%.2f" % (1.0 - ratio))
# #                         res = Image.fromarray(seg_img)
# #                         res.save('./res/result.png')
#                         end = time.time()
#                         print("deeplab total cost %.2f s" % (end - start))
#                         deep_score = 1.0 - ratio
#                         ########################   打分策略    $$$$$$$$$$$$$$$$$$$$$$$
#             else:
#                 with yolo_sess.as_default():
#                     with yolo_graph.as_default():
#                         frame_size = frame.shape[:2]
#                         image_data = utils.image_preporcess(np.copy(frame), [yolo_input_size, yolo_input_size])
#                         image_data = image_data[np.newaxis, ...]
#                         prev_time = time.time()
#                         pred_sbbox, pred_mbbox, pred_lbbox = yolo_sess.run(
#                             [yolo_return_tensors[1], yolo_return_tensors[2], yolo_return_tensors[3]],
#                             feed_dict={yolo_return_tensors[0]: image_data})
#
#                         pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + yolo_num_classes)),
#                                                     np.reshape(pred_mbbox, (-1, 5 + yolo_num_classes)),
#                                                     np.reshape(pred_lbbox, (-1, 5 + yolo_num_classes))], axis=0)
#
#                         bboxes = utils.postprocess_boxes(pred_bbox, frame_size, yolo_input_size, 0.3)
#                         bboxes = utils.nms(bboxes, 0.45, method='nms')
#                         scores.append(utils.cal_dis_ratio(original_image=frame, bboxes=bboxes))
#                         image = utils.draw_bbox(frame, bboxes)
#
#                         curr_time = time.time()
#                         exec_time = curr_time - prev_time
#                         result = np.asarray(image)
#                         info = "fps: %.2f " % (1.0 / exec_time)
#                         cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
#                         result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#                         font = cv2.FONT_HERSHEY_SIMPLEX
#                         result = cv2.putText(result, info, (25, 25), font, 1.2, (255, 255, 0))
#                         cv2.imshow("result", result)
#                         if cv2.waitKey(1) & 0xFF == ord('q'):
#                             break
#                         #print(exec_time)
#                         ########################       打分策略       #############################
#
#         count = 0
#         video_path = 0
#         yolo_score = max(scores)
#         print("dete_score is " + str(yolo_score))
#         eval_score = deep_score + yolo_score
#         print("eval_score is "+ str(eval_score))


count = 0
video_path = 0
#vid = cv2.VideoCapture(video_path)
it = iter(video_list)
while(True):
    if video_path == 0:
        try:
            video_path = next(it)
        except:
            print("评估完成")
            break
        continue
    else:
        init_time = time.time()
        vid = cv2.VideoCapture(video_path)
        count = 0
        while (time.time() - init_time < eval_time * 60):
                return_value, frame = vid.read()
                if return_value:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(frame)
                else:
                    print("Try to catch frame again...")
                    vid = cv2.VideoCapture(video_path)
                    continue
                if count == 0:
                    with deep_sess.as_default():
                        with deep_graph.as_default():
                            count += 1
                            start = time.time()
                            resized_image = utils.deep_image_process(image, deep_input_size)
                            batch_seg_map = deep_sess.run(deep_output, feed_dict={deep_placeholder: resized_image})
                            seg_img = utils.label_to_color_image(batch_seg_map[0][0]).astype(np.uint8)
                            ratio = utils.cal_ratio(seg_img, mask)
                            res = Image.fromarray(seg_img)
                            res.save("./res/result_" + video_path.split('.')[-1] + ".png")
                            end = time.time()
                            print("deeplab total cost %.2f s" % (end - start))
                            deep_score = (1.0 - ratio) * 5
                            print('deep_score is ' + str(deep_score))
                            ########################   打分策略    $$$$$$$$$$$$$$$$$$$$$$$
                else:
                    with yolo_sess.as_default():
                        with yolo_graph.as_default():
                            frame_size = frame.shape[:2]
                            image_data = utils.image_preporcess(np.copy(frame), [yolo_input_size, yolo_input_size])
                            image_data = image_data[np.newaxis, ...]
                            prev_time = time.time()
                            pred_sbbox, pred_mbbox, pred_lbbox = yolo_sess.run(
                                [yolo_return_tensors[1], yolo_return_tensors[2], yolo_return_tensors[3]],
                                feed_dict={yolo_return_tensors[0]: image_data})

                            pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + yolo_num_classes)),
                                                        np.reshape(pred_mbbox, (-1, 5 + yolo_num_classes)),
                                                        np.reshape(pred_lbbox, (-1, 5 + yolo_num_classes))], axis=0)

                            bboxes = utils.postprocess_boxes(pred_bbox, frame_size, yolo_input_size, 0.3)
                            bboxes = utils.nms(bboxes, 0.45, method='nms')
                            scores.append(utils.cal_dis_ratio(original_image=frame, bboxes=bboxes))
                            image = utils.draw_bbox(frame, bboxes)
                            bboxes.clear()
                            curr_time = time.time()
                            exec_time = curr_time - prev_time
                            result = np.asarray(image)
                            info = "fps: %.2f " % (1.0 / exec_time)
                            cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
                            result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            result = cv2.putText(result, info, (25, 25), font, 1.2, (255, 255, 0))
                            cv2.imshow("result", result)
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                break
        count = 0
        video_path = 0
        yolo_score = max(scores)
        print("dete_score is " + str(yolo_score))
        eval_score = deep_score + yolo_score
        print("eval_score is "+ str(eval_score))


