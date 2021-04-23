#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : utils.py
#   Author      : YunYang1994
#   Created date: 2019-02-28 13:14:19
#   Description :
#
#================================================================

import cv2
import random
import colorsys
import numpy as np
import tensorflow as tf
from core.config import cfg
from PIL import Image

def read_class_names(class_file_name):
    '''loads class name from a file'''
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names


def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = np.array(anchors.split(','), dtype=np.float32)
    return anchors.reshape(3, 3, 2)


def image_preporcess(image, target_size, gt_boxes=None):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

    ih, iw    = target_size
    h,  w, _  = image.shape

    scale = min(iw/w, ih/h)
    nw, nh  = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0)
    dw, dh = (iw - nw) // 2, (ih-nh) // 2
    image_paded[dh:nh+dh, dw:nw+dw, :] = image_resized
    image_paded = image_paded / 255.

    if gt_boxes is None:
        return image_paded

    else:
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
        return image_paded, gt_boxes


def draw_bbox(image, bboxes, classes=read_class_names(cfg.YOLO.CLASSES), show_label=True):
    """
    bboxes: [x_min, y_min, x_max, y_max, probability, cls_id] format coordinates.
    """

    num_classes = len(classes)
    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    for i, bbox in enumerate(bboxes):
        coor = np.array(bbox[:4], dtype=np.int32)
        fontScale = 0.5
        score = bbox[4]
        class_ind = int(bbox[5])
        bbox_color = colors[class_ind]
        bbox_thick = int(0.6 * (image_h + image_w) / 600)
        c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])
        cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)

        if show_label:
            class_map = {0: 'person',  1: 'person',
                         2: 'viechel', 3: 'viechel',
                         4: 'viechel', 5: 'face',
                         6: 'plate',   7: 'viechel'}
            bbox_mess = '%s: %.2f' % (class_map[class_ind], score)
            t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick//2)[0]
            cv2.rectangle(image, c1, (c1[0] + t_size[0], c1[1] - t_size[1] - 3), bbox_color, -1)  # filled

            cv2.putText(image, bbox_mess, (c1[0], c1[1]-2), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale, (0, 0, 0), bbox_thick//2, lineType=cv2.LINE_AA)

    return image



def bboxes_iou(boxes1, boxes2):

    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up       = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down    = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area    = inter_section[..., 0] * inter_section[..., 1]
    union_area    = boxes1_area + boxes2_area - inter_area
    ious          = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)

    return ious

def get_deep_tensors(graph, pb_file, placeholder):
    with open(pb_file, 'rb') as f:
        deep_graph_def = tf.GraphDef()
        deep_graph_def.ParseFromString(f.read())
    with graph.as_default():
        deep_output = tf.import_graph_def(deep_graph_def,input_map={"ImageTensor:0":placeholder},
                                          return_elements=["SemanticPredictions:0"])
    return deep_output

def read_pb_return_tensors(graph, pb_file, return_elements):

    with tf.gfile.FastGFile(pb_file, 'rb') as f:
        frozen_graph_def = tf.GraphDef()
        frozen_graph_def.ParseFromString(f.read())

    with graph.as_default():
        return_elements = tf.import_graph_def(frozen_graph_def,
                                              return_elements=return_elements)
    return return_elements


def nms(bboxes, iou_threshold, sigma=0.3, method='nms'):
    """
    :param bboxes: (xmin, ymin, xmax, ymax, score, class)

    Note: soft-nms, https://arxiv.org/pdf/1704.04503.pdf
          https://github.com/bharatsingh430/soft-nms
    """
    classes_in_img = list(set(bboxes[:, 5]))
    best_bboxes = []

    for cls in classes_in_img:
        cls_mask = (bboxes[:, 5] == cls)
        cls_bboxes = bboxes[cls_mask]

        while len(cls_bboxes) > 0:
            max_ind = np.argmax(cls_bboxes[:, 4])
            best_bbox = cls_bboxes[max_ind]
            best_bboxes.append(best_bbox)
            cls_bboxes = np.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
            iou = bboxes_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
            weight = np.ones((len(iou),), dtype=np.float32)

            assert method in ['nms', 'soft-nms']

            if method == 'nms':
                iou_mask = iou > iou_threshold
                weight[iou_mask] = 0.0

            if method == 'soft-nms':
                weight = np.exp(-(1.0 * iou ** 2 / sigma))

            cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
            score_mask = cls_bboxes[:, 4] > 0.
            cls_bboxes = cls_bboxes[score_mask]

    return best_bboxes


def postprocess_boxes(pred_bbox, org_img_shape, input_size, score_threshold):

    valid_scale=[0, np.inf]
    pred_bbox = np.array(pred_bbox)

    pred_xywh = pred_bbox[:, 0:4]
    pred_conf = pred_bbox[:, 4]
    pred_prob = pred_bbox[:, 5:]

    # # (1) (x, y, w, h) --> (xmin, ymin, xmax, ymax)
    pred_coor = np.concatenate([pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5,
                                pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5], axis=-1)
    # # (2) (xmin, ymin, xmax, ymax) -> (xmin_org, ymin_org, xmax_org, ymax_org)
    org_h, org_w = org_img_shape
    resize_ratio = min(input_size / org_w, input_size / org_h)

    dw = (input_size - resize_ratio * org_w) / 2
    dh = (input_size - resize_ratio * org_h) / 2

    pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
    pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio

    # # (3) clip some boxes those are out of range
    pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),
                                np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])], axis=-1)
    invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
    pred_coor[invalid_mask] = 0

    # # (4) discard some invalid boxes
    bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
    scale_mask = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))

    # # (5) discard some boxes with low scores
    classes = np.argmax(pred_prob, axis=-1)
    scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
    score_mask = scores > score_threshold
    mask = np.logical_and(scale_mask, score_mask)
    coors, scores, classes = pred_coor[mask], scores[mask], classes[mask]

    return np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)






def deep_image_process(image, deep_input_size):
    width, height = image.size
    resize_ratio = 1.0 * deep_input_size / max(width, height)
    target_size = (int(resize_ratio * width), int(resize_ratio * height))
    resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
    height, width = resized_image.size
    padded_image = np.full(shape=[deep_input_size, deep_input_size, 3], fill_value=0)
    dw, dh = (deep_input_size - width) // 2, (deep_input_size - height) // 2
    padded_image[dw:dw + width, dh:dh + height, :] = resized_image
    resized_image = np.expand_dims(padded_image, axis=0).astype(np.uint8)
    return resized_image


colormap = np.asarray([
  [149, 58, 157],
  [149, 58, 157],
  [149, 58, 157],
  [70, 70, 70],
  [70, 70, 70],
  [70, 70, 70],
  [107, 142, 35],
  [152,251,152],
  [70, 130, 180],
  [  0,  0,  0],
  ])


dict = {'road'       :    [149, 58, 157],
        'sidewalk'   :    [149, 58, 157],
        'parking'    :    [149, 58, 157],
        'building'   :    [ 70, 70, 70],
        'wall'       :    [ 70, 70, 70],
        'fence'      :    [ 70, 70, 70],
        'vegetation' :    [107,142, 35],
        'terrain'    :    [152,251,152],
        'sky'        :    [ 70,130,180],
        'background' :    [  0,  0,  0],
        }


def cal_ratio(image, index):
    count = 0
    target = []
    for each in index:
        target.append(dict[each])
    for col in image[:,:,:3]:  #获取一个像素点的三通道值
        for pixel in col:
            if pixel.tolist() in target:
               count += 1
    #print('%.2f ' % (count / (image.shape[0]*image.shape[1]) * 100.0) + '%')
    return (count / (image.shape[0] * image.shape[1]))

#bboxes: [x_min, y_min, x_max, y_max, probability, cls_id] format coordinates.
def cal_dis_ratio(original_image, bboxes):
    labels = []
    image_h, _, _ = original_image.shape
    face_score = 0
    plate_score = 0
    result = 0
    for _, bbox in enumerate(bboxes):
        class_id = int(bbox[5])
        #如果不是人脸或车牌信息， 仅加入标签列表， 用于计算人车比例
        if not (class_id == 5 or class_id == 6):
            labels.append(class_id)
            continue
        #是人脸或车牌信息时， 计算回归框中心与图像中心的比例， 作为间接的距离描述
        #最终的分数采用距离信息与回归框得分的加权获得
        #人脸得分始终以最高分来计算
        ltrb =bbox[:4]
        score = float(bbox[4])
        center_h = (ltrb[1] + ltrb[3]) / 2
        dis_ratio = center_h / image_h
        temp = score * 3 + dis_ratio * 2
        if class_id == 5 and temp >face_score:
            face_score = temp
        elif class_id == 6 and temp > plate_score:
            plate_score = temp
    #0 1 是否在标签列表中代表是否包含人
    #2 3 4则代表有无车辆
    if (0 or 1) in labels and (2 or 3 or 4) in labels:
        result = 0.6 * face_score + 0.4 * plate_score
    elif (0 or 1) in labels and not (2 and 3 and 4)  in labels:
        result = face_score
    elif not (0 and 1) in labels and (2 or 3 or 4) in labels:
        result = plate_score
    else:
        result = 0

    return result



def label_to_color_image(label):
  """Adds color defined by the dataset colormap to the label.

  Args:
    label: A 2D array with integer type, storing the segmentation label.

  Returns:
    result: A 2D array with floating type. The element of the array
      is the color indexed by the corresponding element in the input label
      to the PASCAL color map.

  Raises:
    ValueError: If label is not of rank 2 or its value is larger than color
      map maximum entry.
  """
  if label.ndim != 2:
    raise ValueError('Expect 2-D input label')

  #colormap = create_pascal_label_colormap()


  if np.max(label) >= len(colormap):
    raise ValueError('label value too large.')

  return colormap[label]


