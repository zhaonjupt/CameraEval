import cv2
from PIL import Image
# import numpy as np
# import time
#
# dict = {'road'       :    [128, 64,128],
#         'sidewalk'   :    [244, 35,232],
#         'parking'    :    [250,170,160],
#         'building'   :    [ 70, 70, 70],
#         'wall'       :    [102,102,156],
#         'fence'      :    [190,153,153],
#         'bridge'     :    [150,100,100],
#         'tunnel'     :    [150,120, 90],
#         'vegetation' :    [107,142, 35],
#         'terrain'    :    [152,251,152],
#         'sky'        :    [ 70,130,180],
#         'background' :    [255,255,255],
#         }
#
# def cal_ratio(image, index):
#     count = 0
#     target = []
#     for each in index:
#         target.append(dict[each])
#     for col in image[:,:,:3]:  #获取一个像素点的三通道值
#         for pixel in col:
#             if pixel.tolist() in target:
#                count += 1
#     print('%.2f ' % (count / (image.shape[0]*image.shape[1]) * 100.0) + '%')
#
#
#
# if __name__ == '__main__':
#     try:
#         image = Image.open('./img/res_123.png')
#     except:
#         raise Exception("Can\'t open this image... ")
#     start = time.time()
#     cal_ratio(np.copy(image), ['vegetation', 'terrain'])
#     end = time.time()
#     print("%.2f s" % (end-start))
#     # list = [[1,2,3],[2,3,4]]
#     # a = [1,2,4]
#     # print(a in list)


# labels = [2,2,2,2,6,6,6]
# if (0 or 1) in labels and (2 or 3 or 4) in labels:
#     print('case 1')
#     # result = 0.6 * face_score + 0.4 * plate_score
# elif (0 or 1) in labels and not (2 and 3 and 4) in labels:
#     print('case 2')
#     # result = face_score
# elif not (0 and 1) in labels and (2 or 3 or 4) in labels:
#     print('case 3')
#     # result = plate_score
# else:
#     print('case 4')
#     # result = 0

# list = [1,2,3,4];
# it = iter(list)
# try:
#     while(True):
#         print(next(it))
# except:
#     print("迭代完成")

image = Image.open('./img/123.jpg')
img = cv2.imread('img/123.jpg')
print(image.size)
print(img.shape)
