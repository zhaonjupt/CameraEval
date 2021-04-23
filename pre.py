#时间：2018.12.10
#作用：调用pb文件批量生成deeplabv3预测图
#输入：图像文件夹，pb模型文件
#输出：文件预测图
#存在问题：矩阵操作繁琐
#软件版本：win10,python3.5,tensorflow1.6

import tensorflow as tf
import numpy as np
import os
from PIL import Image


Image.MAX_IMAGE_PIXELS = 400000000
imagedir = './img'
savepath = './res'
imagelist = os.listdir(imagedir)
rgbim = np.zeros((513,513,3), 'uint8')
R = rgbim[:,:,0]
G = rgbim[:,:,1]
B = rgbim[:,:,2]

img = tf.placeholder(tf.uint8, [1,334, 540, 3], name="img")
with open("./checkpoint/frozen_inference_graph.pb", "rb") as f:
	graph_def = tf.GraphDef()
	graph_def.ParseFromString(f.read())
	output = tf.import_graph_def(graph_def, input_map={"ImageTensor:0": img},
									return_elements=["SemanticPredictions:0"])

sess = tf.Session()

for i in range(0,len(imagelist)):
	path = os.path.join(imagedir,imagelist[i])
	filename = os.path.splitext(imagelist[i])[0]
	filetype = os.path.splitext(imagelist[i])[1]
	print('正在处理'+filename)
	image = Image.open(path)
	image = np.asarray(image)
	image = np.expand_dims(image, axis=0).astype(np.uint8)

	result = sess.run(output, feed_dict = {img:image})
	
	grayim = np.squeeze(result[0])

	#将灰度图像转换为rgb图像,根据自己的类别数目修改相应的颜色
	for k in range(0,500):
		for j in range(0,500):
			if grayim[k][j] == 1:
				R[k,j] = 128
				B[k,j] = 0
				G[k,j] = 0
			elif grayim[k][j] == 2:
				R[k,j] = 0
				B[k,j] = 128
				G[k,j] = 0
			elif grayim[k][j] == 3:
				R[k,j] = 128
				B[k,j] = 128
				G[k,j] = 0
			elif grayim[k][j] == 4:
				R[k,j] = 0
				B[k,j] = 0
				G[k,j] = 128
			elif grayim[k][j] == 0:
				R[k,j] = 0
				B[k,j] = 0
				G[k,j] = 0


	rgbim = Image.fromarray(np.uint8(rgbim))
	savedir = os.path.join(savepath,filename + '.png')
	rgbim.save(savedir)
	rgbim = np.zeros((500,500,3), 'uint8')
	R = rgbim[:,:,0]
	G = rgbim[:,:,1]
	B = rgbim[:,:,2]


