import tensorflow as tf
import numpy as np
from PIL import  Image
import time
import core.utils as utils

deep_input_size = 513
deep_graph = tf.get_default_graph()
img = tf.placeholder(tf.uint8, [1, 513, 513, 3], name="img")
with open('./checkpoint/frozen_inference_graph.pb', 'rb') as f:
    deep_graph_def = tf.GraphDef()
    deep_graph_def.ParseFromString(f.read())
with deep_graph.as_default():
    output = tf.import_graph_def(deep_graph_def, input_map={"ImageTensor:0": img},
									return_elements=["SemanticPredictions:0"])

#sess = tf.Session()
#with tf.Session(graph=deep_graph) as sess:
sess = tf.Session(graph=deep_graph)
while(True):
    start = time.time()
    image = Image.open('./img/123.jpg')
    resized_image = utils.deep_image_process(image, deep_input_size)
    batch_seg_map = sess.run(output, feed_dict={img: resized_image})
    seg_img = utils.label_to_color_image(batch_seg_map[0][0]).astype(np.uint8)
    print(type(seg_img))
    res = Image.fromarray(seg_img)
    end = time.time()
    print("total cost %.2f s" % (end - start))
    # res.show()



