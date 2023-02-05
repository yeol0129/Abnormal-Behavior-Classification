import cv2
import tensorflow as tf

new_model = tf.keras.models.load_model('/Users/siyeolkim/PycharmProjects/pythonProject4/Efficientnet_model/10-0.0338.hdf5')
labels=["assault","child","escalator_fall","person","public_intoxication","spy_camera","surrounding_fall","theft"]

im=cv2.imread('/Users/siyeolkim/Downloads/r0_276_1292_1005_w1200_h678_fmax.jpg')
print(im.shape)
im.resize(15,224,224,3)
print(im.shape)
r=new_model.predict(im,batch_size=32, verbose=1)

print(r)
res = r[0]
for i, acc in enumerate(res) :
    print(labels[i], "=", acc*100)
print("---")
print("result = " , labels[res.argmax()])