import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
x = 0
while x < 4:
    print(x)
    x += 1
    print(x)

def sort (x,y):
    if x < y:
        return y,x
    else:
        return x,y

print(sort(1,3).__class__)
def foo():
    global x
    x = x+1
    return x

def sigmoid(z):
    z = float(z)
    sig = tf.sigmoid(z)
    with tf.Session() as session:
        result = session.run(sig)
    return result
#print(sigmoid(0))
r = np.array([0,0,255,255,0,0,0])
R = np.array([r,r,r,r,r,r])
G = R.T
B = np.zeros((6,6))
pic = np.array([R,G,B])
arr = [0,-3,-3,3,3,0]
a = np.array([arr,arr,arr,arr,arr,arr])
kernel = a + a.T
x = [arr[i] for i in range(3)]
print(x)
#plt.imshow(pic)
print(kernel)




