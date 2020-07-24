import numpy as np
import time


def sig(z):  # input z as a matrix, output a as sigmoid(z)
    exp = np.exp(-z)
    return 1/(1 + exp)

w = np.random.rand(3, 4)
print(np.shape(w.sum(axis=0)), w.sum(axis=0))
print(np.shape(w.sum(axis=0).reshape(1, 4)), w.sum(axis=0).reshape(1, 4))
pi = np.pi
s = np.linspace(0,2*pi,100)
print(np.sin(s))
x = np.random.rand(4, 5)
E = np.array([1,1,1,1])
E2 = np.zeros((4,1))
print(np.shape(np.dot(w, E)))
print(np.shape(np.dot(w, E2)))


'''print(np.shape(np.random.rand(1000)))'''
start = time.time()
result = np.dot(w, x)
end = time.time()

print("total time " + str(start*100000000-end*100000000) + " s with result " + str(result))

start = time.time()
result2 = np.zeros((3, 5))
for rows in range(0,3):
    for col in range(0,5):
        result2[rows][col] = w[rows] @ x[:, col]

print("result2: " + str(result2))

t = np.array([[1,2],[3,4]])
a = sig(t)
print(str(a))


