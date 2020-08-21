import numpy as np
import os
from os.path import join
w = np.zeros((2, 3))
print(w.dtype)

x = np.ones((2, 3), dtype=int)

print(x.dtype)

s = x + np.ones((1, 3))
a = np.array([1,2,3])
for n in range(0,10):
    a = np.append(a, [4,5,6])
    print(a)
print(a.reshape((3, -1)))


a = np.concatenate((w, x, s, np.ones((1, 3))), axis=0)
print(a)

i = np.array( [ [0,1],                        # indices for the first dim of a
                 [1,2] ] )
j = np.array( [ [2,1],                        # indices for the second dim
                [3,3] ] )

print([i,j])
s = np.array([i,j])
print(s)
print(tuple(s))
x = np.array([1, 2, 3, 4, 5])
x[[1, 3, 4]] = 0
print(x)
# x[2,5] = 0
# print(x) error raised
a = np.arange(12).reshape(3, 4)
print(a)
ixgrid = np.ix_([True, False, True], [2, 4])
print(ixgrid)
print(np.ix_([1,3],[2,5]))
print(np.trace(np.arange(4).reshape((2,2))))
norm = np.random.normal(2, 0.01, 30)
print(norm)

a = np.arange(9).reshape((3, 3))
c = np.multiply(a,a)
print(c)
print(a*a)
x = np.arange(6).reshape((1,2,3))
print(x)
print(x.transpose((1,0,2)))
y = np.array([[1,2],[3,4]])
z = np.arange(4)+1
z = z.reshape((2,2))
if(np.all(y==z)):
    print('yes')
else:
    print('no')

y = np.arange(30720).reshape((10,3072))
print(y.reshape((-1,3,32,32)).transpose(0,2,3,1))
z = np.array([np.array([np.array([np.array([y[num,i+j+c*1024] for c in range(3)]) for j in range(32)]) for i in range(32)]) for num in range(y.shape[0])])
print(z.shape)
files = range(10000)
print('estimate: '+ str(len(files)/200) + ' min')
count = 1
p = 'f_out_path/'+'%d.png' % count
print(p.__class__)
print(p)
if p in ['f_out_path/1.png']:
    print(join('f_out_path', '%d.png') % count)