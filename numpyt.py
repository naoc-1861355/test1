import numpy as np

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