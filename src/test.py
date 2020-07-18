import numpy as np,sys

from scipy.ndimage.filters import maximum_filter
import skimage.measure
from scipy.signal import convolve2d


np.random.seed(56789)

def ReLu(x):
    mask = (x>0) * 1.0 
    return x * mask
def d_ReLu(x):
    mask = (x>0) * 1.0 
    return  mask

def arctan(x):
    return np.arctan(x)
def d_arctan(x):
    return 1 / (1+ x ** 2)

def log(x):
    return 1 / ( 1 + np.exp( -1 *x))
def d_log(x):
    return log(x) * (1 - log(x))

x1 = np.array([
    [1,1,0,1,0,1],
    [1,1,0,1,0,1],
    [1,1,0,1,0,1],
    [1,1,1,1,0,1],
    [1,1,1,1,0,1],
    [1,1,1,1,0,1]    
])

x2 = np.array([
    [-1,0,-1,0,0,1],
    [-1,0,-1,0,0,1],
    [-1,0,-1,1,0,1],
    [-1,-1,-1,0,0,-1],
    [-1,0,-1,0,0,-1],
    [-1,0,-1,0,0,-1]    
])
X = np.array([x1,x2])
y = np.array([
    [arctan(x1.sum())],
    [arctan(x2.sum())]
])

num_epoch = 500
learing_rate = 0.01

w1 = np.random.randn(3,3) * 5.66
w2 = np.random.randn(4,1)* 5.7

prediction = np.array([])
for image_index in range(len(X)):
    
    current_image = X[image_index]
    current_label = y[image_index]

    l1 = convolve2d(current_image,w1,mode='valid')
    l1A = ReLu(l1)
    print(l1A)
    print("==================")
    l1M = skimage.measure.block_reduce(l1A, (2,2), np.max)
    print(l1M)
    print("==================a")
    l2IN = np.reshape(l1M,(1,4))
    print(l2IN)
    print("==================b")
    l2 = l2IN.dot(w2)
    print(l2)
    l2A = arctan(l2)
    print("c")
    print(l2A)

    prediction = np.append(prediction,l2A)
    print("==================")
print("--- Ground Truth -----")
print(y.T)
print("--- Before Training -----")
print(prediction.T)


for iter in range(num_epoch):
    
    for image_index in range(len(X)):
        
        current_image = X[image_index]
        current_label = y[image_index]

        l1 = convolve2d(current_image,w1,mode='valid')
        l1A = ReLu(l1)
        l1M = skimage.measure.block_reduce(l1A, (2,2), np.max)

        l2IN = np.reshape(l1M,(1,4))
        l2 = l2IN.dot(w2)
        l2A = arctan(l2)

        cost = np.square(l2A - current_label).sum() * 0.5
        # print("Current Iter: ", iter, " current cost :", cost ,end='\r')

        grad_2_part_1 = l2A - current_label
        grad_2_part_2 = d_arctan(l2)
        grad_2_part_3 = l2IN
        grad_2 = grad_2_part_3.T.dot(grad_2_part_1 * grad_2_part_2)

        grad_1_part_1 =  np.reshape((grad_2_part_1 * grad_2_part_2).dot(w2.T),(2,2))
        grad_1_mask =  np.equal(l1A, l1M.repeat(2, axis=0).repeat(2, axis=1)).astype(int) 
        grad_1_window = grad_1_mask * grad_1_part_1.repeat(2, axis=0).repeat(2, axis=1) 
        grad_1_part_2 = d_ReLu(l1)
        grad_1_part_3 = current_image
        grad_1 = np.rot90(convolve2d(grad_1_part_3,np.rot90(grad_1_window *grad_1_part_2,2 ),mode='valid'),2)

        w2 = w2 - learing_rate * grad_2
        w1 = w1 - learing_rate * grad_1


print("\n-------------")
prediction = np.array([])
for image_index in range(len(X)):
    
    current_image = X[image_index]
    current_label = y[image_index]

    l1 = convolve2d(current_image,w1,mode='valid')
    l1A = ReLu(l1)
    l1M = skimage.measure.block_reduce(l1A, (2,2), np.max)

    l2IN = np.reshape(l1M,(1,4))
    l2 = l2IN.dot(w2)
    l2A = arctan(l2)

    prediction = np.append(prediction,l2A)

print("--- Ground Truth -----")
print(y.T)
print("--- Before Training -----")
print(prediction.T)
