
from numpy import random
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sc
from scipy.stats import norm, binom
from mpl_toolkits.mplot3d import Axes3D

radian_in_degrees = 180/np.pi



def vector_angle(v1, v2):
    return radian_in_degrees*np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))




def dotproduct(v1, v2):
    return np.dot(v1, v2)



def etothex(x):
    return np.e**x



def sigmoid(x):
    return 1/(1+(np.e**(-x)))



def naturallog(x):
    return np.log(x)



def plott(function):
    x = np.linspace(-10,10,100)
    y = function(x)
    plt.plot(x,y)



def make_grid(axis, limit=4, step =1, color='gray'):
    for x in np.arange(-limit, limit + step, step):
        axis.plot([x,x],[limit,-limit], color=color, lw=1)
    for y in np.arange(-limit, limit + step, step):
        axis.plot([limit, -limit],[y,y], color=color, lw=1)



def make_transformed_grid(axis, A, limit = 4, step = 1, color = 'skyblue'):
    xs = np.arange(-limit, limit + step, step)
    ys = np.arange(-limit, limit + step, step)

    for x in xs:
        points = np.array([[x, x], [-limit, limit]])
        transformed = A @ points
        axis.plot(transformed[0], transformed[1], color=color, lw=1)
    for y in ys:
        points = np.array([[-limit, limit], [y,y]])
        transformed = A @ points
        axis.plot(transformed[0], transformed[1], color=color, lw=1)



def gradient_descent(function, derivative, lr= 0.01, maxiter=100):
    point = random.randint(100)

    for _ in range(maxiter):
        point -= derivative(point)*lr


    return point







def covarience(dataset):

    return (dataset.T@dataset)/(dataset.shape[0]-1)



def plotdataset(axis, dataset, color='red'):
    transdataset = dataset.T
    if dataset.ndim != 1:
        for i in range(transdataset.shape[1]):
            axis.scatter(transdataset[0,i], transdataset[1,i], color=color)









def leastsquares(dataset):
    X_data = dataset[:,0]
    Y_data = dataset[:,1]
    X = np.column_stack((np.ones(len(X_data)), X_data))
    beta = np.linalg.inv(X.T @ X) @ X.T @ Y_data
    expected_y = X @ beta
    return expected_y




def rotationmatrix(angle):

    return np.array([[np.cos(angle), -np.sin(angle)],
                    [np.sin(angle), np.cos(angle)]  ])


def linear_regression():
    np.random.seed(0)
    iterations = 100000
    learning_rate = 0.0001
    true_w, true_b = 0.1 ,5
    size = 10
    X = np.linspace(0, size, size*10)
    y = ((true_w * X)  + true_b) + np.random.randn(100) * 2
    w = 0.0
    b = 0.0
    n = len(X)
    for i in range(iterations):
      expected_y = w * X + b
      error = y - expected_y
      dw = (-2/n)* np.sum(X * error)
      db = (-2/n)*np.sum(error)
      w = w - learning_rate*dw
      b = b - learning_rate*db
      if i % 100 == 0:
          cost = np.mean(error ** 2)
          print(f"Epoch {i}: w={w:.4f}, b={b:.4f}, cost={cost:.4f}")


    w  = np.linspace(-5,5,50)
    b  = np.linspace(-5,5,50)
    W, B = np.meshgrid(w, b)

    J=np.zeros_like(W)
    for i in range(W.shape[0]):
        for j in range(B.shape[1]):
            pred_y = W[i,j] * X + B[i,j]
            J[i,j] = np.mean((y - pred_y)**2)
    return X, J


def EIGEN_DECOMPOSITION(dataset):
    return np.linalg.eig(dataset)

def CENTER_DATA(dataset):
    center = dataset - np.mean(dataset)
    return center


def PCA(dataset, top_k):
    original_dataset = dataset
    
    centered_data = CENTER_DATA(dataset)

    covarience_matrix = centered_data.T @ centered_data

    eigenvalues, eigenvectors = EIGEN_DECOMPOSITION(covarience_matrix)

    accuracy = np.array([(eigenvalues/np.sum(eigenvalues))*100])

    sorted_indices = np.flip(np.argsort(eigenvalues))
    
    V_sorted = eigenvectors[: , sorted_indices]

    top_V = V_sorted[:, :top_k]

    PCA_data = centered_data @ top_V
    

    

    return accuracy, PCA_data


# Dataset for PCA
dataset = np.array([[7,5,4],
                    [3,6,9],
                    [1,6,2],
                    [7,3,2],
                    [5,2,3],
                    [5,2,4],
                    [1,7,4],
                    [3,5,4]])

# Accuracy of principal components / data points after PCA
accuracy, points = PCA(dataset,2)


fig = plt.figure(figsize=(12,5))

ax_3d = fig.add_subplot(1, 2, 2, projection='3d')
ax_2d = fig.add_subplot(1, 2, 1)


ax_3d.set_xlim(10)
ax_3d.set_ylim(10)
ax_3d.set_zlim(10)
ax_3d.scatter(dataset[:, 0], dataset[:,1], dataset[:,2])
ax_3d.set_title('Data before PCA')


ax_2d.set_xlim(-6, 6)
ax_2d.set_xlabel('PC1')
ax_2d.set_ylim(-6, 6)
ax_2d.set_ylabel('PC2')
ax_2d.axvline(color ='gray', alpha = 0.5)
ax_2d.axhline(color ='gray', alpha = 0.5)
ax_2d.scatter(points[:, 0], points[:, 1], color = 'tan')
ax_2d.set_title('Data after PCA')







print(PCA(dataset,2))





plt.show()

