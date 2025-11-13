
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
dataset = np.array([[ 92.48357077,  88.30867849,  86.23844269],
       [ 97.61514928,  91.82923378,  82.89606407],
       [ 93.65846698,  88.43511652,  87.21326209],
       [ 96.01115012,  93.96147421,  89.19275092],
       [ 95.38509033,  89.06121434,  88.89882453],
       [ 93.11238843,  93.92730204,  87.46349436],
       [ 91.85438137,  85.67996842,  87.3532042 ],
       [ 91.76848411,  88.35393253,  81.83391593],
       [ 88.69856864,  83.47682618,  83.04760759],
       [ 87.35260228,  88.65846698,  85.04314312],
       [ 89.34402995,  86.46818939,  84.92227347],
       [ 90.12282368,  84.43093215,  87.04754805],
       [ 92.49598977,  91.20529907,  86.11249398],
       [ 91.44265326,  88.08981276,  85.15335103],
       [ 90.93778835,  86.59194527,  87.56420467],
       [ 74.42806413,  70.18460798,  60.17554426],
       [ 66.82923378,  64.50486315,  58.95678781],
       [ 68.43511652,  61.21326209,  61.89776207],
       [ 63.96147421,  63.19275092,  59.12341847],
       [ 69.06121434,  62.89882453,  63.11438873],
       [ 73.92730204,  68.46349436,  61.86744391],
       [ 65.67996842,  64.3532042 ,  60.45794302],
       [ 68.35393253,  65.83391593,  62.49842278],
       [ 63.47682618,  61.04760759,  58.73241968],
       [ 68.65846698,  66.04314312,  63.21326209],
       [ 66.46818939,  64.92227347,  61.73259832],
       [ 64.43093215,  63.04754805,  59.89523143],
       [ 71.20529907,  67.11249398,  62.38452013],
       [ 68.08981276,  65.15335103,  61.22349845],
       [ 66.59194527,  63.56420467,  60.94835761],
       [ 52.42806413,  46.18460798,  42.17554426],
       [ 56.82923378,  48.50486315,  45.95678781],
       [ 58.43511652,  49.21326209,  44.89776207],
       [ 53.96147421,  47.19275092,  42.12341847],
       [ 59.06121434,  50.89882453,  46.11438873],
       [ 53.92730204,  45.46349436,  43.86744391],
       [ 55.67996842,  46.3532042 ,  44.45794302],
       [ 58.35393253,  48.83391593,  45.49842278],
       [ 53.47682618,  44.04760759,  42.73241968],
       [ 58.65846698,  49.04314312,  46.21326209],
       [ 56.46818939,  47.92227347,  44.73259832],
       [ 54.43093215,  46.04754805,  43.89523143],
       [ 61.20529907,  50.11249398,  45.38452013],
       [ 58.08981276,  48.15335103,  44.22349845],
       [ 56.59194527,  46.56420467,  43.94835761],
       [ 92.0,  91.0,  85.0],  # possible high outlier
       [48.0,  44.0,  41.0],   # possible low outlier
       [65.0,  67.0,  63.0],
       [70.0,  60.0,  58.0],
       [55.0,  50.0,  45.0],
       [60.0,  55.0,  50.0],
       [75.0,  70.0,  65.0],
       [80.0,  75.0,  70.0]])    


# Accuracy of principal components / data points after PCA
accuracy, points = PCA(dataset,2)


fig = plt.figure(figsize=(12,5))

ax_3d = fig.add_subplot(1, 2, 2, projection='3d')
ax_2d = fig.add_subplot(1, 2, 1)


ax_3d.set_xlim(40,100)
ax_3d.set_ylim(40,100)
ax_3d.set_zlim(40,100)
ax_3d.scatter(dataset[:, 0], dataset[:,1], dataset[:,2])
ax_3d.set_title('Data before PCA')

ax_3d.set_xlabel('Math score')
ax_3d.set_ylabel('Physics score')
ax_3d.set_zlabel('Chemistry score')


ax_2d.set_xlim(-40, 45)
ax_2d.set_xlabel('PC1')
ax_2d.set_ylim(-40, 40)
ax_2d.set_ylabel('PC2')
#ax_2d.axvline(color ='gray', alpha = 0.5)
#ax_2d.axhline(color ='gray', alpha = 0.5)
ax_2d.scatter(points[:, 0], points[:, 1], color = 'tan')
ax_2d.set_title('Data after PCA')







print(PCA(dataset,2))





plt.show()

