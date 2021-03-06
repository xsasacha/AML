from sklearn.datasets import load_digits
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

def load_data():
    X = np.empty((357,64))

    digits = load_digits()

    data = digits['data']
    images = digits['images']
    target = digits['target']
    target_names = digits["target_names"]

    index_list = []
    y = []

    for index in enumerate(digits.target):
        if index[1] == 3:
            index_list = np.append(index_list,index[0])
            y.append(1)
        elif index[1] == 8:
            index_list = np.append(index_list, index[0])
            y.append(-1)

    new_data = [digits.data[int(i)] for i in index_list]
    new_data = np.c_[new_data,np.ones(357)]
    new_target = [digits.target[int(i)] for i in index_list]

    return new_data, new_target, y

def sk_class(new_data, new_target):

    c_array = [0.0001,0.001,0.01,0.1,1.0]

    for i in c_array:
        logisticRegr = LogisticRegression(max_iter=200, C=i)
        score = cross_val_score(logisticRegr,new_data, new_target,cv=2)
        print("%0.2f accuracy with a standard deviation of %0.2f" % (score.mean(), score.std()))

Lambda = 1

def sigmoid(z):
    return 1/(1+np.exp(-z))


# What is c?
def gradient(beta, x, y):
    sum = np.log(1+np.exp(-y*x[0].dot(beta)))
    loss = (np.transpose(beta).dot(beta))/(2*c)+1/len(x)*np.sum(sum)
    grad = beta/c + 1/len(x)*np.sum((-y*x[0])/(np.exp(y*x[0].dot(beta))+1))
    return grad

def predict(beta, x):
    y_predict = []
    for i in x:
        if x[0].dot(beta) >= 0:
            y_predict.append(1)
        else:
            y_predict.append(-1)
    return predict

def zero_one_loss(y_predict, y_truth):
    return len(y_predict)/len(y_truth)


def gradient_descent(beta, X, y, tau0, mu=None, gamma=None, m=10, test=False):
    for t in range(m):
        beta -= tau0 * gradient(beta, X, y)
    return beta

def stochastic_gradient_descent(beta, X, y, tau0, gamma, mu=None, m=150, test=False):
    for t in range(m):
        index = np.random.choice(y.shape[0], size=1, replace=False)
        tau = tau0 / (1 + gamma * t)
        beta -= tau * gradient(beta, X[index], y[index])
    return beta

def sg_minibatch(beta, X, y, tau0, gamma, mu=None, m=150, batch_size=20, test=False):
    for t in range(m):
        B = np.random.choice(y.shape[0], size=batch_size, replace=False)
        tau = tau0 / (1 + gamma * t)
        beta -= tau * gradient(beta, X[B], y[B]) #edited
    return beta

def sg_momentum(beta, X, y, tau0, mu, gamma, m=150, test=False):
    g = np.zeros(beta.shape)
    for t in range(m):
        index = np.random.choice(y.shape[0], size=1, replace=False)
        tau = tau0 / (1 + gamma * t)
        g = mu * g + (1 - mu) * gradient(beta, X[index], y[index])
        beta -= tau * g
    return beta

def ADAM(beta, X, y, tau0 = 1e-4, mu = None, gamma = None, m=150, mu1 = 0.9, mu2 = 0.999, eps = 1e-8, test=False):
    g = np.zeros(beta.shape)
    q = np.zeros(beta.shape)
    for t in range(m):
        index = np.random.choice(y.shape[0], size=1, replace=False)
        l = gradient(beta, X[index], y[index])
        g = mu1 * g + (1 - mu1) * l
        q = mu2 * q + (1 - mu2) * l**2
        g_til = g / (1 - mu1 ** (t+1))
        q_til = q / (1 - mu2 ** (t+1))
        beta -= tau0 / (q_til**0.5 + eps ) * g_til
    return beta

def stochastic_average_gradient(beta, X, y, tau0, gamma, mu=None, m=150, test=False):

    gi_stored = (-y * X) * sigmoid(-y * np.dot(X, beta))
    g = np.mean(gi_stored, axis= 0, keepdims= True)

    for t in range(m):
        tau = tau0 / (1 + gamma * t)
        i = np.random.choice(y.shape[0], size=1, replace=False)
        gi = -y[i] * X[i] * sigmoid(-y[i] * np.dot(X[i], beta))
        g += (gi - gi_stored[i,:]) / y.shape[0]
        gi_stored[i,:] = gi
        beta = beta * (1 - tau / Lambda) - tau * g.T
    return beta

def dual_coordinate_ascent(beta, X, y, tau0=None, mu=None, gamma=None, m=150, eps = 0.01, test=False):
    N = y.shape[0]
    alpha = np.random.uniform(0, 1, y.shape)
    beta = Lambda / N * np.dot(X.T, y * alpha)
    for t in range(m):
        i = np.random.choice(y.shape[0], size=1, replace=False)
        f1 = y[i] * np.dot(X[i], beta) + np.log(alpha[i] / (1 - alpha[i]))
        f2 = Lambda / N * np.dot(X[i], X[i].T) + 1 / ((alpha[i])*(1 - alpha[i]))
        alpha_old = alpha[i]
        alpha[i] = np.clip(alpha[i] - f1 / f2, eps, 1 - eps)
        beta += (alpha[i] - alpha_old) * y[i] * X[i].T * Lambda / N
    return beta

def Newton_Raphson(beta, X, y, tau0=None, mu=None, gamma=None, m=10, test=False):
    for t in range(m):
        z = np.dot(X, beta)
        ytil = y / sigmoid(y * z)
        W = np.diag(((sigmoid(z) * sigmoid(-z)) * Lambda / X.shape[0]).reshape(-1,))
        beta = (np.linalg.inv(np.eye(X.shape[1]) + np.dot(X.T,W).dot(X))).dot(X.T).dot(W).dot(z + ytil)
    return beta

def main():
    return 0

if __name__ == "__main__":
    main()
'''
plt.figure(figsize=(20,4))

for index, (image, label) in enumerate(zip(new_data[0:10], digits.target[0:10])):
 plt.subplot(1, 10, index + 1)
 plt.imshow(np.reshape(image, (8,8)), cmap=plt.cm.gray)
 plt.title('Training: %i\n' % label, fontsize = 20)


plt.show()
'''








