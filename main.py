from sklearn.datasets import load_digits
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score


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

c_array = [0.0001,0.001,0.01,0.1,1.0]

for i in c_array:
    logisticRegr = LogisticRegression(max_iter=200, C=i)
    score = cross_val_score(logisticRegr,new_data, new_target,cv=2)
    print("%0.2f accuracy with a standard deviation of %0.2f" % (score.mean(), score.std()))

'''plt.figure(figsize=(20,4))

for index, (image, label) in enumerate(zip(new_data[0:10], digits.target[0:10])):
 plt.subplot(1, 10, index + 1)
 plt.imshow(np.reshape(image, (8,8)), cmap=plt.cm.gray)
 plt.title('Training: %i\n' % label, fontsize = 20)


plt.show()'''







