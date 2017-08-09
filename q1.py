import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import time

class Tree:
    def __init__(self, data, left=None, right=None):
        self.data = data
        self.ba = None
        self.median = None
        self.left  = left
        self.right = right
        self.label = None

def predict(X, example):
    if(X.left == None and X.right == None):
        return (X.label)
    if(example[X.ba] <= X.median):
        return predict(X.left, example)
    else:
        return predict(X.right, example)

def find_accuracy(data, labels):
    global RootNode
    acc = 0
    size = np.shape(data)
    for i in range(size[0]):
        tmp = data[i,:]
        if(predict(RootNode, tmp) == labels[i]):
            acc = acc + 1
    return (acc * 100) / size[0]

def accuracy(y, pred):
    c = np.count_nonzero((y == pred))
    return (c * 100 / len(y))

def entropy(count):
    res = 0
    if(np.sum(count) != 0):
        probs = count / np.sum(count)
        for p in probs:
            res -= p * np.log2(p)
    return res

def conditional_entropy(Y, x):
    global med, y
    j = [x, 54]
    tmp = trainD[Y,:][:,j]
    y_xneg = tmp[np.argwhere(tmp[:,0] <= med[x]).flatten(),:][:,1]				#indices that go to the left child
    y_xpos = tmp[np.argwhere(tmp[:,0] > med[x]).flatten(),:][:,1]				#indices that go to the right child
    tmp = y[y_xpos]
    labels, count_pos = np.unique(tmp, return_counts = True)					#count of each class for y_xpos
    tmp = y[y_xneg]
    labels, count_neg = np.unique(tmp, return_counts = True)					#count of each class for y_xneg
    if(np.sum(count_pos) != 0 and np.sum(count_pos) != 0):						#checks if whole data goes to only one split
        prob_pos = np.sum(count_pos) / (np.sum(count_pos) + np.sum(count_neg))	#probability of positive split
        prob_neg = np.sum(count_neg) / (np.sum(count_pos) + np.sum(count_neg))	#probability of negative split
        res = (prob_pos * entropy(count_pos) + prob_neg * entropy(count_neg))
    else:
        res = np.Inf
    return res, y_xneg, y_xpos

def choose_best_attr(X, attr):
    ce, left, right = [],[],[]
    for x in attr:
        a, b, c = conditional_entropy(X.data, x)
        ce.append(a);left.append(b);right.append(c)
    idx = np.argmin(ce)									#chooses the attribute whose split gives minimum conditional entropy
    x = attr[idx]
    return x, left[idx], right[idx]

def id3(X, attr, idx_test, idx_valid):
    global nodes, trainD, testD, validD, y, y_test, y_valid, med, predTrain, predTest, predValid 
    tmp = y[X.data]
    labels, count = np.unique(tmp, return_counts = True)
    idx = np.argwhere(count != 0).flatten()
    if(len(idx) == 1):
        return X
    for j in range(10):
        med[j] = np.median(trainD[X.data,j])
    x, left, right = choose_best_attr(X, attr)
    print(nodes, x)
    X.ba = x
    X.median = med[x]
    X.left = Tree(left)
    X.right = Tree(right);
    tmp = y[left]
    labels, count = np.unique(tmp, return_counts = True) 
    X.left.label = labels[np.argmax(count)]
    tmp = y[right]
    labels, count = np.unique(tmp, return_counts = True) 
    X.right.label = labels[np.argmax(count)]
    predTrain[left] = X.left.label
    predTrain[right] = X.right.label
    j = [x, 54]
    tmp = testD[idx_test,:][:,j]
    test_left = tmp[np.argwhere(tmp[:,0] <= med[x]).flatten(),:][:,1]
    test_right = tmp[np.argwhere(tmp[:,0] > med[x]).flatten(),:][:,1]
    predTest[test_left] = X.left.label
    predTest[test_right] = X.right.label
    tmp = validD[idx_valid,:][:,j]
    val_left = tmp[np.argwhere(tmp[:,0] <= med[x]).flatten(),:][:,1]
    val_right = tmp[np.argwhere(tmp[:,0] > med[x]).flatten(),:][:,1]
    predValid[val_left] = X.left.label
    predValid[val_right] = X.right.label
    nodes = nodes + 2
    accuracy_matrix[1][[nodes,nodes+1]] = accuracy(y, predTrain)
    accuracy_matrix[2][[nodes,nodes+1]] = accuracy(y_test, predTest)
    accuracy_matrix[3][[nodes,nodes+1]] = accuracy(y_valid, predValid)
    if(x > 9):
        id3(X.left, np.setdiff1d(attr,np.array([x])), test_left, val_left)
        id3(X.right, np.setdiff1d(attr,np.array([x])), test_right, val_right)
    else:
        id3(X.left, attr, test_left, val_left)
        id3(X.right, attr, test_right, val_right)
    return X

start_time = time.time()
examples = np.asarray(pd.read_csv('train.dat', dtype=int, skiprows=0, delimiter=','))	#loading train data
trainD1 = examples[:, 0:(np.shape(examples)[1]-1)]
idx = np.array([x for x in range(np.shape(trainD1)[0])])
trainD = np.asarray(np.concatenate([trainD1, np.asmatrix(idx).T], axis=1))				#adding a column of indices 
y = examples[:,np.shape(examples)[1]-1]													#actual labels of train data

examples_test = np.asarray(pd.read_csv('test.dat', dtype=int, skiprows=0, delimiter=','))	#loading test data
testD1 = examples_test[:, 0:(np.shape(examples_test)[1]-1)]
idx = np.array([x for x in range(np.shape(testD1)[0])])
testD = np.asarray(np.concatenate([testD1, np.asmatrix(idx).T], axis=1))				#adding a column of indices
y_test = examples_test[:,np.shape(examples_test)[1]-1]									#actual labels of test data

examples_valid = np.asarray(pd.read_csv('valid.dat', dtype=int, skiprows=0, delimiter=','))	#loading validation data
validD1 = examples_valid[:, 0:(np.shape(examples_valid)[1]-1)]
idx = np.array([x for x in range(np.shape(validD1)[0])])
validD = np.asarray(np.concatenate([validD1, np.asmatrix(idx).T], axis=1))				#adding a column of indices
y_valid = examples_valid[:,np.shape(examples_valid)[1]-1]								#actual labels of validation data

accuracy_matrix = np.array([[i for i in range(90988)] for j in range(4)]).astype(float)	#matrix to hold accuracies with each growing node

root = np.arange(y.size)						#train data indices
testi = np.arange(y_test.size)			#test data indices
validi = np.arange(y_valid.size)			#validation data indices
RootNode = Tree(root)
tmp = y[root]
labels, count = np.unique(tmp, return_counts = True) 
RootNode.label = labels[np.argmax(count)]
predTrain = np.array([RootNode.label for i in range(len(y))])		#holds the predicted labels for training data
predTest = np.array([RootNode.label for i in range(len(y_test))])	#holds the predicted labels for test data
predValid = np.array([RootNode.label for i in range(len(y_valid))])	#holds the predicted labels for validation data
nodes = 0
accuracy_matrix[1][[nodes,nodes+1]] = accuracy(y, predTrain)
accuracy_matrix[2][[nodes,nodes+1]] = accuracy(y_test, predTest)
accuracy_matrix[3][[nodes,nodes+1]] = accuracy(y_valid, predValid)

med = np.zeros(54)							#holds the median values for the attributes
attr = np.arange(54)							#holds the attributes

dTree = id3(RootNode, attr, testi, validi)
print("Count:", nodes)
print("--- %s seconds ---" % (time.time() - start_time))

#----------------Prediction----------------
print("Training Data Accuracy: ", find_accuracy(trainD, y))
print("Test Data Accuracy: ", find_accuracy(testD, y_test))
print("Validation Data Accuracy: ", find_accuracy(validD, y_valid))

#----------------Plotting w.r.t. no. of nodes----------------
plt.plot(accuracy_matrix[0,:], accuracy_matrix[1,:],'b', label = 'Train Data Accuracy')
plt.plot(accuracy_matrix[0,:], accuracy_matrix[2,:], 'g', label = 'Test Data Accuracy')
plt.plot(accuracy_matrix[0,:], accuracy_matrix[3,:], 'r', label = 'Validation Data Accuracy')
plt.axis([0, 90000, 0, 100])
plt.xlabel('No. of nodes')
plt.legend(loc = 'upper left')
plt.show()

#---------------------------------------------------------------(b) Pruning---------------------------------------------------------

start_time = time.time()
def prune(X, idx_test, idx_valid):
    global validD, y_valid, testD, y_test, y, c
    c += 1
    print(c)
    tmp = y_valid[idx_valid]
    labels, count = np.unique(tmp, return_counts = True)
    i = np.argwhere(labels == X.label).flatten()
    if(len(i) == 0):
        n_valid = 0;
    else:
        n_valid = count[i]
    tmp = y_test[idx_test]
    labels, count = np.unique(tmp, return_counts = True)
    i = np.argwhere(labels == X.label).flatten()
    if(len(i) == 0):
        n_test = 0;
    else:
        n_test = count[i]
    tmp = y[X.data]
    labels, count = np.unique(tmp, return_counts = True)
    i = np.argwhere(labels == X.label).flatten()
    if(len(i) == 0):
        n_train = 0;
    else:
        n_train = count[i]
    if(X.left == None and X.right == None):
        return n_train, n_test, n_valid, 1
    else:
        j = [X.ba, 54]
        tmp = validD[idx_valid,:][:,j]
        val_left = tmp[np.argwhere(tmp[:,0] <= X.median).flatten(),:][:,1]		#validation data indices that should go to the left child
        val_right = tmp[np.argwhere(tmp[:,0] > X.median).flatten(),:][:,1]		#validation data indices that should go to the right child
        tmp = testD[idx_valid,:][:,j]
        test_left = tmp[np.argwhere(tmp[:,0] <= X.median).flatten(),:][:,1]		#test data indices that should go to the left child
        test_right = tmp[np.argwhere(tmp[:,0] > X.median).flatten(),:][:,1]		#test data indices that should go to the right child
        l_train, l_test, l_valid, l_count = prune(X.left, test_left, val_left)
        r_train, r_test, r_valid, r_count = prune(X.right, test_right, val_right)
        if(n_valid > (l_valid + r_valid)):											#condition for pruning
            X.left = None
            X.right = None
            n = nodes[len(nodes) - 1]; nodes.append(n - (l_count + r_count))
            n = train_acc[len(train_acc) - 1]; train_acc.append(n + (n_train - (l_train + r_train)))
            n = test_acc[len(test_acc) - 1]; test_acc.append(n + (n_test - (l_test + r_test)))
            n = valid_acc[len(valid_acc) - 1]; valid_acc.append(n + (n_valid - (l_valid + r_valid)))
            return n_train, n_test, n_valid, 1
        else:
            return (l_train + r_train), (l_test + r_test), (l_valid + r_valid), (l_count + r_count + 1)

c = 0
nodes, train_acc, test_acc, valid_acc = [90987], [accuracy_matrix[1][90987] * len(y) / 100], [accuracy_matrix[2][90987] * len(y_test) / 100], [accuracy_matrix[3][90987] * len(y_valid) / 100]
prune(RootNode, testi, validi)

print("--- %s seconds ---" % (time.time() - start_time))
train_acc = np.array(train_acc) * 100 / len(y)
test_acc = np.array(test_acc) * 100 / len(y_test)
valid_acc = np.array(valid_acc) * 100 / len(y_valid)
plt.plot(nodes, train_acc, 'b', label = 'Train Data Accuracy')
plt.plot(nodes, test_acc, 'g', label = 'Test Data Accuracy')
plt.plot(nodes, valid_acc, 'r', label = 'Validation Data Accuracy')
plt.axis([77000, 92000, 80, 102])
plt.xlabel('No. of nodes')
plt.legend(loc = 'lower right')
plt.show()

#--------------------------------------------(c) Decision Tree using SciKit Learn-------------------------------------------------------------

clf = DecisionTreeClassifier(criterion = 'entropy', min_samples_split = 2, min_samples_leaf = 1)
clf = clf.fit(trainD1, y)
print("No. of nodes: ", clf.tree_.node_count)
print("Training Data Accuracy: ", accuracy(y, clf.predict(trainD1)))
print("Test Data Accuracy: ", accuracy(y_test, clf.predict(testD1)))
print("Validation Data Accuracy: ", accuracy(y_valid, clf.predict(validD1)))

#--------------------------------------------(d) Random Forest using SciKit Learn-------------------------------------------------------

clf = RandomForestClassifier(criterion = 'entropy', n_estimators = 50, max_features = 'auto', bootstrap = False)
clf = clf.fit(trainD1, y)

print("Training Data Accuracy: ", accuracy(y, clf.predict(trainD1)))
print("Test Data Accuracy: ", accuracy(y_test, clf.predict(testD1)))
print("Validation Data Accuracy: ", accuracy(y_valid, clf.predict(validD1)))