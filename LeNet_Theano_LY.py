####################### import packages ############################

import os
import time
import csv
import shelve
from datetime import datetime

import numpy as np
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle

import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d
print theano.config.device # We're using CPUs (for now)
#print theano.config.floatX # Should be 64 bit for CPUs

np.random.seed(0)

####################### import data ############################
FTRAIN = '../Data/FKD_Train.csv'
FTEST = '../Data/FKD_Test.csv'

def load(test=False, cols=None):
    """Loads data from FTEST if *test* is True, otherwise from FTRAIN.
    Pass a list of *cols* if you're only interested in a subset of the
    target columns.
    """
    fname = FTEST if test else FTRAIN
    df = read_csv(os.path.expanduser(fname))  # load pandas dataframe

    # The Image column has pixel values separated by space; convert
    # the values to numpy arrays:
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

    if cols:  # get a subset of columns
        df = df[list(cols) + ['Image']]

    X = np.vstack(df['Image'].values) / 255.  # scale pixel values to [0, 1]
    X = X.astype(np.float32)

    if not test:  # only FTRAIN has any target columns
        y = df[df.columns[:-1]].values
        # scale target coordinates to [-1, 1] - need because we don't have bias on the net
        y = (y - 48) / 48  # 96/2=48
#        X, y = shuffle(X, y, random_state=42)  # shuffle train data
        y = y.astype(np.float32)
        shuffle = np.random.permutation(np.arange(X.shape[0]))
        X, y = X[shuffle], y[shuffle]
    else:
        y = None

    return X, y, np.array(df.columns[:-1])

X, y, y_name = load()
X_t, trash, junk = load(test=True)
print("X.shape == {}; X.min == {:.3f}; X.max == {:.3f}".format(X.shape, X.min(), X.max()))
print("y.shape == {}; y.min == {:.3f}; y.max == {:.3f}".format(y.shape, y.min(), y.max()))
print("X_t.shape == {}; X_t.min == {:.3f}; X_t.max == {:.3f}".format(X_t.shape, X_t.min(), X_t.max()))

####################### grouping data ############################
X_group, y_group = {}, {}
i = 0
for x,f in zip(X,y):
    picker = ~np.isnan(f)
    id = str.join(',', y_name[picker])
    if id not in X_group:
        X_group[id] = []
        y_group[id] = []
    X_group[id].append(x)
    y_group[id].append(f[picker])

feature_size = np.array([np.array(x[1]).shape for x in y_group.items()])
big_group = feature_size[:,0].argsort()
n_top = 7
print '\nimages # vs. features # for top %d groups:' %n_top
print feature_size[big_group][-n_top:]

X1 = np.reshape(X_group.items()[big_group[-1]][1], (-1, 1, 96, 96))
X2 = np.reshape(X_group.items()[big_group[-2]][1], (-1, 1, 96, 96))
y1 = np.array(y_group.items()[big_group[-1]][1])
y2 = np.array(y_group.items()[big_group[-2]][1])
feature1 = y_group.keys()[big_group[-1]].split(',')
feature2 = y_group.keys()[big_group[-2]].split(',')
x_t = np.reshape(X_t, (-1, 1, 96, 96))

print '1st training set: X1:%s - y1:%s, y1.min:%.3f, y1.max:%.3f' %(X1.shape, y1.shape, y1.min(), y1.max())
print '2nd training set: X2:%s - y2:%s, y2.min:%.3f, y2.max:%.3f' %(X2.shape, y2.shape, y2.min(), y2.max())
print 'testing set: x_t:%s' %str(x_t.shape)

####################### define LeNet for X1 training set ############################
## (1) Parameters
numHiddenNodes = 600
patchWidth = 3
patchHeight = 3
featureMapsLayer1 = 32
featureMapsLayer2 = 64
featureMapsLayer3 = 128

# For convonets, we will work in 2d rather than 1d.  The facial images are 96x96 in 2d.
imageWidth = 96
n_train = np.round(0.9*X1.shape[0])
train_X, train_y = X1[:n_train], y1[:n_train]
test_X, test_y = X1[n_train:], y1[n_train:]

# Convolution layers.
w_1 = theano.shared(np.asarray((np.random.randn(*(featureMapsLayer1, 1, patchWidth, patchHeight))*.01)))
w_2 = theano.shared(np.asarray((np.random.randn(*(featureMapsLayer2, featureMapsLayer1, patchWidth-1, patchHeight-1))*.01)))
w_3 = theano.shared(np.asarray((np.random.randn(*(featureMapsLayer3, featureMapsLayer2, patchWidth-1, patchHeight-1))*.01)))
# Fully connected NN. - 12x12 - dimension of L3 (11) plus bias (1)
w_4 = theano.shared(np.asarray((np.random.randn(*(featureMapsLayer3 * 12 * 12, numHiddenNodes))*.01)))
w_5 = theano.shared(np.asarray((np.random.randn(*(numHiddenNodes, train_y.shape[1]))*.01)))
params = [w_1, w_2, w_3, w_4, w_5]

## (2) Model
theano.config.floatX = 'float64'
X = T.tensor4() # conv2d works with tensor4 type
Y = T.matrix()

srng = RandomStreams()
def dropout(X, p=0.):
    if p > 0:
        X *= srng.binomial(X.shape, p=1 - p)
        X /= 1 - p
    return X

# Theano provides built-in support for add convolutional layers
def model(X, w_1, w_2, w_3, w_4, w_5, p_1, p_2):
    # T.maximum is the rectify activation
    l1 = dropout(max_pool_2d(T.maximum(conv2d(X, w_1, border_mode='full'), 0.), (2, 2)), p_1)
    l2 = dropout(max_pool_2d(T.maximum(conv2d(l1, w_2), 0.), (2, 2)), p_1)
    # flatten to switch back to 1d layers - with "outdim = 2" (2d) output
    l3 = dropout(T.flatten(max_pool_2d(T.maximum(conv2d(l2, w_3), 0.), (2, 2)), outdim=2), p_1)
    l4 = dropout(T.maximum(T.dot(l3, w_4), 0.), p_2)
    return T.dot(l4, w_5) #T.nnet.softmax(T.dot(l4, w_5))

#y_hat_train = model(X, w_1, w_2, w_3, w_4, w_5, 0.2, 0.5)
y_hat_train = model(X, w_1, w_2, w_3, w_4, w_5, 0., 0.)
y_hat_predict = model(X, w_1, w_2, w_3, w_4, w_5, 0., 0.)

## (3) Cost
#cost = T.sqrt(T.mean(T.sqr(Y - y_hat_train))) # T.mean(T.nnet.categorical_crossentropy(y_hat_train, Y))
cost = T.sum(T.sqr(Y - y_hat_train))

## (4) Minimization.
def backprop(cost, w, alpha=0.01, rho=0.8, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=w)
    updates = []
    for w1, grad in zip(w, grads):

        # adding gradient scaling
        acc = theano.shared(w1.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * grad ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        grad = grad / gradient_scaling
        updates.append((acc, acc_new))

        updates.append((w1, w1 - grad * alpha))
    return updates

update = backprop(cost, params)
train = theano.function(inputs=[X, Y], outputs=cost, updates=update, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=y_hat_predict, allow_input_downcast=True)

miniBatchSize = 1
def gradientDescentStochastic(epochs):
    print 'Training started @%s, for %d epochs with batch size %d' %(datetime.now(), epochs, miniBatchSize)
    print 'Training set: %s, dev set: %s' %(train_y.shape, test_y.shape)
    start_time = time.time()
    for i in range(epochs):
        for start, end in zip(range(0, len(train_X), miniBatchSize), range(miniBatchSize, len(train_X), miniBatchSize)):
            cost = train(train_X[start:end], train_y[start:end])
#             print 'cost: %.3f' %cost
        print '%d) %s: RMSE = %.4f' %(i+1, datetime.now(), np.sqrt(np.mean(np.square(test_y - predict(test_X)))))
    print 'Training completed in %.2f seconds' %(time.time() - start_time)

print 'LeNet model defined!\n'


####################### training and predciting ############################
gradientDescentStochastic(30)

print 'Predicting...'
start_time = time.time()
y_pred1 = predict(x_t)
print 'Predicting completed in %.2f seconds' %(time.time() - start_time)

####################### define LeNet for X2 training set ############################
## (1) Parameters
numHiddenNodes = 600
patchWidth = 3
patchHeight = 3
featureMapsLayer1 = 32
featureMapsLayer2 = 64
featureMapsLayer3 = 128

# For convonets, we will work in 2d rather than 1d.  The facial images are 96x96 in 2d.
imageWidth = 96
n_train = np.round(0.95*X2.shape[0])
train_X, train_y = X2[:n_train], y2[:n_train]
test_X, test_y = X2[n_train:], y2[n_train:]

# Convolution layers.
w_1 = theano.shared(np.asarray((np.random.randn(*(featureMapsLayer1, 1, patchWidth, patchHeight))*.01)))
w_2 = theano.shared(np.asarray((np.random.randn(*(featureMapsLayer2, featureMapsLayer1, patchWidth-1, patchHeight-1))*.01)))
w_3 = theano.shared(np.asarray((np.random.randn(*(featureMapsLayer3, featureMapsLayer2, patchWidth-1, patchHeight-1))*.01)))
# Fully connected NN. - 12x12 - dimension of L3 (11) plus bias (1)
w_4 = theano.shared(np.asarray((np.random.randn(*(featureMapsLayer3 * 12 * 12, numHiddenNodes))*.01)))
w_5 = theano.shared(np.asarray((np.random.randn(*(numHiddenNodes, train_y.shape[1]))*.01)))
params = [w_1, w_2, w_3, w_4, w_5]

## (2) Model
theano.config.floatX = 'float64'
X = T.tensor4() # conv2d works with tensor4 type
Y = T.matrix()

srng = RandomStreams()
def dropout(X, p=0.):
    if p > 0:
        X *= srng.binomial(X.shape, p=1 - p)
        X /= 1 - p
    return X

# Theano provides built-in support for add convolutional layers
def model(X, w_1, w_2, w_3, w_4, w_5, p_1, p_2):
    # T.maximum is the rectify activation
    l1 = dropout(max_pool_2d(T.maximum(conv2d(X, w_1, border_mode='full'), 0.), (2, 2)), p_1)
    l2 = dropout(max_pool_2d(T.maximum(conv2d(l1, w_2), 0.), (2, 2)), p_1)
    # flatten to switch back to 1d layers - with "outdim = 2" (2d) output
    l3 = dropout(T.flatten(max_pool_2d(T.maximum(conv2d(l2, w_3), 0.), (2, 2)), outdim=2), p_1)
    l4 = dropout(T.maximum(T.dot(l3, w_4), 0.), p_2)
    return T.dot(l4, w_5) #T.nnet.softmax(T.dot(l4, w_5))

#y_hat_train = model(X, w_1, w_2, w_3, w_4, w_5, 0.2, 0.5)
y_hat_train = model(X, w_1, w_2, w_3, w_4, w_5, 0., 0.)
y_hat_predict = model(X, w_1, w_2, w_3, w_4, w_5, 0., 0.)

## (3) Cost
#cost = T.sqrt(T.mean(T.sqr(Y - y_hat_train))) # T.mean(T.nnet.categorical_crossentropy(y_hat_train, Y))
cost = T.sum(T.sqr(Y - y_hat_train))

## (4) Minimization.
def backprop(cost, w, alpha=0.01, rho=0.8, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=w)
    updates = []
    for w1, grad in zip(w, grads):

        # adding gradient scaling
        acc = theano.shared(w1.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * grad ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        grad = grad / gradient_scaling
        updates.append((acc, acc_new))

        updates.append((w1, w1 - grad * alpha))
    return updates

update = backprop(cost, params)
train = theano.function(inputs=[X, Y], outputs=cost, updates=update, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=y_hat_predict, allow_input_downcast=True)

miniBatchSize = 1
def gradientDescentStochastic(epochs):
    print 'Training started @%s, for %d epochs with batch size %d' %(datetime.now(), epochs, miniBatchSize)
    print 'Training set: %s, dev set: %s' %(train_y.shape, test_y.shape)
    start_time = time.time()
    for i in range(epochs):
        for start, end in zip(range(0, len(train_X), miniBatchSize), range(miniBatchSize, len(train_X), miniBatchSize)):
            cost = train(train_X[start:end], train_y[start:end])
#             print 'cost: %.3f' %cost
        print '%d) %s: RMSE = %.4f' %(i+1, datetime.now(), np.sqrt(np.mean(np.square(test_y - predict(test_X)))))
    print 'Training completed in %.2f seconds' %(time.time() - start_time)

print 'LeNet model defined!\n'

####################### training and predciting ############################
gradientDescentStochastic(30)

print 'Predicting...'
start_time = time.time()
y_pred2 = predict(x_t)
print 'Predicting completed in %.2f seconds' %(time.time() - start_time)

####################### saving results ############################
def getSubmission(LookupTable):
    # create a dictionary for feature name indexing
    index2 = {feature2[x]:x for x in range(len(feature2))}
    index1 = {feature1[x]:x for x in range(len(feature1))}
    lookupRow = []
    with open(LookupTable) as csvfile:
        # read the lookup file
        lookupReader = csv.reader(csvfile, delimiter=',')
        lookupRow.append(lookupReader.next())
        for row in lookupReader:
            # get the prediction based on image ID and feature name, and attach to the row
            image_id, fea = int(row[1])-1, row[2]
            location = y_pred2[image_id, index2[fea]]
            if fea in index1:
                location = (location + y_pred1[image_id, index1[fea]])/2
            lookupRow.append(np.append(row, location))
    lookupRow = np.array(lookupRow)
    # save row ID and location ID columns only to the submission file
    saveFile = 'submission_'+datetime.now().strftime("%Y%m%d%H%M%S")+'.csv'
    with open(saveFile, 'wb') as f:
        writer = csv.writer(f)
        writer.writerows(lookupRow[:,[0,3]])
    print 'Submission file saved as: %s' %saveFile

getSubmission('../Data/FKD_IdLookupTable.csv')


####################### session ############################
####################### session ############################
####################### session ############################
