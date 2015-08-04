############################################
#     W207: Summer 2015                    #
#     Final Project: Keypoint Recognition  #
#     Marguerite Oneto                     #
#     TEST                                 #
############################################

from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet

import os
import time
import csv
import shelve
from datetime import datetime

import numpy as np
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle

import matplotlib.pyplot as plt
from matplotlib import cm

np.random.seed(0)

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

#     print(df.count())  # prints the number of values for each column
#     df = df.dropna()  # drop all rows that have missing values in them

    X = np.vstack(df['Image'].values) / 255.  # scale pixel values to [0, 1]
    X = X.astype(np.float32)

    if not test:  # only FTRAIN has any target columns
        y = df[df.columns[:-1]].values
        # scale target coordinates to [-1, 1] - need because we don't have bias on the net
        y = (y - 48) / 48  # 96/2=48
#         X, y = shuffle(X, y, random_state=42)  # shuffle train data
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


def HistogramStretching(image):
#     a, b = min(image), max(image)
    a, b = np.percentile(image, 5), np.percentile(image, 95)
    l, u = 0, 1
    const = 1.0*(b*l - a*u)/(b - a)
    k = 1.0*(u-l)/(b-a)
    return [k*p+const for p in image]

X = [HistogramStretching(x) for x in X]
X_t = [HistogramStretching(x) for x in X_t]
print 'Histogram stretching completed!'


# define the Gaussian weights of neighbors as constant variable
sigma2 = 1.25**2
neighborIndex = [[i,j] for i in range(-1,2) for j in range(-1,2)]
gaussianWeight = np.array([np.exp(-(i**2+j**2)/(2*sigma2))/(2*np.pi*sigma2) for i,j in neighborIndex])
gaussianWeight = gaussianWeight / sum(gaussianWeight)

# function to return the index of neighborhood pixels for pixel at n
def getNeighborAndWeight(n, ncolumn, nrow):
    # get row and column id first for index i
    (r, c) = divmod(n, ncolumn)
    # get indices for the neighbors (including self)
    neighbors = [[r+i,c+j] for i,j in neighborIndex]
    # get neighbor index and the associated Gauusian weigth
    neighborWeights = []
    for nb, gw in zip(neighbors, gaussianWeight):  # range(len(neighbors)):
        r,c = nb
        if r>=0 and r<nrow and c>=0 and c<ncolumn:
            neighborWeights.append([r*ncolumn + c, gw])
    return neighborWeights

# apply Gaussian blur to one image
def gaussianBlurOneSample(x):
    y = np.empty(len(x))
    for i in range(len(x)):
        neighbors = getNeighborAndWeight(i,96,96)
        y[i] = sum([x[j[0]]*j[1] for j in neighbors])
    return y

start_time = time.time()
X = [gaussianBlurOneSample(x) for x in X]
X_t = [gaussianBlurOneSample(x) for x in X_t]
print 'Gaussian blur completed in %.f seconds!' %(time.time()-start_time)


# flip the image
X_flip = np.reshape(np.reshape(X, (-1,1,96,96))[:, :, :, ::-1], (-1, 96*96))

# flip the x coordinate value
multiplier = [-1,1]*(y.shape[1]/2)
y_flip = np.multiply([multiplier,]*y.shape[0], y)

# flip the x coordinates/column name
y_name_flip = []
for name in y_name:
    if 'left' in name.lower():
#         print name + ' --> ' + name.replace('left','right')
        y_name_flip.append(name.replace('left','right'))
    elif 'right' in name.lower():
#         print name + ' --> ' + name.replace('right','left')
        y_name_flip.append(name.replace('right','left'))
    else:
        y_name_flip.append(name)
y_name_flip=np.array(y_name_flip)
isort = [np.where(y_name_flip==x)[0][0] for x in y_name]
print 'Flipping images complete ...'

# combine data and align with original column
y = np.concatenate((y, y_flip[:, isort]), axis=0)
X = np.concatenate((X, X_flip), axis=0)
print 'After merge X:%s, y:%s' %(X.shape, y.shape)


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

print '1st training set: X1:%s - y1:%s, y1.min:%.3f, y1.max:%.3f' %(str(X1.shape), str(y1.shape), y1.min(), y1.max())
print '2nd training set: X2:%s - y2:%s, y2.min:%.3f, y2.max:%.3f' %(str(X2.shape), str(y2.shape), y2.min(), y2.max())
print 'testing set: x_t:%s' %str(x_t.shape)


# def plot(image, points, pred=[]):
#     # print a picture to see
#     plt.figure(figsize=(8, 8))
#     if len(image)==96:
#         plt.imshow(image, cmap = cm.gray)
#     else:
#         plt.imshow(np.reshape(image,(96,96)), cmap = cm.gray)
#     plt.axis('off')
#     if len(points)>0:
#         for i in range(len(points)/2):
#             plt.plot(points[2*i], points[2*i+1],'r.')
#     if len(pred)>0:
#         for i in range(len(pred)/2):
#             plt.plot(pred[2*i],pred[2*i+1],'c.')

# testing example
# ranges = [np.ptp(x) for x in X2]
# id = np.argsort(ranges)[0]
# plot(X2[0], 48*y2[0]+48)
# plot(X2[id], 48*y2[id]+48)
# id=13
# plot(X[id], 48*y[id]+48)
# plot(X[id+X.shape[0]/2], 48*y[id+X.shape[0]/2]+48)


net1 = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('conv1', layers.Conv2DLayer),
        ('pool1', layers.MaxPool2DLayer),
        ('conv2', layers.Conv2DLayer),
        ('pool2', layers.MaxPool2DLayer),
        ('conv3', layers.Conv2DLayer),
        ('pool3', layers.MaxPool2DLayer),
        ('hidden4', layers.DenseLayer),
        ('hidden5', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    input_shape=(None, 1, 96, 96),
    # 3 convoluational layer
    conv1_num_filters=32, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
    conv2_num_filters=64, conv2_filter_size=(2, 2), pool2_pool_size=(2, 2),
    conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_pool_size=(2, 2),
    # 2 fully connected hidden layer
    hidden4_num_units=500, hidden5_num_units=500,
    # fully connected output layer, no activation function to give continuous output
    output_num_units=y1.shape[1], output_nonlinearity=None,

    update_learning_rate=0.02,
    update_momentum=0.9,

    regression=True,
    max_epochs=30,
    verbose=1,
    )

start_time = time.time()
net1.fit(X1, y1)
print 'Net1 training completed in %.f seconds!' %(time.time()-start_time)

net2 = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('conv1', layers.Conv2DLayer),
        ('pool1', layers.MaxPool2DLayer),
        ('conv2', layers.Conv2DLayer),
        ('pool2', layers.MaxPool2DLayer),
        ('conv3', layers.Conv2DLayer),
        ('pool3', layers.MaxPool2DLayer),
        ('hidden4', layers.DenseLayer),
        ('hidden5', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    input_shape=(None, 1, 96, 96),
    # 3 convoluational layer
    conv1_num_filters=32, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
    conv2_num_filters=64, conv2_filter_size=(2, 2), pool2_pool_size=(2, 2),
    conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_pool_size=(2, 2),
    # 2 fully connected hidden layer
    hidden4_num_units=500, hidden5_num_units=500,
    # fully connected output layer, no activation function to give continuous output
    output_num_units=y2.shape[1], output_nonlinearity=None,

    update_learning_rate=0.01,
    update_momentum=0.9,

    regression=True,
    max_epochs=30,
    verbose=1,
    )

start_time = time.time()
net2.fit(X2, y2)
print 'Net2 training completed in %.f seconds!' %(time.time()-start_time)

start_time = time.time()
y_hat2 = net2.predict(x_t)*48+48 # rescale it back
print 'Net2 Prediction Time: %.2f sec, y_hat2.%s' %(time.time()-start_time, str(y_hat2.shape))

start_time = time.time()
y_hat1 = net1.predict(x_t)*48+48
print 'Net1 Prediction Time: %.2f sec, y_hat1.%s' %(time.time()-start_time, str(y_hat1.shape))

# group 2 has everything
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
            location = y_hat2[image_id, index2[fea]]
            if fea in index1:
                location = (location + y_hat1[image_id, index1[fea]])/2
            lookupRow.append(np.append(row, location))
    lookupRow = np.array(lookupRow)
    # save row ID and location ID columns only to the submission file
    saveFile = 'submission_'+datetime.now().strftime("%Y%m%d%H%M%S")+'.csv'
    with open(saveFile, 'wb') as f:
        writer = csv.writer(f)
        writer.writerows(lookupRow[:,[0,3]])
    print 'Submission file saved as: %s' %saveFile

getSubmission(‘../Data/FKD_IdLookupTable.csv')


# id = 1732
# plot(X_t[id],y_hat1[id],y_hat2[id])


# net0 = NeuralNet(
#     layers=[  # three layers: one hidden layer
#         ('input', layers.InputLayer),
#         ('hidden', layers.DenseLayer),
#         ('output', layers.DenseLayer),
#         ],
#     # layer parameters:
#     input_shape=(None, 9216),  # 96x96 input pixels per batch
#     hidden_num_units=100,  # number of units in hidden layer
#     output_nonlinearity=None,  # output layer uses identity function
#     output_num_units=30,  # 30 target values
#
#     # optimization method:
#     update=nesterov_momentum,
#     update_learning_rate=0.01,
#     update_momentum=0.9,
#
#     regression=True,  # flag to indicate we're dealing with regression problem
#     max_epochs=100,  # we want to train this many epochs
#     verbose=1,
#     )
#
# net0.fit(np.array(X_group.items()[big_group[-2]][1]), y2)
#
#
# train_loss = np.array([i["train_loss"] for i in net0.train_history_])
# valid_loss = np.array([i["valid_loss"] for i in net0.train_history_])
# # print train_loss
# plt.figure(figsize=(8, 8))
# plt.plot(train_loss, linewidth=3, label="train")
# plt.plot(valid_loss, linewidth=3, label="valid")
# plt.grid()
# plt.legend()
# plt.xlabel("epoch")
# plt.ylabel("loss")
# # plt.ylim(1e-3, 1e-1)
# plt.yscale("log")
# plt.show()


filename='./save2.out'
my_shelf = shelve.open(filename,'n') # 'n' for new

for key in dir():
    try:
        if key in ['y_hat1', 'y_hat2']:
            my_shelf[key] = globals()[key]
    except TypeError:
        #
        # __builtins__, my_shelf, and imported modules can not be shelved.
        #
        print('ERROR shelving: {0}'.format(key))
my_shelf.close()
print 'Saving complete!'


filename='./save1.out.db'
my_shelf = shelve.open(filename)
vNames = ""
for key in my_shelf:
    globals()[key]=my_shelf[key]
    vNames += key + ", "
my_shelf.close()
print 'loaded ' + vNames