import os
import urllib
from zipfile import ZipFile
import PIL
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Sequential
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.layers import Conv2D, BatchNormalization, UpSampling2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
import tensorflow.python.keras.backend as K
from tensorflow.keras.regularizers import l2

from sklearn.model_selection import train_test_split


def mean_dice(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    dice = K.mean((2. * intersection + smooth) / (union + smooth), axis=0)
    return dice


def decodeImgs(encoded_batch):
    decodedBatch = []
    for k in range(0, encoded_batch.shape[0]):
        decodedBatch.append(np.argmax(encoded_batch[k], axis=-1))
    return np.asarray(decodedBatch)


def download(url, name='wbc.zip'):
    if os.path.isfile('./' + name):
        return

    # download file
    dir = './'
    filename = os.path.join(dir, name)
    if not os.path.isfile(filename):
        urllib.request.urlretrieve(url, filename)

    # unzip downloaded file
    with ZipFile(name, 'r') as zipObj:
        zipObj.extractall()


def load_images(dir='./segmentation_WBC-master/Dataset 1/'):
    screens = []
    masks = []
    # foreach img in folder
    list = os.listdir(dir)
    # sort files names in list
    list.sort()
    for f in list:
        img_format = os.path.splitext(f)[1]
        img = PIL.Image.open(dir + f)
        img.load()
        img = img.resize((128, 128),  PIL.Image.NEAREST)
        img = np.asarray(img)
        if img_format == '.bmp':
            screens.append(img)
        else:
            masks.append(img)
    return np.asarray(screens), np.asarray(masks)


def batch_show(A1, A2, A3):
    cols = 3
    rows = 5
    for j in range(1):
        fig = plt.figure(1, figsize=(10, 8))
        for i in range(j*rows, (j*rows)+rows):
            ax1 = fig.add_subplot(rows, cols, ((i - j * rows) * cols) + 1)
            plt.imshow(A1[i])
            ax2 = fig.add_subplot(rows, cols, ((i - j * rows) * cols) + 2)
            plt.imshow(A2[i])
            ax3 = fig.add_subplot(rows, cols, ((i - j * rows) * cols) + 3)
            plt.imshow(A3[i])
            if i == 0:
                ax1.set(title='screens')
                ax2.set(title='gt masks')
                ax3.set(title='predictions')
        plt.show()
        plt.clf()


def metric_plot(test, train, epochs):
    x = range(1, epochs+1)
    plt.figure()
    plt.title("Training and test MeanDICE")
    plt.plot(x, train, 'y', label='train')
    plt.plot(x, test, 'r', label='test')
    plt.ylabel('dice coeff')
    plt.xlabel('epochs')
    plt.legend()
    plt.show()


num_classes = 3
epochs = 20
batch_size = 2
show_interval = 2

model = Sequential([
    Conv2D(filters=16, kernel_size=(5, 5), padding='same',
           activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    MaxPool2D(pool_size=(2, 2)),
    Conv2D(filters=32, kernel_size=(5, 5), padding='same',
           activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    MaxPool2D(pool_size=(2, 2)),

    Conv2D(filters=32, kernel_size=(5, 5), padding='same',
           activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    UpSampling2D(size=(2, 2)),
    Conv2D(filters=16, kernel_size=(5, 5), padding='same',
           activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    UpSampling2D(size=(2, 2)),

    Conv2D(filters=num_classes, kernel_size=(1, 1), activation='softmax'),
])

# download and unzip dataset
filepath = 'wbc.zip'
download('https://github.com/zxaoyou/segmentation_WBC/archive/master.zip', name=filepath)

# load and standartize dataset
screens, masks = load_images()
print(screens.shape)
print(masks.shape)

# normalize, categorize and split dataset
screens = screens / 255.
np.place(masks, masks == 128, [1])
np.place(masks, masks == 255, [2])
(train_x, test_x, train_y, test_y) = train_test_split(
    screens, masks, test_size=0.3)

# compile model
opt = Adam()
loss = CategoricalCrossentropy()
model.compile(optimizer=opt, loss=loss, metrics=[mean_dice])

his_dice_test = []
his_dice_train = []

# fit model
for i in range(1, epochs+1):
    print('Epoch:', i)
    train_loss = 0
    train_dice = 0

    # permute the indexes to  get random batch
    batch_indexes = np.random.permutation(len(train_x))
    # one epoch fitting
    for k in range(0, len(train_x)-batch_size, batch_size):
        # train model by current batch
        batch_x = train_x[batch_indexes[k:(k+batch_size)]]
        batch_y = to_categorical(
            train_y[batch_indexes[k:(k+batch_size)]], num_classes=num_classes)
        his = model.train_on_batch(batch_x, batch_y)
        train_loss += his[0]

        # evaluate DICE
        batch_preds = model.predict(batch_x)
        train_dice += his[1]

    print('Train CCE loss:', train_loss * batch_size/len(train_x))
    print('Train DICE metric:', train_dice * batch_size/len(train_x))
    his = model.evaluate(test_x, to_categorical(test_y), verbose=0)
    his_dice_train.append(train_dice * batch_size/len(train_x))
    his_dice_test.append(his[1])
    print('Test DICE metric:', his[1])
    print('---------------------------------------')

    # each N-th epoch show test predictions
    if i % show_interval == 0:
        outputs = model.predict(test_x)
        decoded = decodeImgs(outputs)
        # show some screens/masks/preds
        batch_show(test_x, test_y, decoded)

metric_plot(his_dice_test, his_dice_train, epochs)
