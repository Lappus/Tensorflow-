import tensorflow_datasets as tfds 
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math 
import datetime 
import os
import shutil
import tensorboard


(train_ds, test_ds), ds_info = tfds.load('cifar10', split=['train','test'], as_supervised=True, with_info=True)

# print(ds_info)
# 32x32 with 10 classes 
# 50000 training images 10000 test images
# image shape = 32 x 32 x 3 dtype=unit8 --> 256 pixel value range

# tfds.show_examples(train_ds, ds_info)

def prepare_cifar10_data(cifar10):
	#flatten the images (2d höhe und breite) into vectors (1d)
    #cifar10 = cifar10.map(lambda img, target: (tf.reshape(img, (-1,)), target)) 
	#convert data from uint8 to float32									
    cifar10 = cifar10.map(lambda img, target: (tf.cast(img, tf.float32), target))
	#rescale the image values from range [0, 255] to [-1, 1]
    cifar10 = cifar10.map(lambda img, target: ((img/128.)-1., target))
	#create one-hot targets
    cifar10 = cifar10.map(lambda img, target: (img, tf.one_hot(target, depth=10)))
	#cache this progress in memory, as there is no need to redo it
    cifar10 = cifar10.cache()
	#shuffle, batch = how many data samples are given at once, prefetch
    cifar10 = cifar10.shuffle(1000)
    cifar10 = cifar10.batch(32) 
    cifar10 = cifar10.prefetch(20)
    return cifar10

train_dataset = train_ds.apply(prepare_cifar10_data)
test_dataset = test_ds.apply(prepare_cifar10_data)

def try_model(model, ds):
    for x, t in ds.take(5):
        y = model(x)

# class DenslyConnectedCNNLayer(tf.keras.layers.Layer):
#   def __init__ (self, num_filters):
#        super(DenslyConnectedCNNLayer, self).__init__()
#        self.conv = tf.keras.layers.Conv2D(filters=num_filters, kernel_size=3, padding='same', activation='relu')

class DenslyConnectedCNNLayer(tf.keras.layers.Layer):
    def __init__ (self, num_filters, bottleneck_size):
        super(DenslyConnectedCNNLayer, self).__init__()
        self.bottleneck = tf.keras.layers.Conv2D(filters=bottleneck_size, kernel_size=1, padding='same')
        self.conv = tf.keras.layers.Conv2D(filters=num_filters, kernel_size=3, padding='same', activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(filters=num_filters, kernel_size=3, padding='same', activation='relu')

    @tf.function
    def call(self, x):
        c = self.bottleneck(x)
        c = self.conv(c)
        c = self.conv2(c)
        #c = self.conv(x)
        x = tf.concat((c,x), axis=-1)
        return x

class DenslyConnectedCNNBlock(tf.keras.layers.Layer):
    def __init__ (self, num_filters, num_layers, bottleneck_size):
         super(DenslyConnectedCNNBlock, self).__init__()
         self.layers = [DenslyConnectedCNNLayer(num_filters, bottleneck_size) for _ in range(num_layers)]

    #def __init__ (self, num_filters, num_layers):
     #   super(DenslyConnectedCNNBlock, self).__init__()
      #  self.layers = [DenslyConnectedCNNLayer(num_filters) for _ in range(num_layers)]

    @tf.function
    def call(self, x):
        for layer in self.layers:
            x = layer(x)
            return x

class DenslyConnectedCNN(tf.keras.layers.Layer):
    def __init__ (self):
        super(DenslyConnectedCNN, self).__init__()

        self.denseblock1 = DenslyConnectedCNNBlock(32,10,32)
        # self.denseblock1 = DenslyConnectedCNNBlock(32,10)  #shape: [batch_size, 32, 32, 10]
        self.pooling1 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2) #shape [batch_size, 10]

        self.denseblock2 = DenslyConnectedCNNBlock(32,30,32)
        # self.denseblock2 = DenslyConnectedCNNBlock(32,30) #shape: [batch_size, 32, 32, 10]
        self.pooling2 = tf.keras.layers.MaxPool2D()

        self.denseblock3 = DenslyConnectedCNNBlock(32,60,32)
        # self.denseblock3 = DenslyConnectedCNNBlock(32,60) #shape: [batch_size, 32, 32, 10]
        self.globalpooling = tf.keras.layers.GlobalAveragePooling2D()
        self.out = tf.keras.layers.Dense(10, activation='softmax')

    @tf.function
    def call(self, x):
        x = self.denseblock1(x)
        x = self.pooling1(x)
        x = self.denseblock2(x)
        x = self.pooling2(x)
        x = self.denseblock3(x)
        x = self.globalpooling(x)
        x = self.out(x)

        return x

config_name = "model_log"
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

train_loss_log_path = f"log/{config_name}/{current_time}/train_loss"
train_accuracy_log_path = f"log/{config_name}/{current_time}/train_accuracy"
val_loss_log_path = f"log/{config_name}/{current_time}/val_loss"
val_accuracy_log_path = f"log/{config_name}/{current_time}/val_accuracy"

train_loss_writer = tf.summary.create_file_writer(train_loss_log_path)
train_accuracy_writer = tf.summary.create_file_writer(train_accuracy_log_path)
val_loss_writer = tf.summary.create_file_writer(val_loss_log_path)
val_accuracy_writer = tf.summary.create_file_writer(val_accuracy_log_path)

def train_step(model, input, target, loss_function, optimizer):
    with tf.GradientTape() as tape:
        prediction = model(input)
        loss = loss_function(target, prediction)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss

def test(model, test_data, loss_function):
    
    test_accuracy_aggregator = []
    test_loss_aggregator = []

    for (input, target) in test_data:
        prediction = model(input)
        sample_test_loss = loss_function(target, prediction)
        sample_test_accuracy = np.argmax(target, axis=1) == np.argmax(prediction, axis=1)
        sample_test_accuracy = np.mean(sample_test_accuracy)
        test_loss_aggregator.append(sample_test_loss.numpy())
        test_accuracy_aggregator.append(sample_test_accuracy)

    test_loss = tf.reduce_mean(test_loss_aggregator)
    test_accuracy = tf.reduce_mean(test_accuracy_aggregator)

    return test_loss, test_accuracy

dense_model = DenslyConnectedCNN()
epochs = 10
learning_rate = 0.001
cross_entropy_loss = tf.keras.losses.CategoricalCrossentropy()
# optimizer = tf.keras.optimizers.SGD(learning_rate, momentum=0.9)
optimizer = tf.keras.optimizers.Adam(learning_rate)

train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

train_loss, train_accuracy = test(dense_model, train_dataset, cross_entropy_loss)
train_losses.append(train_loss)
train_accuracies.append(train_accuracy)

test_loss, test_accuracy = test(dense_model, test_dataset, cross_entropy_loss)
test_losses.append(test_loss)
test_accuracies.append(test_accuracy)

for epoch in range(epochs):
    print(f'Epoch {str(epoch)}: starting with accuracy {test_accuracies[-1]}')
    epoch_loss_aggregator = []
    train_accuracy_aggregator = []

    for input, target in train_dataset:
        prediction = dense_model(input)
        train_loss = train_step(dense_model, input, target, cross_entropy_loss, optimizer)
        epoch_loss_aggregator.append(train_loss)
        sample_train_accuracy = np.argmax(target, axis=1) == np.argmax(prediction, axis=1)
        sample_train_accuracy = np.mean(sample_train_accuracy)
        train_accuracy_aggregator.append(sample_train_accuracy)

    train_accuracy = tf.reduce_mean(train_accuracy_aggregator)
    train_losses.append(tf.reduce_mean(epoch_loss_aggregator))
    # train_accuracies.append(train_accuracy)

    with train_loss_writer.as_default():
        tf.summary.scalar('Train_loss', train_losses[epoch], step=epoch)

    with train_accuracy_writer.as_default():
        tf.summary.scalar('Train_accuracy', train_accuracy, step=epoch)

    test_loss, test_accuracy = test(dense_model, test_dataset, cross_entropy_loss)
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)

    with val_loss_writer.as_default():
            tf.summary.scalar('Test_loss', test_losses[epoch], step=epoch)

    with val_accuracy_writer.as_default():
            tf.summary.scalar('Test_accuracy', test_accuracies[epoch], step=epoch)
    

train_loss_writer = tf.summary.create_file_writer(train_loss_log_path)
train_accuracy_writer = tf.summary.create_file_writer(train_accuracy_log_path)
val_loss_writer = tf.summary.create_file_writer(val_loss_log_path)
val_accuracy_writer = tf.summary.create_file_writer(val_accuracy_log_path)

#plt.figure()
#line1, = plt.plot(train_losses)
#line2, = plt.plot(test_losses)
#line3, = plt.plot(test_accuracies)
#plt.xlabel("Training steps")
#plt.ylabel("Loss/Accuracy")
#plt.legend((line1,line2, line3),("training","test", "test accuracy"))
#plt.show()

#try_model(dense_model, train_ds)



# First Try:
# 3 Blocks with each 32*4 bottleneck with 32 filter  structure, Learning Rate = 0.001, optimizer = Adam
# Second Try:
# 2 Blocks with each 32*10 bottleneck with 32 filter and 1 block 32*10 with bottleneck 64 filter structure, Learning Rate = 0.001, optimizer = Adam
# Third try:
# changing the learning Rate to = 0.01
# Fourth try:
# removing the bottleneck and using learning Rate = 0.001
# Fith try:
# using optimizer = SDG with everything else from try four
