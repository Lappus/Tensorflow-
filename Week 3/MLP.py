import tensorflow_datasets as tfds 
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense

(train_ds, test_ds), ds_info = tfds.load('mnist', split=['train', 'test'], as_supervised=True, with_info=True)

#print(ds_info)

max_value = 0
for image, label in train_ds:  # Take one image from the training dataset
    if tf.reduce_max(image) > max_value:
        max_value = tf.reduce_max(image)  # Find the maximum pixel value in the image
#print("Maximum pixel value:", max_value.numpy())
# How many training/test images are there?: Test = 10000 Training = 60000
# Image shape = 28 * 28 in greyscale
# dtype=uint8 --> pixel value range = 255

#tfds.show_examples (train_ds, ds_info)

def prepare_mnist_data(mnist):
	#flatten the images (2d höhe und breite) into vectors (1d)
    mnist = mnist.map(lambda img, target: (tf.reshape(img, (-1,)), target)) 
	#convert data from uint8 to float32									
    mnist = mnist.map(lambda img, target: (tf.cast(img, tf.float32), target))
	#rescale the image values from range [0, 255] to [-1, 1]
    mnist = mnist.map(lambda img, target: ((img/128.)-1., target))
	#create one-hot targets
    mnist = mnist.map(lambda img, target: (img, tf.one_hot(target, depth=10)))
	#cache this progress in memory, as there is no need to redo it
    mnist = mnist.cache()
	#shuffle, batch = how many data samples are given at once, prefetch
    mnist = mnist.shuffle(1000)
    mnist = mnist.batch(32) 
    mnist = mnist.prefetch(20)
    return mnist

class Model(tf.keras.Model):
    # determine how much layers we need with how many units per layer 
    # also establish the activation function
    def __init__ (self):
        super(Model, self).__init__()
        self.dense1 = tf.keras.layers.Dense(256, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(256, activation=tf.nn.relu)
        #self.dense3 = tf.keras.layers.Dense(256, activation=tf.nn.relu)
        self.out = tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    
    # now make a forward network
    # give die first layer into the second and the second (inluding the calculation for the first one) into the output layer
    @tf.function
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        #x = self.dense3(x)
        x = self.out(x)
        return x

def train_step(model, input, target, loss_function, optimizer):
    with tf.GradientTape() as tape:
        prediction = model(input)
        loss = loss_function(target, prediction)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# Test over the complete test data
def test (model, test_data, loss_function):

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

tf.keras.backend.clear_session()
# setting up the data and applying the Preparation 
train_dataset = train_ds.apply(prepare_mnist_data)
test_dataset = test_ds.apply(prepare_mnist_data)

model = Model()

# determine the parameters 
epochs = 5
learning_rate = 0.001

# train_dataset = train_dataset.take(1000)
# test_dataset = test_dataset.take(100)

# choose your lossfunction
cross_entropy_loss = tf.keras.losses.CategoricalCrossentropy()
# optimizers are alogrithms that adjust the models parameter regarding to the learning rate to minimalize the Lossfunction
optimizer = tf.keras.optimizers.SGD(learning_rate, momentum=0.9)

# for visualization 
train_losses = []
test_losses = []
test_accuracies = []

# Lets go!
# first we measure, how good the model is without training 
test_loss, test_accuracy = test(model, test_dataset, cross_entropy_loss)
test_losses.append(test_loss)
test_accuracies.append(test_accuracy)

# and how good it is an training data
train_loss, _ = test(model, train_dataset, cross_entropy_loss)
train_losses.append(train_loss)

# Lets go through the epochs of training
for epoch in range(epochs):
    print (f'Epoch {str(epoch)}: starting with accuracy {test_accuracies[-1]}')
    #training (and checking in with training)
    epoch_loss_aggregator = []
    # for every input - target pair in the train dataset -> train the modell and give us the loss
    for input, target in train_dataset:
        train_loss = train_step(model, input, target, cross_entropy_loss, optimizer)
        epoch_loss_aggregator.append(train_loss)

    train_losses.append(tf.reduce_mean(epoch_loss_aggregator))

    test_loss, test_accuracy = test(model, test_dataset, cross_entropy_loss)
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)


# Visualization
#def visualization (train_losses, test_losses, test_accuracies):
#    plt.figure()
#    line1, = plt.plot(train_losses, "b - ")
#    line2, = plt.plot(test_losses, "r - ")
#    line3, = plt.plot(test_accuracies, "r : ")
#    plt.xlabel("Training steps")
#    plt.ylabel("Loss/Accuracy")
#    plt.legend((line1, line2, line3), ("training loss", "test loss", "test accuracy"))
#    plt.show()

# visualization (train_losses, test_losses, test_accuracies)

plt.figure()
line1, = plt.plot(train_losses)
line2, = plt.plot(test_losses)
line3, = plt.plot(test_accuracies)
plt.xlabel("Training steps")
plt.ylabel("Loss/Accuracy")
plt.legend((line1,line2, line3),("training","test", "test accuracy"))
plt.show()


# Adjusting parameters:
# Learning rate: 
#       0.1 
        # pretty fast aggregation with good success. After only 1 epoch we already have ca. 96% accuracy
#       0.001
        # the loss is through all epochs bigger than with the higher learning rate. After one epoch we had 85% 
# That is, because we take a smaller propotion of the loss with a smaller learning rate. So our model adjusts itself slower. 

# Layer size: 
#       two layer with each 256 units vs three layer with each 256 units
        # two layer are already suffient. There is no significant difference between two or three layers. There is no gain in another layer
#       one layer with 256 units vs. two layer with each 256 units
        # there was again no difference in the efficiency

# SGD Optimizer momentum
#       0.0 vs 0.2 momentum vs 0.9 momentum
        # 0.2 momentum had a better accuracy than 0.0 but just by a tiny bit (0.01)
        # 0.9 momentum had the best overall results by over 0.5 higher accuracy 
        # a higher momentum helps in preventing the optimization process from getting stuck in local minima or oscillating around the minimum

# SGD Optimizer and learning rate
#       mometum 0.7 and learning rate 0.001
        # Epoch 0: starting with accuracy 0.08875798722044728
        # Epoch 1: starting with accuracy 0.9019568690095847
        # Epoch 2: starting with accuracy 0.9169329073482428
        # Epoch 3: starting with accuracy 0.9268170926517572
        # Epoch 4: starting with accuracy 0.9328075079872205
#       momentum 0.7 and learning rate 0.1
        # Epoch 0: starting with accuracy 0.11112220447284345
        # Epoch 1: starting with accuracy 0.862220447284345
        # Epoch 2: starting with accuracy 0.9130391373801917
        # Epoch 3: starting with accuracy 0.9122404153354633
        # Epoch 4: starting with accuracy 0.920826677316294
    # so with a higher momentum and a lower learning rate, the model actually had better accuracy through the iterations
    # so maybe because the lower learning rate is slowing down the process of updating of weights and biases but the momentum helps preventing local optima the combination can be quite strong
    # because there is no overshooting?
