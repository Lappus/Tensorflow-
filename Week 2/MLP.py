import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

digits_set = load_digits()
data = digits_set.data
target = digits_set.target 

# Define the number of rows and columns for the grid of images
n_rows = 2
n_cols = 5

# Plot the first 10 images in the dataset
plt.figure(figsize=(10, 5))
for i in range(n_rows * n_cols):
    plt.subplot(n_rows, n_cols, i + 1)
    plt.imshow(data[i].reshape(8, 8), cmap='gray')
    plt.title(f"Label: {target[i]}")
    plt.axis('off')

plt.show()

data_reshaped = data.reshape(data.shape[0], -1)
# 8-bit Grayscale has 256 pixel (from 0 to 255)
data_reshaped = data_reshaped / 255.0

#One-Hot Encoding 
one_hot_encoded = []

for digits in target:
    one_hot = [0]*10
    one_hot[digits] = 1
    one_hot_encoded.append(one_hot)

def generator_shuffle(inputs, targets, size):
    num_samples = len(inputs)
    assert num_samples == len(targets)

    # Shuffle the data indices
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    for start in range(0, num_samples, minibatch_size):
        end = min(start + minibatch_size, num_samples)
        batch_indices = indices[start:end]

        # Create minibatch arrays for inputs and targets
        minibatch_inputs = inputs[batch_indices]
        minibatch_targets = targets[batch_indices]

        yield minibatch_inputs, minibatch_targets

class Sigmoid:
    def __init__(self):
        pass

    def call(input):
        return 1/(1+np.exp(-input))
    
    def backward(activation, gradient): #not included the preactivation 
        d_sigmoid = activation * (1 - activation)
        dL_dpre_activation = gradient * d_sigmoid
        return dL_dpre_activation
    
class Softmax:
    def __init__(self):
        pass

    def call(input):
        exp_inputs = np.exp(input)
        softmax_output = exp_inputs / np.sum(exp_inputs, axis=1, keepdims=True)
        return softmax_output
    
class Lossfunction:
    def __init___(self):
        pass

    def call(prediction, target):
        epsilon = 1e-15
        prediction = np.clip(prediction, epsilon, 1-epsilon)
        loss = -np.sum(target * np.log(prediction)) / prediction.shape[0]
        return loss 

    def backward(predicted_probs, true_labels):
        gradient = predicted_probs - true_labels
        return gradient
    
class MLP_Layer:
    def __init__(self, input_size, num_units, activation_function="sigmoid"):
        self.input_size = input_size
        self.num_units = num_units
        self.activation_function = activation_function
        
        # Initialize weights with small random values and bias with zeros
        self.weights = np.random.normal(loc=0.0, scale=0.2, size=(input_size, num_units))
        self.bias = np.zeros(num_units)

    def forward(self, input_data):
        # Compute the weighted sum and add bias
        weighted_sum = np.dot(input_data, self.weights) + self.bias

        # Apply the activation function
        if self.activation_function == "sigmoid":
            activation_output = Sigmoid.call(weighted_sum)
        elif self.activation_function == "softmax":
            activation_output = Softmax.call(weighted_sum)
        else:
            raise ValueError("Choose either sigmoid or softmax as an activationfunction")

        return activation_output
    
    def backward(self, activation, dL_dactivation):
        num_examples = len(activation)
        
        # Calculate dL/dW by multiplying dL/dpre_activation and dpre_activation/dW
        dL_dW = np.dot(activation.T.reshape(-1, 1), dL_dactivation.reshape(1, -1)) / num_examples

        # Calculate dL/dinput by multiplying dL/dpre_activation and dpre_activation/dinput
        dL_dinput = np.dot(dL_dactivation, self.weights.T)
        
        return dL_dW, dL_dinput

class MLP:
    def __init__(self, layer_sizes, activation_functions):
        
        self.layers = []
        for i in range(1, len(layer_sizes)):
            input_size = layer_sizes[i - 1]
            num_units = layer_sizes[i]
            activation_function = activation_functions[i]
            self.layers.append(MLP_Layer(input_size, num_units, activation_function))

    def forward(self, input_data):
    
        current_output = input_data
        for layer in self.layers:
            current_output = layer.forward(current_output)
        return current_output
    
    def backward(self, activations, dL_dactivation, learning_rate):
        num_layers = len(self.layers)
        layer_gradients = [{} for _ in range(num_layers)]

        # Calculate the gradient for the final layer (CCE Loss)
        last_layer = self.layers[-1]
        last_layer_activation = activations[-1]
        dL_dactivation_last = dL_dactivation
        dL_dW_last, dL_dinput_last = last_layer.backward(last_layer_activation, dL_dactivation_last)
        layer_gradients[-1]['weights'] = dL_dW_last

        for i in range(num_layers - 2, -1, -1):
            # Backpropagate the error signal through the remaining layers
            layer = self.layers[i]
            activation = activations[i]
            dL_dactivation = dL_dinput_last  # Error signal from the next layer
            dL_dW, dL_dinput_last = layer.backward(activation, dL_dactivation)
            layer_gradients[i]['weights'] = dL_dW

        # Update the weights of all MLP layers
        for i in range(num_layers):
            layer = self.layers[i]
            layer.weights -= learning_rate * layer_gradients[i]['weights'].T

    def train(self, input_data, target_data, learning_rate, epochs):
        loss_history = []

        for epoch in range(epochs):
            total_loss = 0.0

            for minibatch_inputs, minibatch_target in generator_shuffle(input_data, target_data, minibatch_size):
                # Forward pass
                activations = self.forward(minibatch_inputs)

                # Calculate the CCE loss and its gradient
                loss = Lossfunction.call(activations[-1], minibatch_target)
                total_loss += loss

                # Calculate the gradient for the final layer
                dL_dactivation = Lossfunction.backward(activations[-1], minibatch_target)

                # Backward pass
                self.backward(activations, dL_dactivation, learning_rate)

            # Calculate and store the average loss for this epoch
            average_loss = total_loss / (len(input_data) / minibatch_size)
            loss_history.append(average_loss)

            print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {average_loss}")

        return loss_history


# Set hyperparameters and data
minibatch_size = 64
learning_rate = 0.01
epochs = 150

# Create an MLP with layer sizes and activation functions
layer_sizes = [64, 128, 10]  # Input size, hidden size, output size
activation_functions = ["sigmoid", "sigmoid", "softmax"]
mlp = MLP(layer_sizes, activation_functions)

# Convert one-hot encoded labels to numpy array
target_data = np.array(one_hot_encoded)

# Define the loss function
loss_function = Lossfunction()

# Call the training function
loss_history = mlp.train(data_reshaped, target_data, learning_rate, epochs)

# Plot the average loss vs. the epoch number
plt.plot(range(1, epochs + 1), loss_history, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.title('Training Loss vs. Epoch')
plt.grid(True)
plt.show()

