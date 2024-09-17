# Neural Network From Scratch
## **Perceptrons**
- A perceptron takes several binary inputs and produces a single binary output.
- Weights express the importance of the respective inputs to the output.
- The output is 0 or 1 depending on whether the weighted sum is less than or greater than some threshold value.

## **Sigmoid Neurons**
- Sigmoid neurons are similar to perceptrons, but modified so that small changes in their weights and bias cause only a small change in their output.
- Inputs can take on any values between 0 and 1.
- There are weights for each input and overall bias.
- Output is not 0 or 1, making it suitable for binary classification problems.
## **The Architecture of Neural Networks**
- The leftmost layer in the network is called the *input layer*, and the neurons within the layer are called *input neurons*.
-  The rightmost or *output layer* contains the *output neurons*.
- The middle layer is called a *hidden layer*, since the neurons in this layer are neither inputs nor outputs.
### Feedforward Neural Network
- Neural networks where the output from one layer is used as input to the next layer.
- No loops in the network.
### Recurrent Neural Network
- Feedback loops are possible.
- The idea in these models is to have neurons which fire for some limited duration of time, before becoming quiescent. That firing can stimulate other neurons, which may fire a little while later, also for a limited duration. That causes still more neurons to fire, and so over time we get a cascade of neurons firing. 
- Loops don't cause problems in such a model, since a neuron's output only affects its input at some later time, not instantaneously.




## A Simple Network to Classify Handwritten Digits


### Classifying Individual Digits
1. Design Neural Network
   - Use of 3 layer neural network.
   - The input layer of the network contains neurons encoding the values of the input pixels. 
   - The second layer of the network is a hidden layer. We denote the number of neurons in this hidden layer by $n$.
   - The output layer of the network contains 10 neurons. 
2. Learn from a Training Dataset
   - Use the [MNIST dataset](https://yann.lecun.com/exdb/mnist/), which contains tens of thousands of scanned images of handwritten digits, together with their correct classifications. 
   - The dataset comes in 2 parts; the first part contains 60,000 images to be used as training data.
   - The images are greyscale and 28 by 28 pixels in size.
   - Each training input $x$ is regarded as a 28 x 28 = 784 - dimensional vector. Each entry represents the grey value for a single pixel in the image
3. Adjust the network's weights and biases to make its predictions, denoted as $𝑎$, as close as possible to the actual labels $y(x)$, by minimizing the cost function:
   $$
   C(w,b)=\frac{1}{2n}\sum||y(x)-a||^2
   $$
      Where:
      - **w**: Weights in the neural network.
      - **b**: Biases in the neural network.
      - **n**: Total number of training examples.
      - **x**: A training input (in this case, a 784-dimensional vector for MNIST).
      - **y(x)**: The desired output vector for input **x** (one-hot encoded vector representing the digit label).
      - **a**: The actual output vector from the network for input **x**.
      - \(\|v\|\): The length (norm) of the vector **v**.

   
- The aim is to make $C(w,b)$ small, which happens when the network's output $a$ is close to the desired output $y(x)$ for all training inputs.


### Training the Model
1. **Data Preparation**
   - **Dataset:** The first step is to gather a large dataset of examples. For handwriting recognition, this is a collection of images of handwritten digits, each labeled with the correct digit.
   - **Normalization:** The data is often normalized to ensure that the inputs are on a similar scale, which helps the network learn more effectively.
2. **Initialization**
   - **Weights and Biases:** The network’s weights and biases are initialized, typically with small random values. This randomness helps break symmetry and ensures that the neurons learn different features.
3. **Forward Propagation**
   - **Input to Output:** The input data is fed through the network. Each neuron processes its inputs, applies its weights and bias, and passes the result through an activation function to produce an output.
   - **Layer by Layer:** This process is repeated layer by layer until the final output is produced.
4. **Loss Calculation**
   - **Error Measurement:** The network’s output is compared to the true labels using a loss function. Common loss functions include Mean Squared Error (MSE) for regression tasks and Cross-Entropy Loss for classification tasks.
   - **Loss Function:** The loss function quantifies the difference between the predicted output and the actual output. A higher loss indicates a larger error.
5. **Backward Propagation (Backpropagation)**
   - **Gradient Calculation:** The network calculates the gradient of the loss function with respect to each weight and bias. This involves applying the chain rule of calculus to propagate the error backward through the network.
   - **Weight Updates:** Using the gradients, the network updates its weights and biases to reduce the loss. This is typically done using an optimization algorithm like Stochastic Gradient Descent (SGD).
6. **Optimization Algorithm**
   - **Stochastic Gradient Descent (SGD):** In SGD, the weights are updated incrementally for each training example
   - **Learning Rate:** The learning rate controls the size of the weight updates. A smaller learning rate results in smaller updates, which can lead to more precise convergence but may take longer.
7. **Iteration**
   - **Epochs:** The entire dataset is passed through the network multiple times, each pass is called an epoch. During each epoch, the network continues to adjust its weights and biases to minimize the loss.
   - **Convergence:** The training process continues until the loss converges to a minimum value, indicating that the network has learned to make accurate predictions.
8. **Validation**
   - **Validation Set:** A separate validation set is used to evaluate the network’s performance during training. This helps to monitor for overfitting, where the network performs well on the training data but poorly on unseen data.
9. **Testing**
   - **Test Set:** After training, the network is evaluated on a test set to assess its generalization performance. This provides an unbiased estimate of how well the network will perform on new, unseen data.

---
# Comparing Models
## Model 1
1. **Weight Initialization:** Weights and biases are initialized randomly using a normal distribution. This helps the network start with diverse values, ensuring better gradient flow during training.
2. **Feedforward Pass:** The network uses the sigmoid activation function in the hidden layer to introduce non-linearity, enabling it to model complex relationships in the data. In the final output layer, it uses the `argmax` function to make predictions by selecting the neuron with the highest output.
3. **Cost Function:** The cost function used is the quadratic cost (mean squared error), with backpropagation implemented to compute the gradient and update weights.
4. Training: The network is trained using **Stochastic Gradient Descent (SGD)**. Key parameters for training include:
   - Epochs: 30 epochs.
   - Mini-batch size: 10.
   - Learning rate (η): 3.0.
5. **Backpropagation:** The model calculates the gradient of the cost function with respect to weights and biases, using backpropagation to update the parameters.
6. **Evaluation:** The model's performance is evaluated on test data, returning the number of correct predictions and calculating the accuracy.
### Results:
The model achieves a final accuracy of **10.41%** on the test data.
## Model 2
1. **Feedforward Pass:**
The feedforward process happens through several layers:
   - The network starts with an input layer receiving normalized MNIST images (28x28 pixels).
   - It passes through multiple dense layers (fully connected layers), with ReLU activations after each layer except the final one.
   - The final output layer uses logits to make predictions about the class (digit 0-9). The network computes these probabilities by feeding data through the stacked layers.
2. **Cost Function:** `softmax_crossentropy_with_logits` calculates the cross-entropy loss, which is a popular cost function used in classification problems by combining the correct logit and the normalization factor from the softmax. It is used to measure the difference between the model's predictions and the actual class labels during training.
3. **Training:** The model trains using batch gradient descent:
   - The dataset is divided into batches (size 64).
   - The network is trained over 32 epochs, meaning it goes through the entire dataset 32 times.
   - After each epoch, the model's accuracy is logged and plotted to track training progress.
4. **Backpropagation:** `grad_softmax_crossentropy_with_logits` calculates the gradient of the cross-entropy loss with respect to the logits. The gradient is essential for backpropagation, which is the process used to update the model’s weights during training. This function outputs the gradient, which tells the model how to adjust its weights to reduce the cost in the next iteration. 
5. **Evaluation:** After training, the network is evaluated on the test set:
   - The network uses its predict() method to make predictions on the test set. 
   - The accuracy is calculated as the ratio of correct predictions to the total number of test samples.
### Results: 
The model achieves a final accuracy of **0.971** on the test data.
## Model 3
1. **Library Imports:**The model uses `tensorflow` and `keras` from TensorFlow 2.17.0 for building and training the neural network. Other imports include `numpy` for numerical operations and `matplotlib.pyplot` for visualizing data.
2. **Data Loading:** The MNIST dataset is loaded using `keras.datasets.mnist`. It consists of 28x28 grayscale images of handwritten digits (0-9) and corresponding labels.
3. **Data Preprocessing:** Training and testing datasets are split using `(training_images, training_labels), (test_images, test_labels) = mnist.load_data()`.
4. **Model Architecture:** This code defines a simple feedforward neural network using TensorFlow's Keras API.
   - **Layer 1 (Flatten Layer) :** 
      - `tf.keras.layers.Flatten()`: This layer flattens the input data from a 2D array (28x28 pixel image in this case) into a 1D array. For example, a 28x28 image would become a 784-element vector. This is a necessary step before feeding the data into the dense (fully connected) layers.
   - **Layer 2 (Dense Layer) :**
      - `tf.keras.layers.Dense(1024, activation=tf.nn.relu)`: This is a fully connected layer with 1024 neurons. Each neuron applies a linear transformation followed by the ReLU activation function (`tf.nn.relu`), which helps introduce non-linearity. The large number of neurons (1024) gives the model capacity to learn complex patterns in the input data.
   - **Layer 3 (Output Layer) :**
      - `tf.keras.layers.Dense(10, activation=tf.nn.softmax)`: This is the output layer with 10 neurons, corresponding to the 10 digit classes (0-9). The `softmax` activation function is used to convert the output into class probabilities, where the sum of probabilities across all classes is 1.
5. **Model Compilation:**
   - **Optimizer:**
      - `optimizer='adam`': The Adam optimizer is used for gradient descent. It's an adaptive learning rate optimizer that combines the advantages of two other extensions of gradient descent, namely RMSProp and Momentum.
   - **Loss Function:**
      - `loss='sparse_categorical_crossentropy'`: This loss function is used for multi-class classification problems where the target labels are integers. Since the labels in MNIST are single integers (0-9), sparse_categorical_crossentropy is appropriate. It computes the cross-entropy loss between the true labels and the predicted probabilities.
   - **Metrics:**
      - `metrics=['accuracy']`: The model is evaluated based on accuracy, which measures the percentage of correctly classified samples.
6. **Model Training:**
   - Training Data: `training_images` and `training_labels` are the input images and their corresponding labels from the MNIST dataset.
   - Epochs: `epochs=5` The model will go through the entire training dataset 5 times during training.  After each epoch, the weights of the model are updated based on the optimizer and loss function.

   During training, the model will adjust its weights to minimize the loss function, gradually improving accuracy on the training set.
7. **Predictions and Results:** 
   - The model outputs predictions for 10000 test images. Each image is classified into one of 10 classes (digits 0-9).
   - The predictions are compared to the true labels, and classification results are printed for inspection.
### Results:
The model achieves a final accuracy of **0.972** on the test data.

## Conclusion
1. **Model 3 is significantly easier to use, especially for quick development and prototyping.**
   - Model 1 and Model 2 require manual coding for almost every aspect of the neural network, making it useful for those who want to deeply understand the mechanics of the neural networks.
   - TensorFlow and Keras abstract much of the complexity. You only need to define the architecture, choose an optimizer, and a loss function. Everything else, including weight initialization and backpropagation, is handled automatically.
   - The high-level API of Keras is easy to understand and implement, even for beginners. You can build and train a neural network in just a few lines of code.
   - TensorFlow/Keras does a lot of the heavy lifting. This makes it more user-friendly for prototyping and experimentation.
2. **Model 3 (TensorFlow/Keras) is far more efficient and optimized, especially when scaling to larger networks and datasets.**
   - TensorFlow/Keras is highly optimized for speed. It has efficient implementations for forward pass and backpropagation, and it makes use of hardware accelerators like GPUs or TPUs when available.
3. **Model 2 offers more granular control, which is useful for specialized models or research purposes, but at the cost of ease of use and performance.**
   - Keras offers flexibility to an extent, but it abstracts many details behind the scenes. While you can tweak most parts of the model (like optimizers and layer types), the high-level interface limits full control.

