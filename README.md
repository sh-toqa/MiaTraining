# Neural Network From Scratch
## **Perceptrons**
- A perceptron takes several binary inputs and produces a single binary output.
- Weights express the importance of the respective inputs to the output.
- The output is 0 or 1 depending on whether the weighted sum is less than or greater than some threshold value.

## **Sigmoid Neurons**
- Sigmoid neurons are similar to perceptrons, but modified so that small changes in their weights and bias cause only a small change in their output.
- Inputs can take on any values between 0 and 1.
- There are weights for each input and overall bias.
- Output is not 0 or 1,.....................
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

### Problem Breakdown:
1. Segmentation Problem
2. Classification Problem

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
