#Artifial neural networks

Künstliches Neuronales Netzwerk: (eng.: artificial neural network ): (short: ANN)

ANNs are no new Invention, in the opposite they are comparable old, but the increasing operation power enables us to calculate bigger and bigger networks, what enables us to attack bigger and bigger problems.

##Neurons

Since a ANN is a network or collection of neurons, the question occurs what is a neuron.

A Neuron is inspired by the structure of a Neuron in a human brain a object that receivs an or mostly several impulses and react more or less strong to them and in the end returns a corresponding impulse again.

###Realization

A Neuron is recieving an input-vector (the impulse):

$$$

\overrightarrow{x} = \left(\begin{array}{c}x_{0}\\ x_{1}\\ ...\\ x_{n}\end{array}\right)

$$$

After the neuron recieved the input it's starts to evaluate the input, by multiply each input value with an internal saved **Weight**. When we create a new Neural network we normally initialize the Weights with random values.

$$$

\sum_{i=0}^{n} w_{i} * x_{i}

$$$

After the input is evaluated the **offset(Bias)** gets added, the offset is used as additional parameter to increase the efficency of the network. The resulting value is the internal state z. When we creat a new neural network we normally initialize each bias with the value zero (0).

$$$

z = (\sum_{i=0}^{n} w_{i} * x_{i})+b

$$$

This so calculated **internat state** get than passed on to the **Activity funtion** to calculate the **Activation(y)** of the Neuron.

$$$

y =  f(z)

$$$

In the end the so calculated Activaion is passed on to **other Neurons**, commonly of the next layer.

##Activation Function
After we evaluluated the input we have a value, that has no defined range, this value can be literally every value between $$$ -\infty $$$ and $$$ +\infty $$$. But we still have to decide if this value is enough to trigger the neuron to fire or not. Therfore we use the actviation function, so the Activation function will decide for which range of values the connected neurons will consider this neuron as fired and for which range they will consider the neuron as not fired.
There are different kinds of activation function, which bring different advantages and disadvantages. Sometimes activation functions are even linked to learn algorithm so you have to use the linked activation function to use this learning algorithm.

The most common acticity functions are the Sigmoid function, the Tangens Hyperbolicus and especially the ReLu (Rectifier Linear Unit) function (max(0,x)).

![common activity functions](http://www.cbcity.de/wp-content/uploads/2016/03/ActivationFunctions-770x154.png "common activity functions")

####Rectifier linear unit
$$
f(x)= max(x,0)
$$
One of the simplest activation functions, that let the Neuron only fire if the activation is bigger than zero.
> return values in range: [0, +$$$\infty$$$]
> not everywhere differentiable
> continuous

#####Exponential linear unit
$$
f(x)= \left\{\begin{matrix}
 x& \text{if x}\geq 0 \\
 (e^{x}-1)& \text{otherwise}
\end{matrix}\right.
$$
Exponential modification of the Rectifier, so the Neuron also fires when negativ values are given (but weaker).
> returns value in range: [-a, +$$$\infty$$$]
> everywhere differentiable
> smooth nonlinearities

####Softplus
$$
f(x)= log(e^{x}+1)
$$
Tries to perform a smooth approximation on the standard relu-functions.
> return values in range: [0, +$$$\infty$$$]
> everywhere differentiable
> smooth nonlinearities

####Softsign
$$
f(x)= \frac{x}{\left | x \right |+1}
$$
> return values in range: [-1, +1]
> everywhere differentiable
> smooth nonlinearities

####Sigmoid
$$
f(x)= \frac{1}{1+ e^{-x}}
$$
One of the most spreaded activation functions and the standard activation function in combination with Backpropagation.
> return values in range: [0, +1]
> everywhere differential
> smooth nonlinearities

####hyperbolic tangent

$$
f(x)= tanh(x) = \frac{2}{1+e^{-2x}}-1
$$
> return values in range: [-1, 1]
> everywhere differential
> smooth nonlinearities

##Layer

A neural network consisting of only one neuron is called a Perceptron, but since such a neural network is limited in its use cases, we tend to use neural networks with several layers.

In general you can differ between three kinds of layers

1. **The input layer**

*This Layer is passive, doing nothing but passing the input vector on to the hidden layer.*

2. **The hidden layer**

*Every layer that exist between the input layer and output layer is called hidden.*

3. **The ouput layer.**

*This Layer return the values, that represent the output of the whole neural network, therefore the activation function doesn't get applied to this layer, because the evaluation if a neuron should fire or not is unimportant, since the output of the neuron is used as output for the whole network.*



![visualisation simple ANN](http://www.dspguide.com/graphics/F_26_5.gif)

A normal neural network has corresponding only one input and one output layer, but you can use as much hidden layer as you want. But please note that the amount of layer will raise the amount of CPU and time, needed to train your network. If yoou network includes a spefice number of Layers (10 ... 20) you can speak of it as Depp neural network.

In a simple artificial neural network each neuron is with every neuron of the next layer connected. Each of such layers, where each neuron is connected with each neuron of the next layer, is called Fully Connected Layer.

**Extract the following to ++different kinds of ANNs++**

It's also possible to integrate a case seperation into a ANN, this option is common for the extraction of feautures from a data set. This case seperation is called convolution and is realized through a  simple function, but it turned out to be a really strong functionality. Such networks are named Convolutional neural networks (or short **ConvNets**)



##Loss function

We now can pass values on to our network and calculate a Activation based on the random weights but we could do the same thing with a piece of paper and a few dieces. But before we implemenent the best known ability of an AI, namely to learn, we first need to know if the thing the AI does before it started learning was right or wrong. But we still have to differenciate between several cases and how we can evaluate how wrong or right the network at this case was.

###Classification

In this case the programmer tries to teach the ANN to classify different inputs into different classes. In this case are only two states at the end possible true or false, either the ANN guessed the class correctly or not. In the easiest case you would just count how many classes your AI guessed correctly and the more the better is your AI trained.

For a classifier it also offers themself to use a softmax function is the last layer (as target function) to transform the real activations of the previous layers into a probability.

#####Softmax Function

This function is used to reduce a K-dimensional vector z of arbitary real numbers:

$$$

z = \left(\begin{array}{c}z_{0}\\ z_{1}\\ ...\\ z_{K}\end{array}\right) e.g.: \left(\begin{array}{c}80.06\\ -200\\ -50.2\\ 0\end{array}\right)

$$$

to a K-dimensional vector of real values between [0, 1] that add upp to one.

$$$

\sigma(z) = \left(\begin{array}{c} \sigma(z)_{0}\\ \sigma(z)_{1}\\ ...\\ \sigma(z)_{K}\end{array}\right) e.g.: \left(\begin{array}{c}0.5\\ 0.05\\0.05\\ 0.4\end{array}\right)

$$$

$$$

1 = \sum_{i = 0}^{K} \sigma(z)_{i}

$$$

$$$

e.g. : 1 = 0.5 + 0.05 +0.05 +0.4

$$$

###Regression

In this case we expect our network to return an estimated or predicted response (mostly a value or a set of values). We therefore need a different loss function for calculate the loss. Commonly we will therefore use L1 were we just use the **differents between the expected and the returned output**. Mostly to create a more neutral loss we take the **square of the differnce** this loss function is then called L2. In some cases it can also become  handy to use Cross Entropy as loss function.

##Learning

After we defined when our network did right or wrong, we now can start to teach it. Note that you natural have to first collect fitting trainings and test data for your network. After that you tran your network by give the network one set of your train data to process. Since we initialized the wheights of the neurons with random values it is unlikely that the produced output is similar to the expected output. To change that we start to "punish" our network using out loss function. There are several different technics how to do that, I will introduce some of them in a different file.

##Representation as Matrix
Its also possible to represent a neural network as matrix. Therefore we extract the weight-vector of the neurons and put them together in a matrix. Each column of these matrix represents now the weigths of one neurons. Each Bias gets extracted into a external vector that gets added during the evaluation.
These principle can be applied to each hidden and the output layer.
$$n : \text{length input Vactor} $$
$$nH : \text{number hidden neurons} $$
$$
\begin{bmatrix}
 w_{1|1}& w_{1|2} & ... & w_{1|n} \\
 w_{2|1}& w_{2|2} & ... & w_{2|n} \\
 ... & ... &  ... & ... \\
 w_{nH|1} & w_{nH|2} & ... & w_{nH|n}
\end{bmatrix}
$$
$$
nO : \text{number of output neurons}
$$
$$
\begin{bmatrix}
 w_{1|1}& w_{1|2} & ... & w_{1|nH} \\
 w_{2|1}& w_{2|2} & ... & w_{2|nH} \\
 ... & ... &  ... & ... \\
 w_{nO|1} & w_{nO|2} & ... & w_{nO|nH}
\end{bmatrix}
$$
