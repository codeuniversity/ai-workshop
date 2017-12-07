import os
import tensorflow as tf
import numpy as np
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

n_input = 361
n_hidden = 650
n_output = 2

# Load data
# empacementsBlack = np.loadtxt("GoEmpacemantsBlack.rtf")
# movesBlack = np.loadtxt("GoMovesBlack.rtf")
# empacementsWhite = np.loadtxt("GoEmpacemantsWhite.rtf")
# movesWhite = np.loadtxt("GoMovesWhite.rtf")

# Placeholder:
x = tf.placeholder(dtype=tf.float32, shape=(1, n_input))
y = tf.placeholder(dtype=tf.float32)

# Initializer:
weight_initializer = tf.random_normal_initializer()
bias_initializer = tf.zeros_initializer(dtype=tf.float32)

# Weights/ Bias
# Hidden:
w_hidden = tf.Variable(weight_initializer([n_input, n_hidden]))
bias_hidden = tf.Variable(bias_initializer([n_hidden]))
# Output:
w_out = tf.Variable(weight_initializer([n_hidden, n_output]))
bias_out = tf.Variable(bias_initializer([n_output]))

# Topology
# hidden layer:
z_hidden = tf.add(tf.matmul(x, w_hidden), bias_hidden)
activation_hidden = tf.minimum(tf.nn.relu(z_hidden), 1)

# output layer:
out_vector = tf.add(tf.matmul(activation_hidden, w_out), bias_out)
out = out_vector

# Loss function
loss = tf.losses.mean_squared_error(out, y)

# Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
opt_operation = optimizer.minimize(loss)

# Initialization
sess = tf.Session()
sess.run(tf.global_variables_initializer())

file = open("GoAI/GoEmplacemantsBlack.rtf", "r")
line = np.fromstring(file.readline(), dtype=float, sep=' ')
print(line)
print(sess.run(z_hidden, feed_dict={x: np.array([line]), y: np.array(line)}))


# Definition shuffle algorithm
def doubleArrayShuffle(array1, array2, shuffle_Iterations=5):
    if (len(array1) != len(array2)):
        raise ValueError("Arrays have to have the same length")
    for i in range(shuffle_Iterations):
        shuffle_indices = np.random.randint(low=0, high=len(array1), size=(len(array1) // 2))
        for num in range(len(shuffle_indices)):
            temp = array1[num]
            array1[num] = array1[shuffle_indices[num]]
            array1[shuffle_indices[num]] = temp
            temp = array2[num]
            array2[num] = array2[shuffle_indices[num]]
            array2[shuffle_indices[num]] = temp

# Batch Training

# (Testing)

# Save It !
