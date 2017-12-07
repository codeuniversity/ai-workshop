import os
import tensorflow as tf
import numpy as np
import sys
import plotly.offline as py
import plotly.graph_objs as go
from math import sqrt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

n_input = 361
n_hidden = 650
n_output = 2

# Load data
# empacementsBlack = np.loadtxt("GoEmpacemantsBlack.txt")
# movesBlack = np.loadtxt("GoMovesBlack.txt")
# empacementsWhite = np.loadtxt("GoEmpacemantsWhite.txt")
# movesWhite = np.loadtxt("GoMovesWhite.txt")

# Placeholder:
x = tf.placeholder(dtype=tf.float32, shape=(1, n_input))
y = tf.placeholder(dtype=tf.float32)

# Initializer:
weight_initializer = tf.variance_scaling_initializer()
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
activation_hidden = tf.nn.softmax(z_hidden)

# output layer:
out_vector = tf.add(tf.matmul(activation_hidden, w_out), bias_out)
out = out_vector

# Loss function
loss = tf.losses.mean_squared_error(out, y)

# Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
opt_operation = optimizer.minimize(loss)

# Initialization
sess = tf.Session()
sess.run(tf.global_variables_initializer())


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


# Initialization
sess = tf.Session()
sess.run(tf.global_variables_initializer())
n_epochs = 500


# Train

def linecount(name):
    file = open(name)
    length = len(file.readlines())
    file.close()
    return length


len_M_W = linecount('GoMovesBlack.txt')
len_E_W = linecount('GoEmpacemantsBlack.txt')

stat_black = open("GoEmpacemantsBlack.txt")
mov_black = open("GoMovesBlack.txt")

loss_Values = []

for i in range(len_M_W):
    status = [np.fromstring(stat_black.readline(), sep=" ")]
    moves = np.fromstring(mov_black.readline(), sep=" ")
    sess.run(opt_operation, feed_dict={x: status, y: moves})
    #if (i % 100 == 0):
    loss_Values.append(sess.run(loss, feed_dict={x: status, y: moves}))


def linreg_a(x, y):
    # Returns the value of a to calculate the linear regression for a Scatterplot
    return (sum(y) * sum([x ** 2 for x in x]) - sum(x) * sum([x * y for x, y in zip(x, y)])) / (
            len(x) * sum([x ** 2 for x in x]) - (sum(x) ** 2))


def linreg_b(x, y):
    # Returns the value of b to calculate the linear regression for a Scatterplot
    return (len(x) * sum([x * y for x, y in zip(x, y)]) - sum(x) * sum(y)) / (
            len(x) * sum([x ** 2 for x in x]) - (sum(x) ** 2))


def linreg_value(a, b, x):
    # Returns the value for y of a linear regression at point x in a Scatterplot
    return a + b * x


def linreg_correlation(x, y):
    # Returns the correlation coefficient of a x and y
    return (len(x) * sum([x*y for x, y in zip(x, y)]) - sum(x) * sum(y)) / sqrt((len(x) * sum(x**2 for x in x) - (sum(x)**2)) * (len(x) * sum(y**2 for y in y) - (sum(y)**2)))

def generate_html():
    chart_list = []
    color_values = np.random.randn(len(loss_Values))
    x = [i for i in range(len(loss_Values))]
    y = [element for element in loss_Values]
    a = linreg_a(x, y)
    b = linreg_b(x, y)

    scatterplot1 = go.Scatter(
        x=x,
        y=y,
        mode='markers',
        marker=dict(
            size=12,
            color=color_values,
            colorscale='Viridis',
            showscale=False
        ))

    linreg_scatterplot1 = go.Scatter(
        x=[0, max(x)],
        y=[a, linreg_value(a, b, max(x))],
        mode='lines',
        text="Linear Regression, Correlation Coefficient: {0:.2f}".format(linreg_correlation(x, y)))

    data = [scatterplot1, linreg_scatterplot1]

    layout = go.Layout(
        title='Loss Rate',
        yaxis=dict(
            title='Difference',
        ),
        xaxis=dict(
            title='Iteration'
        ),
        showlegend=False,
    )

    fig = go.Figure(data=data, layout=layout)
    chart_list.append(py.plot(fig, show_link=False, output_type='div'))

    with open('report.html', 'w') as file:
        for chart in chart_list:
            file.write(chart)


generate_html()
# https://www.tensorflow.org/versions/r0.12/api_docs/python/nn/activation_functions_