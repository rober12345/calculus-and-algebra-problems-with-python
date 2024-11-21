# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 09:18:24 2024
calculus-and-algebra-problems-with-python/notebook
/problems.ipynb
@author: rober ugalde
"""


#Exercise 1
#Let's say, in my office, it takes me 10 seconds (time) to travel 25 meters (distance) to the coffee machine. If we want to express the above situation as a function, then it would be:

#distance = speed * time

#So for this case, speed is the first derivative of the distance function above. As speed describes the rate of change of distance over time, when people say taking the first derivative of a certain function, they mean finding out the rate of change of a function.

#Find the speed and build the linear function on distance d
# over time t
#, when t e  from 00 to 10
#.


# Define the variables
distance = 25  # meters
time = 10      # seconds

# Calculate the speed (rate of change of distance over time)
speed = distance / time

# Define the linear function for distance over time
def distance_function(t):
    """
    Calculate distance as a function of time.
    :param t: Time in seconds
    :return: Distance in meters
    """
    return speed * t

# Example: Calculate distance for a given time
t = 5  # Time in seconds
calculated_distance = distance_function(t)

# Display the function and example calculation
print(f"Speed: {speed} meters per second")
print(f"The linear function is: distance(t) = {speed} * t")
print(f"For t = {t} seconds, the distance is: {calculated_distance} meters")


# Plot the distance function on domain (t)
# Create a DataFrame

import pandas as pd
import matplotlib.pyplot as plt

# Define the variables
distance = 25  # meters
time = 10      # seconds

# Calculate the speed (rate of change of distance over time)
speed = distance / time

# Define the linear function for distance over time
def distance_function(t):
    """
    Calculate distance as a function of time.
    :param t: Time in seconds
    :return: Distance in meters
    """
    return speed * t

# Create a DataFrame for plotting
t_values = range(0, 21)  # Time values from 0 to 20 seconds
distance_values = [distance_function(t) for t in t_values]

# Create a DataFrame
data = pd.DataFrame({
    'Time (seconds)': t_values,
    'Distance (meters)': distance_values
})

# Plot the distance function
plt.figure(figsize=(10, 6))
plt.plot(data['Time (seconds)'], data['Distance (meters)'], label=f'Distance = {speed} * t', color='blue', marker='o')
plt.title('Distance Function Over Time', fontsize=16)
plt.xlabel('Time (seconds)', fontsize=14)
plt.ylabel('Distance (meters)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)
plt.tight_layout()

# Show the plot
plt.show()

# Display the DataFrame
print(data)



# Exercise 2
"""
It turned out that I wasn't walking a constant speed towards getting my coffee, but I was accelerating (my speed increased over time). If my initial speed = 0, it still took me 10 seconds to travel from my seat to my coffee, but I was walking faster and faster.

 = initial speed = 

t = time

a = acceleration

distance = 

speed = 

The first derivative of the speed function is acceleration. I realize that the speed function is closely related to the distance function.

Find the acceleration value and build the quadratic function 
. Also, create a graph and a table.
"""

import pandas as pd
import matplotlib.pyplot as plt

# Given values
distance = 25  # meters
time = 10      # seconds
initial_speed = 0  # meters/second

# Calculate acceleration using the distance formula: distance = (1/2) * a * t^2
acceleration = (2 * distance) / (time ** 2)

# Define the quadratic function for distance over time
def distance_function(t):
    """
    Calculate distance as a function of time with acceleration.
    :param t: Time in seconds
    :return: Distance in meters
    """
    return (1/2) * acceleration * (t ** 2)

# Create a DataFrame for plotting
t_values = range(0, time + 1)  # Time values from 0 to 10 seconds
distance_values = [distance_function(t) for t in t_values]

# Create a DataFrame
data = pd.DataFrame({
    'Time (seconds)': t_values,
    'Distance (meters)': distance_values
})

# Plot the quadratic distance function
plt.figure(figsize=(10, 6))
plt.plot(data['Time (seconds)'], data['Distance (meters)'], label=f'Distance = 0.5 * {acceleration:.2f} * t²', color='blue', marker='o')
plt.title('Quadratic Distance Function Over Time', fontsize=16)
plt.xlabel('Time (seconds)', fontsize=14)
plt.ylabel('Distance (meters)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)
plt.tight_layout()

# Show the plot
plt.show()



# Display the DataFrame
import ace_tools as tools; tools.display_dataframe_to_user(name="Quadratic Distance Function Data", dataframe=data)

# Print the acceleration value
print(f"Acceleration: {acceleration:.2f} meters/second²")




#Exercise 3
"""
When I arrive to the coffee machine, I hear my colleague talking about the per-unit costs of producing 'product B' for the company. As the company produces more units, the per-unit costs continue to decrease until a point where they start to increase.

To optimize the per-unit production cost at its minimum to optimize efficiency, the company would need to find the number of units to be produced where the per-unit production costs begin to change from decreasing to increasing.

Build a quadratic function f(x)=0.1x2 - 9x + 4500.   0 to 100
 on 
 to create the per-unit cost function, and make a conclusion.
 
 """
 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Define the quadratic cost function and its derivative (gradient)
def cost_function(x):
    """
    Quadratic cost function: f(x) = 0.1x^2 - 9x + 4500
    """
    return 0.1 * (x ** 2) - 9 * x + 4500

def cost_function_gradient(x):
    """
    Gradient of the cost function: f'(x) = 0.2x - 9
    """
    return 0.2 * x - 9

# Gradient Descent Algorithm
def gradient_descent(start_x, learning_rate, iterations):
    """
    Perform gradient descent to minimize the cost function.
    """
    x = start_x
    x_history = [x]
    cost_history = [cost_function(x)]

    for _ in range(iterations):
        gradient = cost_function_gradient(x)
        x = x - learning_rate * gradient  # Update x based on the gradient
        x_history.append(x)
        cost_history.append(cost_function(x))
    
    return x_history, cost_history

# Parameters for gradient descent
start_x = 0  # Starting point (initial x value)
learning_rate = 0.1
iterations = 50

# Run gradient descent
x_history, cost_history = gradient_descent(start_x, learning_rate, iterations)

# Plot the cost function and the gradient descent progress
x_values = np.linspace(0, 100, 500)
cost_values = cost_function(x_values)

plt.figure(figsize=(10, 6))
plt.plot(x_values, cost_values, label='Cost Function', color='blue')
plt.scatter(x_history, cost_history, color='red', label='Gradient Descent Path', zorder=5)
plt.title('Gradient Descent on Cost Function', fontsize=16)
plt.xlabel('Units of Production (x)', fontsize=14)
plt.ylabel('Cost (f(x))', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)
plt.tight_layout()

# Show the plot
plt.show()

# Create a DataFrame for gradient descent results
data = pd.DataFrame({
    'Iteration': range(len(x_history)),
    'x': x_history,
    'Cost': cost_history
})

# Display the DataFrame
import ace_tools as tools; tools.display_dataframe_to_user(name="Gradient Descent Results", dataframe=data)



"""
LINEAR ALGEBRA

Exercise 1: Sum of two matrices
Suppose we have two matrices A and B.

A = [[1,2],[3,4]]
B = [[4,5],[6,7]]

then we get
A+B = [[5,7],[9,11]]
A-B = [[-3,-3],[-3,-3]]
Make the sum of two matrices using Python with NumPy


"""
import numpy as np

# Define the matrices
A = np.array([[1, 2], [3, 4]])
B = np.array([[4, 5], [6, 7]])

# Perform matrix addition and subtraction
A_plus_B = A + B
A_minus_B = A - B

# Display the results
print("Matrix A:")
print(A)
print("\nMatrix B:")
print(B)
print("\nA + B:")
print(A_plus_B)
print("\nA - B:")
print(A_minus_B)



"""
Exercise 2: Sum of two lists
There will be many situations in which we'll have to find an index-wise summation of two different lists. This can have possible applications in day-to-day programming. In this exercise, we will solve the same problem in various ways in which this task can be performed.

We have the following two lists:

list1 = [2, 5, 4, 7, 3]
list2 = [1, 4, 6, 9, 10]
Now let's use Python code to demonstrate addition of two lists.
"""

# Naive method

# Initializing lists
list1 = [2, 5, 4, 7, 3]
list2 = [1, 4, 6, 9, 10]
 
# Printing original lists
print ("Original list 1 : " + str(list1))
print ("Original list 2 : " + str(list2))
 
# Using naive method to add two lists 
res_list = []
for i in range(0, len(list1)):
    res_list.append(list1[i] + list2[i])
 
# Printing resulting list 
print ("Resulting list is : " + str(res_list))



# Use list comprehension to perform addition of the two lists:

# Initializing lists
list1 = [2, 5, 4, 7, 3]
list2 = [1, 4, 6, 9, 10]

# Printing original lists
print("Original list 1:", list1)
print("Original list 2:", list2)

# Using list comprehension to add two lists
res_list = [x + y for x, y in zip(list1, list2)]

# Printing resulting list
print("Resulting list using list comprehension:", res_list)



xxxx

# Importing the add operator
from operator import add

# Initializing lists
list1 = [2, 5, 4, 7, 3]
list2 = [1, 4, 6, 9, 10]

# Using map() + add() to add two lists
res_list = list(map(add, list1, list2))

# Printing resulting list
print("Resulting list using map() + add():", res_list)


xxxx



# Initializing lists
list1 = [2, 5, 4, 7, 3]
list2 = [1, 4, 6, 9, 10]

# Using zip() + sum() to add two lists
res_list = [sum(pair) for pair in zip(list1, list2)]

# Printing resulting list
print("Resulting list using zip() + sum():", res_list)



xxxx



"""
Exercise 3: Dot multiplication
We have two matrices:

matrix1 = [[1,7,3],
 [4,5,2],
 [3,6,1]]
matrix2 = [[5,4,1],
 [1,2,3],
 [4,5,2]]
A simple technique but expensive method for larger input datasets is using for loops. In this exercise, we will first use nested for loops to iterate through each row and column of the matrices, and then we will perform the same multiplication using NumPy.

"""

# Define the matrices
matrix1 = [[1, 7, 3],
           [4, 5, 2],
           [3, 6, 1]]

matrix2 = [[5, 4, 1],
           [1, 2, 3],
           [4, 5, 2]]

# Nested for loop to compute dot product
result = [[0 for _ in range(len(matrix2[0]))] for _ in range(len(matrix1))]

for i in range(len(matrix1)):
    for j in range(len(matrix2[0])):
        for k in range(len(matrix2)):
            result[i][j] += matrix1[i][k] * matrix2[k][j]

# Print the result
print("Dot multiplication using nested loops:")
for row in result:
    print(row)

# Dot product using NumPy
import numpy as np

matrix1_np = np.array(matrix1)
matrix2_np = np.array(matrix2)

result_np = np.dot(matrix1_np, matrix2_np)

# Print the NumPy result
print("\nDot multiplication using NumPy:")
print(result_np)




