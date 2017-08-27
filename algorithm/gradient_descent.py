# Following tutorial from: https://github.com/mattnedrich/GradientDescentExample
# More explanations on: http://mccormickml.com/2014/03/04/gradient-descent-derivation/

import os
import pandas as pd
import numpy as np

def mse(b, m, points):
    total_error = 0
    for i, point in points.iterrows():
        total_error += ((b + m*point.x) - point.y)**2
    return total_error

def step_gradient(b_current, m_current, points, learning_rate=0.05):
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points.loc[i, 'x']
        y = points.loc[i, 'y']
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))
    new_b = round(b_current - (learning_rate * b_gradient), 6)
    new_m = round(m_current - (learning_rate * m_gradient), 6)
    return (new_b, new_m)

def gradient_descent_runner(points, learning_rate, starting_b, starting_m, num_iterations=1000):
    b = starting_b
    m = starting_m
    for i in range(num_iterations):
        b, m = step_gradient(b, m, points, learning_rate)
        print("Iteration {:3d} | b: {:5f} | m: {:5f}".format(i+1, b, m))
    return (b, m)

if __name__ == '__main__':
    current_dir = os.path.dirname(__file__)
    read_dir = './data/gradient_descent_data.csv'
    full_path = os.path.join(current_dir, read_dir)

    df = pd.read_csv(full_path, names=['x', 'y'])
    print(df.head())

    # random_y = np.random.randint(low=math.floor(df.y.min()), high=math.ceil(df.y.max()), size=df.shape[0])
    # print(random_y)
    
    print('Running gradient descent algorithm')
    b, m = gradient_descent_runner(df, 0.0001, 0, 0, 450)
    print(f"y = {b} + {m} * x")
    print(mse(b, m, df))
    
    pred_y = df.x.map(lambda x: b + m * x)
        
    # 100 = 11220.4094955
    