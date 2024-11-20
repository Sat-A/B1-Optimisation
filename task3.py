import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def rosenbrock(x):
    return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2

# Initial points for optimization
initial_points = [np.random.rand(2) * 4 - 2 for _ in range(3)]

# Generate the meshgrid for plotting
x = np.linspace(-2, 2, 400)
y = np.linspace(-2, 2, 400)
X, Y = np.meshgrid(x, y)
Z = rosenbrock([X, Y])

for i, x0 in enumerate(initial_points):
    result = minimize(rosenbrock, x0, method='Nelder-Mead', options={'disp': True})
    path = result['x']
    
    plt.figure(figsize=(10, 8))
    cp = plt.contour(X, Y, Z, levels=50, cmap='viridis')
    plt.colorbar(cp)
    plt.title(f'Nelder-Mead Simplex Algorithm Path - Start Point {i+1}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    
    plt.plot(path[0], path[1], 'r-')
    plt.scatter(x0[0], x0[1], color='blue', marker='o')  # Start point
    plt.scatter(path[0], path[1], color='red', marker='x')  # End point
    
    plt.show()