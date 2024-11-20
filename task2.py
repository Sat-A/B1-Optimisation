import numpy as np
import matplotlib.pyplot as plt

def f(x, y):
    return 100 * (y - x**2)**2 + (1 - x)**2

def gradient(x, y):
    df_dx = -400 * x * (y - x**2) - 2 * (1 - x)
    df_dy = 200 * (y - x**2)
    return np.array([df_dx, df_dy])

def hessian(x, y):
    d2f_dx2 = 1200 * x**2 - 400 * y + 2
    d2f_dy2 = 200
    d2f_dxdy = -400 * x
    return np.array([[d2f_dx2, d2f_dxdy], [d2f_dxdy, d2f_dy2]])

def gradient_descent(x0, y0, lr=0.001, tol=1e-6, max_iter=10000):
    x, y = x0, y0
    path = [(x, y)]
    for _ in range(max_iter):
        grad = gradient(x, y)
        x_new, y_new = x - lr * grad[0], y - lr * grad[1]
        path.append((x_new, y_new))
        if np.linalg.norm([x_new - x, y_new - y]) < tol:
            break
        x, y = x_new, y_new
    return np.array(path)

def newton_method(x0, y0, tol=1e-6, max_iter=10000):
    x, y = x0, y0
    path = [(x, y)]
    for _ in range(max_iter):
        grad = gradient(x, y)
        hess = hessian(x, y)
        delta = np.linalg.solve(hess, grad)
        x_new, y_new = x - delta[0], y - delta[1]
        path.append((x_new, y_new))
        if np.linalg.norm([x_new - x, y_new - y]) < tol:
            break
        x, y = x_new, y_new
    return np.array(path)

def gauss_newton_method(x0, y0, tol=1e-6, max_iter=10000):
    x, y = x0, y0
    path = [(x, y)]
    for _ in range(max_iter):
        grad = gradient(x, y)
        hess = hessian(x, y)
        delta = np.linalg.lstsq(hess, grad, rcond=None)[0]
        x_new, y_new = x - delta[0], y - delta[1]
        path.append((x_new, y_new))
        if np.linalg.norm([x_new - x, y_new - y]) < tol:
            break
        x, y = x_new, y_new
    return np.array(path)

# Generate the meshgrid for plotting
x = np.linspace(-2, 2, 400)
y = np.linspace(-2, 2, 400)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

# Initial points for optimization
initial_points = [(-1.5, 1.5), (1.5, -1.5), (0, 0)]

# Plot for each method and each initial point
methods = {
    'Gradient Descent': gradient_descent,
    'Newton\'s Method': newton_method,
    'Gauss-Newton Method': gauss_newton_method
}

for method_name, method in methods.items():
    for i, (x0, y0) in enumerate(initial_points):
        plt.figure(figsize=(10, 8))
        cp = plt.contour(X, Y, Z, levels=100, cmap='viridis')
        plt.colorbar(cp)
        plt.title(f'{method_name} Path from Initial Point {i+1}')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        path = method(x0, y0)
        plt.plot(path[:, 0], path[:, 1], 'r-')
        plt.scatter(path[0, 0], path[0, 1], color='blue', marker='o')  # Start point
        plt.scatter(path[-1, 0], path[-1, 1], color='red', marker='x')  # End point
        # plt.show()
        plt.savefig(f'{method_name} path {i+1}.png', bbox_inches='tight')
