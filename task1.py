import numpy as np
import matplotlib.pyplot as plt

def f(x, y):
    return 100 * (y - x**2)**2 + (1 - x)**2

x = np.linspace(-2, 2, 400)
y = np.linspace(-2, 2, 400)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

# 2D Contour Plot
plt.figure(figsize=(8, 6))
cp = plt.contourf(X, Y, Z, levels=50, cmap='viridis')
plt.colorbar(cp)
plt.title('Contour plot of $f(x, y) = 100(y − x^2)^2 + (1 − x)^2$')
plt.xlabel('x')
plt.ylabel('y')

# Plot and label point (1, 1)
plt.scatter(1, 1, color='red', label='Point (1, 1)')
plt.text(1.1, 1.1, 'Point (1, 1)', color='red')
plt.legend()
# plt.show()
plt.savefig('2D plot.png', bbox_inches='tight')

# 3D Surface Plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
ax.set_title('Surface plot of $f(x, y) = 100(y − x^2)^2 + (1 − x)^2$')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x, y)')

# Plot and label point (1, 1)
z_point = f(1, 1)  # Calculate z value for point (1, 1)
ax.plot([1], [1], [z_point], color='red', marker='o', markersize=8, label='Point (1, 1)')
ax.text(1.1, 1.1, z_point, 'Point (1, 1)', color='red')
ax.legend()

# Change perspective
ax.view_init(elev=30, azim=60)  # Set elevation to 30 degrees and azimuth to 60 degrees
# plt.show()
plt.savefig('3D plot.png', bbox_inches='tight')
