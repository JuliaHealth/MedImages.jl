import numpy as np
import matplotlib.pyplot as plt

def bresenham_3d(x1, y1, z1, x2, y2, z2):
    points = []
    points.append((x1, y1, z1))
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    dz = abs(z2 - z1)
    xs = 1 if x2 > x1 else -1
    ys = 1 if y2 > y1 else -1
    zs = 1 if z2 > z1 else -1

    if dx >= dy and dx >= dz:
        p1 = 2 * dy - dx
        p2 = 2 * dz - dx
        while x1 != x2:
            x1 += xs
            if p1 >= 0:
                y1 += ys
                p1 -= 2 * dx
            if p2 >= 0:
                z1 += zs
                p2 -= 2 * dx
            p1 += 2 * dy
            p2 += 2 * dz
            points.append((x1, y1, z1))
    elif dy >= dx and dy >= dz:
        p1 = 2 * dx - dy
        p2 = 2 * dz - dy
        while y1 != y2:
            y1 += ys
            if p1 >= 0:
                x1 += xs
                p1 -= 2 * dy
            if p2 >= 0:
                z1 += zs
                p2 -= 2 * dy
            p1 += 2 * dx
            p2 += 2 * dz
            points.append((x1, y1, z1))
    else:
        p1 = 2 * dy - dz
        p2 = 2 * dx - dz
        while z1 != z2:
            z1 += zs
            if p1 >= 0:
                y1 += ys
                p1 -= 2 * dz
            if p2 >= 0:
                x1 += xs
                p2 -= 2 * dz
            p1 += 2 * dy
            p2 += 2 * dx
            points.append((x1, y1, z1))
    return points

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

filled = np.zeros((32, 32, 32), dtype=bool)
colors = np.zeros((32, 32, 32, 4), dtype=np.float32)

pts1 = bresenham_3d(0, 0, 0, 31, 31, 31)
for p in pts1:
    filled[p] = True
    colors[p] = [1, 0, 0, 0.5]

pts2 = bresenham_3d(0, 31, 0, 31, 0, 31)
for p in pts2:
    filled[p] = True
    colors[p] = [0, 1, 0, 0.5]

ax.voxels(filled, facecolors=colors, edgecolors='k', linewidth=0.5)
plt.savefig('test_voxels.png')
