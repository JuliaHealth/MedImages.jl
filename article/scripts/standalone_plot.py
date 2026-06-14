import matplotlib.pyplot as plt
import numpy as np
import math

# --- HARDCODED DATA ---
# Original data in [0, 31] coordinate space
GOLD_X, GOLD_Y, GOLD_Z = [14.0, 14.0], [15.0, 15.0], [0.0, 31.0]
UNCOR_X, UNCOR_Y, UNCOR_Z = [23.0, 9.0], [18.0, 12.0], [29.0, 2.0]

# Selected iteration data (Epoch, [p1x, p2x], [p1y, p2y], [p1z, p2z])
EPOCHS_DATA = [
    (1, [8.0, 23.0], [11.0, 18.0], [3.0, 29.0]),
    (101, [14.0, 16.0], [16.0, 14.0], [1.0, 31.0]),
    (201, [13.0, 16.0], [15.0, 13.0], [1.0, 31.0]),
    (301, [13.0, 16.0], [15.0, 13.0], [1.0, 31.0]),
    (401, [14.0, 16.0], [16.0, 14.0], [1.0, 31.0]),
    (500, [13.0, 16.0], [14.0, 15.0], [1.0, 31.0])
]

# --- UTILITIES ---
def rotate_points(x_arr, y_arr, cx, cy, angle_deg):
    angle_rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    new_x, new_y = [], []
    for x, y in zip(x_arr, y_arr):
        tx, ty = x - cx, y - cy
        nx = tx * cos_a - ty * sin_a + cx
        ny = tx * sin_a + ty * cos_a + cy
        new_x.append(nx)
        new_y.append(ny)
    return new_x, new_y

def bresenham_3d(x1, y1, z1, x2, y2, z2):
    points = []
    x1, y1, z1 = int(round(max(0, min(127, x1)))), int(round(max(0, min(127, y1)))), int(round(max(0, min(127, z1))))
    x2, y2, z2 = int(round(max(0, min(127, x2)))), int(round(max(0, min(127, y2)))), int(round(max(0, min(127, z2))))
    points.append((x1, y1, z1))
    dx, dy, dz = abs(x2 - x1), abs(y2 - y1), abs(z2 - z1)
    xs, ys, zs = (1 if x2 > x1 else -1), (1 if y2 > y1 else -1), (1 if z2 > z1 else -1)

    if dx >= dy and dx >= dz:
        p1, p2 = 2 * dy - dx, 2 * dz - dx
        while x1 != x2:
            x1 += xs
            if p1 >= 0: y1 += ys; p1 -= 2 * dx
            if p2 >= 0: z1 += zs; p2 -= 2 * dx
            p1 += 2 * dy; p2 += 2 * dz
            points.append((x1, y1, z1))
    elif dy >= dx and dy >= dz:
        p1, p2 = 2 * dx - dy, 2 * dz - dy
        while y1 != y2:
            y1 += ys
            if p1 >= 0: x1 += xs; p1 -= 2 * dy
            if p2 >= 0: z1 += zs; p2 -= 2 * dy
            p1 += 2 * dx; p2 += 2 * dz
            points.append((x1, y1, z1))
    else:
        p1, p2 = 2 * dy - dz, 2 * dx - dz
        while z1 != z2:
            z1 += zs
            if p1 >= 0: y1 += ys; p1 -= 2 * dz
            if p2 >= 0: x1 += xs; p2 -= 2 * dz
            p1 += 2 * dy; p2 += 2 * dx
            points.append((x1, y1, z1))
    return points

def hex_to_rgba(hex_color, alpha=1.0):
    hex_color = hex_color.lstrip('#')
    return [int(hex_color[i:i+2], 16)/255.0 for i in (0, 2, 4)] + [alpha]

def scale_coords(arr):
    return [v * (127.0 / 31.0) for v in arr]

# --- PLOTTING CORE ---
def plot_voxels(output_name="differentiability_jupyter.png", elev=20, azim=45, color_scheme="blue_gradient"):
    # Center calculation for rotation (use original space)
    cx_orig, cy_orig = 16.0, 16.0 
    
    # Grid initialization
    filled = np.zeros((128, 128, 128), dtype=bool)
    colors_grid = np.zeros((128, 128, 128, 4), dtype=np.float32)

    # Zoom window (128 space)
    ws = 50 # Increased window size for better context
    center_128 = 16.0 * (127.0/31.0)
    x_lims = (max(0, center_128 - ws/2), min(128, center_128 + ws/2))
    y_lims = (max(0, center_128 - ws/2), min(128, center_128 + ws/2))
    z_lims = (max(0, center_128 - ws/2), min(128, center_128 + ws/2))

    def add_line(x, y, z, color, alpha=1.0):
        # Rotate 30 degrees clockwise
        rx, ry = rotate_points(x, y, cx_orig, cy_orig, -30)
        sx, sy, sz = scale_coords(rx), scale_coords(ry), scale_coords(z)
        pts = bresenham_3d(sx[0], sy[0], sz[0], sx[1], sy[1], sz[1])
        rgba = hex_to_rgba(color, alpha)
        for p in pts:
            nx, ny, nz = p
            if x_lims[0] <= nx < x_lims[1] and y_lims[0] <= ny < y_lims[1] and z_lims[0] <= nz < z_lims[1]:
                filled[nx, ny, nz] = True
                colors_grid[nx, ny, nz] = rgba

    # Colors
    if color_scheme == "blue_gradient":
        gold_c, uncor_c = '#00008b', '#e74c3c'
        iter_colors = ['#d1e5f0', '#92c5de', '#4393c3', '#2166ac', '#053061', '#000000']
    else:
        gold_c, uncor_c = '#2166ac', '#b2182b'
        iter_colors = ['#ef8a62', '#fddbc7', '#f7f7f7', '#d1e5f0', '#67a9cf', '#2166ac']

    # Draw elements
    add_line(GOLD_X, GOLD_Y, GOLD_Z, gold_c, alpha=0.9)
    add_line(UNCOR_X, UNCOR_Y, UNCOR_Z, uncor_c, alpha=0.7)
    for i, (epoch, px, py, pz) in enumerate(EPOCHS_DATA):
        alpha = 0.5 + 0.5 * (i / (len(EPOCHS_DATA) - 1))
        add_line(px, py, pz, iter_colors[i], alpha=alpha)

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(filled, facecolors=colors_grid, edgecolors='k', linewidth=0.1)

    # Aesthetics
    ax.set_xlim(x_lims); ax.set_ylim(y_lims); ax.set_zlim(z_lims)
    ax.grid(True)
    ax.xaxis.set_pane_color((0.95, 0.95, 0.95, 1.0))
    ax.yaxis.set_pane_color((0.95, 0.95, 0.95, 1.0))
    ax.zaxis.set_pane_color((0.95, 0.95, 0.95, 1.0))
    
    ax.set_xticks(np.linspace(x_lims[0], x_lims[1], 5, dtype=int))
    ax.set_yticks(np.linspace(y_lims[0], y_lims[1], 5, dtype=int))
    ax.set_zticks(np.linspace(z_lims[0], z_lims[1], 5, dtype=int))
    ax.set_xlabel("X-axis"); ax.set_ylabel("Y-axis"); ax.set_zlabel("Z-axis")

    # Legend
    import matplotlib.patches as mpatches
    legend_elements = [
        mpatches.Patch(color=gold_c, alpha=0.9, label='Gold Standard'),
        mpatches.Patch(color=uncor_c, alpha=0.7, label='Uncorrected')
    ]
    for i, (epoch, px, py, pz) in enumerate(EPOCHS_DATA):
        legend_elements.append(mpatches.Patch(color=iter_colors[i], alpha=0.7, label=f'Epoch {epoch}'))
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 0.9), fontsize=9)

    ax.view_init(elev=elev, azim=azim)
    plt.title(f"3D Voxel Differentiability Proof ({color_scheme})", fontsize=14, fontweight='bold')
    plt.show()

# --- RUN ---
# Change color_scheme to "red_blue_transition" for the alternative view
# Change elev=90, azim=-90 for Top View
plot_voxels(color_scheme="blue_gradient")
