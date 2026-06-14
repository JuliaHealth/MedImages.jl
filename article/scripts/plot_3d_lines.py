import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import csv

def extract_endpoints_nifti(filepath, threshold=0.1):
    img = nib.load(filepath).get_fdata()
    coords = np.array(np.where(img > threshold)).T # (N, 3)
    if len(coords) == 0:
        return [16, 16], [16, 16], [16, 16]
        
    mean = np.mean(coords, axis=0)
    cov = np.cov(coords.T)
    evals, evecs = np.linalg.eigh(cov)
    principal_axis = evecs[:, np.argmax(evals)]
    
    projections = np.dot(coords - mean, principal_axis)
    p1 = coords[np.argmin(projections)]
    p2 = coords[np.argmax(projections)]
    
    return [p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]]

def bresenham_3d(x1, y1, z1, x2, y2, z2):
    points = []
    x1, y1, z1 = int(round(x1)), int(round(y1)), int(round(z1))
    x2, y2, z2 = int(round(x2)), int(round(y2)), int(round(z2))
    
    # Grid size is 128
    x1, y1, z1 = max(0, min(127, x1)), max(0, min(127, y1)), max(0, min(127, z1))
    x2, y2, z2 = max(0, min(127, x2)), max(0, min(127, y2)), max(0, min(127, z2))

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

def hex_to_rgba(hex_color, alpha=1.0):
    hex_color = hex_color.lstrip('#')
    return [int(hex_color[i:i+2], 16)/255.0 for i in (0, 2, 4)] + [alpha]

def scale_coords(arr):
    # Scale from [0, 31] to [0, 127]
    return [v * (127.0 / 31.0) for v in arr]

def rotate_points(x_arr, y_arr, cx, cy, angle_deg):
    import math
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    
    new_x = []
    new_y = []
    for x, y in zip(x_arr, y_arr):
        # Translate to origin
        tx, ty = x - cx, y - cy
        # Rotate (clockwise for positive deg if we use -sin for clockwise, but let's just use standard)
        # "Rotate 30 degrees right" = clockwise = angle_deg = -30
        nx = tx * cos_a - ty * sin_a + cx
        ny = tx * sin_a + ty * cos_a + cy
        new_x.append(nx)
        new_y.append(ny)
    return new_x, new_y

def plot_3d_lines(artifact_dir="data/validation", output_name="differentiability_3d_lines.png", elev=20, azim=45, color_scheme="blue_gradient"):
    try:
        # Load endpoints from CSV
        epochs = []
        endpoints = []
        csv_path = os.path.join(artifact_dir, 'endpoints.csv')
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                epoch = int(row[0])
                p1x, p1y, p1z, p2x, p2y, p2z = map(float, row[1:])
                epochs.append(epoch)
                endpoints.append( ([p1x, p2x], [p1y, p2y], [p1z, p2z]) )
                
        n = len(epochs)
        indices = [0, n//5, 2*n//5, 3*n//5, 4*n//5, n-1]
        
        gx, gy, gz = extract_endpoints_nifti(os.path.join(artifact_dir, "gold_standard.nii.gz"))
        ux, uy, uz = extract_endpoints_nifti(os.path.join(artifact_dir, "uncorrected.nii.gz"))

        # Determine the central region first
        all_x = gx + ux + [p[0][0] for p in endpoints] + [p[0][1] for p in endpoints]
        all_y = gy + uy + [p[1][0] for p in endpoints] + [p[1][1] for p in endpoints]
        all_z = gz + uz + [p[2][0] for p in endpoints] + [p[2][1] for p in endpoints]
        
        cx_orig = (min(all_x) + max(all_x)) / 2.0
        cy_orig = (min(all_y) + max(all_y)) / 2.0
        
        center_x = cx_orig * (127.0/31.0)
        center_y = cy_orig * (127.0/31.0)
        center_z = (min(all_z) + max(all_z)) / 2.0 * (127.0/31.0)
        
        # Zoom window size in 128 space
        ws = 40 
        x_lims = (max(0, center_x - ws/2), min(128, center_x + ws/2))
        y_lims = (max(0, center_y - ws/2), min(128, center_y + ws/2))
        z_lims = (max(0, center_z - ws/2), min(128, center_z + ws/2))

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        filled = np.zeros((128, 128, 128), dtype=bool)
        colors_grid = np.zeros((128, 128, 128, 4), dtype=np.float32)

        def add_line(x_arr, y_arr, z_arr, color, alpha=1.0, thickness=0):
            # Rotate 30 degrees right (clockwise) before scaling
            rx, ry = rotate_points(x_arr, y_arr, cx_orig, cy_orig, -30)
            sx = scale_coords(rx)
            sy = scale_coords(ry)
            sz = scale_coords(z_arr)
            pts = bresenham_3d(sx[0], sy[0], sz[0], sx[1], sy[1], sz[1])
            rgba = hex_to_rgba(color, alpha)
            for p in pts:
                nx, ny, nz = p[0], p[1], p[2]
                if x_lims[0] <= nx < x_lims[1] and \
                   y_lims[0] <= ny < y_lims[1] and \
                   z_lims[0] <= nz < z_lims[1]:
                    filled[nx, ny, nz] = True
                    colors_grid[nx, ny, nz] = rgba

        if color_scheme == "blue_gradient":
            gold_color = '#00008b' # Dark Blue
            uncor_color = '#e74c3c' # Red
            # Iteration colors: Light to Dark Blue
            iter_colors = ['#d1e5f0', '#92c5de', '#4393c3', '#2166ac', '#053061', '#000000']
        else: # "red_blue_transition"
            gold_color = '#2166ac' # Blue
            uncor_color = '#b2182b' # Deep Red
            # Transition from Reddish to Bluish
            iter_colors = ['#ef8a62', '#fddbc7', '#f7f7f7', '#d1e5f0', '#67a9cf', '#2166ac']

        # Gold standard
        add_line(gx, gy, gz, gold_color, alpha=0.9, thickness=0)
        # Uncorrected
        add_line(ux, uy, uz, uncor_color, alpha=0.7, thickness=0)

        for i, idx in enumerate(indices):
            ep = epochs[idx]
            px, py, pz = endpoints[idx]
            alpha = 0.5 + 0.5 * (i / (len(indices) - 1))
            add_line(px, py, pz, iter_colors[i], alpha=alpha, thickness=0)

        ax.voxels(filled, facecolors=colors_grid, edgecolors='k', linewidth=0.1)

        import matplotlib.patches as mpatches
        legend_elements = [
            mpatches.Patch(color=gold_color, alpha=0.9, label='Gold Standard'),
            mpatches.Patch(color=uncor_color, alpha=0.7, label='Uncorrected')
        ]
        for i, idx in enumerate(indices):
            ep = epochs[idx]
            legend_elements.append(mpatches.Patch(color=iter_colors[i], alpha=0.7, label=f'Epoch {ep}'))

        ax.set_title(f"3D Voxel View: {output_name.replace('.png','')}", fontsize=14, fontweight='bold')
        ax.set_xlim(x_lims)
        ax.set_ylim(y_lims)
        ax.set_zlim(z_lims)
        
        # Restore background axes and grid
        ax.grid(True)
        ax.xaxis.set_pane_color((0.95, 0.95, 0.95, 1.0))
        ax.yaxis.set_pane_color((0.95, 0.95, 0.95, 1.0))
        ax.zaxis.set_pane_color((0.95, 0.95, 0.95, 1.0))
        
        ax.set_xticks(np.linspace(x_lims[0], x_lims[1], 5, dtype=int))
        ax.set_yticks(np.linspace(y_lims[0], y_lims[1], 5, dtype=int))
        ax.set_zticks(np.linspace(z_lims[0], z_lims[1], 5, dtype=int))
        
        ax.set_xlabel("X-axis", fontsize=10)
        ax.set_ylabel("Y-axis", fontsize=10)
        ax.set_zlabel("Z-axis", fontsize=10)
        
        # Move legend closer
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.95, 0.9), fontsize=9, frameon=True)

        ax.view_init(elev=elev, azim=azim)
        plt.savefig(output_name, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved {output_name}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    import sys
    artifact_dir = "data/validation"
    if len(sys.argv) > 1:
        artifact_dir = sys.argv[1]
        
    os.makedirs("article/viz_options", exist_ok=True)
    # Option 1: Blue Gradient (as requested)
    plot_3d_lines(artifact_dir, "article/viz_options/option_blue_gradient.png", elev=20, azim=45, color_scheme="blue_gradient")
    # Option 2: Red-Blue Transition
    plot_3d_lines(artifact_dir, "article/viz_options/option_red_blue.png", elev=20, azim=45, color_scheme="red_blue_transition")
    # Option 3: Top View (to see 60 degrees clearly)
    plot_3d_lines(artifact_dir, "article/viz_options/option_top_view.png", elev=90, azim=-90, color_scheme="blue_gradient")
    
    # Also overwrite the default one for the article
    plot_3d_lines(artifact_dir, "article/figures/differentiability_3d_lines.png", elev=30, azim=40, color_scheme="blue_gradient")
