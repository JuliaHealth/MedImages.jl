import os
import numpy as np
import nibabel as nib
from PIL import Image, ImageDraw, ImageFont, ImageOps
import matplotlib.pyplot as plt
from scipy.ndimage import rotate, zoom

# -----------------------------------------------------------------------------
# 1. CONFIG & PATHS
# -----------------------------------------------------------------------------
OUT_DIR = "article/figures/images"
FINAL_OUT = "article/figures/study_flowchart.png"
os.makedirs(OUT_DIR, exist_ok=True)

BASE_DIR = "data/dosimetry_data/FDM_DPI-2024-7-KRN_Lu177_PSMA__SPECT_Tc_0__Pat47"
CT_PATH = os.path.join(BASE_DIR, "ct.nii.gz")
SPECT_PATH = os.path.join(BASE_DIR, "spect.nii.gz")
DOSE_PATH = os.path.join(BASE_DIR, "dosemap_mc.nii.gz")

# Soft Tissue Window
L, W = 40, 400
VMIN, VMAX = L - W/2, L + W/2

def apply_window(data, vmin, vmax):
    return np.clip((data - vmin) / (vmax - vmin), 0, 1)

def save_slice(data, name, cmap="gray", vmin=None, vmax=None):
    plt.figure(figsize=(4, 4))
    plt.imshow(np.flipud(data.T), cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    plt.axis('off')
    plt.savefig(os.path.join(OUT_DIR, name), bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()

def scale_fov(img_2d, scale_factor, pad_val=0):
    h, w = img_2d.shape
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)
    scaled = zoom(img_2d, scale_factor, order=1)
    
    out = np.full((h, w), pad_val, dtype=img_2d.dtype)
    if scale_factor > 1:
        start_h = (new_h - h) // 2
        start_w = (new_w - w) // 2
        out = scaled[start_h:start_h+h, start_w:start_w+w]
    else:
        start_h = (h - new_h) // 2
        start_w = (w - new_w) // 2
        out[start_h:start_h+new_h, start_w:start_w+new_w] = scaled
    return out

# -----------------------------------------------------------------------------
# 2. ASSET GENERATION
# -----------------------------------------------------------------------------
print("Loading data...")
ct_data = nib.load(CT_PATH).get_fdata()
spect_data = nib.load(SPECT_PATH).get_fdata()
dose_data = nib.load(DOSE_PATH).get_fdata()
if dose_data.ndim == 4:
    dose_data = dose_data[..., 0]

# Crop from liver to top of head (approx Z=250 to 508)
Z_START = 250
Z_END = 508
MID_Y = 128

ct_coronal = ct_data[:, MID_Y, Z_START:Z_END]
spect_coronal = spect_data[:, MID_Y, Z_START:Z_END]
dose_coronal = dose_data[:, MID_Y, Z_START:Z_END]

# Intensity Scaling (Balanced percentiles to make functional signal bright and clear)
spect_vmax = np.percentile(spect_coronal, 98.0)
dose_vmax = np.percentile(dose_coronal, 99.0)

# PHASE 2: Aligned (Clean Coronal)
print("Generating Phase 2 assets...")
save_slice(ct_coronal, "ct_aligned.png", vmin=VMIN, vmax=VMAX)
save_slice(spect_coronal, "spect_aligned.png", cmap="magma", vmin=0, vmax=spect_vmax)
save_slice(dose_coronal, "dose_aligned.png", cmap="hot", vmin=0, vmax=dose_vmax)

# PHASE 1: Raw (Independent Misalignments)
print("Generating Phase 1 assets...")

# SPECT raw: Rotated -15, Scaled down to 50% FOV, even brighter
spect_raw = rotate(spect_coronal, -15, reshape=False, mode='nearest')
spect_raw = scale_fov(spect_raw, 0.5, pad_val=0)
save_slice(spect_raw, "spect_raw.png", cmap="magma", vmin=0, vmax=spect_vmax * 0.5)

# CT raw: Rotated +20, Scaled up to 150% FOV
ct_raw = rotate(ct_coronal, 20, reshape=False, mode='nearest')
ct_raw = scale_fov(ct_raw, 1.5, pad_val=-1000) # -1000 HU is air
save_slice(ct_raw, "ct_raw.png", vmin=VMIN, vmax=VMAX)

# Dose raw: Rotated +35, Scaled down to 40% FOV, slightly less pronounced
dose_raw = rotate(dose_coronal, 35, reshape=False, mode='nearest')
dose_raw = scale_fov(dose_raw, 0.4, pad_val=0)
save_slice(dose_raw, "dose_raw.png", cmap="hot", vmin=0, vmax=dose_vmax * 0.8)

# -----------------------------------------------------------------------------
# 3. PIL RECONSTRUCTION
# -----------------------------------------------------------------------------
print("Reconstructing flowchart via PIL...")
W_FLOW, H_FLOW = 1200, 1800
img = Image.new('RGB', (W_FLOW, H_FLOW), color='white')
draw = ImageDraw.Draw(img)

try:
    f_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    f_title = ImageFont.truetype(f_path, 40)
    f_phase = ImageFont.truetype(f_path, 30)
    f_label = ImageFont.truetype(f_path, 22)
    f_math = ImageFont.truetype(f_path, 28)
except:
    f_title = f_phase = f_label = f_math = ImageFont.load_default()

def draw_phase_header(num, text, y):
    draw.rectangle([50, y, 1150, y+60], fill="#f3f4f6", outline="#d1d5db")
    draw.text((70, y+10), f"{num}", fill="black", font=f_phase)
    draw.text((120, y+10), text, fill="#374151", font=f_phase)

# Title
draw.text((W_FLOW//2, 40), "End-to-End Multi-Modal SciML Dosimetry Pipeline", fill="#1f2937", font=f_title, anchor="mm")

# Adjust Y-positions to move images "up" within their sections
img_y_offset = 70 # Higher images relative to header
label_y_offset = 380 # Labels moved lower

# Phase 1
y_p1 = 120
draw_phase_header(1, "Raw Clinical Modalities (Unorganized Batches)", y_p1)
images_p1 = ["spect_raw.png", "ct_raw.png", "dose_raw.png"]
labels_p1 = ["Raw SPECT", "Raw CT", "Pseudo-MC"]
for i, (name, label) in enumerate(zip(images_p1, labels_p1)):
    try:
        im = Image.open(os.path.join(OUT_DIR, name)).resize((250, 250))
        # Add a subtle border so stacked dark images don't blend into one blob
        im_border = ImageOps.expand(im, border=2, fill="#bdc3c7")
        
        # Haphazard offsets: 
        # Base: slightly left and up
        # Middle: slightly bottom and left
        # Top: center
        offsets = [(-20, -15), (-10, 20), (0, 0)]
        for dx, dy in offsets:
            img.paste(im_border, (100 + i*350 + dx, y_p1 + img_y_offset + dy))
            
        draw.text((100 + i*350 + 125, y_p1 + label_y_offset), label, fill="#4b5563", font=f_label, anchor="mm")
    except Exception as e: print(f"Error loading {name}: {e}")

# Arrow
draw.text((W_FLOW//2, y_p1 + 400), "↓", fill="#9ca3af", font=f_title, anchor="mm")

# Phase 2
y_p2 = 550
draw_phase_header(2, "Preprocessed & Batched (Organized Target Grid)", y_p2)
images_p2 = ["spect_aligned.png", "ct_aligned.png", "dose_aligned.png"]
labels_p2 = ["Aligned SPECT", "Aligned CT", "Aligned Target Dose"]
for i, (name, label) in enumerate(zip(images_p2, labels_p2)):
    try:
        im = Image.open(os.path.join(OUT_DIR, name)).resize((250, 250))
        im_border = ImageOps.expand(im, border=2, fill="#bdc3c7")
        
        # Orderly offsets: each moved 15px down and left (base is (-30, 30), middle is (-15, 15), top is (0,0))
        # Wait, user requested: "moved 3 mm up and right" for the next layer. 
        # If top is at (0,0), then the ones UNDER it should be down and left.
        # Let's say top is (0,0), middle is (-15, 15), bottom is (-30, 30).
        # Wait, if bottom is pasted first, then middle, then top:
        # Bottom: (-30, 30) (left and down)
        # Middle: (-15, 15)
        # Top: (0, 0)
        # This looks like they stack "up and right".
        offsets = [(-30, 30), (-15, 15), (0, 0)]
        for dx, dy in offsets:
            img.paste(im_border, (100 + i*350 + dx, y_p2 + img_y_offset + dy))
            
        draw.text((100 + i*350 + 125, y_p2 + label_y_offset), label, fill="#4b5563", font=f_label, anchor="mm")
    except Exception as e: print(f"Error loading {name}: {e}")

# Arrow
draw.text((W_FLOW//2, y_p2 + 400), "↓", fill="#9ca3af", font=f_title, anchor="mm")

# Phase 3: SciML Module
y_p3 = 1000
draw.rectangle([100, y_p3, 1100, y_p3+250], fill="#f8fafc", outline="#64748b", width=2)
draw.text((W_FLOW//2, y_p3 + 30), "3. The Julia Ecosystem (End-to-End Differentiable Imaging)", fill="#1e293b", font=f_phase, anchor="mm")

# Draw a stylized Julia logo
jx, jy = 180, y_p3 + 100
r = 22
draw.ellipse([jx, jy, jx+r*2, jy+r*2], fill="#cb3c33") # Red
draw.ellipse([jx-25, jy+40, jx-25+r*2, jy+40+r*2], fill="#389826") # Green
draw.ellipse([jx+25, jy+40, jx+25+r*2, jy+40+r*2], fill="#9558b2") # Purple

# Library chips
chips = ["Zygote.jl", "Enzyme.jl", "SciML", "Radiomics.jl", "Makie.jl", "Lux.jl"]
chip_x, chip_y = 330, y_p3 + 80
for i, chip in enumerate(chips):
    cx = chip_x + (i%3)*220
    cy = chip_y + (i//3)*60
    draw.rectangle([cx, cy, cx+180, cy+45], fill="#e2e8f0", outline="#94a3b8", width=1)
    draw.text((cx+90, cy+22), chip, fill="#334155", font=f_label, anchor="mm")

# Simplified Equation
math_txt = "D'(r, t) = Physical_Priors + Neural_Corrector(Data)"
draw.text((W_FLOW//2, y_p3 + 210), math_txt, fill="#0f172a", font=f_math, anchor="mm")

# Branching Arrows
y_arr = y_p3 + 250
draw.line([(W_FLOW//2, y_arr), (W_FLOW//2, y_arr+30)], fill="#9ca3af", width=4) # Down
draw.line([(350, y_arr+30), (850, y_arr+30)], fill="#9ca3af", width=4) # Horizontal
draw.line([(350, y_arr+30), (350, y_arr+60)], fill="#9ca3af", width=4) # Down Left
draw.line([(850, y_arr+30), (850, y_arr+60)], fill="#9ca3af", width=4) # Down Right

# Arrow heads
draw.polygon([(340, y_arr+50), (360, y_arr+50), (350, y_arr+65)], fill="#9ca3af")
draw.polygon([(840, y_arr+50), (860, y_arr+50), (850, y_arr+65)], fill="#9ca3af")

# Phase 4
y_p4 = y_arr + 90
draw_phase_header(4, "Endless Applications: Dosimetry, Biomarkers, and Beyond", y_p4)

# Left Image (Dosimetry UDE)
try:
    im = Image.open(os.path.join(OUT_DIR, "ude_pat46.png"))
    w_im, h_im = im.size
    new_h = 280
    new_w = int(w_im * (new_h / h_im))
    im = im.resize((new_w, new_h))
    img.paste(im, (350 - new_w//2, y_p4 + 80))
    draw.text((350, y_p4 + 390), "SciML Dosimetry Inference", fill="#4b5563", font=f_label, anchor="mm")
except Exception as e: print(f"Error loading ude: {e}")

# Right Dots
try:
    f_huge = ImageFont.truetype(f_path, 140)
except:
    f_huge = ImageFont.load_default()
draw.text((850, y_p4 + 200), "...", fill="#9ca3af", font=f_huge, anchor="mm")
draw.text((850, y_p4 + 390), "Novel Scientific Workflows", fill="#4b5563", font=f_label, anchor="mm")

img.save(FINAL_OUT)
print(f"Flowchart successfully saved to {FINAL_OUT}")
