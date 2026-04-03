import os
from PIL import Image, ImageDraw, ImageFont
import textwrap

# -----------------------------------------------------------------------------
# 1. THEME & ROBUST UTILS
# -----------------------------------------------------------------------------
W, H = 2400, 2400
COLORS = {
    "primary": "#2c3e50",
    "legacy": "#e74c3c",
    "medimages": "#27ae60",
    "python": "#3498db",
    "cpp": "#95a5a6",
    "analytical": "#f39c12",
    "bg": "#f8f9fa",
    "white": "#ffffff"
}

try:
    f_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    if not os.path.exists(f_path): f_path = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
    f_h = ImageFont.truetype(f_path, 90)
    f_q = ImageFont.truetype(f_path, 70)
    f_t = ImageFont.truetype(f_path, 55)
    f_b = ImageFont.truetype(f_path, 45)
    f_s = ImageFont.truetype(f_path.replace("-Bold", ""), 32)
except:
    f_h = f_q = f_t = f_b = f_s = ImageFont.load_default()

def draw_rounded_rect(draw, coords, radius, outline="#333", fill="white", width=10):
    draw.rounded_rectangle(coords, radius=radius, fill=fill, outline=outline, width=width)

def draw_quadrant_frame(draw, quad_num, title, y_range, color="#2c3e50"):
    y_s, y_e = y_range
    draw.rectangle([40, y_s, 2360, y_s+130], fill="#eee", outline=color, width=10)
    draw.text((100, y_s+65), f"{quad_num}. {title}", fill=color, font=f_q, anchor="lm")
    draw.rectangle([40, y_s+130, 2360, y_e], outline="#ccc", width=5)

def draw_text_wrapped(draw, text, center_xy, max_width, font, fill="black"):
    limit = max(1, int(max_width / (font.getlength("a") or 15)))
    lines = textwrap.wrap(text, width=limit)
    line_h = font.getbbox("Ay")[3] + 25
    curr_y = center_xy[1] - (len(lines)*line_h)//2
    for line in lines:
        draw.text((center_xy[0], curr_y), line, fill=fill, font=font, anchor="ma")
        curr_y += line_h

def draw_arrow(draw, start, end, color="#333", width=25, dashed=False):
    if dashed:
        dist = ((end[0]-start[0])**2 + (end[1]-start[1])**2)**0.5
        steps = int(dist / 50)
        for i in range(steps):
            if i % 2 == 0:
                s = (start[0]+(end[0]-start[0])*i/steps, start[1]+(end[1]-start[1])*i/steps)
                e = (start[0]+(end[0]-start[0])*(i+1)/steps, start[1]+(end[1]-start[1])*(i+1)/steps)
                draw.line([s, e], fill=color, width=width)
    else: draw.line([start, end], fill=color, width=width)
    x2, y2 = end
    x1, y1 = start
    if abs(x1 - x2) < 40: 
        if y2 > y1: draw.polygon([(x2-50, y2-70), (x2+50, y2-70), (x2, y2)], fill=color)
        else: draw.polygon([(x2-50, y2+70), (x2+50, y2+70), (x2, y2)], fill=color)
    else:
        if x2 > x1: draw.polygon([(x2-70, y2-50), (x2-70, y2+50), (x2, y2)], fill=color)
        else: draw.polygon([(x2+70, y2-50), (x2+70, y2+50), (x2, y2)], fill=color)

# -----------------------------------------------------------------------------
# 2. CHALLENGE 1
# -----------------------------------------------------------------------------
def create_challenge_1():
    img = Image.new('RGB', (W, H), color='white')
    draw = ImageDraw.Draw(img)
    draw.text((W//2, 70), "CHALLENGE 1: BIOBANK VOLUME BOTTLENECK", fill=COLORS["primary"], font=f_h, anchor="mm")
    
    draw_quadrant_frame(draw, 1, "Challenge: Massive Biobank Scaling", (150, 600))
    draw_text_wrapped(draw, "10,000+ Multimodal studies. High-throughput preprocessing is the primary hardware I/O bottleneck.", (1200, 400), 2000, f_t)

    draw_quadrant_frame(draw, 2, "Gap: Serialization Friction (Legacy)", (650, 1150), color=COLORS["legacy"])
    draw.polygon([(200, 750), (500, 750), (450, 950), (250, 950)], fill="#7f8c8d", outline="black", width=8) # Funnel
    draw_text_wrapped(draw, "MONAI PersistentDataset (~650 ms). Pickle/Pt caching creates immense memory buildup.", (1300, 900), 1400, f_b, fill=COLORS["legacy"])

    draw_quadrant_frame(draw, 3, "How Addressed: Zero-Serialization Path", (1200, 1750), color=COLORS["medimages"])
    draw.rectangle((200, 1350, 500, 1500), fill="#bdc3c7", outline="black", width=10); draw.text((350, 1425), "HDF5", font=f_b, anchor="mm")
    draw.rectangle((600, 1300, 850, 1550), fill="#34495e", outline="#f1c40f", width=12); draw.text((725, 1425), "GPU", fill="#f1c40f", font=f_b, anchor="mm")
    draw_text_wrapped(draw, "MedImages.jl implements direct HDF5 persistence + Fused Affine GPU kernels. (~90 ms throughput).", (1600, 1450), 1200, f_b, fill=COLORS["medimages"])

    draw_quadrant_frame(draw, 4, "Experiments: 7.2× Faster biobank turnaround", (1800, 2350))
    try:
        mip = Image.open('elsarticle/figures_new/clinical_assets/mip_wholebody.png').resize((600, 750))
        img.paste(mip, (300, 1850))
    except: pass
    draw_text_wrapped(draw, "RESULT: 7.2× SPEEDUP. Biobank scale enabled.", (1450, 2100), 1400, f_t, fill=COLORS["medimages"])
    img.save('elsarticle/figures_new/challenge_1.png')

# -----------------------------------------------------------------------------
# 3. CHALLENGE 2
# -----------------------------------------------------------------------------
def create_challenge_2():
    img = Image.new('RGB', (W, H), color='white')
    draw = ImageDraw.Draw(img)
    draw.text((W//2, 70), "CHALLENGE 2: THE TWO-LANGUAGE BARRIER", fill=COLORS["primary"], font=f_h, anchor="mm")
    
    draw_quadrant_frame(draw, 1, "Challenge: Performance vs Prototyping", (150, 600))
    draw_text_wrapped(draw, "The 'Two-Language Problem' forces research in high-level scripts while relying on opaque low-level binaries.", (1200, 400), 2000, f_t)

    draw_quadrant_frame(draw, 2, "Gap: Opaque Binary Walls", (650, 1150), color=COLORS["legacy"])
    draw.rectangle((200, 850, 450, 950), fill=COLORS["python"], outline="black"); draw.text((325, 900), "PY", fill="white", font=f_b, anchor="mm")
    draw.rectangle((460, 850, 710, 950), fill=COLORS["cpp"], outline="black"); draw.text((585, 900), "C++", font=f_b, anchor="mm")
    for i in range(8): draw.rectangle((850, 700+i*55, 1000, 750+i*55), fill="#7f8c8d", outline="black")
    draw_text_wrapped(draw, "SITK binaries act as a 'brick wall' preventing native GPU acceleration.", (1600, 900), 1100, f_b, fill=COLORS["legacy"])

    draw_quadrant_frame(draw, 3, "How Addressed: Unified Julia Ecosystem", (1200, 1750), color=COLORS["medimages"])
    draw.ellipse((200, 1350, 550, 1650), outline=COLORS["medimages"], width=25); draw.text((375, 1500), "JULIA", fill=COLORS["medimages"], font=f_b, anchor="mm")
    draw_arrow(draw, (600, 1500), (900, 1500), color=COLORS["medimages"])
    draw.rectangle((1000, 1375, 1300, 1625), fill="#34495e", outline="#f1c40f", width=12); draw.text((1150, 1500), "GPU", fill="#f1c40f", font=f_b, anchor="mm")
    draw_text_wrapped(draw, "Unified compiled approach unlocks native hardware acceleration without speed penalties.", (1750, 1500), 900, f_b, fill=COLORS["medimages"])

    draw_quadrant_frame(draw, 4, "Experiments: 135× Native Acceleration", (1800, 2350))
    draw_text_wrapped(draw, "RESULT: 135× Fused Affine Speedup (0.83 ms) vs Python CPU baselines (6.69 ms).", (1200, 2100), 2000, f_t, fill=COLORS["medimages"])
    img.save('elsarticle/figures_new/challenge_2.png')

# -----------------------------------------------------------------------------
# 4. CHALLENGE 3
# -----------------------------------------------------------------------------
def create_challenge_3():
    img = Image.new('RGB', (W, H), color='white')
    draw = ImageDraw.Draw(img)
    draw.text((W//2, 70), "CHALLENGE 3: DIFFERENTIABLE PHYSICS (UDEs)", fill=COLORS["primary"], font=f_h, anchor="mm")
    
    draw_quadrant_frame(draw, 1, "Challenge: Physics-in-the-Loop ML", (150, 600))
    draw_text_wrapped(draw, "Accurate dosimetry requires integrating mechanistic scientific equations directly into machine learning training loops.", (1200, 400), 2000, f_t)

    draw_quadrant_frame(draw, 2, "Gap: Locked Walled Gardens", (650, 1150), color=COLORS["legacy"])
    for i in range(12): draw.line([(300+i*140, 780), (300+i*140, 1120)], fill="black", width=15)
    draw_text_wrapped(draw, "Frameworks (PyTorch/JAX) fail to differentiate through arbitrary mechanistic simulators.", (1300, 950), 1400, f_b, fill=COLORS["legacy"])

    draw_quadrant_frame(draw, 3, "How Addressed: 4-State UDE Integrator", (1200, 1750), color=COLORS["medimages"])
    draw.rectangle((150, 1350, 500, 1600), fill="#ebf5fb", outline="#3498db", width=12); draw.text((325, 1475), "PHYSICS", fill="#2980b9", font=f_b, anchor="mm")
    draw.ellipse((1000, 1350, 1300, 1650), outline="#34495e", width=20); draw.text((1150, 1500), "∫", fill="#d4ac0d", font=f_h, anchor="mm")
    draw_text_wrapped(draw, "SciML UDE natively connects Mechanistic Knowns with Learned Residuals.", (1700, 1500), 900, f_b, fill=COLORS["medimages"])

    draw_quadrant_frame(draw, 4, "Experiments: MC Fidelity (r=0.957)", (1800, 2350))
    try:
        dose = Image.open('elsarticle/figures_new/clinical_assets/dose_overlay_ct.png').resize((550, 650))
        img.paste(dose, (300, 1850))
    except: pass
    draw_text_wrapped(draw, "RESULT: State-of-the-Art accuracy (r=0.957) while solving the Expression Problem.", (1450, 2100), 1400, f_t, fill=COLORS["medimages"])
    img.save('elsarticle/figures_new/challenge_3.png')

# -----------------------------------------------------------------------------
# 5. CHALLENGE 4
# -----------------------------------------------------------------------------
def create_challenge_4():
    img = Image.new('RGB', (W, 2800), color='white')
    draw = ImageDraw.Draw(img)
    draw.text((W//2, 70), "CHALLENGE 4: METADATA INTEGRITY", fill=COLORS["primary"], font=f_h, anchor="mm")
    
    draw_quadrant_frame(draw, 1, "Challenge: Multi-modal Coregistration", (150, 650))
    draw_text_wrapped(draw, "Theranostic workflows require coregistering heterogeneous spatial data onto a single shared grid.", (1200, 425), 2000, f_t)

    draw_quadrant_frame(draw, 2, "Gap: Metadata Drift (Tags Lost)", (700, 1250), color=COLORS["legacy"])
    draw.text((250, 950), "✂", fill="black", font=f_h, anchor="mm")
    draw_text_wrapped(draw, "GetArrayFromImage() slices Spacing/Origin tags. Mapping is lost during NumPy conversion.", (1300, 1000), 1600, f_b, fill=COLORS["legacy"])

    draw_quadrant_frame(draw, 3, "How Addressed: Protected BatchedMedImage", (1300, 2050), color=COLORS["medimages"])
    slices = ['ct_slice.png', 'dosemap_slice.png', 'spect_nac_slice.png', 'spect_ac_slice.png']
    for i, s in enumerate(slices):
        try:
            sl = Image.open(f'elsarticle/figures_new/clinical_assets/{s}').convert("RGBA").resize((550, 550))
            sl.putalpha(sl.getchannel('A').point(lambda p: 180 if p > 0 else 0))
            img.paste(sl, (300+i*70, 1400+i*80), sl)
        except: pass
    draw_text_wrapped(draw, "🛡️ Julia Type System rigidly binds metadata to the 4D Tensor.", (1800, 1700), 1000, f_b, fill=COLORS["medimages"])

    draw_quadrant_frame(draw, 4, "Experiments: Flawless SUV Consistency", (2100, 2750))
    draw_text_wrapped(draw, "RESULT: SUV Consistency < 1.5% Deviation across rotation batches.", (1200, 2450), 2000, f_t, fill=COLORS["medimages"])
    img.save('elsarticle/figures_new/challenge_4.png')

# -----------------------------------------------------------------------------
# 6. DOSIMETRY EXPERIMENT
# -----------------------------------------------------------------------------
def create_dosimetry_comparison():
    img = Image.new('RGB', (W, 2800), color='white')
    draw = ImageDraw.Draw(img)
    draw.text((W//2, 70), "QUANTITATIVE DOSIMETRY BENCHMARK", fill=COLORS["primary"], font=f_h, anchor="mm")
    
    draw_quadrant_frame(draw, 1, "Challenge: SPECT/CT to Ground Truth", (150, 650))
    draw_quadrant_frame(draw, 2, "Gap: Competitor Performance Failure", (700, 1250), color=COLORS["legacy"])
    draw_quadrant_frame(draw, 3, "How Addressed: SciML UDE Champion Model", (1300, 1950), color=COLORS["medimages"])
    draw_quadrant_frame(draw, 4, "Experiments: Fidelity and Precision", (2000, 2750))

    lanes = [
        {"title": "DL (CNN)", "color": COLORS["legacy"], "asset": "dl_artifacts.png", "metric": "r=0.557"},
        {"title": "ANALYTICAL", "color": COLORS["analytical"], "asset": "vsv_homo.png", "metric": "r=0.912"},
        {"title": "SciML UDE", "color": COLORS["medimages"], "asset": "ude_highfi.png", "metric": "r=0.957"}
    ]
    y_lane = 2050
    for i, lane in enumerate(lanes):
        x = 100 + i * 760
        draw_rounded_rect(draw, (x, y_lane, x + 720, 2700), 40, outline=lane["color"], fill="white", width=12)
        draw.text((x + 360, y_lane + 80), lane["title"], font=f_b, fill=lane["color"], anchor="mm")
        try:
            ast = Image.open(f'elsarticle/figures_new/clinical_assets/{lane["asset"]}').resize((650, 750))
            img.paste(ast, (x + 35, y_lane + 150))
        except: pass
        draw.rectangle((x + 50, y_lane + 500, x + 670, y_lane + 630), fill=lane["color"])
        draw.text((x + 360, y_lane + 565), lane["metric"], fill="white", font=f_q, anchor="mm")
    img.save('elsarticle/figures_new/dosimetry_experiment.png')

if __name__ == "__main__":
    create_challenge_1(); create_challenge_2(); create_challenge_3(); create_challenge_4(); create_dosimetry_comparison()
    print("Definitive high-fidelity pictographic 4-quadrant infographics generated successfully.")
