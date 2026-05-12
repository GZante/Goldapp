"""
PCB Gold Detection — Streamlit Web App
Upload an image, tune HSV thresholds, get an annotated result.
"""

import cv2
import numpy as np
import streamlit as st
from PIL import Image
import io

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PCB Gold Detector",
    page_icon="🟡",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}
.mono { font-family: 'JetBrains Mono', monospace; }

.title-block {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border: 1px solid #ffd700;
    border-radius: 12px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
}
.title-block h1 {
    color: #ffd700;
    font-size: 2.4rem;
    font-weight: 800;
    letter-spacing: -1px;
    margin: 0 0 0.3rem 0;
}
.title-block p {
    color: #a0aec0;
    margin: 0;
    font-size: 0.95rem;
}

.metric-card {
    background: #1a1a2e;
    border: 1px solid #2d3748;
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    text-align: center;
}
.metric-card .label {
    color: #718096;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 0.4rem;
    font-family: 'JetBrains Mono', monospace;
}
.metric-card .value {
    color: #e2e8f0;
    font-size: 1.5rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
}
.metric-card .value.gold { color: #ffd700; }
.metric-card .value.green { color: #68d391; }

.verdict-positive {
    background: linear-gradient(135deg, #2d1b00, #3d2600);
    border: 2px solid #ffd700;
    border-radius: 12px;
    padding: 1.5rem 2rem;
    text-align: center;
    color: #ffd700;
    font-size: 1.8rem;
    font-weight: 800;
    letter-spacing: 1px;
    margin: 1rem 0;
}
.verdict-negative {
    background: linear-gradient(135deg, #1a1a1a, #2d2d2d);
    border: 2px solid #4a5568;
    border-radius: 12px;
    padding: 1.5rem 2rem;
    text-align: center;
    color: #718096;
    font-size: 1.8rem;
    font-weight: 800;
    letter-spacing: 1px;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# ── Physical constants ─────────────────────────────────────────────────────────
RULER_MM           = 10.0
FALLBACK_MM_PER_PX = 0.04
GOLD_THICKNESS_UM  = 0.05
GOLD_DENSITY_G_CM3 = 19.32


# ── Core functions (ported from production script) ────────────────────────────

def calibrate_from_H(image, ruler_mm=RULER_MM):
    h, w = image.shape[:2]
    crop = image[int(h*0.07):int(h*0.42), int(w*0.25):int(w*0.75)]
    grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(grey, 80, 255, cv2.THRESH_BINARY_INV)
    num, _, stats, _ = cv2.connectedComponentsWithStats(bw)
    best = None
    for i in range(1, num):
        x, y, cw, ch, area = stats[i]
        aspect = cw / max(ch, 1)
        if 0.7 < aspect < 1.6 and 80 < ch < 400 and 80 < cw < 400 and area > 3000:
            roi = bw[y:y+ch, x:x+cw]
            spans = [dark[-1]-dark[0] for r in range(roi.shape[0])
                     if len(dark := np.where(roi[r] > 0)[0]) > 5]
            if spans:
                crossbar = max(spans)
                if best is None or crossbar > best:
                    best = crossbar
    if best and best > 30:
        return ruler_mm / best
    return FALLBACK_MM_PER_PX


def isolate_pcb(image):
    L = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2LAB))[0]
    _, dark = cv2.threshold(L, 200, 255, cv2.THRESH_BINARY_INV)
    sat = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))[1]
    _, coloured = cv2.threshold(sat, 40, 255, cv2.THRESH_BINARY)
    mask = cv2.bitwise_or(dark, coloured)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=4)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=2)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if num <= 1:
        return np.ones(image.shape[:2], dtype=np.uint8) * 255
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (labels == largest).astype(np.uint8) * 255


def detect_gold(image, mm_per_px, hsv_lower, hsv_upper, mass_threshold):
    pcb_mask    = isolate_pcb(image)
    pcb_area_px = int(np.sum(pcb_mask > 0))

    if pcb_area_px == 0:
        return dict(
            label='negative', gold_mass_ug=0, gold_fraction=0,
            pcb_area_mm2=0, gold_area_mm2=0,
            pcb_mask=pcb_mask,
            gold_mask=np.zeros_like(pcb_mask),
            annotated_image=image.copy(),
            mm_per_px=mm_per_px,
        )

    hsv       = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gold_mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
    gold_mask = cv2.bitwise_and(gold_mask, pcb_mask)

    ke        = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    gold_mask = cv2.morphologyEx(gold_mask, cv2.MORPH_OPEN,  ke)
    gold_mask = cv2.morphologyEx(gold_mask, cv2.MORPH_CLOSE, ke)

    px_area      = mm_per_px ** 2
    pcb_mm2      = pcb_area_px * px_area
    gold_px      = int(np.sum(gold_mask > 0))
    gold_mm2     = gold_px * px_area
    gold_frac    = gold_px / pcb_area_px
    gold_mass_ug = gold_mm2 * 0.01 * GOLD_THICKNESS_UM * 1e-4 * GOLD_DENSITY_G_CM3 * 1e6

    label = 'positive' if gold_mass_ug >= mass_threshold else 'negative'

    ann = image.copy()
    for mask, colour in [(pcb_mask, (0, 220, 0)), (gold_mask, (0, 0, 255))]:
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(ann, cnts, -1, colour, 3)

    return dict(
        label        = label,
        gold_mass_ug = gold_mass_ug,
        gold_fraction= gold_frac,
        pcb_area_mm2 = pcb_mm2,
        gold_area_mm2= gold_mm2,
        pcb_mask     = pcb_mask,
        gold_mask    = gold_mask,
        annotated_image = ann,
        mm_per_px    = mm_per_px,
    )


def bgr_to_pil(bgr_img):
    return Image.fromarray(cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB))

def gray_to_pil(gray_img):
    return Image.fromarray(gray_img)


# ── UI ─────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="title-block">
  <h1>🟡 PCB Gold Detector</h1>
  <p>Upload a PCB image · Tune HSV thresholds · Get annotated results + mass estimate</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar: controls ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ HSV Thresholds")
    st.markdown("**Lower bound**")
    h_low = st.slider("H min", 0, 179, 20, help="Hue lower bound")
    s_low = st.slider("S min", 0, 255, 100, help="Saturation lower bound")
    v_low = st.slider("V min", 0, 255, 140, help="Value lower bound")

    st.markdown("**Upper bound**")
    h_high = st.slider("H max", 0, 179, 26, help="Hue upper bound")
    s_high = st.slider("S max", 0, 255, 255, help="Saturation upper bound")
    v_high = st.slider("V max", 0, 255, 255, help="Value upper bound")

    st.markdown("---")
    st.markdown("### ⚖️ Mass Threshold")
    mass_thresh = st.number_input(
        "Gold mass threshold (µg)",
        min_value=0.0001, max_value=100.0,
        value=0.01, step=0.001, format="%.4f",
        help="Minimum estimated gold mass to classify as POSITIVE"
    )

    st.markdown("---")
    st.markdown("### ℹ️ Physical constants")
    st.markdown(f"""
    <div style='font-family: JetBrains Mono, monospace; font-size:0.75rem; color:#718096; line-height:1.8'>
    ENIG thickness: {GOLD_THICKNESS_UM} µm<br>
    Gold density: {GOLD_DENSITY_G_CM3} g/cm³<br>
    Ruler: {RULER_MM} mm (H marker)<br>
    Fallback scale: {FALLBACK_MM_PER_PX} mm/px
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🔄 Reset to defaults", use_container_width=True):
        st.rerun()

# ── Main: upload ──────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Drop a PCB image here",
    type=["jpg", "jpeg", "png", "tif", "tiff", "bmp"],
    help="Works best with images that include an H marker (1 cm ruler)"
)

if uploaded is None:
    st.markdown("""
    <div style='text-align:center; padding: 4rem 2rem; color: #4a5568;
                border: 2px dashed #2d3748; border-radius: 12px; margin-top:1rem;'>
        <div style='font-size:3rem; margin-bottom:1rem;'>📷</div>
        <div style='font-size:1.1rem;'>Upload a PCB image to get started</div>
        <div style='font-size:0.85rem; margin-top:0.5rem;'>
            Supports JPG, PNG, TIFF, BMP
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Process ───────────────────────────────────────────────────────────────────
file_bytes = np.frombuffer(uploaded.read(), np.uint8)
img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

if img_bgr is None:
    st.error("Could not decode image. Please try a different file.")
    st.stop()

hsv_lower = np.array([h_low, s_low, v_low], dtype=np.uint8)
hsv_upper = np.array([h_high, s_high, v_high], dtype=np.uint8)

with st.spinner("Analysing…"):
    mm_pp  = calibrate_from_H(img_bgr)
    result = detect_gold(img_bgr, mm_pp, hsv_lower, hsv_upper, mass_thresh)

# ── Verdict banner ─────────────────────────────────────────────────────────────
if result['label'] == 'positive':
    st.markdown('<div class="verdict-positive">🟡 GOLD POSITIVE</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="verdict-negative">⚫ GOLD NEGATIVE</div>', unsafe_allow_html=True)

# ── Metrics row ───────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)

def metric_card(col, label, value, css_class=""):
    col.markdown(f"""
    <div class="metric-card">
        <div class="label">{label}</div>
        <div class="value {css_class}">{value}</div>
    </div>
    """, unsafe_allow_html=True)

metric_card(c1, "Gold Mass",     f"{result['gold_mass_ug']:.3f} µg",    "gold")
metric_card(c2, "Gold Area",     f"{result['gold_area_mm2']:.2f} mm²",  "")
metric_card(c3, "Gold Fraction", f"{result['gold_fraction']*100:.3f}%", "")
metric_card(c4, "PCB Area",      f"{result['pcb_area_mm2']:.1f} mm²",   "")
metric_card(c5, "Scale",         f"{1/mm_pp:.0f} px/mm",                "green")

st.markdown("<div style='margin-top:1.5rem'></div>", unsafe_allow_html=True)

# ── Images ────────────────────────────────────────────────────────────────────
col_orig, col_mask, col_ann = st.columns(3)

with col_orig:
    st.markdown("**Original**")
    st.image(bgr_to_pil(img_bgr), use_container_width=True)

with col_mask:
    st.markdown("**PCB Mask**")
    st.image(gray_to_pil(result['pcb_mask']), use_container_width=True)

with col_ann:
    st.markdown("**Annotated** *(green = PCB, blue = gold)*")
    st.image(bgr_to_pil(result['annotated_image']), use_container_width=True)

# ── HSV preview ───────────────────────────────────────────────────────────────
with st.expander("🎨 HSV gold mask (raw, before morphology)"):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    raw_mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
    st.image(gray_to_pil(raw_mask), caption="Raw HSV mask (full image, no PCB clip)", use_container_width=True)

# ── Download annotated image ──────────────────────────────────────────────────
st.markdown("---")
ann_pil = bgr_to_pil(result['annotated_image'])
buf = io.BytesIO()
ann_pil.save(buf, format="PNG")
st.download_button(
    label="⬇️ Download annotated image",
    data=buf.getvalue(),
    file_name=f"{uploaded.name.rsplit('.',1)[0]}_gold_result.png",
    mime="image/png",
    use_container_width=True,
)

st.markdown(f"""
<div style='text-align:center; margin-top:2rem; color:#2d3748; font-size:0.8rem;
            font-family: JetBrains Mono, monospace;'>
  HSV [{h_low},{s_low},{v_low}]→[{h_high},{s_high},{v_high}]  ·  
  threshold {mass_thresh} µg  ·  
  ENIG {GOLD_THICKNESS_UM}µm  ·  ρ={GOLD_DENSITY_G_CM3} g/cm³
</div>
""", unsafe_allow_html=True)