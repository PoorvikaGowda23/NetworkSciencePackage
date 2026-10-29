"""
generate_ppt.py
===============
Generates a clean PowerPoint presentation (no background images/elements)
for the Network Science course project on memory in temporal networks.
Speaker notes are written for two presenters (Speaker A and Speaker B).
"""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pathlib import Path
import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS_DIR = Path(r"d:\NS Package\NetworkSciencePackage\results")
OUTPUT_PATH = Path(r"d:\NS Package\NetworkSciencePackage\presentation.pptx")

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------
SLIDE_WIDTH = Inches(13.333)
SLIDE_HEIGHT = Inches(7.5)

# Colors
DARK_BG      = RGBColor(0x1A, 0x1A, 0x2E)   # Deep navy
ACCENT_BLUE  = RGBColor(0x4E, 0xA8, 0xDE)   # Bright blue
ACCENT_TEAL  = RGBColor(0x2E, 0xCC, 0xA8)   # Teal green
WHITE        = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT_GRAY   = RGBColor(0xCC, 0xCC, 0xCC)
MID_GRAY     = RGBColor(0x99, 0x99, 0x99)
DARK_TEXT     = RGBColor(0x2D, 0x2D, 0x2D)
ORANGE       = RGBColor(0xE8, 0x8D, 0x3F)
SOFT_RED     = RGBColor(0xE0, 0x5A, 0x5A)


def set_slide_bg(slide, color):
    """Set solid background color for a slide."""
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_text_box(slide, left, top, width, height, text, font_size=18,
                 color=WHITE, bold=False, alignment=PP_ALIGN.LEFT,
                 font_name="Calibri", line_spacing=1.15):
    """Add a text box with specified formatting."""
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(font_size)
    p.font.color.rgb = color
    p.font.bold = bold
    p.font.name = font_name
    p.alignment = alignment
    p.line_spacing = Pt(int(font_size * line_spacing))
    return tf


def add_para(tf, text, font_size=18, color=WHITE, bold=False,
             alignment=PP_ALIGN.LEFT, font_name="Calibri",
             space_before=Pt(6), space_after=Pt(4), level=0):
    """Add a paragraph to an existing text frame."""
    p = tf.add_paragraph()
    p.text = text
    p.font.size = Pt(font_size)
    p.font.color.rgb = color
    p.font.bold = bold
    p.font.name = font_name
    p.alignment = alignment
    p.level = level
    if space_before:
        p.space_before = space_before
    if space_after:
        p.space_after = space_after
    return p


def add_accent_line(slide, left, top, width, color=ACCENT_BLUE):
    """Add a thin horizontal accent line."""
    shape = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, left, top, width, Pt(3)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()


def add_image_safe(slide, img_name, left, top, width, height=None):
    """Add an image if it exists, otherwise add a placeholder box."""
    img_path = RESULTS_DIR / img_name
    if img_path.exists():
        if height:
            slide.shapes.add_picture(str(img_path), left, top, width, height)
        else:
            slide.shapes.add_picture(str(img_path), left, top, width)
    else:
        shape = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height or Inches(3)
        )
        shape.fill.solid()
        shape.fill.fore_color.rgb = RGBColor(0x33, 0x33, 0x50)
        tf = shape.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = f"[Figure: {img_name}]"
        p.font.size = Pt(12)
        p.font.color.rgb = MID_GRAY
        p.alignment = PP_ALIGN.CENTER


def set_notes(slide, text):
    """Set the speaker notes for a slide."""
    notes_slide = slide.notes_slide
    notes_slide.notes_text_frame.text = text


# ============================================================================
# BUILD PRESENTATION
# ============================================================================

prs = Presentation()
prs.slide_width = SLIDE_WIDTH
prs.slide_height = SLIDE_HEIGHT
blank_layout = prs.slide_layouts[6]  # Blank layout


# ====================== SLIDE 1 — TITLE ====================================
slide = prs.slides.add_slide(blank_layout)
set_slide_bg(slide, DARK_BG)

add_accent_line(slide, Inches(1.5), Inches(2.3), Inches(3), ACCENT_TEAL)

add_text_box(slide, Inches(1.5), Inches(2.5), Inches(10), Inches(1.5),
             "Modeling Memory in\nTime-Respecting Paths on Temporal Networks",
             font_size=38, color=WHITE, bold=True, font_name="Calibri Light")

add_text_box(slide, Inches(1.5), Inches(4.2), Inches(9), Inches(0.6),
             "Based on: Guerrini, Cattuto & Dall'Amico — EPJ Data Science (2026)",
             font_size=18, color=ACCENT_BLUE, font_name="Calibri")

add_text_box(slide, Inches(1.5), Inches(5.0), Inches(9), Inches(0.6),
             "Network Science Course Project",
             font_size=16, color=LIGHT_GRAY, font_name="Calibri")

add_text_box(slide, Inches(1.5), Inches(5.6), Inches(9), Inches(0.6),
             "Presented by: Speaker A & Speaker B",
             font_size=15, color=MID_GRAY, font_name="Calibri")

set_notes(slide, """[SPEAKER A]

Good afternoon everyone. Today we'll be walking you through our course project on modeling memory in temporal networks.

This work is based on a recent 2026 paper by Guerrini, Cattuto, and Dall'Amico published in EPJ Data Science. We didn't just replicate the paper — we built a full Python package around it and added several extensions of our own, including epidemic simulations and targeted intervention strategies.

Let me hand it over to [Speaker B] who'll start by setting up the problem.""")


# ====================== SLIDE 2 — MOTIVATION ================================
slide = prs.slides.add_slide(blank_layout)
set_slide_bg(slide, DARK_BG)

add_text_box(slide, Inches(0.8), Inches(0.4), Inches(8), Inches(0.7),
             "Why Does Memory Matter in Networks?",
             font_size=32, color=WHITE, bold=True, font_name="Calibri Light")
add_accent_line(slide, Inches(0.8), Inches(1.15), Inches(2.5), ACCENT_TEAL)

tf = add_text_box(slide, Inches(0.8), Inches(1.5), Inches(5.5), Inches(5.5),
                  "• Real human contact networks are temporal — interactions happen at specific times, not all at once",
                  font_size=17, color=LIGHT_GRAY)
add_para(tf, "", font_size=8, color=LIGHT_GRAY)
add_para(tf, "• A static graph loses the arrow of time: who met whom first matters for disease spread",
         font_size=17, color=LIGHT_GRAY)
add_para(tf, "", font_size=8, color=LIGHT_GRAY)
add_para(tf, "• People don't walk randomly — they tend to revisit recent contacts (memory)",
         font_size=17, color=WHITE, bold=True)
add_para(tf, "", font_size=8, color=LIGHT_GRAY)
add_para(tf, "• This memory effect is invisible in standard models (Markov random walks)",
         font_size=17, color=LIGHT_GRAY)
add_para(tf, "", font_size=8, color=LIGHT_GRAY)
add_para(tf, "• Key question: Can we measure and model this memory? What are its consequences?",
         font_size=17, color=ACCENT_BLUE, bold=True)

# Right side — conceptual box
shape = slide.shapes.add_shape(
    MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.0), Inches(1.5), Inches(5.5), Inches(4.5)
)
shape.fill.solid()
shape.fill.fore_color.rgb = RGBColor(0x25, 0x25, 0x40)
shape.line.color.rgb = ACCENT_BLUE
shape.line.width = Pt(1.5)
tf2 = shape.text_frame
tf2.word_wrap = True
p = tf2.paragraphs[0]
p.text = "Temporal Network: Nodes + Time-Stamped Edges"
p.font.size = Pt(16)
p.font.bold = True
p.font.color.rgb = ACCENT_BLUE
p.alignment = PP_ALIGN.CENTER

add_para(tf2, "", font_size=6, color=LIGHT_GRAY)
add_para(tf2, "    t₁: Alice — Bob", font_size=15, color=WHITE, font_name="Consolas")
add_para(tf2, "    t₂: Bob — Carol", font_size=15, color=WHITE, font_name="Consolas")
add_para(tf2, "    t₃: Carol — Alice  ← return!", font_size=15, color=ACCENT_TEAL, font_name="Consolas", bold=True)
add_para(tf2, "    t₄: Alice — Bob    ← return!", font_size=15, color=ACCENT_TEAL, font_name="Consolas", bold=True)
add_para(tf2, "", font_size=6, color=LIGHT_GRAY)
add_para(tf2, "People revisit recent contacts more", font_size=14, color=ORANGE, alignment=PP_ALIGN.CENTER)
add_para(tf2, "often than a random walk would predict.", font_size=14, color=ORANGE, alignment=PP_ALIGN.CENTER)

set_notes(slide, """[SPEAKER B]

So let's start with why this matters. When we study networks — say, who's in contact with whom in a school or hospital — most models treat these as static graphs. But real contact patterns unfold over time.

Think about your own day: you talk to a colleague, go to a meeting, come back, and talk to the same colleague again. You're revisiting recent contacts. This isn't random — it's a form of memory.

Standard Markov random walk models completely miss this. They assume every step is independent of the history. But as you can see on the right, in reality people return to recently visited nodes far more often than chance would predict.

The central question of this paper is: can we quantify this memory as a single parameter p, and does it actually matter for things like epidemic spreading?""")


# ====================== SLIDE 3 — PAPER OVERVIEW ===========================
slide = prs.slides.add_slide(blank_layout)
set_slide_bg(slide, DARK_BG)

add_text_box(slide, Inches(0.8), Inches(0.4), Inches(10), Inches(0.7),
             "The Paper at a Glance",
             font_size=32, color=WHITE, bold=True, font_name="Calibri Light")
add_accent_line(slide, Inches(0.8), Inches(1.15), Inches(2.5), ACCENT_TEAL)

# Left column
tf = add_text_box(slide, Inches(0.8), Inches(1.5), Inches(5.8), Inches(5.5),
                  "Core Contributions",
                  font_size=20, color=ACCENT_BLUE, bold=True)
add_para(tf, "", font_size=6, color=LIGHT_GRAY)
add_para(tf, "1.  Time-Respecting Paths (TRPs)", font_size=17, color=WHITE, bold=True)
add_para(tf, "     Non-backtracking walks that respect the arrow of time (t₁ ≤ t₂ ≤ ...)", font_size=15, color=LIGHT_GRAY, level=1)
add_para(tf, "", font_size=6, color=LIGHT_GRAY)
add_para(tf, "2.  Memory Model (MEM)", font_size=17, color=WHITE, bold=True)
add_para(tf, "     At each step, with probability p the walker returns to a node in its", font_size=15, color=LIGHT_GRAY, level=1)
add_para(tf, "     memory set M (the last m unique visited nodes)", font_size=15, color=LIGHT_GRAY, level=1)
add_para(tf, "", font_size=6, color=LIGHT_GRAY)
add_para(tf, "3.  MEM+SBM Extension", font_size=17, color=WHITE, bold=True)
add_para(tf, "     Combines memory with community structure via an affinity matrix C", font_size=15, color=LIGHT_GRAY, level=1)
add_para(tf, "", font_size=6, color=LIGHT_GRAY)
add_para(tf, "4.  Null Models", font_size=17, color=WHITE, bold=True)
add_para(tf, "     Erdős–Rényi (destroys topology) and Time-Shuffled (destroys temporal order)", font_size=15, color=LIGHT_GRAY, level=1)

# Right column — formula box
shape = slide.shapes.add_shape(
    MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.2), Inches(1.5), Inches(5.3), Inches(3.0)
)
shape.fill.solid()
shape.fill.fore_color.rgb = RGBColor(0x25, 0x25, 0x40)
shape.line.color.rgb = ACCENT_TEAL
shape.line.width = Pt(1.5)
tf2 = shape.text_frame
tf2.word_wrap = True
p = tf2.paragraphs[0]
p.text = "MEM Transition Probability"
p.font.size = Pt(16)
p.font.bold = True
p.font.color.rgb = ACCENT_TEAL
p.alignment = PP_ALIGN.CENTER
add_para(tf2, "", font_size=8, color=LIGHT_GRAY)
add_para(tf2, "  If next node ∈ Memory set M:", font_size=15, color=WHITE, font_name="Consolas")
add_para(tf2, "    P = p / |M| + (1−p) / (n−2)", font_size=15, color=ACCENT_TEAL, font_name="Consolas", bold=True)
add_para(tf2, "", font_size=6, color=LIGHT_GRAY)
add_para(tf2, "  If next node ∉ Memory set M:", font_size=15, color=WHITE, font_name="Consolas")
add_para(tf2, "    P = (1−p) / (n−2)", font_size=15, color=ORANGE, font_name="Consolas", bold=True)
add_para(tf2, "", font_size=6, color=LIGHT_GRAY)
add_para(tf2, "  p = 0 → pure random walk (no memory)", font_size=14, color=MID_GRAY, font_name="Consolas")
add_para(tf2, "  p = 1 → always return to a recent contact", font_size=14, color=MID_GRAY, font_name="Consolas")

# Datasets box
shape2 = slide.shapes.add_shape(
    MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.2), Inches(4.8), Inches(5.3), Inches(2.2)
)
shape2.fill.solid()
shape2.fill.fore_color.rgb = RGBColor(0x25, 0x25, 0x40)
shape2.line.color.rgb = ORANGE
shape2.line.width = Pt(1.5)
tf3 = shape2.text_frame
tf3.word_wrap = True
p = tf3.paragraphs[0]
p.text = "8 Real-World Proximity Datasets"
p.font.size = Pt(16)
p.font.bold = True
p.font.color.rgb = ORANGE
p.alignment = PP_ALIGN.CENTER
add_para(tf3, "", font_size=4, color=LIGHT_GRAY)
add_para(tf3, "  Primary School  •  High School (2011, 2012, 2013)", font_size=14, color=LIGHT_GRAY)
add_para(tf3, "  Workplace  •  Hospital  •  Conference  •  Malawi Village", font_size=14, color=LIGHT_GRAY)
add_para(tf3, "  Face-to-face contacts recorded via wearable sensors", font_size=13, color=MID_GRAY)

set_notes(slide, """[SPEAKER A]

Let me give you a quick overview of what this paper does.

The fundamental object is a Time-Respecting Path, or TRP. Imagine tracing a chain of contacts through time: person A meets B at time 1, then B meets C at time 2, and so on. The key rule is that time can only move forward, and you can't immediately go back to where you just came from — that's the non-backtracking constraint.

The paper's main model is called MEM. At each step along a TRP, the walker either returns to a recently visited node — one that's in its memory set of the last m contacts — with probability p, or it goes to any other node uniformly. So p is literally the probability of "remembering" a recent contact.

They also extend this to MEM+SBM, which accounts for community structure. If you're in a school, you're more likely to interact within your own class. The affinity matrix C captures how much more likely within-community vs. between-community transitions are.

To validate that the measured memory is real and not just an artifact, they use two null models: an Erdős-Rényi random graph that destroys all structure, and a time-shuffled null that keeps the topology but scrambles the temporal order.

All of this is tested on 8 real proximity datasets from schools, workplaces, hospitals, and more.""")


# ====================== SLIDE 4 — OUR IMPLEMENTATION =======================
slide = prs.slides.add_slide(blank_layout)
set_slide_bg(slide, DARK_BG)

add_text_box(slide, Inches(0.8), Inches(0.4), Inches(10), Inches(0.7),
             "Our Implementation: Package Architecture",
             font_size=32, color=WHITE, bold=True, font_name="Calibri Light")
add_accent_line(slide, Inches(0.8), Inches(1.15), Inches(2.5), ACCENT_TEAL)

# Left — module list
tf = add_text_box(slide, Inches(0.8), Inches(1.6), Inches(5.5), Inches(5.5),
                  "Core Modules (src/memory_trp/)",
                  font_size=18, color=ACCENT_BLUE, bold=True)
add_para(tf, "", font_size=4, color=LIGHT_GRAY)

modules = [
    ("loading.py", "Parse 8 raw CSV datasets, remap node IDs"),
    ("aggregation.py", "Bin contacts into t_res snapshots, split by day"),
    ("trp.py", "Sample non-backtracking TRPs (Algorithm 1)"),
    ("records.py", "Extract (current, next, memory_set) transition records"),
    ("models.py", "MEM fixed-point solver, MEM+SBM (L-BFGS-B), BIC"),
    ("nulls.py", "Erdős–Rényi and Time-Shuffled null graph generators"),
    ("bootstrap.py", "Non-parametric bootstrap CIs (B = 200)"),
    ("timevarying.py", "Clock-time binned memory p(t)"),
    ("generator.py", "Synthetic memory network generator"),
    ("diffusion.py", "Heat diffusion on synthetic networks"),
    ("sir.py", "SIR epidemic + rate calibration + interventions"),
    ("plotting.py", "All Matplotlib figure routines"),
]
for mod, desc in modules:
    add_para(tf, f"  {mod:20s}  {desc}", font_size=13, color=LIGHT_GRAY, font_name="Consolas",
             space_before=Pt(2), space_after=Pt(1))

# Right — stats box
shape = slide.shapes.add_shape(
    MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.2), Inches(1.6), Inches(5.3), Inches(2.5)
)
shape.fill.solid()
shape.fill.fore_color.rgb = RGBColor(0x25, 0x25, 0x40)
shape.line.color.rgb = ACCENT_TEAL
shape.line.width = Pt(1.5)
tf2 = shape.text_frame
tf2.word_wrap = True
p = tf2.paragraphs[0]
p.text = "Package at a Glance"
p.font.size = Pt(18)
p.font.bold = True
p.font.color.rgb = ACCENT_TEAL
p.alignment = PP_ALIGN.CENTER
add_para(tf2, "", font_size=6, color=LIGHT_GRAY)
add_para(tf2, "  12 core modules   •   11 experiment scripts", font_size=15, color=WHITE, alignment=PP_ALIGN.CENTER)
add_para(tf2, "  44 unit tests  •  pip-installable package", font_size=15, color=WHITE, alignment=PP_ALIGN.CENTER)
add_para(tf2, "  Interactive Streamlit dashboard (app.py)", font_size=15, color=ACCENT_TEAL, alignment=PP_ALIGN.CENTER)
add_para(tf2, "  Jupyter notebook demo", font_size=15, color=WHITE, alignment=PP_ALIGN.CENTER)

# Design decisions box
shape2 = slide.shapes.add_shape(
    MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.2), Inches(4.4), Inches(5.3), Inches(2.8)
)
shape2.fill.solid()
shape2.fill.fore_color.rgb = RGBColor(0x25, 0x25, 0x40)
shape2.line.color.rgb = ORANGE
shape2.line.width = Pt(1.5)
tf3 = shape2.text_frame
tf3.word_wrap = True
p = tf3.paragraphs[0]
p.text = "Key Design Choices"
p.font.size = Pt(18)
p.font.bold = True
p.font.color.rgb = ORANGE
p.alignment = PP_ALIGN.CENTER
add_para(tf3, "", font_size=4, color=LIGHT_GRAY)
add_para(tf3, "  ✓  All parameters in config.py (no magic numbers)", font_size=14, color=LIGHT_GRAY)
add_para(tf3, "  ✓  Explicit RNG seeds for full reproducibility", font_size=14, color=LIGHT_GRAY)
add_para(tf3, "  ✓  Damped Mann iteration for guaranteed convergence", font_size=14, color=LIGHT_GRAY)
add_para(tf3, "  ✓  L-BFGS-B on log(C) for numeric stability", font_size=14, color=LIGHT_GRAY)
add_para(tf3, "  ✓  Type hints + docstrings on every function", font_size=14, color=LIGHT_GRAY)
add_para(tf3, "  ✓  Comprehensive pytest test suite", font_size=14, color=LIGHT_GRAY)

set_notes(slide, """[SPEAKER B]

Here's how we structured the implementation. We built a proper installable Python package called memory_trp with 12 core modules.

The pipeline flows naturally: loading parses the raw CSV files, aggregation bins contacts into discrete time windows, trp.py samples the paths, records.py extracts the transition data, and models.py does the statistical inference.

A few things we're proud of in terms of engineering:
- Every single tunable parameter sits in one config.py file. There are zero magic numbers anywhere else.
- We use explicit NumPy random generators with seeds, so every result is exactly reproducible.
- The MEM fixed-point solver uses Mann damping to guarantee convergence, which the paper doesn't explicitly detail.
- For the community affinity matrix C in MEM+SBM, we use L-BFGS-B optimization in log-space rather than the paper's implicit equations, because it's more numerically stable.
- We wrote 44 unit tests covering everything from basic data loading to SIR epidemic correctness.""")


# ====================== SLIDE 5 — TRP ALGORITHM ============================
slide = prs.slides.add_slide(blank_layout)
set_slide_bg(slide, DARK_BG)

add_text_box(slide, Inches(0.8), Inches(0.4), Inches(10), Inches(0.7),
             "Time-Respecting Paths: How We Sample Them",
             font_size=32, color=WHITE, bold=True, font_name="Calibri Light")
add_accent_line(slide, Inches(0.8), Inches(1.15), Inches(2.5), ACCENT_TEAL)

# Algorithm steps
tf = add_text_box(slide, Inches(0.8), Inches(1.6), Inches(5.8), Inches(5.5),
                  "Algorithm (from paper's Algorithm 1)",
                  font_size=18, color=ACCENT_BLUE, bold=True)
add_para(tf, "", font_size=6, color=LIGHT_GRAY)
add_para(tf, "1.  Pick a random snapshot t₀ and a random active edge (u, v)", font_size=16, color=WHITE)
add_para(tf, "     → This is the seed: path = [u, v] starting at time t₀", font_size=14, color=MID_GRAY, level=1)
add_para(tf, "", font_size=6, color=LIGHT_GRAY)
add_para(tf, "2.  Find the end-of-interaction: the last consecutive snapshot", font_size=16, color=WHITE)
add_para(tf, "     where u and v are still in contact", font_size=16, color=WHITE)
add_para(tf, "", font_size=6, color=LIGHT_GRAY)
add_para(tf, "3.  At the next available snapshot (t ≥ t_end), look at all", font_size=16, color=WHITE)
add_para(tf, "     neighbors of v that are NOT u (non-backtracking)", font_size=16, color=WHITE)
add_para(tf, "", font_size=6, color=LIGHT_GRAY)
add_para(tf, "4.  Pick the next node w proportional to contact weights", font_size=16, color=WHITE)
add_para(tf, "", font_size=6, color=LIGHT_GRAY)
add_para(tf, "5.  Repeat until path reaches length L = 30 steps", font_size=16, color=ACCENT_TEAL, bold=True)

# Right box — parameters
shape = slide.shapes.add_shape(
    MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.2), Inches(1.6), Inches(5.3), Inches(2.5)
)
shape.fill.solid()
shape.fill.fore_color.rgb = RGBColor(0x25, 0x25, 0x40)
shape.line.color.rgb = ACCENT_TEAL
shape.line.width = Pt(1.5)
tf2 = shape.text_frame
tf2.word_wrap = True
p = tf2.paragraphs[0]
p.text = "Our Parameters"
p.font.size = Pt(18)
p.font.bold = True
p.font.color.rgb = ACCENT_TEAL
p.alignment = PP_ALIGN.CENTER
add_para(tf2, "", font_size=6, color=LIGHT_GRAY)
add_para(tf2, "  Path length L = 30 steps", font_size=15, color=WHITE)
add_para(tf2, "  20,000 paths per measurement", font_size=15, color=WHITE)
add_para(tf2, "  t_res = 20 seconds (snapshot window)", font_size=15, color=WHITE)
add_para(tf2, "  Memory horizon m ∈ {1, 2, ..., 20}", font_size=15, color=WHITE)
add_para(tf2, "  Max 1000 retries per path", font_size=15, color=MID_GRAY)

# Constraint box
shape2 = slide.shapes.add_shape(
    MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.2), Inches(4.5), Inches(5.3), Inches(2.5)
)
shape2.fill.solid()
shape2.fill.fore_color.rgb = RGBColor(0x25, 0x25, 0x40)
shape2.line.color.rgb = ORANGE
shape2.line.width = Pt(1.5)
tf3 = shape2.text_frame
tf3.word_wrap = True
p = tf3.paragraphs[0]
p.text = "Two Key Constraints"
p.font.size = Pt(18)
p.font.bold = True
p.font.color.rgb = ORANGE
p.alignment = PP_ALIGN.CENTER
add_para(tf3, "", font_size=6, color=LIGHT_GRAY)
add_para(tf3, "  ⏱  Time-respecting:", font_size=15, color=WHITE, bold=True)
add_para(tf3, "     t_{a+1} ≥ t_a  (time only moves forward)", font_size=14, color=LIGHT_GRAY)
add_para(tf3, "", font_size=6, color=LIGHT_GRAY)
add_para(tf3, "  ↩  Non-backtracking:", font_size=15, color=WHITE, bold=True)
add_para(tf3, "     v_{a+1} ≠ v_{a−1}  (can't go right back)", font_size=14, color=LIGHT_GRAY)

set_notes(slide, """[SPEAKER A]

Now let's get into the meat of it. The first thing we need is to actually sample these time-respecting paths from the data.

The algorithm is pretty intuitive. You pick a random moment in time — a random snapshot — and a random edge that's active at that moment. That gives you the first two nodes of your path.

Then you follow the "end of interaction" — basically, if Alice and Bob are in contact for 5 consecutive snapshots, you wait until that interaction ends before looking for the next step.

At the next snapshot, you look at all of Bob's contacts except Alice — that's the non-backtracking part — and you pick the next person proportional to how many contacts Bob has with them.

You keep doing this for 30 steps. We sample 20,000 such paths per measurement to get stable statistics. If a path hits a dead end — someone with no future contacts — we just discard it and try again, up to 1000 retries per path.""")


# ====================== SLIDE 6 — MEMORY MODEL =============================
slide = prs.slides.add_slide(blank_layout)
set_slide_bg(slide, DARK_BG)

add_text_box(slide, Inches(0.8), Inches(0.4), Inches(10), Inches(0.7),
             "The MEM Model: Estimating Memory Parameter p",
             font_size=32, color=WHITE, bold=True, font_name="Calibri Light")
add_accent_line(slide, Inches(0.8), Inches(1.15), Inches(2.5), ACCENT_TEAL)

# Left — model description
tf = add_text_box(slide, Inches(0.8), Inches(1.6), Inches(5.8), Inches(5.5),
                  "From Paths to Records to Likelihood",
                  font_size=18, color=ACCENT_BLUE, bold=True)
add_para(tf, "", font_size=6, color=LIGHT_GRAY)
add_para(tf, "Step 1: Build Transition Records", font_size=17, color=WHITE, bold=True)
add_para(tf, "  For each step in each path, record:", font_size=15, color=LIGHT_GRAY)
add_para(tf, "    • Current node (where are we now)", font_size=15, color=LIGHT_GRAY)
add_para(tf, "    • Next node (where do we go)", font_size=15, color=LIGHT_GRAY)
add_para(tf, "    • Memory set M (last m unique visited nodes)", font_size=15, color=LIGHT_GRAY)
add_para(tf, "    • Is the next node in M?  → \"memory return\" event", font_size=15, color=ACCENT_TEAL)
add_para(tf, "", font_size=6, color=LIGHT_GRAY)
add_para(tf, "Step 2: Maximum Likelihood Estimation", font_size=17, color=WHITE, bold=True)
add_para(tf, "  Write the log-likelihood over all records", font_size=15, color=LIGHT_GRAY)
add_para(tf, "  Differentiate, set to zero → fixed-point equation for p", font_size=15, color=LIGHT_GRAY)
add_para(tf, "", font_size=6, color=LIGHT_GRAY)
add_para(tf, "Step 3: Iterate Until Convergence", font_size=17, color=WHITE, bold=True)
add_para(tf, "  p_new = (fraction of returns) adjusted for overlap", font_size=15, color=LIGHT_GRAY)
add_para(tf, "  Apply Mann damping (α = 0.15) for stability", font_size=15, color=ACCENT_TEAL)

# Right — results highlight
shape = slide.shapes.add_shape(
    MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.2), Inches(1.6), Inches(5.3), Inches(5.0)
)
shape.fill.solid()
shape.fill.fore_color.rgb = RGBColor(0x25, 0x25, 0x40)
shape.line.color.rgb = ACCENT_TEAL
shape.line.width = Pt(1.5)
tf2 = shape.text_frame
tf2.word_wrap = True
p = tf2.paragraphs[0]
p.text = "Inferred Memory p (m = 10)"
p.font.size = Pt(18)
p.font.bold = True
p.font.color.rgb = ACCENT_TEAL
p.alignment = PP_ALIGN.CENTER
add_para(tf2, "", font_size=6, color=LIGHT_GRAY)

results_data = [
    ("Primary School", "0.51", "~0.001"),
    ("High School 2011", "0.54", "~0.001"),
    ("High School 2012", "0.65", "~0.001"),
    ("High School 2013", "0.64", "~0.001"),
    ("Workplace", "0.61", "~0.002"),
    ("Hospital", "0.64", "~0.001"),
    ("Conference", "0.56", "~0.001"),
    ("Malawi Village", "0.97", "~0.004"),
]
add_para(tf2, f"  {'Dataset':<22} {'p':>6}   {'ER null':>8}", font_size=13, color=ACCENT_BLUE, font_name="Consolas", bold=True)
for ds, pv, er in results_data:
    add_para(tf2, f"  {ds:<22} {pv:>6}   {er:>8}", font_size=13, color=WHITE, font_name="Consolas",
             space_before=Pt(1), space_after=Pt(1))
add_para(tf2, "", font_size=6, color=LIGHT_GRAY)
add_para(tf2, "  All datasets: p >> 0 (strong memory)", font_size=14, color=ORANGE, bold=True, alignment=PP_ALIGN.CENTER)
add_para(tf2, "  ER null: p ≈ 0 (no memory, as expected)", font_size=14, color=MID_GRAY, alignment=PP_ALIGN.CENTER)

set_notes(slide, """[SPEAKER B]

So how do we actually estimate p? It's a three-step process.

First, we convert our sampled paths into "transition records." For each step in each path, we record where we are, where we go next, and what the memory set looks like — that is, the last m unique nodes we visited. The key binary label is: did the next node fall inside the memory set or not?

Second, we write out the full log-likelihood. Each record contributes either log(p/|M| + (1-p)/(n-2)) if it's a return, or log((1-p)/(n-2)) if it's not. We differentiate this with respect to p, set to zero, and get a fixed-point equation.

Third, we iterate. We use Mann damping with alpha = 0.15 — this means p_new = alpha * p_old + (1-alpha) * update. This guarantees convergence, which a naive iteration doesn't always do.

On the right you can see our results at memory horizon m = 10 across all 8 datasets. Every single dataset shows strong memory — p ranges from 0.51 to 0.97. Meanwhile the Erdős-Rényi null gives p essentially zero. This confirms that the memory we're measuring is real and not an artifact of the network's degree distribution.""")


# ====================== SLIDE 7 — MEMORY CURVES ===========================
slide = prs.slides.add_slide(blank_layout)
set_slide_bg(slide, DARK_BG)

add_text_box(slide, Inches(0.8), Inches(0.4), Inches(10), Inches(0.7),
             "Results: Memory p vs. Horizon m",
             font_size=32, color=WHITE, bold=True, font_name="Calibri Light")
add_accent_line(slide, Inches(0.8), Inches(1.15), Inches(2.5), ACCENT_TEAL)

add_text_box(slide, Inches(0.8), Inches(1.4), Inches(11), Inches(0.5),
             "Memory increases with look-back window m and plateaus — people have a finite contact memory horizon",
             font_size=16, color=LIGHT_GRAY)

# Add the combined plot
add_image_safe(slide, "step15_p_vs_m_all_datasets.png",
               Inches(1.0), Inches(2.0), Inches(11.3), Inches(5.0))

set_notes(slide, """[SPEAKER A]

This is one of the key results. Each subplot shows one dataset, and we're plotting the inferred memory p on the y-axis against the memory horizon m on the x-axis.

The blue line with confidence bands is the empirical data. The green dashed line is the time-shuffled null, and the orange dotted line is the ER null.

A few things to notice. First, memory p increases with m and then plateaus — this makes sense because people have a finite "contact memory" of maybe 5-10 recent contacts. Looking back further doesn't add much new information.

Second, the empirical p is consistently and significantly above the ER null, which hovers near zero. This is the fundamental result: real networks have memory that random graphs don't.

Third, the time-shuffled null is interesting. For some datasets like the Workplace, shuffling time dramatically reduces memory, meaning temporal ordering matters a lot. For others like Conference, shuffling actually increases apparent memory, suggesting that the static topology itself generates some return patterns.

Fourth, look at Malawi — it has p near 0.97! This is a small rural village where people basically interact with the same handful of neighbors over and over.""")


# ====================== SLIDE 8 — BIC MODEL SELECTION ======================
slide = prs.slides.add_slide(blank_layout)
set_slide_bg(slide, DARK_BG)

add_text_box(slide, Inches(0.8), Inches(0.4), Inches(10), Inches(0.7),
             "Model Selection: MEM vs. MEM+SBM (BIC)",
             font_size=32, color=WHITE, bold=True, font_name="Calibri Light")
add_accent_line(slide, Inches(0.8), Inches(1.15), Inches(2.5), ACCENT_TEAL)

add_text_box(slide, Inches(0.8), Inches(1.4), Inches(11), Inches(0.5),
             "BIC penalises model complexity — MEM+SBM wins decisively for every labeled dataset (ΔBIC > 6,000)",
             font_size=16, color=LIGHT_GRAY)

add_image_safe(slide, "step08_bic_comparison.png",
               Inches(0.5), Inches(2.0), Inches(12.3), Inches(5.0))

set_notes(slide, """[SPEAKER B]

Now, is community structure actually important, or does the simple MEM model already capture everything? We use the Bayesian Information Criterion to answer this rigorously.

BIC balances goodness-of-fit against model complexity. A lower BIC is better. The ΔBIC between MEM and MEM+SBM is the key number — if it's above 10, that's "very strong evidence" for the more complex model.

And we see staggering ΔBIC values. For High School 2013, it's over 90,000. For the Workplace, over 50,000. Even the Hospital, the smallest labeled dataset, has ΔBIC over 6,000.

This means incorporating community structure via the affinity matrix C dramatically improves the model. People aren't just returning to recent contacts randomly — they're more likely to return to contacts within their own community. Students go back to classmates, not random students from other classes.

The MEM+SBM p is typically a bit lower than MEM p, because some of the "apparent memory" in MEM was actually community affinity being misattributed to individual memory.""")


# ====================== SLIDE 9 — RESOLUTION STUDY ========================
slide = prs.slides.add_slide(blank_layout)
set_slide_bg(slide, DARK_BG)

add_text_box(slide, Inches(0.8), Inches(0.4), Inches(10), Inches(0.7),
             "Results: Step-Based vs. Physical Time Resolution",
             font_size=32, color=WHITE, bold=True, font_name="Calibri Light")
add_accent_line(slide, Inches(0.8), Inches(1.15), Inches(2.5), ACCENT_TEAL)

add_text_box(slide, Inches(0.8), Inches(1.3), Inches(11), Inches(0.6),
             "Memory persists across temporal resolutions — it is a robust property, not a measurement artifact",
             font_size=16, color=LIGHT_GRAY)

# Two images side by side
add_image_safe(slide, "step10_fig05_06_heatmaps_side_by_side.png",
               Inches(0.5), Inches(2.0), Inches(6.0), Inches(5.0))

add_image_safe(slide, "step10_tau_curves_combined.png",
               Inches(6.8), Inches(2.0), Inches(6.0), Inches(5.0))

set_notes(slide, """[SPEAKER A]

One important question is: does our measurement of memory depend on how we define the time resolution? If we change the snapshot window from 20 seconds to 5 minutes or even 15 minutes, does memory disappear?

On the left, you see heatmaps of p across different combinations of snapshot resolution Δt and memory horizon m, for both MEM and MEM+SBM. The key takeaway is that memory is consistently present regardless of resolution — it's not an artifact of our choice of time window.

On the right, we convert from step-based memory to physical time duration τ in minutes. We're asking: how far back in physical time does the memory extend? The curves show that memory is strong for the first 10-15 minutes and then gradually saturates. This corresponds to the typical timescale of face-to-face interactions in these settings.""")


# ====================== SLIDE 10 — NULL MODELS ============================
slide = prs.slides.add_slide(blank_layout)
set_slide_bg(slide, DARK_BG)

add_text_box(slide, Inches(0.8), Inches(0.4), Inches(10), Inches(0.7),
             "Null Models: Isolating the Source of Memory",
             font_size=32, color=WHITE, bold=True, font_name="Calibri Light")
add_accent_line(slide, Inches(0.8), Inches(1.15), Inches(2.5), ACCENT_TEAL)

# Left — explanation
tf = add_text_box(slide, Inches(0.8), Inches(1.6), Inches(5.5), Inches(5.5),
                  "Two Null Models, Two Different Questions",
                  font_size=18, color=ACCENT_BLUE, bold=True)
add_para(tf, "", font_size=8, color=LIGHT_GRAY)
add_para(tf, "Erdős–Rényi (ER) Null", font_size=17, color=ACCENT_TEAL, bold=True)
add_para(tf, "  • Destroys ALL structure (topology + time)", font_size=15, color=LIGHT_GRAY)
add_para(tf, "  • Preserves only node count and edge density", font_size=15, color=LIGHT_GRAY)
add_para(tf, "  • Result: p ≈ 0 everywhere → confirms memory is real", font_size=15, color=WHITE, bold=True)
add_para(tf, "", font_size=8, color=LIGHT_GRAY)
add_para(tf, "Time-Shuffled Null", font_size=17, color=ORANGE, bold=True)
add_para(tf, "  • Keeps each snapshot's exact topology", font_size=15, color=LIGHT_GRAY)
add_para(tf, "  • Randomly permutes the order of snapshots", font_size=15, color=LIGHT_GRAY)
add_para(tf, "  • Destroys temporal ordering (causal chains)", font_size=15, color=LIGHT_GRAY)
add_para(tf, "  • Destroys bursty interaction continuity", font_size=15, color=LIGHT_GRAY)
add_para(tf, "  • If p_shuffled < p_real → temporal order matters!", font_size=15, color=WHITE, bold=True)

# Right — null comparison image
add_image_safe(slide, "step09_null_comparison_quick.png",
               Inches(6.5), Inches(1.5), Inches(6.3), Inches(5.5))

set_notes(slide, """[SPEAKER B]

Let's dig deeper into what generates this memory. We use two null models to decompose it.

The ER null destroys everything — it's just a random graph. If we still see memory on a random graph, something's wrong with our method. We don't — p is essentially zero on ER nulls, which validates our approach.

The time-shuffled null is more subtle. It keeps every single snapshot — every instantaneous contact pattern — exactly intact. What it destroys is the temporal ordering. Imagine shuffling the frames of a movie — each frame is unchanged, but the story makes no sense.

The chart on the right shows the comparison across all 8 datasets. For the Workplace, time-shuffling drops p significantly, from 0.61 to 0.55. This means that part of the memory in the Workplace comes from the temporal dynamics — the specific ordering of who meets whom when.

For Conference, interestingly, shuffling actually increases apparent memory. This suggests that in the conference setting, the static network topology itself creates return patterns, and the actual temporal ordering partially disrupts them.""")


# ====================== SLIDE 11 — TIME-VARYING MEMORY ====================
slide = prs.slides.add_slide(blank_layout)
set_slide_bg(slide, DARK_BG)

add_text_box(slide, Inches(0.8), Inches(0.4), Inches(10), Inches(0.7),
             "Time-Varying Memory: Circadian Dynamics",
             font_size=32, color=WHITE, bold=True, font_name="Calibri Light")
add_accent_line(slide, Inches(0.8), Inches(1.15), Inches(2.5), ACCENT_TEAL)

add_text_box(slide, Inches(0.8), Inches(1.3), Inches(11), Inches(0.5),
             "Memory is not constant throughout the day — it follows structured circadian patterns",
             font_size=16, color=LIGHT_GRAY)

# Two images side by side
add_image_safe(slide, "step14_time_varying_hourly.png",
               Inches(0.3), Inches(1.9), Inches(6.3), Inches(5.3))

add_image_safe(slide, "step14_time_varying_15min.png",
               Inches(6.8), Inches(1.9), Inches(6.3), Inches(5.3))

set_notes(slide, """[SPEAKER A]

This is one of our implementation results that extends the paper. We binned the transition records by clock time — both hourly and in 15-minute intervals — and fit MEM at each time window.

On the left is the hourly view across all labeled datasets. You can see clear circadian patterns. In the Primary School, memory peaks during class time and dips during breaks. In the Workplace, there's a distinctive lunch-time dip.

On the right is the finer 15-minute resolution, which reveals even more structure. For the High Schools, you can actually see the class-break-class rhythm in the memory curve.

This tells us something important: memory isn't a fixed property of a network. It's dynamic and depends on what people are doing at that moment. During structured activities like classes, people interact with a fixed group and show high return rates. During breaks, interactions become more exploratory and memory drops.""")


# ====================== SLIDE 12 — PER-COMMUNITY MEMORY ====================
slide = prs.slides.add_slide(blank_layout)
set_slide_bg(slide, DARK_BG)

add_text_box(slide, Inches(0.8), Inches(0.4), Inches(10), Inches(0.7),
             "Per-Community Memory: Who Has the Strongest Memory?",
             font_size=32, color=WHITE, bold=True, font_name="Calibri Light")
add_accent_line(slide, Inches(0.8), Inches(1.15), Inches(2.5), ACCENT_TEAL)

add_text_box(slide, Inches(0.8), Inches(1.3), Inches(11), Inches(0.5),
             "Different communities show different memory levels — patients in hospitals have weaker memory than nurses",
             font_size=16, color=LIGHT_GRAY)

# Two images
add_image_safe(slide, "step16_pc_primaryschool.png",
               Inches(0.3), Inches(2.0), Inches(4.0), Inches(5.0))
add_image_safe(slide, "step16_pc_hospital.png",
               Inches(4.5), Inches(2.0), Inches(4.0), Inches(5.0))
add_image_safe(slide, "step16_pc_workplace.png",
               Inches(8.7), Inches(2.0), Inches(4.0), Inches(5.0))

set_notes(slide, """[SPEAKER B]

Going even deeper, we can estimate a separate memory parameter p_c for each community within a dataset.

Here on the left is the Primary School broken down by class. Each bar is a different class, with 95% bootstrap confidence intervals. You can see that most classes have similar memory around 0.4-0.5, but there's meaningful variation.

The Hospital in the middle is really interesting. Nurses and medical staff show high memory — they return to the same patients and colleagues repeatedly. But patients show lower memory, which makes sense because their contact patterns are less structured.

On the right is the Workplace broken down by department. Some departments are clearly more insular than others — they interact mostly within their group.

Note that we hold the global affinity matrix C fixed from the full dataset fit and only re-estimate p per group. This ensures we're comparing apples to apples — the differences in p_c reflect genuine differences in memory behavior, not differences in community size or structure.""")


# ====================== SLIDE 13 — BOOTSTRAP CIs ===========================
slide = prs.slides.add_slide(blank_layout)
set_slide_bg(slide, DARK_BG)

add_text_box(slide, Inches(0.8), Inches(0.4), Inches(10), Inches(0.7),
             "Statistical Rigor: Bootstrap Confidence Intervals",
             font_size=32, color=WHITE, bold=True, font_name="Calibri Light")
add_accent_line(slide, Inches(0.8), Inches(1.15), Inches(2.5), ACCENT_TEAL)

tf = add_text_box(slide, Inches(0.8), Inches(1.5), Inches(5.5), Inches(3.5),
                  "Non-Parametric Bootstrap (B = 200)",
                  font_size=18, color=ACCENT_BLUE, bold=True)
add_para(tf, "", font_size=6, color=LIGHT_GRAY)
add_para(tf, "1.  Resample transition records with replacement", font_size=16, color=WHITE)
add_para(tf, "2.  Re-fit MEM on each bootstrap sample", font_size=16, color=WHITE)
add_para(tf, "3.  Take 2.5th and 97.5th percentiles → 95% CI", font_size=16, color=WHITE)
add_para(tf, "", font_size=8, color=LIGHT_GRAY)
add_para(tf, "Result: Tight confidence bands confirm", font_size=16, color=ACCENT_TEAL, bold=True)
add_para(tf, "that our estimates are statistically stable", font_size=16, color=ACCENT_TEAL, bold=True)
add_para(tf, "", font_size=6, color=LIGHT_GRAY)
add_para(tf, "Typical CI width: ±0.005 to ±0.015", font_size=15, color=MID_GRAY)

add_image_safe(slide, "step15_p_vs_m_primaryschool.png",
               Inches(6.5), Inches(1.3), Inches(6.3), Inches(5.8))

set_notes(slide, """[SPEAKER A]

All of our results come with uncertainty quantification. We use non-parametric bootstrap with B = 200 resamples.

The procedure is straightforward: we resample the transition records with replacement, re-fit the MEM model on each resample, and take the 2.5th and 97.5th percentiles as our 95% confidence interval.

The plot on the right shows the Primary School example. The shaded bands are the confidence intervals, and you can see they're tight — typically plus or minus 0.005 to 0.015. This means our estimates are reliable; if you repeated the path sampling with a different seed, you'd get essentially the same answer.

This is important because it confirms that the differences we see between datasets, between communities, and between null models are statistically significant, not noise.""")


# ====================== SLIDE 14 — ADD-ON: SYNTHETIC GENERATOR =============
slide = prs.slides.add_slide(blank_layout)
set_slide_bg(slide, DARK_BG)

add_text_box(slide, Inches(0.8), Inches(0.3), Inches(10), Inches(0.7),
             "★ Add-On: Synthetic Network Generator & Diffusion",
             font_size=30, color=ORANGE, bold=True, font_name="Calibri Light")
add_accent_line(slide, Inches(0.8), Inches(1.0), Inches(2.5), ORANGE)

add_text_box(slide, Inches(0.8), Inches(1.15), Inches(11), Inches(0.5),
             "Beyond the paper — we built a generative model to test how memory affects information spreading",
             font_size=16, color=LIGHT_GRAY)

# Left text
tf = add_text_box(slide, Inches(0.8), Inches(1.8), Inches(5.5), Inches(5.0),
                  "Synthetic Network Generator",
                  font_size=18, color=ACCENT_TEAL, bold=True)
add_para(tf, "", font_size=6, color=LIGHT_GRAY)
add_para(tf, "Generates temporal networks with controlled memory:", font_size=15, color=LIGHT_GRAY)
add_para(tf, "  • Mixing parameter α ∈ [0, 1]", font_size=15, color=WHITE)
add_para(tf, "    α = 0: pure Markov (no memory)", font_size=14, color=MID_GRAY, level=1)
add_para(tf, "    α = 1: maximum memory (always revisit)", font_size=14, color=MID_GRAY, level=1)
add_para(tf, "  • Configurable degree, community structure", font_size=15, color=WHITE)
add_para(tf, "  • 250 nodes, 300 time steps per graph", font_size=15, color=WHITE)
add_para(tf, "", font_size=8, color=LIGHT_GRAY)
add_para(tf, "Heat Diffusion Experiment", font_size=18, color=ACCENT_TEAL, bold=True)
add_para(tf, "", font_size=6, color=LIGHT_GRAY)
add_para(tf, "Run diffusion on synthetic graphs across α:", font_size=15, color=LIGHT_GRAY)
add_para(tf, "  • More memory → slower spreading", font_size=15, color=WHITE, bold=True)
add_para(tf, "  • Memory concentrates flow in local clusters", font_size=15, color=WHITE)
add_para(tf, "  • Measured via entropy of diffused mass", font_size=15, color=WHITE)

# Right — diffusion plot
add_image_safe(slide, "step13_diffusion.png",
               Inches(6.5), Inches(1.6), Inches(6.3), Inches(5.5))

set_notes(slide, """[SPEAKER B]

Now let's talk about what we added beyond the paper. This is one of our major extensions.

We built a synthetic temporal network generator where we can precisely control the amount of memory. The parameter alpha goes from 0 — a pure Markov network with no memory — to 1, maximum memory where nodes always revisit recent contacts.

We generated 10 graphs for each combination of alpha and average degree, then ran heat diffusion on each one to see how information spreads.

The key finding, shown in the plot on the right, is that more memory means slower spreading. When alpha is high, diffused information stays trapped in local clusters rather than spreading globally. This is measured by entropy — lower entropy means the information is more concentrated, less spread out.

This has real implications: in a network with strong memory, a piece of gossip or a virus might spread very quickly within a tight-knit group but take much longer to jump between groups. The degree also matters — higher-degree networks spread faster regardless of memory, but memory still acts as a brake.""")


# ====================== SLIDE 15 — ADD-ON: SIR ============================
slide = prs.slides.add_slide(blank_layout)
set_slide_bg(slide, DARK_BG)

add_text_box(slide, Inches(0.8), Inches(0.3), Inches(10), Inches(0.7),
             "★ Add-On: SIR Epidemic Spreading",
             font_size=30, color=ORANGE, bold=True, font_name="Calibri Light")
add_accent_line(slide, Inches(0.8), Inches(1.0), Inches(2.5), ORANGE)

add_text_box(slide, Inches(0.8), Inches(1.15), Inches(11), Inches(0.5),
             "Does temporal memory affect how epidemics spread on real contact networks?",
             font_size=16, color=LIGHT_GRAY)

# Add the summary figure
add_image_safe(slide, "step17_sir_summary.png",
               Inches(0.5), Inches(1.8), Inches(12.3), Inches(5.4))

set_notes(slide, """[SPEAKER A]

Our biggest extension is running actual SIR epidemic simulations — Susceptible, Infected, Recovered — on the real temporal networks and comparing them to the time-shuffled versions.

We calibrate the transmission rate for each dataset using bisection search, targeting a mean outbreak fraction around 40% on the real network. Then we run 200 simulations on both the real and shuffled networks using the same rate.

The results are striking. Look at the left panel — mean outbreak size. For almost every dataset, the time-shuffled network produces much larger outbreaks. The Primary School jumps from 16% to 89%. High School 2013 goes from 3% to 41%.

The middle panel shows time to peak. Epidemics on shuffled networks take longer to peak because the temporal scrambling creates more circuitous spread paths.

The right panel connects epidemic size to memory. There's a clear correlation — datasets with higher memory p tend to have smaller relative outbreaks compared to their shuffled versions. This makes sense: memory concentrates contacts within familiar groups, which creates a natural barrier to epidemic spread. The virus keeps re-infecting the same cluster rather than jumping to new ones.""")


# ====================== SLIDE 16 — ADD-ON: INTERVENTIONS ===================
slide = prs.slides.add_slide(blank_layout)
set_slide_bg(slide, DARK_BG)

add_text_box(slide, Inches(0.8), Inches(0.3), Inches(10), Inches(0.7),
             "★ Add-On: Targeted Intervention Strategies",
             font_size=30, color=ORANGE, bold=True, font_name="Calibri Light")
add_accent_line(slide, Inches(0.8), Inches(1.0), Inches(2.5), ORANGE)

add_text_box(slide, Inches(0.8), Inches(1.15), Inches(11), Inches(0.6),
             "If you can only isolate 10% of people, who should you pick to minimize epidemic spread?",
             font_size=16, color=LIGHT_GRAY)

add_image_safe(slide, "step18_interventions_grid.png",
               Inches(0.5), Inches(1.9), Inches(7.0), Inches(5.3))

# Right — strategy descriptions
tf = add_text_box(slide, Inches(7.8), Inches(1.9), Inches(5.0), Inches(5.3),
                  "Isolation Strategies Compared",
                  font_size=18, color=ACCENT_BLUE, bold=True)
add_para(tf, "", font_size=6, color=LIGHT_GRAY)
add_para(tf, "None", font_size=16, color=SOFT_RED, bold=True)
add_para(tf, "  No intervention (baseline)", font_size=14, color=MID_GRAY)
add_para(tf, "", font_size=4, color=LIGHT_GRAY)
add_para(tf, "Random", font_size=16, color=WHITE, bold=True)
add_para(tf, "  Remove 10% of nodes at random", font_size=14, color=MID_GRAY)
add_para(tf, "", font_size=4, color=LIGHT_GRAY)
add_para(tf, "Contact Time (highest total)", font_size=16, color=WHITE, bold=True)
add_para(tf, "  Remove super-spreaders by contact volume", font_size=14, color=MID_GRAY)
add_para(tf, "", font_size=4, color=LIGHT_GRAY)
add_para(tf, "Return Rate (our novel metric) ★", font_size=16, color=ACCENT_TEAL, bold=True)
add_para(tf, "  Remove nodes most likely to revisit", font_size=14, color=ACCENT_TEAL)
add_para(tf, "  recent contacts (memory-informed)", font_size=14, color=ACCENT_TEAL)
add_para(tf, "", font_size=4, color=LIGHT_GRAY)
add_para(tf, "Community Closure", font_size=16, color=WHITE, bold=True)
add_para(tf, "  Remove an entire community/group", font_size=14, color=MID_GRAY)
add_para(tf, "", font_size=8, color=LIGHT_GRAY)
add_para(tf, "Key finding: Contact Time and", font_size=15, color=ORANGE, bold=True)
add_para(tf, "Return Rate are most effective", font_size=15, color=ORANGE, bold=True)

set_notes(slide, """[SPEAKER B]

This is perhaps our most practically relevant extension. We ask: if a disease is spreading and you can only isolate 10% of the population, who should you isolate?

We compare five strategies. "None" is the baseline — no intervention. "Random" removes 10% at random. "Contact Time" targets people with the most total contact time — the traditional super-spreader approach. "Return Rate" is our novel metric that uses the memory framework — we target people who are most likely to revisit recent contacts. And "Community Closure" removes an entire group, like closing one classroom.

The grid on the left shows the results across all 8 datasets. The key findings are:

First, Contact Time targeting is generally the most effective strategy — removing the heaviest interactors significantly cuts outbreak size.

Second, our Return Rate strategy performs comparably in many cases, which is notable because it's a fundamentally different approach. It targets people based on their memory behavior rather than their raw contact volume.

Third, Community Closure can be very effective in structured environments like schools, where removing one class cuts off a major transmission pathway.

Fourth, random removal helps but is never as effective as targeted approaches — this justifies the effort of identifying high-risk individuals.""")


# ====================== SLIDE 17 — STREAMLIT APP ===========================
slide = prs.slides.add_slide(blank_layout)
set_slide_bg(slide, DARK_BG)

add_text_box(slide, Inches(0.8), Inches(0.3), Inches(10), Inches(0.7),
             "★ Add-On: Interactive Streamlit Dashboard",
             font_size=30, color=ORANGE, bold=True, font_name="Calibri Light")
add_accent_line(slide, Inches(0.8), Inches(1.0), Inches(2.5), ORANGE)

# Description
tf = add_text_box(slide, Inches(0.8), Inches(1.4), Inches(11.5), Inches(5.5),
                  "Live Web Application (app.py)",
                  font_size=20, color=ACCENT_BLUE, bold=True)
add_para(tf, "", font_size=6, color=LIGHT_GRAY)
add_para(tf, "We built an interactive Streamlit web app that lets anyone explore the results without writing code.", font_size=16, color=LIGHT_GRAY)
add_para(tf, "", font_size=8, color=LIGHT_GRAY)
add_para(tf, "Features:", font_size=17, color=WHITE, bold=True)
add_para(tf, "  •  Pick any of the 8 datasets from a dropdown", font_size=15, color=WHITE)
add_para(tf, "  •  Adjust temporal resolution t_res (20s to 600s) with a slider", font_size=15, color=WHITE)
add_para(tf, "  •  Change memory horizon m (1 to 20) in real time", font_size=15, color=WHITE)
add_para(tf, "  •  See inferred p for MEM, MEM+SBM, and both null models as metric cards", font_size=15, color=WHITE)
add_para(tf, "  •  Interactive memory curve (p vs m) across all horizons", font_size=15, color=WHITE)
add_para(tf, "  •  Community affinity matrix C heatmap for labeled datasets", font_size=15, color=WHITE)
add_para(tf, "  •  BIC model selection table with evidence assessment", font_size=15, color=WHITE)
add_para(tf, "  •  Dataset info: node counts, snapshot counts, raw data preview", font_size=15, color=WHITE)
add_para(tf, "", font_size=8, color=LIGHT_GRAY)
add_para(tf, "Technical details:", font_size=17, color=ACCENT_TEAL, bold=True)
add_para(tf, "  •  All computations cached with @st.cache_data — instant response on parameter changes", font_size=15, color=LIGHT_GRAY)
add_para(tf, "  •  Hierarchical caching: changing m doesn't resample paths, only re-builds records", font_size=15, color=LIGHT_GRAY)
add_para(tf, "  •  Launch with: streamlit run app.py", font_size=15, color=ACCENT_TEAL, font_name="Consolas")

set_notes(slide, """[SPEAKER A]

We also built an interactive dashboard using Streamlit. The idea is that someone who hasn't looked at our code — say a professor or a classmate — can explore all the results just by visiting a web page.

You pick a dataset from a dropdown, adjust the temporal resolution and memory horizon with sliders, and the dashboard instantly shows you the inferred p values, a plot of memory vs horizon, the community affinity matrix, and BIC model selection results.

The caching is hierarchical — if you just change m, the app doesn't need to re-sample paths or re-build the graph; it only re-creates the transition records and re-fits the model. This makes parameter exploration feel instant.

We think this kind of interactive exploration tool is really valuable for understanding the results intuitively. You can quickly see how memory changes across datasets, how it responds to different parameter choices, and how the null models compare.

You can launch it with just one command: streamlit run app.py.""")


# ====================== SLIDE 18 — TESTING & ENGINEERING ====================
slide = prs.slides.add_slide(blank_layout)
set_slide_bg(slide, DARK_BG)

add_text_box(slide, Inches(0.8), Inches(0.4), Inches(10), Inches(0.7),
             "Engineering Quality: Testing & Packaging",
             font_size=32, color=WHITE, bold=True, font_name="Calibri Light")
add_accent_line(slide, Inches(0.8), Inches(1.15), Inches(2.5), ACCENT_TEAL)

# Left column
tf = add_text_box(slide, Inches(0.8), Inches(1.6), Inches(5.5), Inches(5.5),
                  "44 Unit Tests Across 10 Modules",
                  font_size=18, color=ACCENT_BLUE, bold=True)
add_para(tf, "", font_size=6, color=LIGHT_GRAY)

tests = [
    ("test_aggregation.py", "Snapshot binning, day splitting"),
    ("test_bootstrap.py", "Resampling, CI computation"),
    ("test_diffusion.py", "Heat flow, entropy bounds"),
    ("test_models.py", "MEM recovery, likelihood, BIC"),
    ("test_nulls.py", "ER/shuffled invariants"),
    ("test_records.py", "Memory set tracking"),
    ("test_sbm.py", "MEM+SBM optimization"),
    ("test_sir.py", "SIR dynamics, calibration"),
    ("test_timevarying.py", "Time binning, fixed-C solver"),
    ("test_trp.py", "Non-backtracking validity"),
]
for mod, desc in tests:
    add_para(tf, f"  ✓ {mod:25s} {desc}", font_size=13, color=LIGHT_GRAY, font_name="Consolas",
             space_before=Pt(2), space_after=Pt(1))
add_para(tf, "", font_size=8, color=LIGHT_GRAY)
add_para(tf, "  All 44 tests pass in ~19 seconds", font_size=15, color=ACCENT_TEAL, bold=True)

# Right column
shape = slide.shapes.add_shape(
    MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.2), Inches(1.6), Inches(5.3), Inches(5.0)
)
shape.fill.solid()
shape.fill.fore_color.rgb = RGBColor(0x25, 0x25, 0x40)
shape.line.color.rgb = ACCENT_TEAL
shape.line.width = Pt(1.5)
tf2 = shape.text_frame
tf2.word_wrap = True
p = tf2.paragraphs[0]
p.text = "Package Features"
p.font.size = Pt(18)
p.font.bold = True
p.font.color.rgb = ACCENT_TEAL
p.alignment = PP_ALIGN.CENTER
add_para(tf2, "", font_size=6, color=LIGHT_GRAY)
add_para(tf2, "  📦  pip install -e . (editable mode)", font_size=15, color=WHITE)
add_para(tf2, "  📋  pyproject.toml (PEP 517/518)", font_size=15, color=WHITE)
add_para(tf2, "  📖  Comprehensive README with all steps", font_size=15, color=WHITE)
add_para(tf2, "  📓  Jupyter demo notebook", font_size=15, color=WHITE)
add_para(tf2, "  🌐  Streamlit interactive dashboard", font_size=15, color=WHITE)
add_para(tf2, "  🔧  config.py: all tunable parameters", font_size=15, color=WHITE)
add_para(tf2, "  🎲  Explicit seeds: reproducible results", font_size=15, color=WHITE)
add_para(tf2, "  📐  Type hints on every function", font_size=15, color=WHITE)
add_para(tf2, "  📝  Docstrings on every function", font_size=15, color=WHITE)
add_para(tf2, "", font_size=8, color=LIGHT_GRAY)
add_para(tf2, "  11 standalone experiment scripts", font_size=15, color=ORANGE)
add_para(tf2, "  82 generated result files", font_size=15, color=ORANGE)

set_notes(slide, """[SPEAKER B]

We put a lot of effort into engineering quality, because we wanted this to be a proper package, not just a collection of scripts.

We have 44 unit tests across 10 test modules. These test everything from basic data aggregation to the correctness of the SIR epidemic simulation. All 44 pass in about 19 seconds.

The package is installable via pip with an editable install. We have a pyproject.toml for modern Python packaging standards. The README documents every experiment step, the folder layout, and all our design decisions.

We also have a Jupyter demo notebook for interactive exploration, and of course the Streamlit dashboard we showed earlier.

Every function has type hints and a docstring. Every parameter lives in config.py. Every random operation uses explicit seeds. We generated 82 result files across 11 experiment scripts.""")


# ====================== SLIDE 19 — KEY TAKEAWAYS ===========================
slide = prs.slides.add_slide(blank_layout)
set_slide_bg(slide, DARK_BG)

add_text_box(slide, Inches(0.8), Inches(0.4), Inches(10), Inches(0.7),
             "Key Takeaways",
             font_size=32, color=WHITE, bold=True, font_name="Calibri Light")
add_accent_line(slide, Inches(0.8), Inches(1.15), Inches(2.5), ACCENT_TEAL)

tf = add_text_box(slide, Inches(0.8), Inches(1.5), Inches(11.5), Inches(5.5),
                  "", font_size=18, color=WHITE)

takeaways = [
    ("From the Paper:", ACCENT_BLUE, True, 20),
    ("", LIGHT_GRAY, False, 6),
    ("1.  Real temporal networks exhibit strong memory (p = 0.5 – 0.97)", WHITE, False, 17),
    ("     People consistently revisit recent contacts far beyond what chance predicts", MID_GRAY, False, 14),
    ("", LIGHT_GRAY, False, 6),
    ("2.  Community structure explains part of the memory (MEM+SBM always wins on BIC)", WHITE, False, 17),
    ("     But individual memory p remains significant even after accounting for communities", MID_GRAY, False, 14),
    ("", LIGHT_GRAY, False, 6),
    ("3.  Memory is robust across temporal resolutions and follows circadian patterns", WHITE, False, 17),
    ("", LIGHT_GRAY, False, 10),
    ("From Our Extensions:", ORANGE, True, 20),
    ("", LIGHT_GRAY, False, 6),
    ("4.  Memory slows epidemic spreading — higher memory = smaller outbreaks vs. shuffled", WHITE, False, 17),
    ("     Time-shuffled networks produce 2× to 12× larger epidemics", MID_GRAY, False, 14),
    ("", LIGHT_GRAY, False, 6),
    ("5.  Memory-informed isolation strategies are competitive with traditional approaches", WHITE, False, 17),
    ("     Return Rate targeting performs comparably to Contact Time targeting", MID_GRAY, False, 14),
    ("", LIGHT_GRAY, False, 6),
    ("6.  The package is production-ready: tested, documented, interactive, reproducible", ACCENT_TEAL, False, 17),
]

for text, color, bold, size in takeaways:
    add_para(tf, text, font_size=size, color=color, bold=bold,
             space_before=Pt(0), space_after=Pt(0))

set_notes(slide, """[SPEAKER A]

Let me summarise our key takeaways.

From the paper: memory is real and strong across all 8 datasets we tested. People revisit recent contacts with probability 0.5 to 0.97, far above what any null model predicts. Community structure explains part of this — MEM+SBM always wins on BIC — but individual memory remains significant even after accounting for communities. And memory is robust to measurement choices and follows real circadian patterns.

From our extensions: memory actually matters for epidemic spreading. Time-shuffled networks, which destroy temporal memory, produce 2 to 12 times larger epidemics. This means memory acts as a natural containment mechanism. We also showed that targeting people based on their memory behavior — their return rate — is a viable isolation strategy, competitive with traditional super-spreader targeting.

And from an engineering perspective, we built a proper Python package with tests, documentation, and an interactive dashboard. This isn't a throwaway script — it's a tool that could be used for further research.""")


# ====================== SLIDE 20 — CONCLUSION / Q&A ========================
slide = prs.slides.add_slide(blank_layout)
set_slide_bg(slide, DARK_BG)

add_accent_line(slide, Inches(1.5), Inches(2.3), Inches(3), ACCENT_TEAL)

add_text_box(slide, Inches(1.5), Inches(2.5), Inches(10), Inches(1.2),
             "Thank You!",
             font_size=44, color=WHITE, bold=True, font_name="Calibri Light")

add_text_box(slide, Inches(1.5), Inches(3.8), Inches(10), Inches(0.7),
             "Questions & Discussion",
             font_size=26, color=ACCENT_BLUE, font_name="Calibri Light")

add_text_box(slide, Inches(1.5), Inches(4.8), Inches(10), Inches(0.6),
             "Code:  github.com/[your-repo]  •  Launch:  streamlit run app.py",
             font_size=16, color=MID_GRAY, font_name="Consolas")

add_text_box(slide, Inches(1.5), Inches(5.5), Inches(10), Inches(1.0),
             "Based on: Guerrini, Cattuto & Dall'Amico,\n\"Modeling memory in time-respecting paths on temporal networks\",\nEPJ Data Science (2026)",
             font_size=14, color=MID_GRAY)

set_notes(slide, """[SPEAKER B]

And that's our presentation. To summarise very briefly: human contact networks have strong temporal memory, and it matters — both for understanding social dynamics and for practical applications like epidemic control.

We'd love to take your questions. We can also do a live demo of the Streamlit dashboard if anyone wants to see it in action.

Thank you!""")


# ============================================================================
# SAVE
# ============================================================================
prs.save(str(OUTPUT_PATH))
print(f"Presentation saved to: {OUTPUT_PATH}")
print(f"Total slides: {len(prs.slides)}")
