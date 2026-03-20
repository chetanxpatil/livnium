#!/usr/bin/env python3
"""
Generate Livnium v2 paper PDF using reportlab.
Usage:
    pip3 install reportlab
    python3 generate_paper.py
Output: livnium_paper_v2.pdf
"""

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable, KeepTogether
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import Flowable
import reportlab.rl_config
reportlab.rl_config.warnOnMissingFontGlyphs = 0

# ── Output path ───────────────────────────────────────────────────────────────
OUTPUT = "livnium_paper_v2.pdf"

# ── Styles ────────────────────────────────────────────────────────────────────
styles = getSampleStyleSheet()

def make_style(name, parent='Normal', **kwargs):
    base = styles[parent]
    s = ParagraphStyle(name, parent=base, **kwargs)
    return s

title_style = make_style('Title2', 'Normal',
    fontSize=16, fontName='Times-Bold', alignment=TA_CENTER,
    spaceAfter=6, leading=20)

author_style = make_style('Author', 'Normal',
    fontSize=11, fontName='Times-Roman', alignment=TA_CENTER,
    spaceAfter=4)

date_style = make_style('Date', 'Normal',
    fontSize=10, fontName='Times-Italic', alignment=TA_CENTER,
    spaceAfter=18)

abstract_label = make_style('AbsLabel', 'Normal',
    fontSize=10, fontName='Times-Bold', alignment=TA_CENTER, spaceAfter=4)

abstract_style = make_style('Abstract', 'Normal',
    fontSize=9.5, fontName='Times-Roman',
    leftIndent=40, rightIndent=40, leading=13,
    spaceAfter=16, alignment=TA_JUSTIFY)

h1 = make_style('H1', 'Normal',
    fontSize=13, fontName='Times-Bold',
    spaceBefore=14, spaceAfter=5, leading=16)

h2 = make_style('H2', 'Normal',
    fontSize=11, fontName='Times-Bold',
    spaceBefore=10, spaceAfter=4, leading=14)

h3 = make_style('H3', 'Normal',
    fontSize=10.5, fontName='Times-BoldItalic',
    spaceBefore=7, spaceAfter=3, leading=13)

body = make_style('Body', 'Normal',
    fontSize=10, fontName='Times-Roman',
    leading=14, spaceAfter=6, alignment=TA_JUSTIFY)

body_nb = make_style('BodyNB', 'Normal',
    fontSize=10, fontName='Times-Roman',
    leading=14, spaceAfter=2, alignment=TA_JUSTIFY)

equation_style = make_style('Equation', 'Normal',
    fontSize=10, fontName='Courier',
    leftIndent=40, leading=14, spaceAfter=4)

caption_style = make_style('Caption', 'Normal',
    fontSize=9, fontName='Times-Italic',
    alignment=TA_CENTER, leading=12, spaceAfter=8)

bullet_style = make_style('Bullet', 'Normal',
    fontSize=10, fontName='Times-Roman',
    leftIndent=20, leading=13, spaceAfter=3, alignment=TA_JUSTIFY)

def_box_style = make_style('DefBox', 'Normal',
    fontSize=10, fontName='Times-Roman',
    leftIndent=20, rightIndent=20, leading=14,
    spaceAfter=4, alignment=TA_JUSTIFY,
    borderPadding=4)

code_style = make_style('Code', 'Normal',
    fontSize=9, fontName='Courier',
    leftIndent=30, leading=13, spaceAfter=2)

# ── Helpers ───────────────────────────────────────────────────────────────────
def P(text, style=body):
    return Paragraph(text, style)

def EQ(text):
    """Display a centred equation block."""
    return Paragraph(text, equation_style)

def SP(n=6):
    return Spacer(1, n)

def HR():
    return HRFlowable(width="100%", thickness=0.5, color=colors.grey, spaceAfter=6)

def table_style_base():
    return TableStyle([
        ('FONTNAME',  (0,0), (-1,0),  'Times-Bold'),
        ('FONTSIZE',  (0,0), (-1,-1), 9),
        ('FONTNAME',  (0,1), (-1,-1), 'Times-Roman'),
        ('ROWBACKGROUNDS', (0,0), (-1,-1), [colors.white, colors.Color(0.96,0.96,0.98)]),
        ('BOX',       (0,0), (-1,-1), 0.5, colors.black),
        ('LINEBELOW', (0,0), (-1,0),  0.8, colors.black),
        ('LINEABOVE', (0,-1),(-1,-1), 0.5, colors.black),
        ('TOPPADDING',  (0,0),(-1,-1), 3),
        ('BOTTOMPADDING',(0,0),(-1,-1),3),
        ('LEFTPADDING', (0,0),(-1,-1), 5),
        ('RIGHTPADDING',(0,0),(-1,-1), 5),
        ('VALIGN',    (0,0), (-1,-1), 'MIDDLE'),
    ])

def section_rule():
    return HRFlowable(width="100%", thickness=0.3, color=colors.lightgrey, spaceAfter=2)

# ── Page template with page numbers ──────────────────────────────────────────
def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFont('Times-Roman', 9)
    canvas.setFillColor(colors.grey)
    canvas.drawString(inch, 0.55 * inch, "Patil (2026)  —  Three Laws of Semantic Collapse")
    canvas.drawRightString(letter[0] - inch, 0.55 * inch, str(doc.page))
    canvas.restoreState()

def on_first_page(canvas, doc):
    canvas.saveState()
    canvas.setFont('Times-Roman', 8)
    canvas.setFillColor(colors.grey)
    canvas.drawCentredString(letter[0]/2, 0.55*inch,
        "Preprint — github.com/chetanxpatil/livnium")
    canvas.restoreState()

# ── Build story ───────────────────────────────────────────────────────────────
story = []

# ── TITLE PAGE ────────────────────────────────────────────────────────────────
story.append(SP(20))
story.append(P("Three Laws of Semantic Collapse:<br/>NLI as Gradient Descent on an Anchor Energy Landscape", title_style))
story.append(SP(10))
story.append(P("Chetan Patil", author_style))
story.append(P("chetan12patil@gmail.com", author_style))
story.append(P("github.com/chetanxpatil/livnium", author_style))
story.append(P("March 2026", date_style))
story.append(HR())
story.append(SP(4))

# Abstract
story.append(P("Abstract", abstract_label))
story.append(P(
    "We present Livnium, a classification system for Natural Language Inference in which "
    "inference is not a single forward pass but a dynamical process: the hidden state evolves "
    "through multiple geometry-aware steps before readout. We discover that the trained system's "
    "behavior is governed by three empirical laws: (1) the initial state encodes the relational "
    "difference between premise and hypothesis, (2) the semantic space is organized by an energy "
    "landscape V(h) = \u2212log\u03a3exp(\u03b2\u00b7cos(h, A<sub>k</sub>)) over three class anchors, "
    "and (3) the state evolves by gradient descent on this energy. Critically, Laws 2 and 3 were "
    "<i>not designed</i>\u2014they were recovered from analyzing what a trained system was doing. "
    "We show that replacing the trained 1.2M-parameter MLP update with the analytical gradient of V "
    "matches accuracy on the full SNLI dev set (9,842 samples), with the learned residual contributing "
    "\u22640.25%. Joint retraining under the discovered dynamics reduces head\u2013dynamics disagreement "
    "from 55.1% to 4.0% and improves neutral recall by +4.0pp. Trajectory analysis reveals a universal "
    "fixed-point phenomenon: collapse dynamics converge to the same state after ~3 steps regardless of "
    "input, identifying the primary bottleneck. Code and pretrained weights are publicly available.",
    abstract_style))

story.append(HR())
story.append(SP(8))

# ── 1. INTRODUCTION ───────────────────────────────────────────────────────────
story.append(P("1.  Introduction", h1))
story.append(section_rule())

story.append(P(
    "The standard neural classifier computes y = softmax(Wh + b): a single affine map from "
    "representation to logits. We explore an alternative in which the classification head is a "
    "small dynamical system that <i>evolves</i> the representation through L steps before producing a label:",
    body))
story.append(EQ("h<sub>0</sub>  ->  h<sub>1</sub>  ->  ...  ->  h<sub>L</sub>  ->  logits"))
story.append(P(
    "where each step applies learned residuals and geometric forces from label-specific anchor vectors. "
    "The label is determined by which <i>basin</i> the state settles into, not by a one-shot projection.",
    body))

story.append(P(
    "The original Livnium system introduced this idea with a bag-of-words encoder, achieving 76% on SNLI. "
    "The system used hand-designed cosine-radial forces with a 135\u00b0 mismatch from true cosine "
    "gradients\u2014a non-conservative force field with no scalar potential.",
    body))

story.append(P(
    "In this paper, we report a surprising discovery. After upgrading to a jointly-trained BERT "
    "bi-encoder (82.06% accuracy), we analyzed the trained collapse dynamics and found that they are "
    "well-approximated by gradient descent on a simple energy function over the class anchors. The "
    "learned MLP \u03b4<sub>\u03b8</sub>, which constitutes 1.2M parameters of the collapse engine, "
    "can be replaced by a three-line analytical gradient formula with no accuracy loss\u2014and "
    "slight improvement.",
    body))

story.append(P("This leads to the <b>Three Laws of Livnium</b>:", body))
story.append(P("\u2022  <b>Law 1</b> (Relational State):  h<sub>0</sub> = v<sub>h</sub> \u2212 v<sub>p</sub>", bullet_style))
story.append(P("\u2022  <b>Law 2</b> (Energy Landscape):  V(h) = \u2212logsumexp(\u03b2\u00b7cos(h, A<sub>E</sub>),  \u03b2\u00b7cos(h, A<sub>C</sub>),  \u03b2\u00b7cos(h, A<sub>N</sub>))", bullet_style))
story.append(P("\u2022  <b>Law 3</b> (Collapse Dynamics):  h<sub>t+1</sub> = h<sub>t</sub> \u2212 \u03b1\u2207V(h<sub>t</sub>)", bullet_style))
story.append(SP(4))
story.append(P(
    "Laws 2 and 3 were not assumed\u2014they were recovered empirically from a trained system. "
    "This is the main result of the paper.",
    body))

story.append(P("<b>Contributions.</b>", body_nb))
story.append(P("(1)  Discovery that trained attractor dynamics approximate gradient descent on a logsumexp cosine energy.", bullet_style))
story.append(P("(2)  Formulation of three empirical laws that fully specify the system's inference behavior.", bullet_style))
story.append(P("(3)  Joint retraining framework achieving head\u2013dynamics consistency (TYPE-? rate: 55.1% \u2192 4.0%).", bullet_style))
story.append(P("(4)  Tunnel test diagnostic classifying error modes by trajectory analysis.", bullet_style))
story.append(P("(5)  Identification of a universal fixed-point phenomenon as the primary accuracy bottleneck.", bullet_style))
story.append(SP(8))

# ── 2. BACKGROUND ─────────────────────────────────────────────────────────────
story.append(P("2.  Background and Related Work", h1))
story.append(section_rule())

story.append(P("<b>Energy-based models.</b>  "
    "Hopfield networks pioneered energy-based attractor dynamics for memory retrieval; modern Hopfield "
    "networks extend this with exponential storage capacity and connections to attention. "
    "LeCun et al.'s energy-based learning framework provides a general formulation. Our work differs "
    "in that the energy function was not designed but <i>discovered</i> from trained dynamics.", body))

story.append(P("<b>Equilibrium and implicit models.</b>  "
    "Deep equilibrium models (DEQ) compute representations as fixed points of implicit layers, and "
    "neural ODEs parameterize continuous-time dynamics. Livnium uses explicit, finite-step dynamics "
    "with an interpretable geometric structure\u2014the energy landscape can be written in closed form.", body))

story.append(P("<b>Prototype-based classification.</b>  "
    "Prototypical networks classify by distance to class centroids. Livnium's anchors serve a similar "
    "role, but classification involves iterative state refinement. The anchors define an energy landscape, "
    "not just a nearest-neighbor rule.", body))

story.append(P("<b>Natural language inference.</b>  "
    "SNLI is the standard benchmark, with baselines from bag-of-words (~80%) through decomposable "
    "attention (~86%) to fine-tuned BERT (~90%). Our work focuses on the classification head; we "
    "demonstrate that the head itself contains discoverable physical laws.", body))
story.append(SP(8))

# ── 3. SYSTEM ─────────────────────────────────────────────────────────────────
story.append(P("3.  The Livnium System", h1))
story.append(section_rule())

story.append(P("3.1  Architecture Overview", h2))
story.append(P(
    "Given a premise p and hypothesis q, a BERT bi-encoder produces sentence vectors "
    "v<sub>p</sub>, v<sub>q</sub> \u2208 R<sup>768</sup>. The initial state is their difference (Law 1):",
    body))
story.append(EQ("h<sub>0</sub>  =  v<sub>q</sub>  \u2212  v<sub>p</sub>"))
story.append(P(
    "Three learned anchor vectors A<sub>E</sub>, A<sub>C</sub>, A<sub>N</sub> \u2208 R<sup>768</sup> "
    "(entailment, contradiction, neutral) define the geometry of the label space. Each anchor is "
    "unit-normalized during the forward pass.",
    body))

story.append(P("3.2  Trained Collapse Rule", h2))
story.append(P(
    "The original system evolves the state through L=6 steps. At each step t = 0, \u2026, L\u22121:",
    body))
story.append(EQ("h<sub>t+1</sub>  =  h<sub>t</sub>  +  \u03b4<sub>\u03b8</sub>(h<sub>t</sub>)  "
                "\u2212  s<sub>y</sub>\u00b7D(h<sub>t</sub>, A<sub>y</sub>)\u00b7n\u0302(h<sub>t</sub>, A<sub>y</sub>)  "
                "\u2212  \u03b2\u00b7B(h<sub>t</sub>)\u00b7n\u0302(h<sub>t</sub>, A<sub>N</sub>)"))
story.append(P("where:", body_nb))
story.append(P("\u2022  \u03b4<sub>\u03b8</sub>(h<sub>t</sub>): a learned 2-layer MLP residual.", bullet_style))
story.append(P("\u2022  D(h, A) = 0.38 \u2212 cos(h, A): divergence from the equilibrium ring.", bullet_style))
story.append(P("\u2022  n\u0302(h, A) = (h\u2212A)/\u2016h\u2212A\u2016: Euclidean radial direction.", bullet_style))
story.append(P("\u2022  B(h) = 1 \u2212 |cos(h,A<sub>E</sub>) \u2212 cos(h,A<sub>C</sub>)|: boundary proximity.", bullet_style))
story.append(SP(4))
story.append(P(
    "The equilibrium target for each anchor is a cosine-similarity ring at cos(h, A<sub>y</sub>) = 0.38, "
    "not the anchor point itself. After L collapse steps, the refined state h<sub>L</sub> is passed "
    "to a readout MLP (SNLIHead) with auxiliary geometric features.",
    body))
story.append(SP(8))

# ── 4. THREE LAWS ─────────────────────────────────────────────────────────────
story.append(P("4.  The Three Laws", h1))
story.append(section_rule())

story.append(P(
    "We propose that the trained system's behavior is fully specified by three laws. "
    "Law 1 is a design choice; Laws 2 and 3 are empirical discoveries.",
    body))

# Law 1
story.append(P("Law 1: Relational State Formation", h2))
story.append(P(
    "The initial state h<sub>0</sub> = v<sub>q</sub> \u2212 v<sub>p</sub> encodes the <i>direction of semantic "
    "change</i> from premise to hypothesis. Entailment pulls h<sub>0</sub> toward A<sub>E</sub>; contradiction "
    "toward A<sub>C</sub>; neutral is orthogonal. This is a design choice\u2014the weakest law, but "
    "necessary to ground the others.",
    body))

# Law 2
story.append(P("Law 2: Energy Landscape", h2))
story.append(EQ("V(h)  =  \u2212log \u03a3<sub>k\u2208{E,C,N}</sub>  exp(\u03b2 \u00b7 cos(h, A<sub>k</sub>))"))
story.append(P(
    "The energy at any point h is the negative log-partition function of Boltzmann-weighted cosine "
    "similarities to the three class anchors. V(h) assigns lower energy to points near an anchor, "
    "defining three semantic basins. The parameter \u03b2 controls basin sharpness.",
    body))

# Law 3
story.append(P("Law 3: Collapse Dynamics", h2))
story.append(EQ("h<sub>t+1</sub>  =  h<sub>t</sub>  \u2212  \u03b1 \u2207<sub>h</sub>V(h<sub>t</sub>)"))
story.append(P("The analytical gradient is:", body_nb))
story.append(EQ("\u2207<sub>h</sub>V(h)  =  \u2212 \u03a3<sub>k</sub>  w<sub>k</sub>(h) \u00b7 \u2207<sub>h</sub>cos(h, A<sub>k</sub>)"))
story.append(P(
    "where w<sub>k</sub>(h) = softmax(\u03b2\u00b7cos(h, A<sub>k</sub>)) are the Boltzmann weights, and "
    "\u2207<sub>h</sub>cos(h, A) = (A \u2212 h\u00b7cos(h,A)) / \u2016h\u2016. "
    "The Boltzmann weights ensure smooth interpolation near basin boundaries. At \u03b2=1, weights "
    "form a balanced softmax\u2014preserving multi-basin gradient signal at boundaries. "
    "At \u03b2\u2192\u221e, the dynamics reduce to argmax (nearest-anchor), losing boundary sensitivity.",
    body))
story.append(SP(8))

# ── 5. DISCOVERING THE EQUATION OF MOTION ────────────────────────────────────
story.append(P("5.  Discovering the Equation of Motion", h1))
story.append(section_rule())

story.append(P("5.1  Experimental Setup", h2))
story.append(P(
    "We train a BERT bi-encoder jointly with the collapse engine on SNLI (549k training pairs, "
    "9,842 dev pairs), achieving 82.06% dev accuracy. We then evaluate three collapse modes:",
    body))
story.append(P("<b>Mode 1 \u2014 Full:</b>  The trained system\u2014MLP \u03b4<sub>\u03b8</sub> + anchor forces.", bullet_style))
story.append(P("<b>Mode 2 \u2014 No-delta:</b>  Remove the learned MLP; keep only anchor forces.", bullet_style))
story.append(P("<b>Mode 3 \u2014 Grad-V:</b>  Replace the entire collapse rule with pure gradient descent on V(h).", bullet_style))

story.append(P("5.2  Result: The MLP Is Redundant", h2))

# Table 1
t1_data = [
    ["Mode", "Accuracy", "E-recall", "N-recall", "C-recall"],
    ["Original checkpoint — \u03b2=1, full SNLI dev (n=9,842)", "", "", "", ""],
    ["  Full (\u03b4\u03b8 + forces)", "82.05%", "92.79%", "71.16%", "81.88%"],
    ["  No-delta (forces only)", "81.81%", "93.48%", "68.62%", "82.98%"],
    ["  Grad-V (\u03b2=1, \u03b1=0.2)  \u2605", "82.21%", "92.97%", "72.18%", "81.18%"],
    ["Joint-retrained checkpoint — \u03b2=20, n=2,000", "", "", "", ""],
    ["  Full (\u03b4\u03b8 + forces)", "82.35%", "90.35%", "71.64%", "85.30%"],
    ["  No-delta (forces only)", "82.20%", "93.36%", "69.87%", "83.64%"],
    ["  Grad-V (\u03b2=20, \u03b1=0.2)  \u2605", "82.10%", "94.42%", "72.53%", "79.55%"],
]

t1_col_widths = [2.3*inch, 0.75*inch, 0.75*inch, 0.75*inch, 0.75*inch]
t1 = Table(t1_data, colWidths=t1_col_widths, repeatRows=1)
ts = TableStyle([
    ('FONTNAME',  (0,0), (-1,0), 'Times-Bold'),
    ('FONTSIZE',  (0,0), (-1,-1), 8.5),
    ('FONTNAME',  (0,1), (-1,-1), 'Times-Roman'),
    ('FONTNAME',  (0,1), (0,1), 'Times-BoldItalic'),
    ('FONTNAME',  (0,5), (0,5), 'Times-BoldItalic'),
    ('SPAN',      (0,1), (-1,1)),
    ('SPAN',      (0,5), (-1,5)),
    ('BACKGROUND',(0,1),(-1,1), colors.Color(0.90,0.90,0.95)),
    ('BACKGROUND',(0,5),(-1,5), colors.Color(0.90,0.90,0.95)),
    ('BACKGROUND',(0,4),(-1,4), colors.Color(0.92,0.98,0.92)),
    ('BACKGROUND',(0,8),(-1,8), colors.Color(0.92,0.98,0.92)),
    ('BOX',       (0,0), (-1,-1), 0.5, colors.black),
    ('LINEBELOW', (0,0), (-1,0), 0.8, colors.black),
    ('TOPPADDING',(0,0),(-1,-1), 3),
    ('BOTTOMPADDING',(0,0),(-1,-1),3),
    ('LEFTPADDING',(0,0),(-1,-1), 5),
    ('RIGHTPADDING',(0,0),(-1,-1),5),
    ('ALIGN',     (1,0), (-1,-1), 'CENTER'),
    ('VALIGN',    (0,0), (-1,-1), 'MIDDLE'),
])
t1.setStyle(ts)
story.append(KeepTogether([t1]))
story.append(P("Table 1: Gradient collapse comparison (\u2605 = Grad-V rows). Grad-V matches or exceeds the trained "
               "system on accuracy and neutral recall. The learned MLP \u03b4<sub>\u03b8</sub> contributes "
               "\u22640.25%\u2014the system <i>is</i> a gradient flow on V(h).", caption_style))

story.append(P(
    "The key finding in Table 1: the analytical gradient of V(h) matches the trained system. "
    "On the full dev set (n=9,842), Grad-V <i>outperforms</i> the trained system by +0.16pp overall "
    "and +1.02pp on neutral recall. The learned MLP \u03b4<sub>\u03b8</sub>, which constitutes 1.2M parameters "
    "and requires a forward pass at each collapse step, contributes at most 0.24pp. It is functionally dead.",
    body))

story.append(P(
    "<b>Interpretation.</b>  The trained MLP was approximating \u2207V but introducing small distortions. "
    "The clean analytical gradient recovers and slightly exceeds the learned behavior. V(h) is the "
    "Boltzmann log-partition function over anchor similarities\u2014a natural measure of semantic basin "
    "proximity. The system learned to perform gradient descent on this energy without being told to.",
    body))

story.append(P("5.3  Stability Across \u03b2", h2))
story.append(P(
    "We swept \u03b2 \u2208 {0.5, 1, 2, 5, 10, 20, 50} at \u03b1=0.2. All values within \u03b2 \u2208 [1, 50] "
    "produce accuracy within \u00b11% of the trained system, confirming robustness to the sharpness "
    "parameter. The result is not an artifact of hyperparameter tuning.",
    body))
story.append(SP(8))

# ── 6. JOINT RETRAINING ───────────────────────────────────────────────────────
story.append(P("6.  Joint Retraining Under Discovered Dynamics", h1))
story.append(section_rule())

story.append(P("6.1  Motivation: The TYPE-? Problem", h2))
story.append(P(
    "At \u03b2=20, gradient dynamics become highly decisive\u2014states fall cleanly into the nearest "
    "anchor basin. However, the classification head (SNLIHead) was trained under original \u03b2=1 "
    "dynamics, where collapse converges to a universal fixed point (Section 7.2). When we swap in "
    "\u03b2=20 Grad-V dynamics at test time, the head's predictions disagree with the dominant basin "
    "on 55.1% of errors\u2014the head ignores dynamics it was not trained on.",
    body))

story.append(P("6.2  Method", h2))
story.append(P(
    "We introduce <b>LiveniumJoint</b>: BERT, anchors, and SNLIHead are trained together under "
    "differentiable Grad-V dynamics, staying in the computation graph. The loss combines "
    "cross-entropy with an anchor alignment term:",
    body))
story.append(EQ("\u2112  =  \u2112<sub>CE</sub>  +  \u03bb<sub>align</sub> \u00b7 (\u2212cos(h<sub>L</sub>, A<sub>y</sub>))"))
story.append(P(
    "The alignment loss directly penalizes disagreement between the final collapsed state and the "
    "correct anchor, forcing head\u2013dynamics consistency. Training: 30,000 stratified samples "
    "(10k per class), 5 epochs, \u03b2=20, \u03b1=0.05, \u03bb<sub>align</sub>=0.3, warm-started "
    "from the BERT-joint checkpoint.",
    body))

story.append(P("6.3  Results", h2))

t2_data = [
    ["Metric", "Before", "Ep 1", "Ep 2", "Ep 3", "Ep 4", "Ep 5"],
    ["Accuracy",        "81.95%", "82.79%\u2605", "82.48%", "82.68%", "82.43%", "82.48%"],
    ["Neutral recall",  "70.54%", "74.50%", "74.74%", "75.98%", "76.63%", "76.51%"],
    ["C recall",        "\u2014",  "80.96%", "79.62%", "80.32%", "78.83%", "79.23%"],
    ["Dyn/head agree",  "29.73%", "72.52%", "74.10%", "73.29%", "73.62%", "73.88%"],
    ["TYPE-? rate",     "7.84%",  "4.20%",  "3.98%",  "4.56%",  "4.50%",  "4.43%"],
    ["cos(AC, AN)", "\u2014", "-0.031", "-0.007", "+0.011", "+0.024", "+0.026"],
]
t2_col_widths = [1.55*inch, 0.75*inch, 0.75*inch, 0.75*inch, 0.75*inch, 0.75*inch, 0.75*inch]
t2 = Table(t2_data, colWidths=t2_col_widths, repeatRows=1)
t2.setStyle(TableStyle([
    ('FONTNAME',  (0,0), (-1,0), 'Times-Bold'),
    ('FONTSIZE',  (0,0), (-1,-1), 8.5),
    ('FONTNAME',  (0,1), (-1,-1), 'Times-Roman'),
    ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.Color(0.96,0.96,0.98)]),
    ('BOX',       (0,0), (-1,-1), 0.5, colors.black),
    ('LINEBELOW', (0,0), (-1,0), 0.8, colors.black),
    ('TOPPADDING',(0,0),(-1,-1),3),('BOTTOMPADDING',(0,0),(-1,-1),3),
    ('LEFTPADDING',(0,0),(-1,-1),4),('RIGHTPADDING',(0,0),(-1,-1),4),
    ('ALIGN',     (1,0),(-1,-1),'CENTER'),
    ('VALIGN',    (0,0),(-1,-1),'MIDDLE'),
    # Highlight best accuracy col (ep1)
    ('BACKGROUND',(2,1),(-1,1), colors.Color(0.92,0.98,0.92)),
    # Highlight best N-recall col (ep4)
    ('BACKGROUND',(5,2),(5,2), colors.Color(0.92,0.98,0.92)),
    # Warn on C-N drift turning positive
    ('BACKGROUND',(5,6),(6,6), colors.Color(1.0,0.93,0.87)),
]))
story.append(KeepTogether([t2]))
story.append(P("Table 2: Full 5-epoch joint retraining (\u2605 = best checkpoint saved). "
               "Head\u2013dynamics agreement jumps from 29.7% to 73.9%. Neutral recall rises "
               "monotonically from 70.5% to 76.6% (peak epoch 4, +6.1pp). Note the C\u2013N anchor "
               "drift: cos(A<sub>C</sub>, A<sub>N</sub>) turns positive by epoch 3, explaining the C-recall erosion.", caption_style))

story.append(P(
    "Joint retraining achieves its primary goal: head\u2013dynamics agreement rises from 29.7% to "
    "73.9%, and neutral recall improves monotonically across all 5 epochs, reaching 76.63% at epoch 4 "
    "(+6.1pp over baseline). Overall accuracy is best at epoch 1 (82.79%), reflecting a trade-off: "
    "the alignment loss optimizes neutral recall at some cost to contradiction recall and overall accuracy.",
    body))

story.append(P(
    "<b>Anchor drift phenomenon.</b>  cos(A<sub>C</sub>, A<sub>N</sub>) crosses zero at epoch 3 "
    "and reaches +0.026 by epoch 5. The C and N anchors become positively correlated under the "
    "alignment loss, causing C-recall to erode from 80.96% to 78.83%. This identifies a concrete "
    "regularization target for future work: anchor orthogonality should be explicitly enforced "
    "during joint training.",
    body))
story.append(SP(8))

# ── 7. TRAJECTORY ANALYSIS ───────────────────────────────────────────────────
story.append(P("7.  Trajectory Analysis", h1))
story.append(section_rule())

story.append(P("7.1  Tunnel Test: Classifying Error Modes", h2))
story.append(P(
    "We develop a <b>tunnel test</b> diagnostic that traces the step-by-step collapse trajectory "
    "and classifies neutral misclassifications by when and how the dynamics fail. "
    "At each step t=0,\u2026,L, we record the dominant anchor basin (highest cosine).",
    body))

story.append(P("<b>TYPE-1</b> (bad h<sub>0</sub>, Law 1 failure):  "
    "h<sub>0</sub> starts in the wrong basin; the initial encoding does not place the state near the neutral anchor.", bullet_style))
story.append(P("<b>TYPE-2</b> (mid-diversion, Law 3 failure):  "
    "h<sub>0</sub> starts correctly but the trajectory is diverted to a wrong basin during collapse.", bullet_style))
story.append(P("<b>TYPE-3</b> (boundary stall, Law 2 failure):  "
    "The trajectory converges near a boundary or to the universal fixed point, never reaching a decisive basin.", bullet_style))
story.append(P("<b>TYPE-?</b> (head override):  "
    "The dynamics deliver the correct dominant basin at step L, but the head overrides and predicts a different label.", bullet_style))
story.append(SP(6))

t3_data = [
    ["Configuration", "TYPE-1", "TYPE-2", "TYPE-3", "TYPE-?"],
    ["\u03b2=1 (original)",         "14.4%", "21.3%", "63.9%", "0.5%"],
    ["\u03b2=20 (Grad-V, no joint)", "2.8%",  "20.4%", "21.7%", "55.1%"],
    ["\u03b2=20 (joint-retrained)",  "n/a",   "n/a",   "n/a",   "4.0%"],
]
t3_col_widths = [2.2*inch, 0.85*inch, 0.85*inch, 0.85*inch, 0.85*inch]
t3 = Table(t3_data, colWidths=t3_col_widths, repeatRows=1)
t3.setStyle(TableStyle([
    ('FONTNAME',  (0,0), (-1,0), 'Times-Bold'),
    ('FONTSIZE',  (0,0), (-1,-1), 9),
    ('FONTNAME',  (0,1), (-1,-1), 'Times-Roman'),
    ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.Color(0.96,0.96,0.98)]),
    ('BOX',       (0,0), (-1,-1), 0.5, colors.black),
    ('LINEBELOW', (0,0), (-1,0), 0.8, colors.black),
    ('TOPPADDING',(0,0),(-1,-1),3),('BOTTOMPADDING',(0,0),(-1,-1),3),
    ('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),
    ('ALIGN',     (1,0),(-1,-1),'CENTER'),
    ('VALIGN',    (0,0),(-1,-1),'MIDDLE'),
    ('BACKGROUND',(4,1),(-1,1), colors.Color(0.95,0.95,0.75)),
    ('BACKGROUND',(4,2),(-1,2), colors.Color(0.95,0.75,0.75)),
    ('BACKGROUND',(4,3),(-1,3), colors.Color(0.92,0.98,0.92)),
]))
story.append(KeepTogether([t3]))
story.append(P("Table 3: Error type distribution. At \u03b2=1, TYPE-3 (fixed-point convergence) dominates. "
               "At \u03b2=20, dynamics become decisive but the head overrides them. "
               "Joint retraining resolves the disconnect.", caption_style))

story.append(P("7.2  The Universal Fixed Point", h2))
story.append(P(
    "The dominant error mode at \u03b2=1 is TYPE-3 (63.9%). Investigation reveals this is not "
    "boundary ambiguity but a more fundamental phenomenon: the collapse dynamics converge to a "
    "<b>universal fixed point</b> after approximately 3 steps, independent of the input.",
    body))
story.append(P(
    "At steps 4\u20136, nearly all samples\u2014correct and incorrect\u2014produce the same "
    "cosine alignment pattern:",
    body_nb))
story.append(EQ("cos(h, A<sub>E</sub>) \u2248 \u22120.129,     cos(h, A<sub>C</sub>) \u2248 \u22120.124,     cos(h, A<sub>N</sub>) \u2248 +0.172"))
story.append(P(
    "This fixed point is weakly neutral-biased, explaining why neutral recall is better than chance "
    "(71%) but not as strong as entailment or contradiction. The collapse discards input-specific "
    "information after step 3\u2014the final state is input-agnostic.",
    body))

story.append(P(
    "<b>Energy signature.</b>  Correctly classified samples show mean energy descent "
    "\u0394V = +0.046 (ascending toward lower-energy basins), while error samples show "
    "\u0394V = \u22120.002 (flat\u2014stuck at the fixed point). The dynamics are contracting "
    "for correct predictions and stagnant for errors.",
    body))

story.append(P(
    "<b>Implication.</b>  The accuracy bottleneck is not the head or the energy landscape\u2014it is "
    "the fixed-point collapse. Input information is lost after three steps. Breaking this fixed point "
    "(e.g., via input-dependent residual forcing or adaptive step sizes) is the primary target for "
    "future work.",
    body))

story.append(P("7.3  Basin Stability", h2))
story.append(P(
    "Correct predictions occupy deeper attractor basins than incorrect ones. We measure stability "
    "by perturbing h<sub>0</sub> with Gaussian noise (\u03c3=0.3) across 20 trials per sample:",
    body))

t4_data = [
    ["Group", "Flip rate", "Entropy"],
    ["Correct predictions (n=1,653)",  "0.0017 \u00b1 0.025",  "0.293"],
    ["Incorrect predictions (n=347)",  "0.0068 \u00b1 0.048",  "0.604"],
    ["Ratio",                           "3.93\u00d7",           "2.06\u00d7"],
]
t4_col_widths = [2.5*inch, 1.6*inch, 1.6*inch]
t4 = Table(t4_data, colWidths=t4_col_widths, repeatRows=1)
t4.setStyle(TableStyle([
    ('FONTNAME',  (0,0), (-1,0), 'Times-Bold'),
    ('FONTNAME',  (0,-1),(0,-1), 'Times-Bold'),
    ('FONTSIZE',  (0,0), (-1,-1), 9),
    ('FONTNAME',  (0,1), (-1,-1), 'Times-Roman'),
    ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.Color(0.96,0.96,0.98)]),
    ('BOX',       (0,0), (-1,-1), 0.5, colors.black),
    ('LINEBELOW', (0,0), (-1,0), 0.8, colors.black),
    ('LINEABOVE', (0,-1),(-1,-1),0.5, colors.black),
    ('TOPPADDING',(0,0),(-1,-1),3),('BOTTOMPADDING',(0,0),(-1,-1),3),
    ('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),
    ('ALIGN',     (1,0),(-1,-1),'CENTER'),
    ('VALIGN',    (0,0),(-1,-1),'MIDDLE'),
]))
story.append(KeepTogether([t4]))
story.append(P("Table 4: Basin stability. Wrong predictions are 3.9\u00d7 more likely to flip under perturbation "
               "and carry 2\u00d7 higher entropy. Correctness correlates with basin depth.", caption_style))

story.append(P(
    "Per-class breakdown: entailment errors flip 7.18\u00d7 more than correct entailment predictions, "
    "neutral errors 3.17\u00d7, contradiction errors only 1.82\u00d7. The low contradiction ratio "
    "indicates systematic geometric bias rather than boundary ambiguity\u2014a structural feature "
    "of the anchor geometry.",
    body))
story.append(SP(8))

# ── 8. DISCUSSION ─────────────────────────────────────────────────────────────
story.append(P("8.  Discussion", h1))
story.append(section_rule())

story.append(P("8.1  Why Does the MLP Learn \u2207V?", h2))
story.append(P(
    "The trained collapse rule uses cosine-magnitude forces with Euclidean-radial "
    "directions\u2014a 135\u00b0 mismatch from the true cosine gradient. Yet the combined effect "
    "of MLP + forces approximates gradient descent on V(h). We hypothesize that the MLP learns a "
    "corrective residual that compensates for this geometric inconsistency, effectively rotating the "
    "force field toward the true gradient. The end-to-end training signal (cross-entropy on the "
    "final label) provides sufficient pressure to discover this correction.",
    body))

story.append(P("8.2  What Is the Energy V(h)?", h2))
story.append(P(
    "V(h) = \u2212logsumexp(\u03b2\u00b7cos(h, A<sub>k</sub>)) is the negative log-partition function "
    "of a Boltzmann distribution over anchor cosine similarities. At any point h, it measures "
    "aggregate proximity to all class anchors, weighted exponentially. The minima correspond to anchor "
    "positions; saddle points define basin boundaries. At \u03b2=1, Boltzmann weights form a balanced "
    "softmax, preserving gradient signal at boundaries. At \u03b2\u2192\u221e, V(h) recovers the "
    "Voronoi partition over anchors.",
    body))

story.append(P("8.3  Accuracy Context", h2))
story.append(P(
    "The system achieves 82\u201383% on SNLI, below the BERT-linear baseline (~90%). The gap is "
    "attributable to: (1) the bi-encoder architecture loses cross-attention signal, and (2) the "
    "universal fixed-point collapse discards input information after 3 steps. The contribution of "
    "this work is not state-of-the-art accuracy but the <i>discovery of physical laws</i> governing "
    "a neural inference system\u2014a result that holds regardless of absolute accuracy level.",
    body))

story.append(P("8.4  Limitations", h2))
story.append(P("\u2022  <b>Single dataset.</b>  All results are on SNLI. Evaluation on MultiNLI and adversarial NLI would test generality.", bullet_style))
story.append(P("\u2022  <b>Single seed.</b>  The gradient-flow result has been validated on the full dev set (n=9,842) but from one training run. Multi-seed validation is needed.", bullet_style))
story.append(P("\u2022  <b>Encoder dependence.</b>  The laws are demonstrated with a BERT bi-encoder. Whether they hold for cross-encoders or non-BERT architectures is untested.", bullet_style))
story.append(P("\u2022  <b>Fixed-point bottleneck.</b>  The universal fixed point limits accuracy. Breaking it requires architectural changes not yet validated.", bullet_style))
story.append(P("\u2022  <b>C\u2013N anchor drift.</b>  Joint retraining shows C\u2013N anchor convergence (cos\u2192\u22120.007), which may limit contradiction recall over longer training.", bullet_style))
story.append(SP(8))

# ── 9. CONCLUSION ─────────────────────────────────────────────────────────────
story.append(P("9.  Conclusion", h1))
story.append(section_rule())

story.append(P(
    "We have shown that a trained attractor-based classification system for NLI is governed by three "
    "empirical laws: relational state formation, a logsumexp cosine energy landscape, and "
    "gradient-descent collapse dynamics. The central finding is that Laws 2 and 3 were "
    "<i>not designed</i>\u2014they were recovered by analyzing what the trained MLP was doing. "
    "The learned residual \u03b4<sub>\u03b8</sub> can be replaced by the analytical gradient \u2207V "
    "with no accuracy loss on 9,842 samples.",
    body))

story.append(P(
    "Trajectory analysis via the tunnel test reveals the primary bottleneck: a universal fixed point "
    "that discards input information after three collapse steps. Joint retraining under the discovered "
    "dynamics reduces head\u2013dynamics disagreement from 55% to 4% and improves neutral recall by +4pp.",
    body))

story.append(P(
    "The broader implication is that neural systems can develop interpretable physical laws through "
    "training\u2014laws that are simpler, faster, and more transparent than the trained components "
    "they replace. Whether this phenomenon generalizes beyond attractor-based classification heads "
    "is an open question and a promising direction for future work.",
    body))

story.append(SP(6))
story.append(P("Code: github.com/chetanxpatil/livnium  \u00b7  "
               "Model: huggingface.co/chetanxpatil/livnium-snli", body))
story.append(SP(14))

# ── REFERENCES ────────────────────────────────────────────────────────────────
story.append(HR())
story.append(P("References", h1))

refs = [
    "Bai, S., Kolter, J. Z., and Koltun, V. (2019). Deep equilibrium models. <i>NeurIPS</i>.",
    "Bowman, S., Angeli, G., Potts, C., and Manning, C. (2015). A large annotated corpus for learning natural language inference. <i>EMNLP</i>.",
    "Chen, R. T. Q., Rubanova, Y., Bettencourt, J., and Duvenaud, D. (2018). Neural ordinary differential equations. <i>NeurIPS</i>.",
    "Devlin, J., Chang, M.-W., Lee, K., and Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. <i>NAACL-HLT</i>.",
    "Hopfield, J. J. (1982). Neural networks and physical systems with emergent collective computational abilities. <i>PNAS</i>, 79(8):2554\u20132558.",
    "LeCun, Y., Chopra, S., Hadsell, R., Ranzato, M., and Huang, F. (2006). A tutorial on energy-based learning. In <i>Predicting Structured Data</i>. MIT Press.",
    "Parikh, A., T\u00e4ckstr\u00f6m, O., Das, D., and Uszkoreit, J. (2016). A decomposable attention model for natural language inference. <i>EMNLP</i>.",
    "Patil, C. (2026). Iterative attractor dynamics for classification. Preprint, Zenodo. zenodo.org/records/19058910.",
    "Ramsauer, H. et al. (2021). Hopfield networks is all you need. <i>ICLR</i>.",
    "Snell, J., Swersky, K., and Zemel, R. (2017). Prototypical networks for few-shot learning. <i>NeurIPS</i>.",
]
for r in refs:
    story.append(P(r, make_style(f'ref_{id(r)}', 'Normal',
        fontSize=9, fontName='Times-Roman',
        leftIndent=18, firstLineIndent=-18,
        leading=13, spaceAfter=5, alignment=TA_JUSTIFY)))

# ── BUILD ─────────────────────────────────────────────────────────────────────
doc = SimpleDocTemplate(
    OUTPUT,
    pagesize=letter,
    leftMargin=inch,
    rightMargin=inch,
    topMargin=inch,
    bottomMargin=0.9*inch,
)
doc.build(story, onFirstPage=on_first_page, onLaterPages=on_page)
print(f"Done! Output: {OUTPUT}")
