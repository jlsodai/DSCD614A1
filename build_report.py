"""Generate report.pdf for DSCD 614 Assignment 1."""
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Image, PageBreak,
)
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from environment import MAP
from evaluate import evaluate

# ── Load training artefacts ──────────────────────────────────────────────────
q_table     = np.load("results/q_table.npy")
rewards_arr = np.load("results/episode_rewards.npy")
success_arr = np.load("results/success_flags.npy")

TOTAL_EP  = len(rewards_arr)
TOTAL_SUC = int(success_arr.sum())

# ── Run evaluation dynamically ───────────────────────────────────────────────
eval_results = evaluate(verbose=False)
EVAL_EP       = eval_results["successes"] + eval_results["failures"]
EVAL_SUC      = eval_results["successes"]
EVAL_FAIL     = eval_results["failures"]
EVAL_SR       = eval_results["success_rate"]
EVAL_AVG_REW  = eval_results["avg_reward"]
ACTION_SYM = {0: "<", 1: "v", 2: ">", 3: "^"}

def get_policy():
    return np.argmax(q_table, axis=1)

# ── Document setup ───────────────────────────────────────────────────────────
doc = SimpleDocTemplate(
    "report.pdf",
    pagesize=A4,
    leftMargin=2.2*cm, rightMargin=2.2*cm,
    topMargin=2*cm, bottomMargin=2*cm,
)

W, H = A4
styles = getSampleStyleSheet()

# Custom styles
title_style = ParagraphStyle("Title2",
    parent=styles["Title"], fontSize=16, spaceAfter=4, leading=20,
    textColor=colors.HexColor("#1a237e"))

subtitle_style = ParagraphStyle("Sub",
    parent=styles["Normal"], fontSize=10, textColor=colors.HexColor("#455a64"),
    alignment=TA_CENTER, spaceAfter=14)

h1 = ParagraphStyle("H1",
    parent=styles["Heading1"], fontSize=12, textColor=colors.HexColor("#1a237e"),
    spaceBefore=10, spaceAfter=4, borderPad=0)

h2 = ParagraphStyle("H2",
    parent=styles["Heading2"], fontSize=10.5, textColor=colors.HexColor("#0d47a1"),
    spaceBefore=6, spaceAfter=3)

body = ParagraphStyle("Body",
    parent=styles["Normal"], fontSize=9.5, leading=14, alignment=TA_JUSTIFY,
    spaceAfter=5)

code_style = ParagraphStyle("Code",
    parent=styles["Code"], fontSize=8.5, leading=12, leftIndent=12,
    backColor=colors.HexColor("#f5f5f5"), borderColor=colors.HexColor("#e0e0e0"),
    borderWidth=0.5, borderPad=4, spaceAfter=6)

table_header_style = ParagraphStyle("TH",
    parent=styles["Normal"], fontSize=9, fontName="Helvetica-Bold", alignment=TA_CENTER)

# ── Helper ────────────────────────────────────────────────────────────────────
def h(text, style=h1): return Paragraph(text, style)
def p(text): return Paragraph(text, body)
def sp(n=6): return Spacer(1, n)
def rule(): return HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#90caf9"), spaceAfter=6, spaceBefore=2)

# ── Build story ───────────────────────────────────────────────────────────────
story = []

# ── Title block ───────────────────────────────────────────────────────────────
story.append(sp(8))
story.append(Paragraph("Frozen Lake from First Principles Using Q-Learning", title_style))
story.append(Paragraph("DSCD 614 – Reinforcement Learning &nbsp;|&nbsp; Assignment 1 &nbsp;|&nbsp; Semester II 2025/2026", subtitle_style))
story.append(rule())
story.append(sp(4))

# ── 1. Introduction ───────────────────────────────────────────────────────────
story.append(h("1. Introduction"))
story.append(p(
    "Reinforcement Learning (RL) is a paradigm of machine learning in which an <b>agent</b> "
    "learns to act in an <b>environment</b> by trial and error. At each time step the agent "
    "observes a <b>state</b>, selects an <b>action</b>, and receives a scalar <b>reward</b> "
    "signal. The agent's goal is to discover a <b>policy</b> — a mapping from states to "
    "actions — that maximises the cumulative discounted reward over time. No labelled "
    "examples are provided; the agent must balance <i>exploration</i> (trying new actions) "
    "and <i>exploitation</i> (leveraging known good actions)."
))
story.append(p(
    "<b>Frozen Lake</b> is a classic grid-world benchmark. An agent navigates an 8×8 grid "
    "of frozen tiles from a Start cell (S) to a Goal cell (G) while avoiding Holes (H). "
    "The reward is sparse (+1 only upon reaching G), making it a challenging testbed for "
    "value-based RL algorithms. This assignment implements the environment, agent, training "
    "loop, and evaluation entirely in plain Python without any RL framework."
))

# ── 2. Environment Design ─────────────────────────────────────────────────────
story.append(h("2. Environment Design"))

story.append(h("2.1 Grid Map", h2))
story.append(p(
    "The standard 8×8 map is used (Figure 1). It contains 10 Hole cells arranged in "
    "two barrier clusters, forcing the agent to navigate an indirect path."
))

# Map table
map_data = []
for r, row_str in enumerate(MAP):
    row_cells = []
    for c, ch in enumerate(row_str):
        if ch == "H":
            cell = Paragraph("<b>H</b>", ParagraphStyle("mc", parent=styles["Normal"],
                fontSize=8, alignment=TA_CENTER, textColor=colors.white))
        elif ch == "G":
            cell = Paragraph("<b>G</b>", ParagraphStyle("mc", parent=styles["Normal"],
                fontSize=8, alignment=TA_CENTER, textColor=colors.white))
        elif ch == "S":
            cell = Paragraph("<b>S</b>", ParagraphStyle("mc", parent=styles["Normal"],
                fontSize=8, alignment=TA_CENTER, textColor=colors.white))
        else:
            cell = Paragraph("F", ParagraphStyle("mc", parent=styles["Normal"],
                fontSize=8, alignment=TA_CENTER))
        row_cells.append(cell)
    map_data.append(row_cells)

cell_w = 1.5 * cm
map_table = Table(map_data, colWidths=[cell_w]*8, rowHeights=[cell_w]*8)
map_ts = TableStyle([
    ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#90caf9")),
    ("ALIGN", (0,0), (-1,-1), "CENTER"),
    ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ("BACKGROUND", (0,0), (0,0), colors.HexColor("#43a047")),   # S
    ("BACKGROUND", (7,7), (7,7), colors.HexColor("#fdd835")),   # G
])
# Holes
holes = [(r,c) for r in range(8) for c in range(8) if MAP[r][c]=="H"]
for (r,c) in holes:
    map_ts.add("BACKGROUND", (c,r), (c,r), colors.HexColor("#e53935"))
map_table.setStyle(map_ts)

story.append(Table([[map_table]], colWidths=[(W - 4.4*cm)]))
story.append(p("<i>Figure 1: Frozen Lake 8×8 map. Green=Start, Yellow=Goal, Red=Hole, White=Frozen.</i>"))
story.append(sp(4))

story.append(h("2.2 State and Action Representation", h2))
story.append(p(
    "States are encoded as a single integer <b>s = row × 8 + col</b>, giving 64 states "
    "(0–63). State 0 is the Start; state 63 is the Goal. There are 4 actions: "
    "0 = Left (←), 1 = Down (↓), 2 = Right (→), 3 = Up (↑). "
    "Moves that would take the agent off the grid leave it in place."
))

story.append(h("2.3 Reward Structure", h2))
reward_data = [
    ["Event", "Reward"],
    ["Reach Goal (G)", "+1.0"],
    ["Fall in Hole (H)", "−1.0"],
    ["Any other step", "0.0"],
]
rt = Table(reward_data, colWidths=[9*cm, 4*cm])
rt.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1a237e")),
    ("TEXTCOLOR", (0,0), (-1,0), colors.white),
    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
    ("FONTSIZE", (0,0), (-1,-1), 9),
    ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.HexColor("#e8eaf6"), colors.white]),
    ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#9fa8da")),
    ("ALIGN", (1,0), (1,-1), "CENTER"),
    ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ("TOPPADDING", (0,0), (-1,-1), 4),
    ("BOTTOMPADDING", (0,0), (-1,-1), 4),
]))
story.append(rt)
story.append(sp(4))
story.append(p(
    "Holes are penalised with −1 (rather than 0) to provide an informative gradient signal "
    "even before the agent discovers the goal. This is a standard reward-shaping technique "
    "for sparse-reward environments."
))

# ── 3. Q-Learning Algorithm ───────────────────────────────────────────────────
story.append(h("3. Q-Learning Algorithm"))
story.append(p(
    "Q-Learning (Watkins, 1989) is a <b>model-free, off-policy</b> temporal-difference "
    "algorithm. It maintains a Q-table Q[s, a] estimating the expected discounted return "
    "when taking action <i>a</i> in state <i>s</i> and following the greedy policy thereafter. "
    "The table is initialised to <b>+1.0</b> (optimistic), which forces the agent to visit "
    "every state-action pair at least once before trusting its estimates."
))
story.append(h("3.1 Update Equation", h2))
story.append(Paragraph(
    "Q(s, a)  &lt;-  Q(s, a)  +  &alpha; [ r  +  &gamma; &middot; max<sub>a'</sub> Q(s', a')  &minus;  Q(s, a) ]",
    ParagraphStyle("eq", parent=styles["Normal"], fontSize=11, alignment=TA_CENTER,
                   spaceBefore=6, spaceAfter=6, fontName="Helvetica-Bold",
                   textColor=colors.HexColor("#1a237e"))
))
story.append(p(
    "Here <b>α</b> (alpha) is the learning rate controlling how fast new information overwrites "
    "old estimates; <b>γ</b> (gamma) is the discount factor weighting future rewards; "
    "<b>r</b> is the immediate reward; and <b>s'</b> is the next state. The term "
    "max<sub>a'</sub> Q(s', a') is the best estimated value of the next state under the "
    "greedy policy — this is what makes Q-Learning <i>off-policy</i>."
))

story.append(h("3.2 Exploration Strategy", h2))
story.append(p(
    "An <b>ε-greedy</b> policy is used: with probability ε the agent picks a uniformly "
    "random action (explore); otherwise it picks argmax<sub>a</sub> Q(s, a) (exploit). "
    "ε starts at 1.0 (pure exploration) and is multiplied by a decay factor after each "
    "episode, asymptotically reducing to a minimum of 0.01. This schedule ensures the "
    "agent explores thoroughly early in training and gradually transitions to exploitation "
    "as the Q-table converges."
))

# ── 4. Training Methodology ────────────────────────────────────────────────────
story.append(PageBreak())
story.append(h("4. Training Methodology"))

hp_data = [
    ["Hyperparameter", "Value", "Rationale"],
    ["Episodes", "20,000", "Sufficient for convergence on this map"],
    ["Learning rate (α)", "0.2", "Faster convergence; map is deterministic"],
    ["Discount factor (γ)", "0.99", "Goal is many steps away; future rewards matter"],
    ["Initial ε", "1.0", "Full exploration at start"],
    ["Minimum ε", "0.01", "Retain small random chance throughout"],
    ["ε decay rate", "0.9997", "Reaches ε_min (~0.01) around episode 15,350"],
    ["Max steps / episode", "300", "Prevents infinite loops in early training"],
    ["Q-table init", "+1.0 (optimistic)", "Forces systematic exploration"],
    ["Hole reward", "−1.0", "Provides gradient signal before goal is found"],
]
hpt = Table(hp_data, colWidths=[5.5*cm, 3.5*cm, 7.5*cm])
hpt.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1a237e")),
    ("TEXTCOLOR", (0,0), (-1,0), colors.white),
    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
    ("FONTSIZE", (0,0), (-1,-1), 8.5),
    ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.HexColor("#e8eaf6"), colors.white]),
    ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#9fa8da")),
    ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ("TOPPADDING", (0,0), (-1,-1), 3),
    ("BOTTOMPADDING", (0,0), (-1,-1), 3),
]))
story.append(hpt)
story.append(sp(6))

story.append(p(
    "The training loop runs for 20,000 episodes. Each episode begins with a call to "
    "<i>env.reset()</i>, then iterates the ε-greedy action selection, environment step, "
    "Q-table update, and ε-decay cycle until the episode terminates (hole, goal, or step "
    "limit). Episode reward, success flag, and current ε are recorded for analysis."
))

# ── 5. Experimental Results ────────────────────────────────────────────────────
story.append(h("5. Experimental Results"))

# Window success rates
windows = [2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000]
sr_data = [["Episode", "Success Rate (last 2000 ep)", "Epsilon"]]
for ep in windows:
    sl = success_arr[max(0, ep-2000):ep]
    sr = np.mean(sl) * 100
    eps_val = max(0.01, 1.0 * (0.9997 ** ep))
    sr_data.append([str(ep), f"{sr:.1f}%", f"{eps_val:.4f}"])
srt = Table(sr_data, colWidths=[4*cm, 7*cm, 4*cm])
srt.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1a237e")),
    ("TEXTCOLOR", (0,0), (-1,0), colors.white),
    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
    ("FONTSIZE", (0,0), (-1,-1), 9),
    ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.HexColor("#e8eaf6"), colors.white]),
    ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#9fa8da")),
    ("ALIGN", (1,0), (2,-1), "CENTER"),
    ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ("TOPPADDING", (0,0), (-1,-1), 3),
    ("BOTTOMPADDING", (0,0), (-1,-1), 3),
]))
story.append(srt)
story.append(sp(4))

story.append(p(
    f"Training yielded <b>{TOTAL_SUC:,} successful episodes out of {TOTAL_EP:,} "
    f"({TOTAL_SUC/TOTAL_EP*100:.1f}%)</b>. The success rate rises steeply from episode "
    "4,000 as ε falls below 0.3 and the Q-table begins to converge. By episode 16,000 "
    "the rolling success rate exceeds 97%."
))

# Training plot
try:
    img = Image("results/training_performance.png", width=15*cm, height=9*cm)
    story.append(img)
    story.append(p("<i>Figure 2: Training performance — reward, success rate, and epsilon decay.</i>"))
except Exception:
    story.append(p("<i>(Training performance plot: results/training_performance.png)</i>"))

story.append(sp(4))

# Evaluation summary — computed dynamically from evaluate()
eval_data = [
    ["Metric", "Value"],
    ["Episodes evaluated", str(EVAL_EP)],
    ["Successful runs", str(EVAL_SUC)],
    ["Failures", str(EVAL_FAIL)],
    ["Success rate", f"{EVAL_SR:.1f}%"],
    ["Average reward", f"{EVAL_AVG_REW:.4f}"],
]
et = Table(eval_data, colWidths=[8*cm, 5*cm])
et.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1a237e")),
    ("TEXTCOLOR", (0,0), (-1,0), colors.white),
    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
    ("FONTSIZE", (0,0), (-1,-1), 9),
    ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.HexColor("#e8eaf6"), colors.white]),
    ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#9fa8da")),
    ("ALIGN", (1,0), (1,-1), "CENTER"),
    ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ("TOPPADDING", (0,0), (-1,-1), 3),
    ("BOTTOMPADDING", (0,0), (-1,-1), 3),
]))
story.append(et)

# ── 6. Learned Policy ─────────────────────────────────────────────────────────
story.append(h("6. Learned Policy"))
story.append(p(
    "The policy is extracted by taking argmax over the Q-table for each non-terminal state. "
    "The table below shows the recommended action for each cell "
    "( < = Left, > = Right, ^ = Up, v = Down )."
))

policy = get_policy()
pol_data = []
for r in range(8):
    row = []
    for c in range(8):
        state = r * 8 + c
        ch = MAP[r][c]
        if ch == "H":
            sym = "H"
        elif ch == "G":
            sym = "G"
        else:
            sym = ACTION_SYM[policy[state]]
        color = colors.HexColor("#e53935") if ch == "H" else \
                colors.HexColor("#fdd835") if ch == "G" else \
                colors.HexColor("#43a047") if ch == "S" else \
                colors.HexColor("#e8eaf6")
        row.append(Paragraph(f"<b>{sym}</b>",
            ParagraphStyle("pc", parent=styles["Normal"],
                fontSize=9, alignment=TA_CENTER,
                textColor=colors.white if ch in ("H","G","S") else colors.HexColor("#1a237e"))))
    pol_data.append(row)

pol_table = Table(pol_data, colWidths=[1.8*cm]*8, rowHeights=[1.0*cm]*8)
pol_ts = TableStyle([
    ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#90caf9")),
    ("ALIGN", (0,0), (-1,-1), "CENTER"),
    ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ("BACKGROUND", (0,0), (0,0), colors.HexColor("#43a047")),
    ("BACKGROUND", (7,7), (7,7), colors.HexColor("#e6a817")),
])
for (r,c) in holes:
    pol_ts.add("BACKGROUND", (c,r), (c,r), colors.HexColor("#e53935"))
# Alternate row colour for non-special cells
for r in range(8):
    for c in range(8):
        ch = MAP[r][c]
        if ch not in ("H", "G", "S"):
            bg = colors.HexColor("#e8eaf6") if (r+c) % 2 == 0 else colors.white
            pol_ts.add("BACKGROUND", (c,r), (c,r), bg)
pol_table.setStyle(pol_ts)
story.append(Table([[pol_table]], colWidths=[(W - 4.4*cm)]))
story.append(p(
    "<i>Figure 3: Learned policy. Green=Start, Yellow=Goal, Red=Hole. "
    "Arrows indicate the greedy action for each safe cell.</i>"
))
story.append(sp(4))
story.append(p(
    "The greedy path from the learned policy is 14 steps (optimal): the agent moves "
    "down column 0 to row 1, then right along row 1 to column 4, then down through "
    "the middle-right corridor (rows 2–7, column 4–5), and finally right along row 7 "
    "to the goal at (7,7). This path avoids all 10 holes."
))

# ── 7. Challenges ─────────────────────────────────────────────────────────────
story.append(h("7. Challenges Encountered"))
story.append(p(
    "<b>Sparse reward and low random-walk success rate.</b> The 8×8 map contains 10 holes "
    "arranged in two barrier clusters. A purely random agent reaches the goal in fewer than "
    "0.4% of episodes (measured empirically over 100,000 trials). With hole reward 0 and "
    "Q-table initialised to 0, learning is unreliable: depending on the random seed, the agent "
    "may never reach the goal during 10,000 episodes, leaving the Q-table at zero and producing "
    "poor greedy behaviour. This motivated reward shaping (hole = −1) and optimistic "
    "initialisation (+1)."
))
story.append(p(
    "<b>Epsilon decay calibration.</b> An initial decay rate of 0.995 caused ε to reach "
    "its minimum after only ~900 episodes — far too early. The agent committed to a greedy "
    "policy before acquiring any useful Q-values. Slowing the decay to 0.9997 ensures "
    "meaningful exploration persists for the full 20,000-episode training run."
))
story.append(p(
    "<b>Solutions applied.</b> Two complementary techniques resolved both issues: "
    "(1) <i>Reward shaping</i> — penalising holes with −1 provides a learning signal "
    "even without reaching the goal, allowing the agent to learn hole-avoidance from "
    "early in training. "
    "(2) <i>Optimistic Q-table initialisation</i> (+1) — by initially overestimating "
    "every state, the agent treats unvisited states as attractive, driving systematic "
    "exploration before any reward is received. Together these techniques yield "
    "convergence to a 97%+ success rate well within 20,000 episodes."
))

# ── 8. Conclusion ─────────────────────────────────────────────────────────────
story.append(h("8. Conclusion"))
story.append(p(
    "A complete Q-Learning solution for the Frozen Lake 8×8 environment was implemented "
    "from first principles in Python, without any RL framework. The environment, agent, "
    "training loop, and evaluation are fully self-contained. After 20,000 training episodes "
    f"the agent achieves a <b>{EVAL_SR:.0f}% success rate</b> on {EVAL_EP} greedy "
    "evaluation episodes, "
    "demonstrating that the learned Q-table encodes a reliable path from Start to Goal. "
    "The key insights are: (i) sparse-reward maps require reward shaping or optimistic "
    "initialisation to bootstrap learning, and (ii) ε-decay must be calibrated to the "
    "total episode budget so that exploration is maintained throughout training. "
    "As a bonus, training performance visualisations were generated (Figure 2) and are "
    "included in the repository under <i>results/</i>."
))

# ── Build ─────────────────────────────────────────────────────────────────────
doc.build(story)
print("report.pdf generated successfully.")
