import argparse
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

SHAPES = ["circle", "triangle", "star", "diamond", "heart", "clover", "square"]
COLORS = ["red", "blue", "black", "green", "yellow", "purple", "orange", "white"]
HEX = {"red": "#D93A3A", "blue": "#3A6FD9", "black": "#222222", "green": "#3AA655",
       "yellow": "#F2C230", "purple": "#8A4FC7", "orange": "#F08A24", "white": "#FFFFFF"}
NEUTRAL, EDGE, INK, BG = "#A0A0A0", "#333333", "#222222", "#EEF0F3"
DPI, X0, STEP, SIZE, Q_Y = 100, 170, 115, 40, 100
CYCLE_X = {"b1": 330, "b2": 900}
Q_LEGEND = {
    "forward": "Q: start from the leftmost shape and press the buttons from left to right; ? is the result.",
    "inverse": "Q: starting from the leftmost shape, pressing the buttons from left to right gives the rightmost "
               "shape; ? is the unknown button.",
}
ROLES = {
    "swap": "swaps the roles of b1 and b2. Pressing it again swaps them back.",
    "reverse": "reverses the cycling order of b1 and b2. Pressing it again restores the original order.",
    "both": "presses b1 and b2 at the same time.",
}
CYCLE_LEN = (3, 5)


def press(state, button, rule):
    shape, color, swapped, reverse = state
    if button == 3:
        if rule["role"] == "swap":
            return shape, color, not swapped, reverse
        if rule["role"] == "reverse":
            return shape, color, swapped, not reverse
        return press(press(state, 1, rule), 2, rule)
    step = -1 if reverse else 1
    if (button == 1) != swapped:
        shape = (shape + step) % len(rule["shapes"])
    else:
        color = (color + step) % len(rule["colors"])
    return shape, color, swapped, reverse


def run(rule, start, buttons, ignore_b3=False):
    state = (*start, False, False)
    for b in buttons:
        if not (ignore_b3 and b == 3):
            state = press(state, b, rule)
    return state[:2]


def fmt(rule, s):
    return f"{rule['shapes'][s[0]]}:{rule['colors'][s[1]]}"


def rules_text(rule):
    def cycle(xs):
        return " → ".join(xs + xs[:1])
    return "\n\n".join([
        "There are three buttons: b1, b2, and b3.",
        f"1. Pressing b1 changes the shape by one step along the cycle {cycle(rule['shapes'])}, "
        "in the direction of the arrows.",
        f"2. Pressing b2 changes the color by one step along the cycle {cycle(rule['colors'])}, "
        "in the direction of the arrows.",
        f"3. Pressing b3 {ROLES[rule['role']]}",
    ])


def _ngon(n, phase, r=1.0):
    t = phase + 2 * np.pi * np.arange(n) / n
    return r * np.stack([np.cos(t), np.sin(t)], 1)


def _star():
    t = np.pi / 2 + np.pi * np.arange(10) / 5
    r = np.where(np.arange(10) % 2 == 0, 1.0, 0.45)
    return np.stack([r * np.cos(t), r * np.sin(t)], 1)


def _heart():
    t = np.linspace(0, 2 * np.pi, 120, endpoint=False)
    x = 16 * np.sin(t) ** 3
    y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
    return np.stack([x, y + 2.5], 1) / 17


SHAPE_PARTS = {
    "circle": [_ngon(64, 0, 0.85)],
    "triangle": [_ngon(3, np.pi / 2) + (0, -0.2)],
    "star": [_star()],
    "diamond": [_ngon(4, np.pi / 2) * (0.75, 1.0)],
    "heart": [_heart()],
    "clover": [_ngon(48, 0, 0.36) + c for c in ((0, 0.42), (-0.42, -0.05), (0.42, -0.05))]
              + [_ngon(32, 0, 0.22) + (0, 0.1),
                 np.array([[-0.08, -0.1], [0.08, -0.1], [0.25, -0.95], [-0.25, -0.95]])],
    "square": [_ngon(4, np.pi / 4, 0.85)],
}


def _pt(px):
    return px * 72 / DPI


def _draw_item(ax, item, x, y):
    kind = item[0]
    if kind == "shape":
        parts = [p * SIZE + (x, y) for p in SHAPE_PARTS[item[1]]]
        for p in parts:
            ax.add_patch(Polygon(p, closed=True, fc="none", ec=EDGE, lw=3, zorder=2))
        for p in parts:
            ax.add_patch(Polygon(p, closed=True, fc=item[2], ec="none", zorder=3))
    elif kind == "chip":
        ax.add_patch(FancyBboxPatch((x - 34, y - 16), 68, 32, boxstyle="round,pad=0,rounding_size=8",
                                    fc=HEX[item[1]], ec=EDGE, lw=1.5, zorder=2))
    else:
        w = 64 if kind == "button" else 84
        ax.add_patch(FancyBboxPatch((x - w / 2, y - 24 * w / 64), w, 48 * w / 64,
                                    boxstyle="round,pad=0,rounding_size=10", fc="white", ec=EDGE,
                                    lw=1.5, ls="-" if kind == "button" else "--", zorder=2))
        ax.text(x, y, item[1], fontsize=_pt(24), fontweight="bold", ha="center", va="center",
                color=INK, zorder=3)


def cycle_radius(n):
    return max(115, 75 / np.sin(np.pi / n))


def cycle_points(n, cx, cy, r):
    t = np.pi / 2 - 2 * np.pi * np.arange(n) / n
    return np.stack([cx + r * np.cos(t), cy + r * np.sin(t)], 1)


def draw_cycle(ax, items, pts):
    for i, item in enumerate(items):
        _draw_item(ax, item, *pts[i])
        a, b = pts[i], pts[(i + 1) % len(items)]
        d = (b - a) / np.linalg.norm(b - a)
        ax.add_patch(FancyArrowPatch(a + 50 * d, b - 50 * d, arrowstyle="-|>", mutation_scale=22, lw=2.2,
                                     color=INK, zorder=1))


def layout(rule):
    ns = {"b1": len(rule["shapes"]), "b2": len(rule["colors"])}
    r_max = max(cycle_radius(n) for n in ns.values())
    cy = Q_Y + 150 + r_max
    pos = {k: cycle_points(n, CYCLE_X[k], cy, cycle_radius(n)) for k, n in ns.items()}
    title_y = cy + r_max + 75
    return pos, title_y, int(title_y + 170)


def render(rule, q, task, path):
    s0, c0 = q["start"].split(":")
    q_row = [("shape", s0, HEX[c0])] + [("button", f"b{b}") for b in q["buttons"]]
    if task == "inverse":
        q_row[q["blank_index"] + 1] = ("button", "?")
        s1, c1 = q["end"].split(":")
        q_row.append(("shape", s1, HEX[c1]))
    else:
        q_row.append(("unknown", "?"))
    pos, title_y, height = layout(rule)
    width = max(1300, X0 + (len(q_row) - 1) * STEP + 110)
    fig = plt.figure(figsize=((width + 1e-6) / DPI, (height + 1e-6) / DPI), dpi=DPI, facecolor=BG)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis("off")
    legend = ["b1: changes the shape by one step in the direction of the arrows.",
              "b2: changes the color by one step in the direction of the arrows.",
              f"b3: {ROLES[rule['role']]}", Q_LEGEND[task]]
    for i, line in enumerate(legend):
        ax.text(40, height - 35 - 30 * i, line, fontsize=_pt(19), va="center", color=INK)
    for key, items in (("b1", [("shape", s, NEUTRAL) for s in rule["shapes"]]),
                       ("b2", [("chip", c) for c in rule["colors"]])):
        ax.text(CYCLE_X[key], title_y, key, fontsize=_pt(28), fontweight="bold", ha="center", va="center", color=INK)
        draw_cycle(ax, items, pos[key])
    ax.text(40, Q_Y, "Q", fontsize=_pt(28), fontweight="bold", va="center", color=INK)
    for i, item in enumerate(q_row):
        _draw_item(ax, item, X0 + i * STEP, Q_Y)
        if i < len(q_row) - 1:
            ax.text(X0 + (i + 0.5) * STEP, Q_Y, "→", fontsize=_pt(28), ha="center", va="center", color=INK)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def multimodal_text(rule, task):
    lines = ["There are three buttons: b1, b2, and b3.",
             "1. Pressing b1 changes the shape by one step in the direction of the arrows in the b1 cycle shown in "
             "the image.",
             "2. Pressing b2 changes the color by one step in the direction of the arrows in the b2 cycle shown in "
             "the image.",
             f"3. Pressing b3 {ROLES[rule['role']]}"]
    if task == "forward":
        lines.append("Question: As shown in row Q of the image, start from the leftmost shape and press the buttons "
                     "in order. What is the result?\n"
                     f"Write the answer as shape:color. The shape is one of {', '.join(SHAPES)}, "
                     f"and the color is one of {', '.join(COLORS)}.")
    else:
        lines.append("Question: As shown in row Q of the image, starting from the leftmost shape, the buttons were "
                     "pressed in order and the result is the rightmost shape. Which button was pressed at the ? "
                     "position?")
    return "\n\n".join(lines)


def _draw(rng, rule, n_press):
    start = (rng.randrange(len(rule["shapes"])), rng.randrange(len(rule["colors"])))
    return start, [rng.randint(1, 3) for _ in range(n_press)]


def make_forward(rng, rule, n_press):
    start, buttons = _draw(rng, rule, n_press)
    end, naive = run(rule, start, buttons), run(rule, start, buttons, ignore_b3=True)
    seq = " -> ".join([fmt(rule, start)] + [f"button {b}" for b in buttons])
    return {"question": f"Question: What is the result after [{seq}]?", "answer": fmt(rule, end),
            "start": fmt(rule, start), "buttons": buttons, "naive_answer": fmt(rule, naive)}


def make_inverse(rng, rule, n_press, blank_pos=None):
    for _ in range(1000):
        start, buttons = _draw(rng, rule, n_press)
        k = rng.randrange(n_press) if blank_pos is None else blank_pos
        end = run(rule, start, buttons)

        def fits(ignore_b3):
            return [b for b in (1, 2, 3)
                    if run(rule, start, buttons[:k] + [b] + buttons[k + 1:], ignore_b3) == end]

        if fits(False) == [buttons[k]]:
            seq = [fmt(rule, start)] + [f"button {b}" for b in buttons] + [fmt(rule, end)]
            seq[k + 1] = "button ?"
            return {"question": f"Question: In [{' -> '.join(seq)}], which button was pressed at the ? position?",
                    "answer": f"button {buttons[k]}", "start": fmt(rule, start), "end": fmt(rule, end),
                    "buttons": buttons, "blank_index": k, "naive_candidates": fits(True)}
    raise RuntimeError("inverse question not found")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--min-presses", type=int, default=3)
    ap.add_argument("--max-presses", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("button_tasks"))
    ap.add_argument("--blank-pos", type=int, default=None)
    args = ap.parse_args()
    if not 2 <= args.min_presses <= args.max_presses:
        ap.error("--min-presses는 2 이상, --max-presses 이하여야 합니다.")
    if args.blank_pos is not None and not 0 <= args.blank_pos < args.min_presses:
        ap.error("--blank-pos는 0 이상, --min-presses 미만이어야 합니다.")

    rng = random.Random(args.seed)
    (args.out / "images").mkdir(parents=True, exist_ok=True)
    with open(args.out / "samples.jsonl", "w", encoding="utf-8") as fh:
        for sid in range(args.n):
            rule = {"shapes": rng.sample(SHAPES, rng.randint(*CYCLE_LEN)),
                    "colors": rng.sample(COLORS, rng.randint(*CYCLE_LEN)),
                    "role": rng.choice(list(ROLES))}
            for task, maker in (("forward", make_forward), ("inverse", make_inverse)):
                n_press = rng.randint(args.min_presses, args.max_presses)
                q = maker(rng, rule, n_press) if task == "forward" else maker(rng, rule, n_press, args.blank_pos)
                rid = f"{sid:05d}_{task}"
                img = f"images/{rid}.png"
                render(rule, q, task, args.out / img)
                rec = {"id": rid, "scenario_id": sid, "task": task,
                       "input": {"text": rules_text(rule) + "\n\n" + q.pop("question"),
                                 "multimodal": {"image": img, "text": multimodal_text(rule, task)}},
                       "label": {**q, "rule": rule}}
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"규칙 세트 {args.n}개, 문항 {2 * args.n}개 생성 완료: {args.out / 'samples.jsonl'}")


if __name__ == "__main__":
    main()
