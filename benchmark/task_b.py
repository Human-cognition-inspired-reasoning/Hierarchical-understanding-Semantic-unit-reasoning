import argparse
import itertools
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
DPI, STEP, SIZE = 100, 115, 40
CYCLE_X = {"b1": 330, "b2": 900}
CYCLE_LEN = (3, 5)
X_LABEL, X_START, ROW_H = 40, 250, 150
HYPOTHESES = [a[:i] + ("swap",) + a[i:] for i in range(3)
              for a in itertools.product(("shape", "color", "none"), repeat=2)]
TRUE_ASSIGNS = [a for a in HYPOTHESES if a.count("none") < 2]
SWAP_LINE = ("* One of the three buttons makes the other two buttons exchange their functions (pressing it again "
             "exchanges them back).")
START_LINE = ("* Each case starts from an initial setting with every button in its original function, and the "
              "buttons are pressed in the order shown. The result is one shape with the resulting shape and color.")


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


def render_cycles(rule, path, ccw):
    ns = {"b1": len(rule["shapes"]), "b2": len(rule["colors"])}
    r_max = max(cycle_radius(n) for n in ns.values())
    cy = 110 + r_max
    title_y = cy + r_max + 75
    height = int(title_y + 60)
    fig = plt.figure(figsize=((1300 + 1e-6) / DPI, (height + 1e-6) / DPI), dpi=DPI, facecolor=BG)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1300)
    ax.set_ylim(0, height)
    ax.axis("off")
    for key, title, items, flip in (("b1", "Shape cycle", [("shape", s, NEUTRAL) for s in rule["shapes"]], ccw[0]),
                                    ("b2", "Color cycle", [("chip", c) for c in rule["colors"]], ccw[1])):
        ax.text(CYCLE_X[key], title_y, title, fontsize=_pt(28), fontweight="bold", ha="center", va="center",
                color=INK)
        pts = cycle_points(ns[key], CYCLE_X[key], cy, cycle_radius(ns[key]))
        if flip:
            pts[:, 0] = 2 * CYCLE_X[key] - pts[:, 0]
        draw_cycle(ax, items, pts)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def render_rows(rule, examples, q, task, path):
    def shape(s):
        return "shape", rule["shapes"][s[0]], HEX[rule["colors"][s[1]]]
    st, bs, end, k = q
    rows = [(f"Example {j}", [shape(a)] + [("button", f"B{b}") for b in b_] + [shape(e)])
            for j, (a, b_, e) in enumerate(examples, 1)]
    qrow = [shape(st)] + [("button", "?" if (task == "inverse" and i == k) else f"B{b}") for i, b in enumerate(bs)]
    rows.append(("Question", qrow + [shape(end) if task == "inverse" else ("unknown", "?")]))
    width = max(1000, X_START + (max(len(r) for _, r in rows) - 1) * STEP + 110)
    height = ROW_H * len(rows) + 40
    fig = plt.figure(figsize=((width + 1e-6) / DPI, (height + 1e-6) / DPI), dpi=DPI, facecolor=BG)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis("off")
    for j, (label, items) in enumerate(rows):
        y = height - 20 - ROW_H * (j + 0.5)
        ax.text(X_LABEL, y, label, fontsize=_pt(26), fontweight="bold", va="center", color=INK)
        for i, item in enumerate(items):
            _draw_item(ax, item, X_START + i * STEP, y)
            if i < len(items) - 1:
                ax.text(X_START + (i + 0.5) * STEP, y, "→", fontsize=_pt(28), ha="center", va="center", color=INK)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def a_run(assign, sizes, start, buttons, skip_swap=False):
    s, c, sw = start[0], start[1], False
    pair = [i for i, f in enumerate(assign) if f != "swap"]
    for b in buttons:
        f = assign[b - 1]
        if f == "swap":
            sw = sw if skip_swap else not sw
            continue
        if sw:
            f = assign[pair[1 - pair.index(b - 1)]]
        if f == "shape":
            s = (s + 1) % sizes[0]
        elif f == "color":
            c = (c + 1) % sizes[1]
    return s, c


def fmt(rule, s):
    return f"{rule['shapes'][s[0]]}-{rule['colors'][s[1]]}"


def seq(buttons, blank=None):
    return ", ".join("?" if i == blank else f"B{b}" for i, b in enumerate(buttons))


def prompt(rule, examples, q, task, multimodal):
    def cycle(xs):
        return " → ".join(xs + xs[:1])
    lines = ["There are three buttons, B1, B2, and B3.", ""]
    if multimodal:
        lines.append("* The state is a shape and a color. The shapes and the colors are each arranged in the cycles "
                     "shown in the first image.")
    else:
        lines.append(f"* The state is a shape and a color. The shapes are arranged in the cycle "
                     f"{cycle(rule['shapes'])}, and the colors are arranged in the cycle {cycle(rule['colors'])}.")
    lines += [SWAP_LINE, START_LINE, "* The buttons work the same way in every case below."]
    names = [f"Example {k}" for k in range(1, len(examples) + 1)]
    if multimodal:
        lines.append(f"* The second image shows {', '.join(names)}, and Question, one per row. Each row starts with "
                     "the initial setting on the left, shows the buttons pressed in order, and ends with the result "
                     "on the right.")
        if task == "forward":
            return "\n".join(lines + ["", "Question: In the Question row, what is the result?",
                                      f"Write the answer as shape:color. The shape is one of {', '.join(SHAPES)}, "
                                      f"and the color is one of {', '.join(COLORS)}."])
        return "\n".join(lines + ["", "Question: In the Question row, which button was pressed at the ? position?"])
    for n, (st, bs, end) in zip(names, examples):
        lines += ["", f"{n}: Starting from {fmt(rule, st)}, the buttons were pressed in the order {seq(bs)}. "
                      f"The result was {fmt(rule, end)}."]
    st, bs, end, k = q
    if task == "forward":
        return "\n".join(lines + ["", f"Question: Starting from {fmt(rule, st)}, the buttons are pressed in the "
                                      f"order {seq(bs)}. What is the result?"])
    return "\n".join(lines + ["", f"Question: Starting from {fmt(rule, st)}, the buttons were pressed in the order "
                                  f"{seq(bs, k)}, and the result was {fmt(rule, end)}. "
                                  "Which button was pressed at the ? position?"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--min-presses", type=int, default=3)
    ap.add_argument("--max-presses", type=int, default=5)
    ap.add_argument("--n-examples", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("task_b"))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    arng = random.Random(f"arrows-{args.seed}")
    (args.out / "images").mkdir(parents=True, exist_ok=True)
    with open(args.out / "samples.jsonl", "w", encoding="utf-8") as fh:
        for sid in range(args.n):
            assign = rng.choice(TRUE_ASSIGNS)
            rule = {"shapes": rng.sample(SHAPES, rng.randint(*CYCLE_LEN)),
                    "colors": rng.sample(COLORS, rng.randint(*CYCLE_LEN)), "assign": list(assign)}
            sizes = (len(rule["shapes"]), len(rule["colors"]))

            def draw():
                st = (rng.randrange(sizes[0]), rng.randrange(sizes[1]))
                bs = [rng.randint(1, 3) for _ in range(rng.randint(args.min_presses, args.max_presses))]
                return st, bs, a_run(assign, sizes, st, bs)

            tries = 0
            while True:
                tries += 1
                examples = [draw() for _ in range(args.n_examples)]
                fq, iq = draw(), draw()
                k = rng.randrange(len(iq[1]))
                cons = [a for a in HYPOTHESES if all(a_run(a, sizes, st, bs) == e for st, bs, e in examples)]
                fwd = {a_run(a, sizes, fq[0], fq[1]) for a in cons}
                inv = {tuple(b for b in (1, 2, 3) if a_run(a, sizes, iq[0], iq[1][:k] + [b] + iq[1][k + 1:]) == iq[2])
                       for a in cons}
                if fwd == {fq[2]} and inv == {(iq[1][k],)}:
                    break
            cyc_img = f"images/{sid:05d}_cycles.png"
            ccw = (arng.random() < 0.5, arng.random() < 0.5)
            render_cycles(rule, args.out / cyc_img, ccw)
            ex_label = [{"start": fmt(rule, st), "buttons": bs, "end": fmt(rule, e)} for st, bs, e in examples]
            for task, (st, bs, end) in (("forward", fq), ("inverse", iq)):
                q = (st, bs, end, k)
                rid = f"{sid:05d}_{task}"
                img = f"images/{rid}.png"
                render_rows(rule, examples, q, task, args.out / img)
                label = {"rule": rule, "examples": ex_label, "start": fmt(rule, st), "buttons": bs,
                         "end": fmt(rule, end), "consistent_assignments": len(cons), "tries": tries, "ccw": list(ccw)}
                if task == "forward":
                    naive = a_run(assign, sizes, st, bs, skip_swap=True)
                    label.update(answer=f"{rule['shapes'][end[0]]}:{rule['colors'][end[1]]}",
                                 naive_answer=f"{rule['shapes'][naive[0]]}:{rule['colors'][naive[1]]}")
                else:
                    label.update(answer=f"button {bs[k]}", blank_index=k,
                                 naive_candidates=[b for b in (1, 2, 3) if a_run(
                                     assign, sizes, st, bs[:k] + [b] + bs[k + 1:], skip_swap=True) == end])
                rec = {"id": rid, "scenario_id": sid, "task": task,
                       "input": {"text": prompt(rule, examples, q, task, False),
                                 "multimodal": {"image": [cyc_img, img], "text": prompt(rule, examples, q, task, True)}},
                       "label": label}
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"{sid:05d}: 배정 {'/'.join(assign)}, 순환 {sizes[0]}/{sizes[1]}, 시도 {tries}회, "
                  f"남은 배정 {len(cons)}개", flush=True)
    print(f"시나리오 {args.n}개, 문항 {2 * args.n}개 생성 완료: {args.out / 'samples.jsonl'}")


if __name__ == "__main__":
    main()
