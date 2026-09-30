import argparse
import itertools
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Polygon

SHAPES = ["circle", "triangle", "star", "diamond", "heart", "clover", "square"]
COLORS = ["red", "blue", "black", "green", "yellow", "purple", "orange", "white"]
HEX = {"red": "#D93A3A", "blue": "#3A6FD9", "black": "#222222", "green": "#3AA655",
       "yellow": "#F2C230", "purple": "#8A4FC7", "orange": "#F08A24", "white": "#FFFFFF"}
EDGE, INK, BG = "#333333", "#222222", "#EEF0F3"
DPI, STEP, SIZE = 100, 115, 40
X_LABEL, X_IN, ROW_H = 40, 300, 150
LEVEL_ROLES = {1: ("shape+", "color+", "none"), 2: ("shape+", "shape-", "color+", "none")}


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
    if item[0] == "shape":
        parts = [p * SIZE + (x, y) for p in SHAPE_PARTS[item[1]]]
        for p in parts:
            ax.add_patch(Polygon(p, closed=True, fc="none", ec=EDGE, lw=3, zorder=2))
        for p in parts:
            ax.add_patch(Polygon(p, closed=True, fc=item[2], ec="none", zorder=3))
    else:
        w = 84
        ax.add_patch(FancyBboxPatch((x - w / 2, y - 24 * w / 64), w, 48 * w / 64,
                                    boxstyle="round,pad=0,rounding_size=10", fc="white", ec=EDGE,
                                    lw=1.5, ls="--", zorder=2))
        ax.text(x, y, item[1], fontsize=_pt(24), fontweight="bold", ha="center", va="center",
                color=INK, zorder=3)


def render(examples, query, path):
    rows = ([(f"Example {k}", e[0], e[1], (e[3], e[4])) for k, e in enumerate(examples, 1)]
            + [("Question", query[0], query[1], None)])
    height = ROW_H * len(rows) + 40
    fig = plt.figure(figsize=((1000 + 1e-6) / DPI, (height + 1e-6) / DPI), dpi=DPI, facecolor=BG)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, height)
    ax.axis("off")
    for j, (label, s, c, out) in enumerate(rows):
        y = height - 20 - ROW_H * (j + 0.5)
        ax.text(X_LABEL, y, label, fontsize=_pt(26), fontweight="bold", va="center", color=INK)
        _draw_item(ax, ("shape", s, HEX[c]), X_IN, y)
        ax.text(X_IN + STEP, y, "→", fontsize=_pt(34), ha="center", va="center", color=INK)
        _draw_item(ax, ("unknown", "?") if out is None else ("shape", out[0], HEX[out[1]]), X_IN + 2 * STEP, y)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def easy_count(rng, max_offset):
    return 100 * rng.randint(2, 50) + rng.randint(0, max_offset)


def effect(roles, counts):
    ds = sum(n if r == "shape+" else -n for r, n in zip(roles, counts) if r in ("shape+", "shape-"))
    return ds, sum(n for r, n in zip(roles, counts) if r == "color+")


def order_table(pal, seen, n):
    rest = [x for x in pal if x not in seen]
    rows = []
    for add in itertools.combinations(rest, n - len(seen)):
        first, *others = sorted(seen) + list(add)
        rows += [(first, *p) for p in itertools.permutations(others)]
    idx = {x: i for i, x in enumerate(pal)}
    arr = np.array([[idx[x] for x in o] for o in rows])
    pos = np.full((len(rows), len(pal)), -1)
    pos[np.arange(len(rows))[:, None], arr] = np.arange(n)
    return arr, pos


def predictions(pal, n, cases, query, steps, qstep):
    idx = {x: i for i, x in enumerate(pal)}
    seen = {a for a, _ in cases} | {b for _, b in cases} | {query}
    if len(seen) > n:
        return set()
    arr, pos = order_table(pal, seen, n)
    rows = np.arange(len(arr))
    mask = np.ones(len(arr), bool)
    for (a, b), d in zip(cases, steps):
        mask &= arr[rows, (pos[:, idx[a]] + d) % n] == idx[b]
    got = arr[rows[mask], (pos[mask, idx[query]] + qstep) % n]
    return {pal[i] for i in got.tolist()}


def unique_answer(level, S, C, examples, query):
    preds = set()
    for roles in set(itertools.permutations(LEVEL_ROLES[level])):
        eff = [effect(roles, e[2]) for e in examples]
        qe = effect(roles, query[2])
        ps = predictions(SHAPES, len(S), [(e[0], e[3]) for e in examples], query[0], [d for d, _ in eff], qe[0])
        if not ps:
            continue
        pc = predictions(COLORS, len(C), [(e[1], e[4]) for e in examples], query[1], [d for _, d in eff], qe[1])
        preds |= {(a, b) for a in ps for b in pc}
        if len(preds) > 1:
            break
    return preds


def presses(counts):
    parts = [f"B{i} was pressed {n:,} times" for i, n in enumerate(counts, 1)]
    return ", then ".join(parts[:-1]) + ", and then " + parts[-1] + "."


def hint(level, n, m):
    lines = [f"* There are {n} kinds of shapes and {m} kinds of colors in total."]
    if level == 1:
        lines.append("* Of the three buttons, one changes the shape, one changes the color, and one does nothing. Each "
                     "press of the shape button (or the color button) moves the shape (or the color) one step forward "
                     "in a fixed cyclic order. Which button does which, and the cyclic orders, are not given.")
    else:
        lines.append("* Of the four buttons, one moves the shape one step forward, one moves the shape one step "
                     "backward in the same cyclic order, one moves the color one step forward, and one does nothing. "
                     "The shapes and the colors each follow a fixed cyclic order. Which button does which, and the "
                     "cyclic orders, are not given.")
    return lines


def prompt(level, S, C, examples, query, multimodal):
    k = len(LEVEL_ROLES[level])
    names = ", ".join(f"B{i}" for i in range(1, k))
    lines = [f"There are {['', '', '', 'three', 'four'][k]} buttons, {names}, and B{k}. What each button does is not "
             "explained.", "",
             "* Before any button is pressed, a shape and a color are set (the initial setting).",
             "* After the buttons are pressed, one shape with the resulting shape and color is shown (the output).",
             *hint(level, len(S), len(C)),
             "* The buttons work the same way in every case below."]
    ex_names = [f"Example {j}" for j in range(1, len(examples) + 1)]
    if multimodal:
        lines += [f"* The image shows {', '.join(ex_names)}, and Question, one per row. Each row shows the initial "
                  "setting on the left and the output on the right.", ""]
        lines += [f"{n}: {presses(e[2])}" for n, e in zip(ex_names, examples)]
        return "\n".join(lines + [f"Question: {presses(query[2])} What is the output?",
                                  f"Write the answer as shape:color. The shape is one of {', '.join(SHAPES)}, "
                                  f"and the color is one of {', '.join(COLORS)}."])
    for n, (s, c, cnt, so, co) in zip(ex_names, examples):
        lines += ["", f"{n}: The initial setting is {s}-{c}. {presses(cnt)} The output was {so}-{co}."]
    return "\n".join(lines + ["", f"Question: The initial setting is {query[0]}-{query[1]}. {presses(query[2])} "
                                  "What is the output?"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, choices=(1, 2), default=1)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--min-cycle", type=int, default=3)
    ap.add_argument("--max-cycle", type=int, default=6)
    ap.add_argument("--max-offset", type=int, default=5)
    ap.add_argument("--n-examples", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    base = LEVEL_ROLES[args.level]
    (args.out / "images").mkdir(parents=True, exist_ok=True)
    with open(args.out / "samples.jsonl", "w", encoding="utf-8") as fh:
        for i in range(args.n):
            S = rng.sample(SHAPES, rng.randint(args.min_cycle, args.max_cycle))
            C = rng.sample(COLORS, rng.randint(args.min_cycle, args.max_cycle))
            roles = rng.sample(base, len(base))
            home = (rng.random() < 1 / len(S), rng.random() < 1 / len(C))
            tries = 0
            while True:
                tries += 1
                cases = []
                for _ in range(args.n_examples + 1):
                    s, c = rng.randrange(len(S)), rng.randrange(len(C))
                    cnt = [easy_count(rng, args.max_offset) for _ in base]
                    ds, dc = effect(roles, cnt)
                    cases.append((S[s], C[c], cnt, S[(s + ds) % len(S)], C[(c + dc) % len(C)]))
                examples, q = cases[:-1], cases[-1]
                if any(q[:3] == e[:3] for e in examples) or ((q[3] == q[0]), (q[4] == q[1])) != home:
                    continue
                if unique_answer(args.level, S, C, examples, q[:3]) == {(q[3], q[4])}:
                    break
            rid = f"{i:05d}"
            img = f"images/{rid}.png"
            render(examples, q, args.out / img)
            label = {"answer": f"{q[3]}:{q[4]}", "level": args.level, "roles": roles, "shapes": S, "colors": C,
                     "examples": [{"start": f"{e[0]}:{e[1]}", "presses": e[2], "output": f"{e[3]}:{e[4]}"} for e in examples],
                     "start": f"{q[0]}:{q[1]}", "presses": q[2], "tries": tries}
            rec = {"id": rid, "task": "forward",
                   "input": {"text": prompt(args.level, S, C, examples, q[:3], False),
                             "multimodal": {"image": img, "text": prompt(args.level, S, C, examples, q[:3], True)}},
                   "label": label}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"{rid}: L{args.level} 역할 {'/'.join(roles)}, 순환 {len(S)}/{len(C)}, 시도 {tries}회", flush=True)
    print(f"문항 {args.n}개 생성 완료: {args.out / 'samples.jsonl'}")


if __name__ == "__main__":
    main()
