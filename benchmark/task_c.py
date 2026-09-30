import argparse
import json
import random
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

COLORS = ["red", "blue", "black", "green", "yellow", "purple", "orange", "white"]
HEX = {"red": "#D93A3A", "blue": "#3A6FD9", "black": "#222222", "green": "#3AA655",
       "yellow": "#F2C230", "purple": "#8A4FC7", "orange": "#F08A24", "white": "#FFFFFF"}
EDGE, INK, BG = "#333333", "#222222", "#EEF0F3"
DPI, STEP, SIZE = 100, 115, 40
X_LABEL, X_START, ROW_H = 40, 250, 150
RULE_LIST = ["A1, A2, and A3 are three different quantities from the following list; which one is which is not given:",
             "- the number of shapes",
             "- the number of times the shape changes between neighboring shapes",
             "- the number of times the color changes between neighboring shapes",
             "- the number of different shapes",
             "- the number of different colors",
             "- the total number of corners of all shapes (a circle has no corners)"]
VERTICES = {"circle": 0, "triangle": 3, "square": 4, "diamond": 4, "pentagon": 5, "hexagon": 6}
C_SHAPES = list(VERTICES)
RULES = ("count", "shape changes", "color changes", "distinct shapes", "distinct colors", "total vertices")


def _ngon(n, phase, r=1.0):
    t = phase + 2 * np.pi * np.arange(n) / n
    return r * np.stack([np.cos(t), np.sin(t)], 1)


SHAPE_PARTS = {
    "circle": [_ngon(64, 0, 0.85)],
    "triangle": [_ngon(3, np.pi / 2) + (0, -0.2)],
    "diamond": [_ngon(4, np.pi / 2) * (0.75, 1.0)],
    "square": [_ngon(4, np.pi / 4, 0.85)],
    "pentagon": [np.array([[0.0, 0.95], [0.95, 0.25], [0.45, -0.9], [-0.55, -0.75], [-0.9, 0.45]])],
    "hexagon": [np.array([[0.15, 0.95], [0.95, 0.45], [0.8, -0.35], [0.05, -0.95], [-0.8, -0.6], [-0.95, 0.35]])],
}


def _pt(px):
    return px * 72 / DPI


def _draw_shape(ax, name, color, x, y):
    parts = [p * SIZE + (x, y) for p in SHAPE_PARTS[name]]
    for p in parts:
        ax.add_patch(Polygon(p, closed=True, fc="none", ec=EDGE, lw=3, zorder=2))
    for p in parts:
        ax.add_patch(Polygon(p, closed=True, fc=color, ec="none", zorder=3))


def features(seq):
    sh, co = [s for s, _ in seq], [c for _, c in seq]

    def ch(xs):
        return sum(a != b for a, b in zip(xs, xs[1:]))

    def repeated(xs):
        cnt = Counter(xs)
        return sum(cnt[x] > 1 for x in xs)

    n, v = len(seq), [VERTICES[s] for s in sh]
    f = {"count": n, "count-1": n - 1, "shape changes": ch(sh), "color changes": ch(co), "any changes": ch(seq),
         "both change": sum(a[0] != b[0] and a[1] != b[1] for a, b in zip(seq, seq[1:])),
         "distinct shapes": len(set(sh)), "distinct colors": len(set(co)), "distinct pairs": len(set(seq)),
         "total vertices": sum(v), "max vertices": max(v), "distinct-shape vertices": sum(VERTICES[s] for s in set(sh)),
         "most common color": max(Counter(co).values()), "most common shape": max(Counter(sh).values()),
         "most common pair": max(Counter(seq).values()),
         "repeated-color shapes": repeated(co), "repeated-shape shapes": repeated(sh)}
    for k in ("shape changes", "color changes", "any changes"):
        f[f"{k}+1"] = f[k] + 1
        f[f"no {k}"] = n - 1 - f[k]
    for k in ("distinct shapes", "distinct colors", "distinct pairs", "most common color", "most common shape"):
        f[f"{k}-1"] = f[k] - 1
    return f


def extended_features(seq):
    sh, co = [s for s, _ in seq], [c for _, c in seq]
    f = dict(features(seq))
    for x in COLORS:
        f[f"{x} count"] = co.count(x)
        f[f"{x} first position"] = co.index(x) + 1 if x in co else 0
    for x in C_SHAPES:
        f[f"{x} count"] = sh.count(x)
        f[f"{x} first position"] = sh.index(x) + 1 if x in sh else 0
    poly = [s for s in sh if s != "circle"]
    f.update({"non-circle count": len(poly), "non-circle distinct shapes": len(set(poly)),
              "same color as first": co.count(co[0]), "same shape as first": sh.count(sh[0]),
              "same as first": seq.count(seq[0]), "same color as last": co.count(co[-1]),
              "same shape as last": sh.count(sh[-1]), "min vertices": min(VERTICES[s] for s in sh),
              "shapes with corners": sum(VERTICES[s] > 0 for s in sh)})
    return f


def equivalents(S, C, rule, seed):
    r = random.Random(seed)
    F = [extended_features(sequence(r, S, C, r.randint(3, 6))) for _ in range(300)]
    return {k for k in F[0] if all(f[k] == f[rule] for f in F)}


def identifies(examples, assign, eq):
    F = [extended_features(s) for s, _ in examples]
    for i, true in enumerate(assign):
        fit = {k for k in F[0] if all(f[k] == v[i] for f, (_, v) in zip(F, examples))}
        if true not in fit or not fit <= eq[i]:
            return False
    return True


def sequence(rng, S, C, n):
    s, c = rng.choice(S), rng.choice(C)
    seq = [(s, c)]
    for _ in range(n - 1):
        if rng.random() < 0.5:
            s = rng.choice([x for x in S if x != s])
        if rng.random() < 0.5:
            c = rng.choice([x for x in C if x != c])
        seq.append((s, c))
    return seq


def fmt_seq(seq):
    return "[" + ", ".join(f"{c} {s}" for s, c in seq) + "]"


def fmt_vals(v):
    return ", ".join(f"A{i}={x}" for i, x in enumerate(v, 1))


HEAD = ["Each sequence below is a list of colored shapes, from left to right.",
        "For each sequence, three numbers A1, A2, and A3 are determined from the sequence. The same rules are used "
        "for every sequence below."]


def prompt(examples, query, multimodal):
    names = [f"Example {k}" for k in range(1, len(examples) + 1)]
    lines = HEAD + RULE_LIST
    if multimodal:
        lines += [f"The image shows {', '.join(names)}, and Question, one sequence per row.", ""]
        lines += [f"{n}: {fmt_vals(v)}" for n, (_, v) in zip(names, examples)]
        return "\n".join(lines + ["", "Question: What are A1, A2, and A3 for the sequence in the Question row?"])
    for n, (s, v) in zip(names, examples):
        lines += ["", f"{n}: {fmt_seq(s)}", fmt_vals(v)]
    return "\n".join(lines + ["", f"Question: {fmt_seq(query)}", "What are A1, A2, and A3?"])


def render(examples, query, path):
    rows = [(f"Example {k}", s) for k, (s, _) in enumerate(examples, 1)] + [("Question", query)]
    width = max(1000, X_START + (max(len(s) for _, s in rows) - 1) * STEP + 110)
    height = ROW_H * len(rows) + 40
    fig = plt.figure(figsize=((width + 1e-6) / DPI, (height + 1e-6) / DPI), dpi=DPI, facecolor=BG)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis("off")
    for j, (label, seq) in enumerate(rows):
        y = height - 20 - ROW_H * (j + 0.5)
        ax.text(X_LABEL, y, label, fontsize=_pt(26), fontweight="bold", va="center", color=INK)
        for i, (s, c) in enumerate(seq):
            _draw_shape(ax, s, HEX[c], X_START + i * STEP, y)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--min-len", type=int, default=3)
    ap.add_argument("--max-len", type=int, default=6)
    ap.add_argument("--n-examples", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("task_c"))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    (args.out / "images").mkdir(parents=True, exist_ok=True)
    with open(args.out / "samples.jsonl", "w", encoding="utf-8") as fh:
        for i in range(args.n):
            S, C = rng.sample(C_SHAPES, rng.randint(3, 5)), rng.sample(COLORS, rng.randint(3, 5))
            assign = rng.sample(RULES, 3)
            eq = [equivalents(S, C, a, f"eq-{args.seed}-{i}-{a}") for a in assign]
            tries = 0
            while True:
                tries += 1
                seqs = [sequence(rng, S, C, rng.randint(args.min_len, args.max_len)) for _ in range(args.n_examples + 1)]
                examples = [(s, tuple(features(s)[a] for a in assign)) for s in seqs[:-1]]
                if identifies(examples, assign, eq) and seqs[-1] not in seqs[:-1]:
                    break
            query = seqs[-1]
            answer = tuple(features(query)[a] for a in assign)
            rid = f"{i:05d}"
            img = f"images/{rid}.png"
            render(examples, query, args.out / img)
            label = {"answer": fmt_vals(answer), "assign": list(assign), "tries": tries,
                     "examples": [{"sequence": [f"{s}:{c}" for s, c in sq], "values": fmt_vals(v)} for sq, v in examples],
                     "sequence": [f"{s}:{c}" for s, c in query]}
            rec = {"id": rid, "task": "seq",
                   "input": {"text": prompt(examples, query, False),
                             "multimodal": {"image": img, "text": prompt(examples, query, True)}},
                   "label": label}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"{rid}: A1~A3 = {' / '.join(assign)}, 시도 {tries}회", flush=True)
    print(f"문항 {args.n}개 생성 완료: {args.out / 'samples.jsonl'}")


if __name__ == "__main__":
    main()
