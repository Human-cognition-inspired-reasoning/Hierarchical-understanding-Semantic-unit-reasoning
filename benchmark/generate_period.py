import argparse
import itertools
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from generate_button_tasks import BG, COLORS, DPI, HEX, INK, SHAPES, STEP, _draw_item, _pt

X_LABEL, X_IN, ROW_H = 40, 300, 150


def perm_tables(n):
    perms = np.array(list(itertools.permutations(range(n))), dtype=np.int64)
    T = np.empty((len(perms), n, n + 1), dtype=np.int64)
    T[:, :, 0] = np.arange(n)
    for r in range(1, n + 1):
        T[:, :, r] = np.take_along_axis(perms, T[:, :, r - 1], 1)
    L = np.argmax(T[:, :, 1:] == np.arange(n)[None, :, None], axis=2) + 1
    return T, L


def predictions(tables, cases, query):
    T, L = tables
    idx = np.arange(len(T))
    out = set()
    for d in range(3):
        mask = np.ones(len(T), bool)
        for v, w, steps in cases:
            mask &= T[idx, v, steps[d] % L[:, v]] == w
        vq, tq = query[0], query[1][d]
        out |= set(T[idx[mask], vq, tq % L[mask, vq]].tolist())
    ws = {w for _, w, _ in cases}
    return out | ws if len(ws) == 1 else out


def easy_count(rng, max_offset=3):
    return 100 * rng.randint(2, 50) + rng.randint(0, max_offset)


def fmt_state(rule, s, c):
    return f"{rule['shapes'][s]}-{rule['colors'][c]}"


def presses(n1, n2):
    return f"B1 was pressed {n1:,} times, and then B2 was pressed {n2:,} times."


HEAD = ["There are two buttons, B1 and B2. What each button does is not explained.", "",
        "* Before any button is pressed, a shape and a color are set (the initial setting).",
        "* After the buttons are pressed, one shape with the resulting shape and color is shown (the output).",
        "* The buttons work the same way in every case below."]


def prompt(rule, examples, query, multimodal, hint=False, revised=False):
    lines = list(HEAD)
    if revised:
        lines[-1:-1] = [f"* There are {len(rule['shapes'])} kinds of shapes and {len(rule['colors'])} kinds of colors "
                        "in total.",
                        "* One of the two buttons changes the shape and the other changes the color. Each press moves "
                        "the shape (or the color) one step forward in a fixed cyclic order. Which button changes "
                        "which, and the cyclic orders, are not given."]
    if hint:
        lines[-1:-1] = [f"* There are {len(rule['shapes'])} kinds of shapes and {len(rule['colors'])} kinds of colors "
                        "in total.",
                        "* Pressing a certain button changes the shape, cycling through the shapes in a fixed order, "
                        "and pressing a certain button changes the color, cycling through the colors in a fixed "
                        "order."]
    names = [f"Example {k}" for k in range(1, len(examples) + 1)]
    if multimodal:
        lines += [f"* The image shows {', '.join(names)}, and Question, one per row. Each row shows the initial "
                  "setting on the left and the output on the right.", ""]
        lines += [f"{n}: {presses(n1, n2)}" for n, (_, _, n1, n2, _) in zip(names, examples)]
        return "\n".join(lines + [f"Question: {presses(query[2], query[3])} What is the output?",
                                  f"Write the answer as shape:color. The shape is one of {', '.join(SHAPES)}, "
                                  f"and the color is one of {', '.join(COLORS)}."])
    for n, (s, c, n1, n2, (so, co)) in zip(names, examples):
        lines += ["", f"{n}: The initial setting is {fmt_state(rule, s, c)}. {presses(n1, n2)} "
                      f"The output was {fmt_state(rule, so, co)}."]
    s, c, n1, n2, _ = query
    return "\n".join(lines + ["", f"Question: The initial setting is {fmt_state(rule, s, c)}. {presses(n1, n2)} "
                                  "What is the output?"])


def render(rule, examples, query, path):
    rows = [(f"Example {k}", e) for k, e in enumerate(examples, 1)] + [("Question", query)]
    height = ROW_H * len(rows) + 40
    fig = plt.figure(figsize=((1000 + 1e-6) / DPI, (height + 1e-6) / DPI), dpi=DPI, facecolor=BG)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, height)
    ax.axis("off")
    for j, (label, (s, c, _, _, out)) in enumerate(rows):
        y = height - 20 - ROW_H * (j + 0.5)
        ax.text(X_LABEL, y, label, fontsize=_pt(26), fontweight="bold", va="center", color=INK)
        _draw_item(ax, ("shape", rule["shapes"][s], HEX[rule["colors"][c]]), X_IN, y)
        ax.text(X_IN + STEP, y, "→", fontsize=_pt(34), ha="center", va="center", color=INK)
        last = ("unknown", "?") if label == "Question" else ("shape", rule["shapes"][out[0]], HEX[rule["colors"][out[1]]])
        _draw_item(ax, last, X_IN + 2 * STEP, y)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--min-cycle", type=int, default=2)
    ap.add_argument("--max-cycle", type=int, default=4)
    ap.add_argument("--n-examples", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("task_period"))
    ap.add_argument("--hint", action="store_true")
    ap.add_argument("--revised", action="store_true")
    ap.add_argument("--max-offset", type=int, default=3)
    args = ap.parse_args()

    tables = {"shape": perm_tables(len(SHAPES)), "color": perm_tables(len(COLORS))}
    rng = random.Random(args.seed)
    (args.out / "images").mkdir(parents=True, exist_ok=True)
    with open(args.out / "samples.jsonl", "w", encoding="utf-8") as fh:
        for i in range(args.n):
            S = rng.sample(SHAPES, rng.randint(args.min_cycle, args.max_cycle))
            C = rng.sample(COLORS, rng.randint(args.min_cycle, args.max_cycle))
            rule = {"shapes": S, "colors": C}
            shape_btn = rng.randrange(2) if args.revised else 0
            if args.revised:
                home = (rng.random() < 1 / len(S), rng.random() < 1 / len(C))
            tries = 0
            while True:
                tries += 1
                cases = []
                for _ in range(args.n_examples + 1):
                    s, c, n1, n2 = rng.randrange(len(S)), rng.randrange(len(C)), easy_count(rng, args.max_offset), easy_count(rng, args.max_offset)
                    ns, nc = (n1, n2) if shape_btn == 0 else (n2, n1)
                    cases.append((s, c, n1, n2, ((s + ns) % len(S), (c + nc) % len(C))))
                examples, query = cases[:-1], cases[-1]
                ok = all(query[:4] != e[:4] for e in examples)
                if args.revised:
                    ok = ok and ((query[4][0] == query[0]), (query[4][1] == query[1])) == home
                for a, (pal, cyc) in enumerate((("shape", S), ("color", C))):
                    full = SHAPES if pal == "shape" else COLORS
                    obs = [(full.index(cyc[e[a]]), full.index(cyc[e[4][a]]), (e[2], e[3], e[2] + e[3])) for e in examples]
                    q = (full.index(cyc[query[a]]), (query[2], query[3], query[2] + query[3]))
                    ok = ok and predictions(tables[pal], obs, q) == {full.index(cyc[query[4][a]])}
                if ok:
                    break
            rid = f"{i:05d}"
            img = f"images/{rid}.png"
            render(rule, examples, query, args.out / img)
            so, co = query[4]
            label = {"answer": f"{S[so]}:{C[co]}", "shapes": S, "colors": C, "tries": tries,
                     "examples": [{"start": f"{S[e[0]]}:{C[e[1]]}", "presses": [e[2], e[3]],
                                   "output": f"{S[e[4][0]]}:{C[e[4][1]]}"} for e in examples],
                     "start": f"{S[query[0]]}:{C[query[1]]}", "presses": [query[2], query[3]]}
            if args.revised:
                label["shape_button"] = f"B{shape_btn + 1}"
            rec = {"id": rid, "task": "forward",
                   "input": {"text": prompt(rule, examples, query, False, args.hint, args.revised),
                             "multimodal": {"image": img,
                                            "text": prompt(rule, examples, query, True, args.hint, args.revised)}},
                   "label": label}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"{rid}: 순환 {len(S)}/{len(C)}, 시도 {tries}회", flush=True)
    print(f"문항 {args.n}개 생성 완료: {args.out / 'samples.jsonl'}")


if __name__ == "__main__":
    main()
