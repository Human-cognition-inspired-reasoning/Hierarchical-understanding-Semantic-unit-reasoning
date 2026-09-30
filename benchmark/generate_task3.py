import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt

from generate_button_tasks import (BG, COLORS, CYCLE_LEN, DPI, HEX, INK, NEUTRAL, SHAPES, STEP, _draw_item, _pt,
                                   cycle_points, cycle_radius, draw_cycle)

X3, OUT_Y, START_Y = 230, 100, 240
CYCLE_CX = {"B2": 330, "B3": 900}
LEGEND = [
    "Initial setting: the shape and color set at the beginning (no shape has been placed yet).",
    "B1: places a shape with the currently set shape and color.",
    "B2: changes the currently set shape by one step in the direction of the arrows.",
    "B3: changes the currently set color by one step in the direction of the arrows.",
]


def solve(rule, outputs):
    s, c = rule["start"]
    b2 = b3 = 0
    for ts, tc in outputs:
        b2 += (ts - s) % len(rule["shapes"])
        b3 += (tc - c) % len(rule["colors"])
        s, c = ts, tc
    return len(outputs), b2, b3


def naive_solve(rule, outputs):
    prev = [tuple(rule["start"])] + outputs[:-1]
    return (len(outputs), sum(p[0] != o[0] for p, o in zip(prev, outputs)),
            sum(p[1] != o[1] for p, o in zip(prev, outputs)))


def fmt_counts(counts):
    return ", ".join(f"B{i}={n}" for i, n in enumerate(counts, 1))


def press_sequence(rule, outputs):
    s, c = rule["start"]
    seq = []
    for ts, tc in outputs:
        seq += ["B2"] * ((ts - s) % len(rule["shapes"])) + ["B3"] * ((tc - c) % len(rule["colors"])) + ["B1"]
        s, c = ts, tc
    return seq


def capped_outputs(rng, rule, n_out, max_b2, max_b3):
    k2, k3 = [0] * n_out, [0] * n_out
    for _ in range(rng.randint(0, max_b2)):
        k2[rng.randrange(n_out)] += 1
    for _ in range(rng.randint(0, max_b3)):
        k3[rng.randrange(n_out)] += 1
    s, c = rule["start"]
    outputs = []
    for a, b in zip(k2, k3):
        s, c = (s + a) % len(rule["shapes"]), (c + b) % len(rule["colors"])
        outputs.append((s, c))
    return outputs


def fmt_outputs(rule, outputs):
    return "[" + ", ".join(f"{rule['colors'][c]} {rule['shapes'][s]}" for s, c in outputs) + "]"


def prompt(rule, outputs, multimodal, forward=False):
    S, C = rule["shapes"], rule["colors"]
    s0, c0 = rule["start"]

    def cycle(xs):
        return " → ".join(xs + xs[:1])

    if multimodal:
        shape_dir = "only in the direction of the arrows in the B2 cycle shown in the image"
        color_dir = "only in the direction of the arrows in the B3 cycle shown in the image"
        start = "The initial setting is shown in the Initial setting row of the image."
        result = ["After pressing the buttons several times, the created shapes, from left to right, "
                  "are shown in row Output of the image."]
    else:
        shape_dir = f"only to the right in the order {cycle(S)}"
        color_dir = f"only in the direction of the arrows in the order {cycle(C)}"
        start = f"The initial setting is {S[s0]}-{C[c0]}."
        result = ["After pressing the buttons several times, the created shapes, from left to right, were:",
                  fmt_outputs(rule, outputs)]
    if forward:
        tail = ["", "The buttons were pressed in the following order: " + ", ".join(press_sequence(rule, outputs)) + ".",
                "Question: What shapes were created, from left to right?"]
        if multimodal:
            tail.append(f"Write each shape as '<color> <shape>'. The shape is one of {', '.join(SHAPES)}, "
                        f"and the color is one of {', '.join(COLORS)}.")
    else:
        tail = ["* Every button was pressed the minimum number of times.", "", *result,
                "Question: How many times were B1, B2, and B3 each pressed?"]
    return "\n".join([
        "There are three buttons.",
        "",
        "* B1 — Create: creates one shape with the currently set shape and color. "
        "Created shapes are placed from left to right in order.",
        f"* B2 — Change shape: each time B2 is pressed, the currently set shape changes by one step, "
        f"{shape_dir}. It cannot move in the opposite direction.",
        f"* B3 — Change color: each time B3 is pressed, the currently set color changes by one step, "
        f"{color_dir}. It cannot move in the opposite direction.",
        f"* {start}",
        "* Shapes that have already been created do not change when B2 or B3 is pressed afterwards.",
        *tail,
    ])


def layout(rule):
    ns = {"B2": len(rule["shapes"]), "B3": len(rule["colors"])}
    rs = {k: cycle_radius(n) for k, n in ns.items()}
    cy = START_Y + 130 + max(rs.values())
    pos = {k: cycle_points(n, CYCLE_CX[k], cy, rs[k]) for k, n in ns.items()}
    title_y = cy + max(rs.values()) + 75
    return pos, title_y, int(title_y + 170)


def render(rule, outputs, path, with_output=True):
    S, C = rule["shapes"], rule["colors"]
    s0, c0 = rule["start"]
    pos, title_y, height = layout(rule)
    width = max(1300, X3 + (len(outputs) - 1) * STEP + 110)
    fig = plt.figure(figsize=((width + 1e-6) / DPI, (height + 1e-6) / DPI), dpi=DPI, facecolor=BG)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis("off")
    for i, line in enumerate(LEGEND):
        ax.text(40, height - 35 - 30 * i, line, fontsize=_pt(19), va="center", color=INK)
    for key, items in (("B2", [("shape", s, NEUTRAL) for s in S]), ("B3", [("chip", c) for c in C])):
        ax.text(CYCLE_CX[key], title_y, key, fontsize=_pt(28), fontweight="bold", ha="center", va="center", color=INK)
        draw_cycle(ax, items, pos[key])
    rows = [("Initial\nsetting", [("shape", S[s0], HEX[C[c0]])], START_Y)]
    if with_output:
        rows.append(("Output", [("shape", S[s], HEX[C[c]]) for s, c in outputs], OUT_Y))
    for label, items, y in rows:
        ax.text(40, y, label, fontsize=_pt(28), fontweight="bold", va="center", color=INK)
        for i, item in enumerate(items):
            _draw_item(ax, item, X3 + i * STEP, y)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--min-shapes", type=int, default=3)
    ap.add_argument("--max-shapes", type=int, default=5)
    ap.add_argument("--min-cycle", type=int, default=CYCLE_LEN[0])
    ap.add_argument("--max-cycle", type=int, default=CYCLE_LEN[1])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("task3"))
    ap.add_argument("--forward", action="store_true")
    ap.add_argument("--max-b2-total", type=int, default=None)
    ap.add_argument("--max-b3-total", type=int, default=None)
    args = ap.parse_args()
    if (args.max_b2_total is None) != (args.max_b3_total is None):
        ap.error("--max-b2-total과 --max-b3-total은 함께 지정해야 합니다.")
    if not 1 <= args.min_shapes <= args.max_shapes:
        ap.error("--min-shapes는 1 이상, --max-shapes 이하여야 합니다.")
    if not 2 <= args.min_cycle <= args.max_cycle <= len(SHAPES):
        ap.error(f"--min-cycle은 2 이상, --max-cycle은 {len(SHAPES)} 이하여야 합니다.")

    rng = random.Random(args.seed)
    (args.out / "images").mkdir(parents=True, exist_ok=True)
    with open(args.out / "samples.jsonl", "w", encoding="utf-8") as fh:
        for i in range(args.n):
            S = rng.sample(SHAPES, rng.randint(args.min_cycle, args.max_cycle))
            C = rng.sample(COLORS, rng.randint(args.min_cycle, args.max_cycle))
            rule = {"shapes": S, "colors": C, "start": (rng.randrange(len(S)), rng.randrange(len(C)))}
            n_out = rng.randint(args.min_shapes, args.max_shapes)
            if args.max_b2_total is None:
                outputs = [(rng.randrange(len(S)), rng.randrange(len(C))) for _ in range(n_out)]
            else:
                outputs = capped_outputs(rng, rule, n_out, args.max_b2_total, args.max_b3_total)
            counts = solve(rule, outputs)
            assert args.max_b2_total is None or (counts[1] <= args.max_b2_total and counts[2] <= args.max_b3_total)
            rid = f"{i:05d}"
            img = f"images/{rid}.png"
            render(rule, outputs, args.out / img, with_output=not args.forward)
            label = {"answer": fmt_counts(counts), "counts": dict(zip(("B1", "B2", "B3"), counts)),
                     "naive_answer": fmt_counts(naive_solve(rule, outputs)),
                     "shapes": S, "colors": C, "start": f"{S[rule['start'][0]]}:{C[rule['start'][1]]}",
                     "outputs": [f"{S[s]}:{C[c]}" for s, c in outputs]}
            if args.forward:
                label = {"answer": fmt_outputs(rule, outputs), "presses": press_sequence(rule, outputs),
                         **{k: v for k, v in label.items() if k not in ("answer", "naive_answer")}}
            rec = {"id": rid, "task": "create" if args.forward else "count",
                   "input": {"text": prompt(rule, outputs, False, args.forward),
                             "multimodal": {"image": img, "text": prompt(rule, outputs, True, args.forward)}},
                   "label": label}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"문항 {args.n}개 생성 완료: {args.out / 'samples.jsonl'}")


if __name__ == "__main__":
    main()
