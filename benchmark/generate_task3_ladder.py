import argparse
import json
import random
import re
from pathlib import Path

import matplotlib.pyplot as plt

from generate_button_tasks import BG, COLORS, DPI, HEX, INK, NEUTRAL, SHAPES, STEP, _draw_item, _pt, draw_cycle
from generate_task3 import (CYCLE_CX, LEGEND, OUT_Y, START_Y, X3, fmt_counts, fmt_outputs, layout, naive_solve,
                            prompt, solve)

LEVELS = ("L1", "L2", "L3")
CASE_H = 380
CYC_OFF = START_Y + 40
MM_L12 = [("in the B2 cycle shown in the image", "in the B2 cycle shown in the first image"),
          ("in the B3 cycle shown in the image", "in the B3 cycle shown in the first image"),
          ("Initial setting row of the image.", "Initial setting row of the second image (Question)."),
          ("are shown in row Output of the image.", "are shown in the Output row of the second image (Question).")]
MIN_LINE = "* Every button was pressed the minimum number of times."
HINT = ("* Hint: B1 equals the number of created shapes. B2 equals the number of times the shape differs from "
        "the one before it, and B3 equals the number of times the color differs from the one before it. "
        "The first created shape is compared with the initial setting.")
FUNCS = ("create", "shape", "color")
FEATURE = {"create": "count", "shape": "shape changes", "color": "color changes"}


def rename(line, assign):
    m = {f"B{i}": f"B{assign.index(f) + 1}" for i, f in enumerate(FUNCS, 1)}
    return re.sub(r"\bB[123]\b", lambda x: m[x.group(0)], line)


def relabel(text, assign):
    lines = [l if l.startswith("Question:") else rename(l, assign) for l in text.split("\n")]
    idx = [i for i, l in enumerate(lines) if re.match(r"\* B[123] — ", l)]
    for i, l in zip(idx, sorted(lines[i] for i in idx)):
        lines[i] = l
    return "\n".join(lines)


def walk(rng, rule, start, n_out, max_b2, max_b3):
    k2 = set(rng.sample(range(n_out), rng.randint(0, min(max_b2, n_out))))
    k3 = set(rng.sample(range(n_out), rng.randint(0, min(max_b3, n_out))))
    s, c = start
    outputs = []
    for i in range(n_out):
        s, c = (s + (i in k2)) % len(rule["shapes"]), (c + (i in k3)) % len(rule["colors"])
        outputs.append((s, c))
    return outputs


def features(start, outputs):
    seq = [tuple(start)] + list(outputs)
    sh, co = [s for s, _ in seq], [c for _, c in seq]

    def ch(xs):
        return sum(a != b for a, b in zip(xs, xs[1:]))

    return {"count": len(outputs), "count-1": len(outputs) - 1, "distinct pairs": len(set(outputs)),
            "shape changes": ch(sh), "shape changes after first": ch(sh[1:]),
            "distinct shapes": len(set(sh[1:])), "distinct shapes-1": len(set(sh[1:])) - 1,
            "distinct shapes with initial-1": len(set(sh)) - 1,
            "color changes": ch(co), "color changes after first": ch(co[1:]),
            "distinct colors": len(set(co[1:])), "distinct colors-1": len(set(co[1:])) - 1,
            "distinct colors with initial-1": len(set(co)) - 1,
            "any changes": ch(seq), "any changes after first": ch(seq[1:]),
            "both change": sum(a[0] != b[0] and a[1] != b[1] for a, b in zip(seq, seq[1:]))}


def identifies(examples, assign):
    F = [features(st, outs) for st, outs, _ in examples]
    for i, true in enumerate(FEATURE[f] for f in assign):
        if [k for k in F[0] if all(f[k] == c[i] for f, (_, _, c) in zip(F, examples))] != [true]:
            return False
    return True


def l3_prompt(rule, examples, query, multimodal, describe=False):
    S, C = rule["shapes"], rule["colors"]
    head = "There are three buttons, B1, B2, and B3." + ("" if describe else " What each button does is not explained.")
    desc = [f"* The three buttons have three different functions: one creates a shape with the current shape and "
            f"color, one changes the current shape to the next one in a fixed cyclic order of {len(S)} shapes, and "
            f"one changes the current color to the next one in a fixed cyclic order of {len(C)} colors. Which "
            "function is assigned to which button, and the cyclic orders, are not given."] if describe else []
    lines = [head, "", *desc,
             "* Before any button is pressed, a shape and a color are set (the initial setting). "
             "No shape has been placed yet.",
             "* Pressing the buttons creates shapes, which are placed from left to right in order.",
             "* The buttons work the same way in every case below. "
             "In every case, each button was pressed the minimum number of times."]
    names = [f"Example {k}" for k in range(1, len(examples) + 1)]
    if multimodal:
        lines += [f"* The images show {', '.join(names)}, and Question, in that order. Each image has an "
                  "Initial setting row and an Output row. The Output row shows the created shapes, "
                  "from left to right.", "", "Button presses in the examples:"]
        lines += [f"{n}: {fmt_counts(c)}" for n, (_, _, c) in zip(names, examples)]
        return "\n".join(lines + ["", "Question: For the case in the Question image, how many times were B1, B2, and B3 each pressed?"])
    for n, (st, outs, c) in zip(names + ["New case"], examples + [query + (None,)]):
        lines += ["", f"{n}:", f"The initial setting is {S[st[0]]}-{C[st[1]]}.",
                  "The created shapes, from left to right, were: " + fmt_outputs(rule, outs)]
        if c:
            lines.append(f"Button presses: {fmt_counts(c)}")
    return "\n".join(lines + ["Question: How many times were B1, B2, and B3 each pressed?"])


def cycle_layout(rule):
    pos, title_y, height = layout(rule)
    return {k: v - (0, CYC_OFF) for k, v in pos.items()}, title_y - CYC_OFF, height - CYC_OFF


def render_cycles(rule, path, assign):
    pos, title_y, height = cycle_layout(rule)
    fig = plt.figure(figsize=((1300 + 1e-6) / DPI, (height + 1e-6) / DPI), dpi=DPI, facecolor=BG)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1300)
    ax.set_ylim(0, height)
    ax.axis("off")
    for i, line in enumerate(LEGEND[:1] + sorted(rename(l, assign) for l in LEGEND[1:])):
        ax.text(40, height - 35 - 30 * i, line, fontsize=_pt(19), va="center", color=INK)
    for key, items in (("B2", [("shape", s, NEUTRAL) for s in rule["shapes"]]),
                       ("B3", [("chip", c) for c in rule["colors"]])):
        ax.text(CYCLE_CX[key], title_y, rename(key, assign), fontsize=_pt(28), fontweight="bold", ha="center",
                va="center",
                color=INK)
        draw_cycle(ax, items, pos[key])
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def render_case(rule, title, start, outputs, path):
    S, C = rule["shapes"], rule["colors"]
    width = max(1300, X3 + (len(outputs) - 1) * STEP + 110)
    fig = plt.figure(figsize=((width + 1e-6) / DPI, (CASE_H + 1e-6) / DPI), dpi=DPI, facecolor=BG)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, width)
    ax.set_ylim(0, CASE_H)
    ax.axis("off")
    ax.text(40, CASE_H - 40, title, fontsize=_pt(28), fontweight="bold", va="center", color=INK)
    for label, items, y in (("Initial\nsetting", [start], START_Y), ("Output", outputs, OUT_Y)):
        ax.text(40, y, label, fontsize=_pt(28), fontweight="bold", va="center", color=INK)
        for k, (s, c) in enumerate(items):
            _draw_item(ax, ("shape", S[s], HEX[C[c]]), X3 + k * STEP, y)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--min-out", type=int, default=4)
    ap.add_argument("--max-out", type=int, default=10)
    ap.add_argument("--min-cycle", type=int, default=3)
    ap.add_argument("--max-cycle", type=int, default=6)
    ap.add_argument("--max-b2-total", type=int, default=6)
    ap.add_argument("--max-b3-total", type=int, default=6)
    ap.add_argument("--n-examples", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("task3_ladder"))
    ap.add_argument("--random-roles", action="store_true")
    ap.add_argument("--describe", action="store_true")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    arng = random.Random(f"roles-{args.seed}")
    (args.out / "images").mkdir(parents=True, exist_ok=True)
    with open(args.out / "samples.jsonl", "w", encoding="utf-8") as fh:
        for i in range(args.n):
            S = rng.sample(SHAPES, rng.randint(args.min_cycle, args.max_cycle))
            C = rng.sample(COLORS, rng.randint(args.min_cycle, args.max_cycle))
            start = (rng.randrange(len(S)), rng.randrange(len(C)))
            rule = {"shapes": S, "colors": C, "start": start}
            assign = arng.sample(FUNCS, 3) if args.random_roles else list(FUNCS)

            def by_button(t):
                return tuple(dict(zip(FUNCS, t))[f] for f in assign)

            outputs = walk(rng, rule, start, rng.randint(args.min_out, args.max_out), args.max_b2_total,
                           args.max_b3_total)
            counts = by_button(solve(rule, outputs))
            while True:
                examples = []
                for _ in range(args.n_examples):
                    st = (rng.randrange(len(S)), rng.randrange(len(C)))
                    outs = walk(rng, rule, st, rng.randint(args.min_out, args.max_out), args.max_b2_total,
                                args.max_b3_total)
                    examples.append((st, outs, by_button(solve({**rule, "start": st}, outs))))
                if identifies(examples, assign) and all((st, o) != (start, outputs) for st, o, _ in examples):
                    break
            rid = f"{i:05d}"
            render_cycles(rule, args.out / f"images/{rid}_cycles.png", assign)
            cases = [(f"Example {k}", st, o, f"images/{rid}_ex{k}.png") for k, (st, o, _) in enumerate(examples, 1)]
            cases.append(("Question", start, outputs, f"images/{rid}_q.png"))
            for title, st, o, img in cases:
                render_case(rule, title, st, o, args.out / img)
            label = {"answer": fmt_counts(counts), "counts": dict(zip(("B1", "B2", "B3"), counts)),
                     "naive_answer": fmt_counts(by_button(naive_solve(rule, outputs))),
                     "shapes": S, "colors": C, "start": f"{S[start[0]]}:{C[start[1]]}",
                     "outputs": [f"{S[s]}:{C[c]}" for s, c in outputs],
                     "examples": [{"start": f"{S[st[0]]}:{C[st[1]]}", "outputs": [f"{S[s]}:{C[c]}" for s, c in o],
                                   "answer": fmt_counts(c)} for st, o, c in examples]}
            if args.random_roles:
                label["assign"] = assign
            texts = {"L2": (prompt(rule, outputs, False), prompt(rule, outputs, True))}
            mm = texts["L2"][1]
            for a, b in MM_L12:
                assert mm.count(a) == 1, a
                mm = mm.replace(a, b)
            texts["L2"] = (texts["L2"][0], mm)
            texts["L1"] = tuple(t.replace(MIN_LINE, MIN_LINE + "\n" + HINT) for t in texts["L2"])
            for lv in ("L1", "L2"):
                texts[lv] = tuple(relabel(t, assign) for t in texts[lv])
            texts["L3"] = (l3_prompt(rule, examples, (start, outputs), False, args.describe),
                           l3_prompt(rule, examples, (start, outputs), True, args.describe))
            for lv in LEVELS:
                img = [c[3] for c in cases] if lv == "L3" else [f"images/{rid}_cycles.png", cases[-1][3]]
                rec = {"id": f"{lv}-{rid}", "item": rid, "level": lv, "task": "count",
                       "input": {"text": texts[lv][0], "multimodal": {"image": img, "text": texts[lv][1]}},
                       "label": label}
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"문항 {args.n}개 × 3단계 생성 완료: {args.out / 'samples.jsonl'}")


if __name__ == "__main__":
    main()
