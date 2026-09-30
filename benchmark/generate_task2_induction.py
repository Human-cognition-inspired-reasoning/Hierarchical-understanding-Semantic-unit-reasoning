import argparse
import itertools
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt

from generate_button_tasks import (BG, COLORS, CYCLE_LEN, CYCLE_X, DPI, HEX, INK, NEUTRAL, ROLES, SHAPES, STEP,
                                   _draw_item, _pt, cycle_points, cycle_radius, draw_cycle, run)

EFFECTS = [None] + [(a, st) for a in (0, 1) for st in (1, -1, 2, -2)]
HYPOTHESES = list(itertools.product(EFFECTS, EFFECTS, ["swap", "reverse", "both", "none"]))
X_LABEL, X_START, ROW_H = 40, 250, 150


def h_press(state, b, h, sizes):
    s, c, sw, rv = state
    if b == 3:
        if h[2] == "swap":
            return s, c, not sw, rv
        if h[2] == "reverse":
            return s, c, sw, not rv
        if h[2] == "both":
            return h_press(h_press(state, 1, h, sizes), 2, h, sizes)
        return state
    e = h[0] if (b == 1) != sw else h[1]
    if e is None:
        return state
    xs = [s, c]
    xs[e[0]] = (xs[e[0]] + (-e[1] if rv else e[1])) % sizes[e[0]]
    return xs[0], xs[1], sw, rv


def h_run(h, sizes, start, buttons):
    state = (*start, False, False)
    for b in buttons:
        state = h_press(state, b, h, sizes)
    return state[:2]


def fmt(rule, s):
    return f"{rule['shapes'][s[0]]}-{rule['colors'][s[1]]}"


def seq(buttons, blank=None):
    return ", ".join("?" if i == blank else f"B{b}" for i, b in enumerate(buttons))


def prompt(rule, examples, q, task, multimodal):
    def cycle(xs):
        return " → ".join(xs + xs[:1])
    lines = ["There are three buttons, B1, B2, and B3. What each button does is not explained.", ""]
    if multimodal:
        lines.append("* The state is a shape and a color. The shapes and the colors are each arranged in the cycles "
                     "shown in the first image.")
    else:
        lines.append(f"* The state is a shape and a color. The shapes are arranged in the cycle "
                     f"{cycle(rule['shapes'])}, and the colors are arranged in the cycle {cycle(rule['colors'])}.")
    lines += ["* Each case starts from an initial setting, and the buttons are pressed in the order shown. "
              "The result is one shape with the resulting shape and color.",
              "* The buttons work the same way in every case below."]
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


def render_cycles(rule, path, ccw=(False, False)):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--min-presses", type=int, default=3)
    ap.add_argument("--max-presses", type=int, default=5)
    ap.add_argument("--n-examples", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("task2_induction"))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    (args.out / "images").mkdir(parents=True, exist_ok=True)
    with open(args.out / "samples.jsonl", "w", encoding="utf-8") as fh:
        for sid in range(args.n):
            rule = {"shapes": rng.sample(SHAPES, rng.randint(*CYCLE_LEN)),
                    "colors": rng.sample(COLORS, rng.randint(*CYCLE_LEN)), "role": rng.choice(list(ROLES))}
            sizes = (len(rule["shapes"]), len(rule["colors"]))

            def draw():
                st = (rng.randrange(sizes[0]), rng.randrange(sizes[1]))
                bs = [rng.randint(1, 3) for _ in range(rng.randint(args.min_presses, args.max_presses))]
                return st, bs, run(rule, st, bs)

            tries = 0
            while True:
                tries += 1
                examples = [draw() for _ in range(args.n_examples)]
                fq, iq = draw(), draw()
                k = rng.randrange(len(iq[1]))
                cons = [h for h in HYPOTHESES if all(h_run(h, sizes, st, bs) == e for st, bs, e in examples)]
                fwd = {h_run(h, sizes, fq[0], fq[1]) for h in cons}
                inv = {tuple(b for b in (1, 2, 3) if h_run(h, sizes, iq[0], iq[1][:k] + [b] + iq[1][k + 1:]) == iq[2])
                       for h in cons}
                if fwd == {fq[2]} and inv == {(iq[1][k],)}:
                    break
            cyc_img = f"images/{sid:05d}_cycles.png"
            render_cycles(rule, args.out / cyc_img)
            ex_label = [{"start": fmt(rule, st), "buttons": bs, "end": fmt(rule, e)} for st, bs, e in examples]
            for task, (st, bs, end) in (("forward", fq), ("inverse", iq)):
                q = (st, bs, end, k)
                rid = f"{sid:05d}_{task}"
                img = f"images/{rid}.png"
                render_rows(rule, examples, q, task, args.out / img)
                label = {"rule": rule, "examples": ex_label, "start": fmt(rule, st), "buttons": bs,
                         "end": fmt(rule, end), "consistent_hypotheses": len(cons), "tries": tries}
                if task == "forward":
                    label.update(answer=f"{rule['shapes'][end[0]]}:{rule['colors'][end[1]]}",
                                 naive_answer=":".join(fmt(rule, run(rule, st, bs, ignore_b3=True)).split("-")))
                else:
                    label.update(answer=f"button {bs[k]}", blank_index=k,
                                 naive_candidates=[b for b in (1, 2, 3)
                                                   if run(rule, st, bs[:k] + [b] + bs[k + 1:], True) == end])
                rec = {"id": rid, "scenario_id": sid, "task": task,
                       "input": {"text": prompt(rule, examples, q, task, False),
                                 "multimodal": {"image": [cyc_img, img], "text": prompt(rule, examples, q, task, True)}},
                       "label": label}
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"{sid:05d}: b3={rule['role']}, 순환 {sizes[0]}/{sizes[1]}, 시도 {tries}회, 남은 가설 {len(cons)}개",
                  flush=True)
    print(f"시나리오 {args.n}개, 문항 {2 * args.n}개 생성 완료: {args.out / 'samples.jsonl'}")


if __name__ == "__main__":
    main()
