import argparse
import itertools
import json
import random
from pathlib import Path

from generate_button_tasks import COLORS, CYCLE_LEN, SHAPES
from generate_task2_induction import fmt, prompt as base_prompt, render_cycles, render_rows

ASSIGNS = list(itertools.permutations(("shape", "color", "swap")))
MIN_ASSIGNS = [a[:i] + ("swap",) + a[i:] for i in range(3)
               for a in itertools.product(("shape", "color", "none"), repeat=2)]
MIN_TRUE = [a for a in MIN_ASSIGNS if a.count("none") < 2]
FUNC_LINE = ("* The three buttons have three different functions: one changes the shape by one step along the shape "
             "cycle, one changes the color by one step along the color cycle, and one swaps the roles of the other "
             "two buttons (pressing it again swaps them back). Which function is assigned to which button is not "
             "given.")
FUNC_HIDDEN = ("* There are {n} kinds of shapes and {m} kinds of colors. The three buttons have three different "
               "functions: one moves the current shape one step forward in a fixed cyclic order of the {n} shapes, "
               "one moves the current color one step forward in a fixed cyclic order of the {m} colors, and one "
               "makes the other two buttons exchange their functions (pressing it again exchanges them back). "
               "Which function is assigned to which button, and the cyclic orders, are not given.")
FUNC_REVEAL = (FUNC_HIDDEN.replace("in a fixed cyclic order of the {n} shapes", "along the shape cycle")
               .replace("in a fixed cyclic order of the {m} colors", "along the color cycle")
               .replace("Which function is assigned to which button, and the cyclic orders, are not given.",
                        "Which function is assigned to which button is not given."))
FUNC_MINIMAL = ("* One of the three buttons makes the other two buttons exchange their functions (pressing it again "
                "exchanges them back).")
START_LINE = ("* Each case starts from an initial setting, and the buttons are pressed in the order shown. "
              "The result is one shape with the resulting shape and color.")
START_HIDDEN = ("* Each case starts from an initial setting with every button in its original function, and the "
                "buttons are pressed in the order shown. The result is one shape with the resulting shape and "
                "color.")


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


def prompt(rule, examples, q, task, multimodal, hide=False, reveal=False, minimal=False):
    lines = base_prompt(rule, examples, q, task, multimodal).split("\n")
    assert lines[0].endswith(" What each button does is not explained.") and lines[2].startswith("* The state is")
    lines[0] = "There are three buttons, B1, B2, and B3."
    if minimal:
        lines.insert(3, FUNC_MINIMAL)
        assert lines[4] == START_LINE
        lines[4] = START_HIDDEN
        return "\n".join(lines)
    if not hide:
        lines.insert(3, FUNC_LINE)
        return "\n".join(lines)
    if not reveal:
        lines[2] = "* The state is a shape and a color."
    lines.insert(3, (FUNC_REVEAL if reveal else FUNC_HIDDEN).format(n=len(rule["shapes"]), m=len(rule["colors"])))
    assert lines[4] == START_LINE
    lines[4] = START_HIDDEN
    text = "\n".join(lines)
    if multimodal and not reveal:
        assert text.count("* The second image shows ") == 1
        text = text.replace("* The second image shows ", "* The image shows ")
    return text


def steps(assign, buttons):
    return a_run(assign, (10 ** 9, 10 ** 9), (0, 0), buttons)


def orders(palette, observed, n):
    extra = [x for x in palette if x not in observed]
    for add in itertools.combinations(extra, n - len(observed)):
        first, *rest = sorted(observed) + list(add)
        for perm in itertools.permutations(rest):
            yield (first, *perm)


def hidden_ok(rule, examples, fq, iq, k):
    S, C = rule["shapes"], rule["colors"]
    cases = [(S[a[0]], C[a[1]], bs, S[e[0]], C[e[1]]) for a, bs, e in examples]
    obs = [{x for c in cases for x in (c[0], c[3])} | {S[fq[0][0]], S[iq[0][0]], S[iq[2][0]]},
           {x for c in cases for x in (c[1], c[4])} | {C[fq[0][1]], C[iq[0][1]], C[iq[2][1]]}]
    if len(obs[0]) > len(S) or len(obs[1]) > len(C):
        return False
    seqs = {b: iq[1][:k] + [b] + iq[1][k + 1:] for b in (1, 2, 3)}
    fwd_true = (S[fq[2][0]], C[fq[2][1]])
    for a in ASSIGNS:
        st = [steps(a, bs) for _, _, bs, _, _ in cases]
        good = []
        for j, (pal, n) in enumerate(((SHAPES, len(S)), (COLORS, len(C)))):
            good.append([o for o in orders(pal, obs[j], n)
                         if all(o[(o.index(c[j]) + t[j]) % n] == c[3 + j] for c, t in zip(cases, st))])
        if not good[0] or not good[1]:
            continue
        tf, ti = steps(a, fq[1]), {b: steps(a, sq) for b, sq in seqs.items()}
        start_f = (S[fq[0][0]], C[fq[0][1]])
        start_i, end_i = (S[iq[0][0]], C[iq[0][1]]), (S[iq[2][0]], C[iq[2][1]])
        fit = []
        for j in (0, 1):
            for o in good[j]:
                n = len(o)
                if o[(o.index(start_f[j]) + tf[j]) % n] != fwd_true[j]:
                    return False
                fit.append({b for b in (1, 2, 3) if o[(o.index(start_i[j]) + ti[b][j]) % n] == end_i[j]})
        shape_fits, color_fits = fit[:len(good[0])], fit[len(good[0]):]
        if any(fs & fc != {iq[1][k]} for fs in shape_fits for fc in color_fits):
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--min-presses", type=int, default=3)
    ap.add_argument("--max-presses", type=int, default=5)
    ap.add_argument("--n-examples", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("task2_assign"))
    ap.add_argument("--hide-cycles", action="store_true")
    ap.add_argument("--reveal-cycles", action="store_true")
    ap.add_argument("--random-arrows", action="store_true")
    ap.add_argument("--minimal-desc", action="store_true")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    arng = random.Random(f"arrows-{args.seed}")
    (args.out / "images").mkdir(parents=True, exist_ok=True)
    with open(args.out / "samples.jsonl", "w", encoding="utf-8") as fh:
        for sid in range(args.n):
            space = MIN_ASSIGNS if args.minimal_desc else ASSIGNS
            assign = rng.choice(MIN_TRUE if args.minimal_desc else ASSIGNS)
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
                cons = [a for a in space if all(a_run(a, sizes, st, bs) == e for st, bs, e in examples)]
                fwd = {a_run(a, sizes, fq[0], fq[1]) for a in cons}
                inv = {tuple(b for b in (1, 2, 3) if a_run(a, sizes, iq[0], iq[1][:k] + [b] + iq[1][k + 1:]) == iq[2])
                       for a in cons}
                if fwd == {fq[2]} and inv == {(iq[1][k],)} and (
                        not args.hide_cycles or hidden_ok(rule, examples, fq, iq, k)):
                    break
            cyc_img = f"images/{sid:05d}_cycles.png"
            show = not args.hide_cycles or args.reveal_cycles
            v3_text = args.hide_cycles or args.reveal_cycles
            ccw = (arng.random() < 0.5, arng.random() < 0.5) if args.random_arrows else (False, False)
            if show:
                render_cycles(rule, args.out / cyc_img, ccw)
            ex_label = [{"start": fmt(rule, st), "buttons": bs, "end": fmt(rule, e)} for st, bs, e in examples]
            for task, (st, bs, end) in (("forward", fq), ("inverse", iq)):
                q = (st, bs, end, k)
                rid = f"{sid:05d}_{task}"
                img = f"images/{rid}.png"
                render_rows(rule, examples, q, task, args.out / img)
                label = {"rule": rule, "examples": ex_label, "start": fmt(rule, st), "buttons": bs,
                         "end": fmt(rule, end), "consistent_assignments": len(cons), "tries": tries}
                if args.random_arrows:
                    label["ccw"] = list(ccw)
                if task == "forward":
                    naive = a_run(assign, sizes, st, bs, skip_swap=True)
                    label.update(answer=f"{rule['shapes'][end[0]]}:{rule['colors'][end[1]]}",
                                 naive_answer=f"{rule['shapes'][naive[0]]}:{rule['colors'][naive[1]]}")
                else:
                    label.update(answer=f"button {bs[k]}", blank_index=k,
                                 naive_candidates=[b for b in (1, 2, 3) if a_run(
                                     assign, sizes, st, bs[:k] + [b] + bs[k + 1:], skip_swap=True) == end])
                mm_text = prompt(rule, examples, q, task, True, v3_text, args.reveal_cycles, args.minimal_desc)
                if args.random_arrows and not args.minimal_desc:
                    for cyc in ("shape", "color"):
                        a = f"along the {cyc} cycle,"
                        assert mm_text.count(a) == 1, a
                        mm_text = mm_text.replace(a, f"along the {cyc} cycle (in the direction of the arrows),")
                rec = {"id": rid, "scenario_id": sid, "task": task,
                       "input": {"text": prompt(rule, examples, q, task, False, v3_text, args.reveal_cycles, args.minimal_desc),
                                 "multimodal": {"image": [cyc_img, img] if show else [img],
                                                "text": mm_text}},
                       "label": label}
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"{sid:05d}: 배정 {'/'.join(assign)}, 순환 {sizes[0]}/{sizes[1]}, 시도 {tries}회, "
                  f"남은 배정 {len(cons)}개", flush=True)
    print(f"시나리오 {args.n}개, 문항 {2 * args.n}개 생성 완료: {args.out / 'samples.jsonl'}")


if __name__ == "__main__":
    main()
