import argparse
import collections
import csv
import json
import random
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle
from PIL import Image

from grid_common import canvas, cell, disc, is_tile, save, text, write

SV = Path("E:/benchmarks/survey_v1")
FLUENT = Path("E:/fluent/png")
N = 6
EX_ICON, Q_ICON = 56, 72
EX_CELL, Q_CELL = round(EX_ICON / 0.76), round(Q_ICON / 0.76)
SMALL_SIDE, SMALL_AREA = 0.8, 0.3
N_OBJ, MAJ = 5, 3
SIZE = {1: (6, 5), 2: (7, 5), 3: (8, 5)}
WALLS = {6: {"short": (1, 2, 2, 3), "maze": (3, 4, 2, 4)},
         7: {"short": (2, 3, 2, 3), "maze": (0, 0, 2, 5)},
         8: {"short": (2, 4, 2, 4), "maze": (0, 0, 2, 5)}}
MAZE_DENSITY = {7: 0.30, 8: 0.35}
T_MIN, F_MAX, STRONG = 8, 1, 9
EMOTION = ["기쁨", "슬픔", "분노"]
UNCLEAR_AFFECT = ["놀람", "공포", "징그러움"]
COLORS = ["빨강", "검정", "흰색", "초록", "파랑"]
LEVEL3 = ["긍정", "부정", "인공물"]
FRAME_GRID = ("* In every room, the agent walks to one of the objects, and the object is chosen by the same rule. The "
              "rule depends only on what the objects are and where they are in the room.")
FACE = set()
GLYPH = {k: v["glyph"] for k, v in json.load(open("E:/fluent_meta.json", encoding="utf-8")).items()}


def configure(n, n_obj):
    global N, N_OBJ, MAJ
    N, N_OBJ, MAJ = n, n_obj, n_obj - 2


def parse(raw, attrs):
    got = []
    for l in raw.splitlines():
        t = re.sub(r"^\s*(?:[-*•·]|\d+[.)])\s*", "", l).strip().strip("*").strip()
        if t not in attrs:
            t = re.sub(r"\s*\(.*?\)\s*$", "", t).strip()
        if t in attrs:
            got.append(t)
    return list(dict.fromkeys(got))[:5]


def load(pool_dir=None):
    attrs = [r["속성"] for r in csv.DictReader(open(SV / "attributes_48.csv", encoding="utf-8-sig"))]
    votes = collections.defaultdict(collections.Counter)
    for m in ("gpt-6-sol", "claude-sonnet-5-5"):
        for l in open(SV / f"features48_{m}_seq_raw.jsonl", encoding="utf-8"):
            c = json.loads(l)
            votes[c["emoji"]].update(parse(c["raw"], attrs))
    if pool_dir:
        base = {p.stem for p in (Path(pool_dir) / "used").glob("*.png")}
        emo = {p.stem for p in (Path(pool_dir) / "emotion").glob("*.png")}
    else:
        base = set(json.load(open(SV / "pool_kappa085.json", encoding="utf-8")))
        emo = {u for u in votes if any(votes[u][a] >= STRONG for a in EMOTION)}
        small = {u for u in base | emo if is_small(u)}
        base, emo = base - small, emo - small
    F = {u: {a: "T" if votes[u][a] >= T_MIN else "F" if votes[u][a] <= F_MAX else "U" for a in attrs} for u in base | emo}
    meta = json.load(open("E:/fluent_meta.json", encoding="utf-8"))
    FACE.update(u for u in base | emo if meta[u]["group"] == "Smileys & Emotion")
    for u in base | emo:
        if meta[u]["group"] == "Symbols" and is_tile(u) and not u.endswith(" square"):
            F[u].update({c: "U" for c in COLORS})
    return attrs, F, (lambda u, a: votes[u][a] >= STRONG and F[u][a] == "T"), sorted(base), sorted(base | emo), len(emo - base)


def is_small(u):
    m = np.asarray(Image.open(FLUENT / f"{u}.png").convert("RGBA"))[..., 3] > 128
    ys, xs = np.nonzero(m)
    return max(ys.ptp() + 1, xs.ptp() + 1) / max(m.shape) < SMALL_SIDE or m.mean() < SMALL_AREA


def bfs(start, blocked):
    dist, frontier = {start: 0}, [start]
    while frontier:
        nxt = []
        for r, c in frontier:
            for q in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if 1 <= q[0] <= N and 1 <= q[1] <= N and q not in blocked and q not in dist:
                    dist[q] = dist[(r, c)] + 1
                    nxt.append(q)
        frontier = nxt
    return dist


def shortest_path(start, goal, blocked):
    prev, frontier = {start: None}, [start]
    while frontier and goal not in prev:
        nxt = []
        for r, c in frontier:
            for q in ((r, c + 1), (r + 1, c), (r, c - 1), (r - 1, c)):
                if 1 <= q[0] <= N and 1 <= q[1] <= N and q not in prev and (q == goal or q not in blocked):
                    prev[q] = (r, c)
                    nxt.append(q)
        frontier = nxt
    route = [goal]
    while prev[route[-1]] is not None:
        route.append(prev[route[-1]])
    return route[::-1]


class Room:
    def __init__(self, agent, walls, objs):
        self.agent, self.walls, self.objs = agent, walls, objs
        path = bfs(agent, walls | {p for _, p in objs})
        near = [[path[q] for q in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)) if q in path] for _, (r, c) in objs]
        self.dist = {"path": [min(d) + 1 if d else None for d in near],
                     "manhattan": [abs(p[0] - agent[0]) + abs(p[1] - agent[1]) for _, p in objs]}


def majority(names, F, attrs):
    """The unique feature splitting off at least MAJ objects, else None."""
    splits = []
    for g in attrs:
        t = {n for n in names if F[n][g] == "T"}
        tu = {n for n in names if F[n][g] != "F"}
        if len(t) == len(names) or len(tu) < MAJ:
            continue
        if len(t) >= MAJ and t == tu:
            splits.append(frozenset(t))
        else:
            return None
    return next(iter(splits)) if len(set(splits)) == 1 else None


def unique_best(cands, key):
    ks = sorted(key(i) for i in cands)
    return None if not ks or (len(ks) > 1 and ks[0] == ks[1]) else min(cands, key=key)


def odd_pick(room, F, attrs, S="near", D="path"):
    names = [o[0] for o in room.objs]
    maj = majority(names, F, attrs)
    if maj is None:
        return None
    cands = [i for i, n in enumerate(names) if n not in maj and room.dist[D][i] is not None]
    return unique_best(cands, lambda i: room.dist[D][i] if S == "near" else -room.dist[D][i])


def feat_pick(room, F, g, want, S, D):
    names = [o[0] for o in room.objs]
    cands = [i for i, n in enumerate(names) if room.dist[D][i] is not None
             and (g is None or F[n][g] != ("F" if want == "T" else "T"))]
    sure = [i for i in cands if g is None or F[names[i]][g] == want]
    best = unique_best(sure, lambda i: room.dist[D][i] if S == "near" else -room.dist[D][i])
    if best is None:
        return None
    k = room.dist[D][best] if S == "near" else -room.dist[D][best]
    doubt = [i for i in cands if i not in sure and (room.dist[D][i] if S == "near" else -room.dist[D][i]) <= k]
    return None if doubt else best


def rivals(attrs):
    hs = [("odd", S, D) for S in ("near", "far") for D in ("path", "manhattan")]
    hs += [("feat", g, w, S, D) for g in [None] + attrs for w in ("T", "F") for S in ("near", "far")
           for D in ("path", "manhattan") if not (g is None and w == "F")]
    return hs + [("alpha", "first"), ("alpha", "last")]


def apply(h, room, F, attrs):
    if h[0] == "odd":
        return odd_pick(room, F, attrs, h[1], h[2])
    if h[0] == "feat":
        return feat_pick(room, F, *h[1:])
    names = [o[0] for o in room.objs]
    return (min if h[1] == "first" else max)(range(len(names)), key=lambda i: names[i])


def wall_target(kind):
    return round(MAZE_DENSITY[N] * N * N) if kind == "maze" and N in MAZE_DENSITY else 0


def fenced(rng, walls, fence, kind):
    """Walls plus fence, trimmed to the density target."""
    target = wall_target(kind)
    if not target:
        return walls | fence
    return fence | set(rng.sample(sorted(walls - fence), max(0, target - len(fence))))


def walls_for(rng, kind):
    cells = [(r, c) for r in range(1, N + 1) for c in range(1, N + 1)]
    a, b, lo, hi = WALLS[N][kind]
    target = wall_target(kind)
    nseg = 0 if target else rng.randint(a, b)
    walls = set()
    while nseg > 0 or len(walls) < target:
        nseg -= 1
        r, c = rng.choice(cells)
        horiz = rng.random() < 0.5
        for k in range(rng.randint(lo, hi)):
            q = (r, c + k) if horiz else (r + k, c)
            if 1 <= q[0] <= N and 1 <= q[1] <= N and (not target or len(walls) < target):
                walls.add(q)
    return walls


def kind_of(room, F, attrs, t):
    names = [o[0] for o in room.objs]
    maj = majority(names, F, attrs)
    odd = [i for i, n in enumerate(names) if n not in maj]
    m = min(odd, key=lambda i: room.dist["manhattan"][i])
    reach = [i for i in odd if room.dist["path"][i] is not None]
    m2 = min(reach, key=lambda i: room.dist["manhattan"][i])
    if room.dist["path"][m] is None:
        return "both" if room.dist["path"][t] > room.dist["manhattan"][t] else "enclosed"
    return "wall" if m != t else "basic"


def make_room(rng, f, F, strong, pool, used, want, wall_kind, attrs):
    n_maj = N_OBJ - 1 if want == "single" else MAJ
    n_min = N_OBJ - n_maj
    emo = f in EMOTION
    maj_pool = [n for n in pool if n not in used and strong(n, f) and (not emo or n in FACE)]
    min_pool = [n for n in pool if n not in used and F[n][f] == "F" and (not emo or n in FACE)]
    if len(maj_pool) < n_maj or len(min_pool) < n_min:
        return None
    cells = [(r, c) for r in range(1, N + 1) for c in range(1, N + 1)]
    for _ in range(4000 if emo else 800):
        major = rng.sample(maj_pool, n_maj)
        if emo:
            by_g = {"face": min_pool}
        else:
            by_g = {g: [n for n in min_pool if F[n][g] == "T"] for g in attrs if g != f and all(F[n][g] == "T" for n in major)}
            by_g = {g: v for g, v in by_g.items() if len(v) >= n_min}
        if not by_g:
            continue
        names = major + rng.sample(by_g[rng.choice(sorted(by_g))], n_min)
        if majority(names, F, attrs) != frozenset(names[:n_maj]):
            continue
        walls = walls_for(rng, wall_kind)
        free = [p for p in cells if p not in walls]
        rng.shuffle(free)
        agent, spots = free[0], free[1:1 + len(names)]
        rng.shuffle(names)
        objs = list(zip(names, spots))
        room = Room(agent, walls, objs)
        if want in ("enclosed", "both"):
            mins = [i for i, n in enumerate(names) if F[n][f] == "F"]
            r, c = objs[min(mins, key=lambda i: room.dist["manhattan"][i])][1]
            around = [q for q in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)) if 1 <= q[0] <= N and 1 <= q[1] <= N]
            if agent in around:
                continue
            room = Room(agent, fenced(rng, walls, {q for q in around if q not in spots}, wall_kind), objs)
        t = odd_pick(room, F, attrs)
        if t is None or feat_pick(room, F, None, "T", "near", "path") == t:
            continue
        kind = kind_of(room, F, attrs, t)
        if want in (None, "single") or kind == want:
            return room, t, kind
    return None


def text_map(room, target=None):
    """Character map of a room; * marks the walked path."""
    marks = {w: "#" for w in room.walls}
    marks.update({p: GLYPH[n] for n, p in room.objs})
    if target is not None:
        route = shortest_path(room.agent, room.objs[target][1], room.walls | {o[1] for o in room.objs})
        marks.update({p: "*" for p in route[1:-1]})
    marks[room.agent] = "@"
    rows = ["   " + "  ".join(str(c) for c in range(1, N + 1))]
    rows += [f"{r}  " + "  ".join(marks.get((r, c), ".") for c in range(1, N + 1)) for r in range(1, N + 1)]
    return "\n".join(rows)


def grid_prompt(examples, query, labels, multimodal):
    head = [f"An agent moves in rooms that are {N} × {N} grids of cells. Rows are numbered 1 to {N} from top to bottom, "
            f"and columns 1 to {N} from left to right. The agent moves one cell up, down, left, or right per step and "
            "cannot pass through walls or other objects.", "", FRAME_GRID]
    if multimodal:
        head.append(f"* The image shows Example 1 to Example {len(examples)} at the top and the Question room at the "
                    "bottom. Dark cells are walls, and the black dot is the agent. In each example, the pink line shows "
                    "the path the agent walked and the circle shows the object it reached. In the Question, four of the "
                    "objects are labeled A to D.")
        return "\n".join(head + ["", "Question: In the Question room, which object (A, B, C, or D) will the agent walk "
                                     "to? Answer with its letter."])
    lines = head + ["* In the maps below, @ is the agent, # is a wall, . is an empty cell, and each emoji is an object. "
                    "In the examples, * marks the cells the agent walked through."]
    for j, (room, t) in enumerate(examples, 1):
        name, (r, c) = room.objs[t]
        lines += ["", f"Example {j}:", text_map(room, t), f"The agent walked to the {GLYPH[name]} at row {r}, column {c}."]
    lines += ["", "Question:", text_map(query),
              ", ".join(f"{'ABCD'[k]}: {GLYPH[query.objs[i][0]]}" for k, i in enumerate(labels)),
              "Which object (A, B, C, or D) will the agent walk to? Answer with its letter."]
    return "\n".join(lines)


@lru_cache(None)
def icon(asset, size):
    return np.asarray(Image.open(FLUENT / f"{asset}.png").convert("RGBA").resize((size, size), Image.LANCZOS))


def draw_room(ax, x0, y0, cw, title, room, target=None, labels=None):
    text(ax, x0, y0 + N * cw + 22, title, size=20 if cw < 70 else 24)

    def xy(r, c):
        return x0 + (c - 0.5) * cw, y0 + (N - r + 0.5) * cw

    for r in range(1, N + 1):
        for c in range(1, N + 1):
            cell(ax, x0 + (c - 1) * cw, y0 + (N - r) * cw, cw, "#3B3F45" if (r, c) in room.walls else "white")
    for i, (name, (r, c)) in enumerate(room.objs):
        cx, cy = xy(r, c)
        h = cw * 0.38
        ax.imshow(icon(name, 128), extent=(cx - h, cx + h, cy - h, cy + h), zorder=3)
        if labels and i in labels:
            lx = cx - cw / 2 + 2
            ax.add_patch(Rectangle((lx, cy + cw / 2 - 24), 24, 22, fc="white", ec="#111111", lw=1.5, zorder=5))
            text(ax, lx + 12, cy + cw / 2 - 13, chr(65 + labels.index(i)), size=16, ha="center")
    if target is not None:
        route = [xy(*p) for p in shortest_path(room.agent, room.objs[target][1], room.walls | {o[1] for o in room.objs})]
        ax.plot([p[0] for p in route[:-1]], [p[1] for p in route[:-1]], color="#E0115F", lw=3, alpha=0.85, zorder=7,
                solid_capstyle="round", solid_joinstyle="round")
        ax.add_patch(FancyArrowPatch(route[-2], route[-1], arrowstyle="-|>", mutation_scale=20, lw=3, color="#E0115F",
                                     shrinkA=0, shrinkB=cw * 0.48, zorder=7))
        disc(ax, *route[-1], cw * 0.5, "none", ec="#E0115F", lw=3.5, zorder=7)
    disc(ax, *xy(*room.agent), cw * 0.19, "#111111", ec="white", lw=3, zorder=8)


def render_grid(examples, query, labels, path):
    ew, qw, gap = N * EX_CELL, N * Q_CELL, 40
    width = max(3 * ew + 2 * gap + 60, qw + 80)
    height = ew + qw + 170
    fig, ax = canvas(width, height)
    x0 = (width - (3 * ew + 2 * gap)) / 2
    for j, (room, t) in enumerate(examples):
        draw_room(ax, x0 + j * (ew + gap), height - ew - 45, EX_CELL, f"Example {j + 1}", room, target=t)
    draw_room(ax, (width - qw) / 2, 20, Q_CELL, "Question", query, labels=labels)
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    save(fig, path)


def kind_same(room, F, f, t):
    names = [o[0] for o in room.objs]
    tg = [i for i, n in enumerate(names) if F[n][f] == "T"]
    m = min(tg, key=lambda i: room.dist["manhattan"][i])
    reach = [i for i in tg if room.dist["path"][i] is not None]
    m2 = min(reach, key=lambda i: room.dist["manhattan"][i])
    if room.dist["path"][m] is None:
        return "both" if m2 != t else "enclosed"
    return "wall" if m != t else "basic"


def make_room_same(rng, f, F, strong, pool, used, want, wall_kind):
    n_t = 1 if want == "single" else 3 if want == "both" else 2
    emo = f in EMOTION
    tpool = [n for n in pool if n not in used and strong(n, f) and (not emo or n in FACE)]
    npool = [n for n in pool if n not in used and F[n][f] == "F" and (not emo or n in FACE)]
    if len(tpool) < n_t or len(npool) < N_OBJ - n_t:
        return None
    cells = [(r, c) for r in range(1, N + 1) for c in range(1, N + 1)]
    for _ in range(800):
        names = rng.sample(tpool, n_t) + rng.sample(npool, N_OBJ - n_t)
        walls = walls_for(rng, wall_kind)
        free = [p for p in cells if p not in walls]
        rng.shuffle(free)
        agent, spots = free[0], free[1:1 + len(names)]
        rng.shuffle(names)
        objs = list(zip(names, spots))
        room = Room(agent, walls, objs)
        if want in ("enclosed", "both"):
            tg = [i for i, n in enumerate(names) if F[n][f] == "T"]
            r, c = objs[min(tg, key=lambda i: room.dist["manhattan"][i])][1]
            around = [q for q in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)) if 1 <= q[0] <= N and 1 <= q[1] <= N]
            if agent in around:
                continue
            room = Room(agent, fenced(rng, walls, {q for q in around if q not in spots}, wall_kind), objs)
        t = feat_pick(room, F, f, "T", "near", "path")
        if t is None or feat_pick(room, F, None, "T", "near", "path") == t:
            continue
        kind = kind_same(room, F, f, t)
        if want in (None, "single") or kind == want:
            return room, t, kind
    return None


def dump_room(r):
    return {"agent": list(r.agent), "walls": sorted(map(list, r.walls)), "objects": [[n, list(p)] for n, p in r.objs]}


def same_item(rng, level, F, strong, pool, fam, attrs, H, qp):
    want_q = "basic" if level == 1 else rng.choice(["wall", "enclosed"]) if level == 2 else "both"
    wall_q = "short" if level == 1 else "maze"
    tries = 0
    while True:
        tries += 1
        if tries > 3000:
            raise RuntimeError(f"같은 것 찾기 L{level} 문항을 만들지 못했습니다")
        f = rng.choice(fam)
        wants = ["wall", "enclosed"]
        rng.shuffle(wants)
        wants = ["single"] + wants
        got, used, ws = [None] * 4, set(), wants + [want_q]
        for j in (3, 0, 1, 2):
            g = make_room_same(rng, f, F, strong, pool, used, ws[j], wall_q if j == 3 else "short")
            if g is None:
                break
            got[j] = g
            used |= {o[0] for o in g[0].objs} if j == 3 else {g[0].objs[g[1]][0]}
        if None in got:
            continue
        examples = [(r, t) for r, t, _ in got[:3]]
        query, qt, qkind = got[3]
        cons = [h for h in H if all(apply(h, r, F, attrs) == t for r, t in examples)]
        if any(apply(h, query, F, attrs) not in (None, qt) for h in cons) or any(h[0] == "odd" for h in cons):
            continue
        names = [o[0] for o in query.objs]
        other = [k for k in range(len(names)) if k != qt and F[names[k]][f] == "T"]
        rest = [k for k in range(len(names)) if F[names[k]][f] != "T"]
        labels = rng.sample(other + rng.sample(rest, 2), 3)
        labels.insert(qp, qt)
        label = {"answer": "ABCD"[qp], "type": "same", "level": level, "deciding": [f] * 4, "query_kind": qkind,
                 "example_kinds": [g[2] for g in got[:3]], "labels": labels, "n_consistent": len(cons), "tries": tries,
                 "examples": [dict(dump_room(r), target=t) for r, t in examples], "query": dump_room(query)}
        return examples, query, labels, label


def odd_item(rng, level, F, strong, pool, fam, attrs, H, qp):
    want_q = "basic" if level == 1 else rng.choice(["wall", "enclosed"]) if level == 2 else "both"
    wall_q = "short" if level == 1 else "maze"
    tries = 0
    while True:
        tries += 1
        if tries > 3000:
            raise RuntimeError(f"다른 것 찾기 L{level} 문항을 만들지 못했습니다")
        fs = rng.sample(fam, 4) if len(fam) >= 4 else [rng.choice(fam) for _ in range(4)]
        wants = ["wall", "enclosed"]
        rng.shuffle(wants)
        wants = ["single"] + wants
        got, used, ws = [None] * 4, set(), wants + [want_q]
        for j in (3, 0, 1, 2):
            g = make_room(rng, fs[j], F, strong, pool, used, ws[j], wall_q if j == 3 else "short", attrs)
            if g is None:
                break
            got[j] = g
            used |= {o[0] for o in g[0].objs} if j == 3 else {g[0].objs[g[1]][0]}
        if None in got:
            continue
        examples = [(r, t) for r, t, _ in got[:3]]
        query, qt, qkind = got[3]
        cons = [h for h in H if all(apply(h, r, F, attrs) == t for r, t in examples)]
        same_rule = any(h[0] == "feat" and h[1] is not None for h in cons)
        if same_rule or any(apply(h, query, F, attrs) not in (None, qt) for h in cons):
            continue
        names = [o[0] for o in query.objs]
        maj = majority(names, F, attrs)
        other = [k for k in range(len(names)) if k != qt and names[k] not in maj]
        labels = rng.sample(other + rng.sample([k for k in range(len(names)) if names[k] in maj], 2), 3)
        labels.insert(qp, qt)
        label = {"answer": "ABCD"[qp], "type": "odd", "level": level, "deciding": fs, "query_kind": qkind,
                 "example_kinds": [g[2] for g in got[:3]], "labels": labels, "n_consistent": len(cons), "tries": tries,
                 "examples": [dict(dump_room(r), target=t) for r, t in examples], "query": dump_room(query)}
        return examples, query, labels, label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("E:/benchmarks/godd_v3/insight_v3_grid200"))
    ap.add_argument("--pool-dir", type=Path)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    attrs, F, strong, base, _, n_added = load(args.pool_dir)
    pools = {1: base, 2: base, 3: base}
    fam1 = [a for a in attrs if a not in EMOTION + LEVEL3 + UNCLEAR_AFFECT]
    probe = random.Random(1)

    def grid_ok(f, pool):
        maj = [u for u in pool if strong(u, f) and (f not in EMOTION or u in FACE)]
        mn = [u for u in pool if F[u][f] == "F" and (f not in EMOTION or u in FACE)]
        return len(maj) >= MAJ and len(mn) >= N_OBJ - MAJ and any(
            majority(names, F, attrs) == frozenset(names[:MAJ])
            for names in (probe.sample(maj, MAJ) + probe.sample(mn, N_OBJ - MAJ) for _ in range(3000)))

    fams, gfams = {}, {}
    for lv, fam in ((1, fam1), (2, fam1), (3, fam1)):
        configure(*SIZE[lv])
        fams[lv] = [f for f in fam if sum(strong(u, f) for u in pools[lv]) >= MAJ and sum(F[u][f] == "F" for u in pools[lv]) >= 3]
        gfams[lv] = [f for f in fams[lv] if lv != 2 or grid_ok(f, pools[lv])]
    print(f"속성 {len(attrs)}개 | 기본 풀 {len(base)}개, 감정 확장 {n_added}개 | 결정 특징: " +
          " / ".join(f"L{k} {len(v)}개" for k, v in fams.items()), flush=True)
    print("다른 것 찾기 L2 결정 특징:", gfams[2], "| 같은 것 찾기 L2 결정 특징:", fams[2], flush=True)
    H = rivals(attrs)
    (args.out / "images").mkdir(parents=True, exist_ok=True)
    recs = []
    for s in range(args.sessions):
        seqs, qpos = {}, {}
        for t, extra in zip(("same", "odd"), ([0, 1], [2, 3]) if s % 2 == 0 else ([2, 3], [0, 1])):
            rest = [1, 2, 2, 2, 2, 3, 3, 3]
            rng.shuffle(rest)
            seqs[t] = [1, 1] + rest
            letters = [0, 1, 2, 3] * 2 + extra
            rng.shuffle(letters)
            qpos[t] = iter(letters)
        slots = ["same"] * 10 + ["odd"] * 10
        rng.shuffle(slots)
        it = {t: iter(v) for t, v in seqs.items()}
        for pos, t in enumerate(slots, 1):
            level = next(it[t])
            rid = f"s{s:02d}_p{pos:02d}_{t}"
            img = f"images/{rid}.png"
            make = same_item if t == "same" else odd_item
            configure(*SIZE[level])
            examples, query, labels, label = make(rng, level, F, strong, pools[level],
                                                  (fams if t == "same" else gfams)[level], attrs, H, next(qpos[t]))
            label["grid"] = N
            render_grid(examples, query, labels, args.out / img)
            inp = {"text": grid_prompt(examples, query, labels, False),
                   "multimodal": {"image": img, "text": grid_prompt(examples, query, labels, True)}}
            recs.append({"id": rid, "task": "label", "session": s, "position": pos, "input": inp, "label": label})
            print(f"{rid}: L{level} {' / '.join(label['deciding'])} -> {label['answer']} | 시도 {label['tries']}회", flush=True)
    write(args.out, recs)
    print(f"문항 {len(recs)}개 생성 완료: {args.out / 'samples.jsonl'}")
    for t in ("same", "odd"):
        rs = [r for r in recs if r["label"]["type"] == t]
        print(t, "난이도", dict(collections.Counter(r["label"]["level"] for r in rs)), "| 정답",
              dict(collections.Counter(r["label"]["answer"] for r in rs)), "| 질문 방 유형",
              dict(collections.Counter(r["label"]["query_kind"] for r in rs)))


if __name__ == "__main__":
    main()
