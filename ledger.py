#!/usr/bin/env python3
"""
ledger.py — Two ledgers inside one conserved collapse system.

This is a SELF-CONTAINED demo (numpy + stdlib only) of the idea behind the
Vector Collapse Engine: a model that does not only *predict* an answer but
moves through a space of meanings — a vector bouncing on words — and keeps a
record of HOW the answer formed. It is faithful to the real engine in
`collapse_retrain/vector_collapse.py`:

  * three semantic anchors  E (entailment) / N (neutral) / C (contradiction),
  * divergence  div = 1 - align   (a true gravity well: attractive everywhere,
    zero only on the anchor — see COLLAPSE_ENGINE_VERDICT.md),
  * the update  h = h + delta - force ,
        force = strength * div * normalize(h - target_anchor),
    i.e. each step SUBTRACTS a direction in meaning space.

On top of that mechanism it adds the part you asked for: two ledgers that live
in ONE conserved system.

  Ledger 1 — Thinking Ledger   (dynamic, editable, time-annotated)
      Every collapse step is a Frame. A frame records the vector state, its
      alignment/divergence to each anchor, WHAT idea was subtracted this step
      and WHY, the conserved weight it holds, and an English-word readout of
      where the "ball" landed. The model can revise, subtract, and annotate
      frames, and the timeline can be compressed into distinct thoughts.

  Ledger 2 — Output Ledger     (clean, structured, readable)
      Where the model *speaks*: the final label, its confidence, and one plain
      sentence. No thinking lives here.

Both ledgers share ONE conserved budget. Early on, almost all the weight sits
in the Thinking Ledger (the model is still uncertain). As the vector commits to
an anchor, weight FLOWS from thinking to output — but the total is conserved
exactly. When the two ledgers agree, a trainable "connection" is registered.

Run:  python3 ledger.py
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# --------------------------------------------------------------------------- #
# 0. Geometry: anchors and a tiny "word well" space (the ball bounces on words)
# --------------------------------------------------------------------------- #

DIM = 16
LABELS = ("entailment", "neutral", "contradiction")
ANCHOR_KEYS = ("E", "N", "C")
_KEY_TO_LABEL = {"E": "entailment", "N": "neutral", "C": "contradiction"}

TOTAL_WEIGHT = 1.0  # the conserved quantity shared by both ledgers


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / (n + 1e-9)


def _seed_vec(text: str, dim: int = DIM) -> np.ndarray:
    """Deterministic pseudo-random unit vector for any token (stable across runs)."""
    h = hashlib.sha256(text.encode()).digest()
    rng = np.random.default_rng(int.from_bytes(h[:8], "big"))
    return _unit(rng.normal(size=dim))


# Fixed anchors, well separated.
_rng = np.random.default_rng(7)
ANCHORS: Dict[str, np.ndarray] = {
    "E": _unit(_rng.normal(size=DIM)),
    "N": _unit(_rng.normal(size=DIM)),
    "C": _unit(_rng.normal(size=DIM)),
}

# A small lexicon of "word wells" the trajectory can land on. Each word sits
# near an anchor so the English readout is meaningful. Arbitrary words still
# work (they get a deterministic vector via _seed_vec).
_WORD_WELLS: Dict[str, np.ndarray] = {}
for _w, _key, _bias in [
    ("same", "E", 0.9), ("means", "E", 0.8), ("therefore", "E", 0.7),
    ("implies", "E", 0.85), ("so", "E", 0.6),
    ("maybe", "N", 0.9), ("could", "N", 0.8), ("unrelated", "N", 0.85),
    ("perhaps", "N", 0.8), ("unknown", "N", 0.7),
    ("not", "C", 0.9), ("never", "C", 0.85), ("opposite", "C", 0.9),
    ("contradicts", "C", 0.8), ("but", "C", 0.6),
]:
    _WORD_WELLS[_w] = _unit(_bias * ANCHORS[_key] + (1 - _bias) * _seed_vec(_w))


def nearest_word(h: np.ndarray) -> Tuple[str, float]:
    hn = _unit(h)
    best, best_sim = "·", -1.0
    for w, wv in _WORD_WELLS.items():
        s = float(hn @ wv)
        if s > best_sim:
            best, best_sim = w, s
    return best, best_sim


def encode_sentence(text: str) -> np.ndarray:
    toks = [t for t in text.lower().replace(".", " ").replace(",", " ").split() if t]
    if not toks:
        return np.zeros(DIM)
    vecs = [_WORD_WELLS.get(t, _seed_vec(t)) for t in toks]
    return np.mean(vecs, axis=0)


# --------------------------------------------------------------------------- #
# 1. The conserved Frame and the two Ledgers
# --------------------------------------------------------------------------- #

@dataclass
class Subtraction:
    """What idea was removed this step, and why."""
    idea: str            # which anchor/word lost support
    magnitude: float     # how strongly it was pushed down
    reason: str          # human-readable why


@dataclass
class Frame:
    """One time-step of thinking. Editable."""
    t: int
    vector: np.ndarray
    align: Dict[str, float]              # alignment to E / N / C
    divergence: Dict[str, float]         # 1 - align, per anchor
    subtracted: Optional[Subtraction]    # what was removed reaching this frame
    thinking_weight: float               # share of the conserved budget held here
    output_weight: float                 # share that has flowed to the output side
    word: str                            # English-word readout (where the ball is)
    word_sim: float
    note: str = ""                       # free annotation (why kept / changed)
    alive: bool = True                   # frames can be subtracted but kept on record

    @property
    def conserved(self) -> float:
        return self.thinking_weight + self.output_weight

    @property
    def committed_key(self) -> str:
        return max(self.align, key=self.align.get)

    def __repr__(self) -> str:
        sub = "—"
        if self.subtracted:
            sub = f"-{self.subtracted.idea}({self.subtracted.magnitude:.2f})"
        flag = "" if self.alive else "  [subtracted]"
        return (f"t={self.t:>2}  word={self.word:<11} "
                f"E={self.align['E']:+.2f} N={self.align['N']:+.2f} C={self.align['C']:+.2f}  "
                f"think={self.thinking_weight:.2f} speak={self.output_weight:.2f}  "
                f"drop {sub}{flag}")


class ThinkingLedger:
    """Ledger 1 — dynamic, editable, time-annotated. Where the model THINKS."""

    def __init__(self) -> None:
        self.frames: List[Frame] = []

    def append(self, frame: Frame) -> None:
        self.frames.append(frame)

    # --- editing operations (the model can act on its own thoughts) ---------
    def annotate(self, t: int, note: str) -> None:
        self.frames[t].note = note

    def subtract_frame(self, t: int, reason: str) -> None:
        """Remove a weak thought but keep it on record (with the reason why)."""
        f = self.frames[t]
        f.alive = False
        f.note = (f.note + " | " if f.note else "") + f"subtracted: {reason}"

    def revise_word(self, t: int, new_word: str, reason: str) -> None:
        self.frames[t].word = new_word
        self.frames[t].note = (self.frames[t].note + " | " if self.frames[t].note else "") + \
            f"revised→{new_word}: {reason}"

    # --- views --------------------------------------------------------------
    def live(self) -> List[Frame]:
        return [f for f in self.frames if f.alive]

    def timeline(self) -> List[Tuple[int, str, str]]:
        """(time, word, committed-anchor) for each live frame — the thought path."""
        return [(f.t, f.word, f.committed_key) for f in self.live()]

    def compress(self) -> List[Dict]:
        """Collapse consecutive frames with the same committed anchor into one
        keyframe. The compressed timeline is the sequence of DISTINCT thoughts —
        easier to read and to train on."""
        keyframes: List[Dict] = []
        for f in self.live():
            if keyframes and keyframes[-1]["key"] == f.committed_key:
                kf = keyframes[-1]
                kf["t_end"] = f.t
                kf["words"].append(f.word)
                if f.subtracted:
                    kf["subtractions"].append(f.subtracted.idea)
            else:
                keyframes.append({
                    "key": f.committed_key,
                    "label": _KEY_TO_LABEL[f.committed_key],
                    "t_start": f.t, "t_end": f.t,
                    "words": [f.word],
                    "subtractions": [f.subtracted.idea] if f.subtracted else [],
                })
        return keyframes


class OutputLedger:
    """Ledger 2 — clean, structured, readable. Where the model SPEAKS."""

    def __init__(self) -> None:
        self.label: str = ""
        self.confidence: float = 0.0
        self.sentence: str = ""
        self.weight: float = 0.0  # conserved weight that has flowed here

    def speak(self, label: str, confidence: float, premise: str, hypothesis: str) -> None:
        self.label = label
        self.confidence = confidence
        verb = {"entailment": "follows from",
                "neutral": "is neither supported nor contradicted by",
                "contradiction": "contradicts"}[label]
        self.sentence = (f'The hypothesis "{hypothesis}" {verb} the premise '
                         f'"{premise}".')

    def __repr__(self) -> str:
        return (f"[{self.label.upper()} @ {self.confidence:.0%}]  {self.sentence}")


# --------------------------------------------------------------------------- #
# 2. The conserved system that holds both ledgers
# --------------------------------------------------------------------------- #

class ConservedSystem:
    """One brain, two sections. Thinking and speaking share a single conserved
    budget (TOTAL_WEIGHT). As the vector commits, weight flows from think→speak;
    the sum is conserved exactly. When the two ledgers agree, a connection
    forms — a trainable signal that the thinking justified the speaking."""

    def __init__(self, premise: str, hypothesis: str) -> None:
        self.premise = premise
        self.hypothesis = hypothesis
        self.thinking = ThinkingLedger()
        self.output = OutputLedger()

    # --- conservation check -------------------------------------------------
    def conservation_error(self) -> float:
        if not self.thinking.frames:
            return 0.0
        return max(abs(f.conserved - TOTAL_WEIGHT) for f in self.thinking.frames)

    # --- the connection between the two ledgers -----------------------------
    def connection(self) -> Dict:
        """Does the thinking path justify what was spoken? Correlation = the
        final commitment strength when the committed anchor equals the spoken
        label. This is the cross-ledger signal that becomes trainable."""
        live = self.thinking.live()
        if not live or not self.output.label:
            return {"connected": False, "strength": 0.0, "reason": "empty ledger"}
        final = live[-1]
        agree = _KEY_TO_LABEL[final.committed_key] == self.output.label
        strength = final.output_weight if agree else 0.0
        return {
            "connected": bool(agree and strength > 0.5),
            "strength": float(strength),
            "thinking_says": _KEY_TO_LABEL[final.committed_key],
            "output_says": self.output.label,
            "reason": ("thinking path and spoken answer agree"
                       if agree else "thinking and output disagree — weak connection"),
        }


# --------------------------------------------------------------------------- #
# 3. The collapse thinker: runs the dynamics and fills both ledgers
# --------------------------------------------------------------------------- #

class CollapseThinker:
    """A faithful, dependency-light reimplementation of the collapse step that
    also writes every step into the Thinking Ledger and resolves into the
    Output Ledger."""

    def __init__(self, num_layers: int = 6, strength: float = 0.35,
                 temperature: float = 6.0) -> None:
        self.num_layers = num_layers
        self.strength = strength
        self.temperature = temperature  # sharpness of the final softmax

    def _aligns(self, h: np.ndarray) -> Dict[str, float]:
        hn = _unit(h)
        return {k: float(hn @ ANCHORS[k]) for k in ANCHOR_KEYS}

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        x = x - x.max()
        e = np.exp(x)
        return e / e.sum()

    def think(self, system: ConservedSystem) -> ConservedSystem:
        u = encode_sentence(system.premise)
        v = encode_sentence(system.hypothesis)
        h = u - v  # the pair vector — exactly as the real engine forms it

        # route: the anchor the pair is already closest to is the target it
        # collapses toward (per-item routing in vector_collapse.py).
        a0 = self._aligns(h)
        target = max(a0, key=a0.get)
        prev_align: Optional[Dict[str, float]] = None

        for t in range(self.num_layers + 1):
            align = self._aligns(h)
            div = {k: 1.0 - align[k] for k in ANCHOR_KEYS}

            # what was subtracted reaching this frame: the anchor whose
            # alignment dropped most since last step (an idea pushed down).
            subtraction: Optional[Subtraction] = None
            if prev_align is not None:
                drops = {k: prev_align[k] - align[k] for k in ANCHOR_KEYS}
                worst = max(drops, key=drops.get)
                if drops[worst] > 5e-3:
                    subtraction = Subtraction(
                        idea=_KEY_TO_LABEL[worst],
                        magnitude=float(drops[worst]),
                        reason=(f"lost support as the vector moved toward "
                                f"{_KEY_TO_LABEL[target]}"),
                    )

            # conserved split: certainty = how far the leading anchor leads.
            # weight flows think→speak as the gap widens; sum stays = TOTAL.
            ordered = sorted(align.values(), reverse=True)
            gap = (ordered[0] - ordered[1])           # margin of the leader
            certainty = float(np.clip(gap / 0.6, 0.0, 1.0))
            out_w = TOTAL_WEIGHT * certainty
            think_w = TOTAL_WEIGHT - out_w

            word, sim = nearest_word(h)
            system.thinking.append(Frame(
                t=t, vector=h.copy(), align=align, divergence=div,
                subtracted=subtraction, thinking_weight=think_w,
                output_weight=out_w, word=word, word_sim=sim,
            ))

            prev_align = align
            if t == self.num_layers:
                break

            # the collapse update: h = h + delta - force.
            # delta starts at identity (0) like the zero-init engine MLP; the
            # force is the gravity-well pull toward the routed anchor.
            anchor = ANCHORS[target]
            force = self.strength * div[target] * _unit(h - anchor)
            h = h + 0.0 - force

            # norm clamp, as in the engine
            n = np.linalg.norm(h)
            if n > 10.0:
                h = h * (10.0 / (n + 1e-9))

        # resolve → speak. Final logits = alignment to each anchor.
        final = system.thinking.live()[-1]
        logits = np.array([final.align[k] for k in ANCHOR_KEYS])
        probs = self._softmax(self.temperature * logits)
        idx = int(probs.argmax())
        system.output.speak(LABELS[idx], float(probs[idx]),
                            system.premise, system.hypothesis)
        system.output.weight = final.output_weight
        return system


# --------------------------------------------------------------------------- #
# 4. Demo
# --------------------------------------------------------------------------- #

def _demo_pair(premise: str, hypothesis: str) -> None:
    sys = ConservedSystem(premise, hypothesis)
    CollapseThinker().think(sys)

    print("=" * 78)
    print(f"PREMISE    : {premise}")
    print(f"HYPOTHESIS : {hypothesis}")
    print("-" * 78)

    print("LEDGER 1 — THINKING  (time-annotated frames; the ball bouncing on words)")
    for f in sys.thinking.frames:
        print("   ", f)

    # demonstrate editability: subtract the weakest mid-thought on record.
    live = sys.thinking.live()
    if len(live) > 2:
        weak_t = min(live[1:-1], key=lambda fr: fr.word_sim).t
        sys.thinking.subtract_frame(weak_t, "low word-support, off the main path")
        print(f"   (edited: subtracted frame t={weak_t} as a weak thought)")

    print("\n   compressed thought-path (distinct thoughts only):")
    for kf in sys.thinking.compress():
        subs = ", ".join(dict.fromkeys(kf["subtractions"])) or "—"
        print(f"     t{kf['t_start']}–{kf['t_end']}  {kf['label']:<13} "
              f"words[{'→'.join(dict.fromkeys(kf['words']))}]  dropped[{subs}]")

    print("\nLEDGER 2 — OUTPUT  (clean; where the model speaks)")
    print("   ", sys.output)

    conn = sys.connection()
    print("\nCONSERVED SYSTEM")
    print(f"    conservation error (max |think+speak - {TOTAL_WEIGHT}|): "
          f"{sys.conservation_error():.2e}")
    print(f"    connection: {'FORMED' if conn['connected'] else 'weak'}  "
          f"(strength {conn['strength']:.2f}) — {conn['reason']}")
    print()


def main() -> None:
    print("\nTWO LEDGERS IN ONE CONSERVED COLLAPSE SYSTEM")
    print("thinking (Ledger 1) and speaking (Ledger 2) share one conserved budget\n")
    _demo_pair("a man is playing a guitar on stage",
               "a person is performing music")          # entailment-ish
    _demo_pair("a dog is running in the park",
               "the animal is sleeping indoors")          # contradiction-ish
    _demo_pair("a woman is buying vegetables at a market",
               "she is preparing dinner tonight")         # neutral-ish


if __name__ == "__main__":
    main()
