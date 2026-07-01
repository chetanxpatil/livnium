#!/usr/bin/env python3
"""
container_stream.py — One conserved tape of dynamically-sized containers.

This is the *unified* version of ledger.py. Instead of two fixed ledgers
(thinking + speaking), there is ONE timeline of containers. Each container is
tagged THINK or SPEAK and is sized in tokens at runtime, so a run looks like:

        THINK[2] · SPEAK[1] · THINK[4] · SPEAK[9] · ...

Thinking and speaking happen *together* in the same stream. The sizes are not
chosen in advance — they fall out of the collapse dynamics:

  * Conserved quantity:  unresolved_mass + resolved_mass = 1.0  (always).
  * The pair vector h = u - v collapses toward an E/N/C anchor exactly as in
    `collapse_retrain/vector_collapse.py`  (div = 1 - align, h = h + delta - force).
  * While the vector is undecided (small margin) the tape grows a THINK
    container — one think-token per collapse step.
  * Each time the collapse RESOLVES a chunk of mass (the leading anchor pulls
    ahead), that chunk is *minted* into SPEAK tokens and a SPEAK container is
    flushed. A small resolution mints a small burst (1); a big jump mints a big
    burst (9). Mass moves unresolved -> resolved; the total stays 1.0.

So "how long to think" and "how long to speak" are decided dynamically by how
much meaning got resolved at each moment — not by a fixed schedule.

Reuses the geometry from ledger.py (same anchors, word wells, encoder).

Run:  python3 container_stream.py
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ledger import (
    ANCHORS, ANCHOR_KEYS, _KEY_TO_LABEL, LABELS, TOTAL_WEIGHT,
    _unit, encode_sentence, nearest_word, _WORD_WELLS,
)

THINK = "THINK"
SPEAK = "SPEAK"

# Each SPEAK token represents this much resolved mass. Smaller quantum => more,
# finer speak tokens; this is what turns a 0.72 jump in certainty into ~9 tokens.
TOKEN_QUANTUM = 0.08


# Words grouped by the anchor they sit nearest — used to give SPEAK tokens
# actual English content drawn from the committed basin.
ANCHOR_WORDS: Dict[str, List[str]] = {k: [] for k in ANCHOR_KEYS}
for _w, _wv in _WORD_WELLS.items():
    _best = max(ANCHOR_KEYS, key=lambda k: float(_unit(_wv) @ ANCHORS[k]))
    ANCHOR_WORDS[_best].append(_w)


@dataclass
class Container:
    """A dynamically-sized unit of the tape. THINK or SPEAK."""
    kind: str
    tokens: List[str] = field(default_factory=list)
    t_start: int = 0
    t_end: int = 0
    mass: float = 0.0                 # conserved mass held/consumed here
    committed: str = ""               # the anchor this stretch leans to (E/N/C)
    align_end: Dict[str, float] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return len(self.tokens)

    def __repr__(self) -> str:
        body = " ".join(self.tokens)
        tag = f"{self.kind}[{self.size}]"
        lean = f" ⟶{_KEY_TO_LABEL[self.committed]}" if self.committed else ""
        return f"{tag:<10} mass={self.mass:.2f}  ({body}){lean}"


class ContainerStream:
    """The one tape. Holds both thinking and speaking, conserved together."""

    def __init__(self, premise: str, hypothesis: str) -> None:
        self.premise = premise
        self.hypothesis = hypothesis
        self.containers: List[Container] = []
        self.resolved_mass = 0.0
        self.unresolved_mass = TOTAL_WEIGHT

    # conservation: the whole point — these two must always sum to TOTAL.
    def conservation_error(self) -> float:
        return abs((self.resolved_mass + self.unresolved_mass) - TOTAL_WEIGHT)

    def pattern(self) -> str:
        """Compact shape of the run, e.g. '2T·1S·4T·9S'."""
        return "·".join(f"{c.size}{'T' if c.kind == THINK else 'S'}"
                        for c in self.containers if c.size)

    def speak_text(self) -> str:
        """The readable conclusion = the last SPEAK container's verdict."""
        speaks = [c for c in self.containers if c.kind == SPEAK and c.committed]
        if not speaks:
            return ""
        label = _KEY_TO_LABEL[speaks[-1].committed]
        verb = {"entailment": "follows from",
                "neutral": "is neither supported nor contradicted by",
                "contradiction": "contradicts"}[label]
        return (f'[{label.upper()}]  The hypothesis "{self.hypothesis}" {verb} '
                f'the premise "{self.premise}".')


class DynamicCollapse:
    """Runs the collapse and lays down the interleaved container tape."""

    def __init__(self, max_steps: int = 12, strength: float = 0.35) -> None:
        self.max_steps = max_steps
        self.strength = strength

    def _aligns(self, h: np.ndarray) -> Dict[str, float]:
        hn = _unit(h)
        return {k: float(hn @ ANCHORS[k]) for k in ANCHOR_KEYS}

    @staticmethod
    def _certainty(align: Dict[str, float]) -> float:
        ordered = sorted(align.values(), reverse=True)
        return float(np.clip((ordered[0] - ordered[1]) / 0.6, 0.0, 1.0))

    def _mint_speak(self, stream: ContainerStream, gain: float, t: int,
                    committed: str, align: Dict[str, float]) -> None:
        """Convert a chunk of resolved mass into a SPEAK container of dynamic
        size. Mass moves unresolved -> resolved; total conserved."""
        n = int(np.floor(gain / TOKEN_QUANTUM))
        if n < 1:
            return
        minted = n * TOKEN_QUANTUM
        minted = min(minted, stream.unresolved_mass)  # never overspend the budget
        n = max(1, int(round(minted / TOKEN_QUANTUM)))

        pool = ANCHOR_WORDS.get(committed) or [_KEY_TO_LABEL[committed]]
        toks = [pool[i % len(pool)] for i in range(n)]

        stream.unresolved_mass -= minted
        stream.resolved_mass += minted
        stream.containers.append(Container(
            kind=SPEAK, tokens=toks, t_start=t, t_end=t, mass=minted,
            committed=committed, align_end=dict(align),
        ))

    def run(self, premise: str, hypothesis: str) -> ContainerStream:
        stream = ContainerStream(premise, hypothesis)
        h = encode_sentence(premise) - encode_sentence(hypothesis)

        a0 = self._aligns(h)
        target = max(a0, key=a0.get)          # routed anchor (per-item routing)
        achieved_certainty = 0.0
        stalled = 0                            # consecutive steps with no resolution
        think: Optional[Container] = None

        for t in range(self.max_steps + 1):
            align = self._aligns(h)
            committed = max(align, key=align.get)
            word, _ = nearest_word(h)

            # grow a THINK container, one token per step, holding unresolved mass
            if think is None:
                think = Container(kind=THINK, t_start=t, committed=committed)
                stream.containers.append(think)
            think.tokens.append(word)
            think.t_end = t
            think.committed = committed
            think.align_end = dict(align)
            think.mass = stream.unresolved_mass

            # how much new certainty did this step buy us?
            cert = self._certainty(align)
            gain = cert - achieved_certainty
            if gain >= TOKEN_QUANTUM:
                # resolve: mint a SPEAK burst sized by the gain, then reopen
                # a fresh THINK container for whatever is still unresolved.
                self._mint_speak(stream, gain * stream.unresolved_mass, t,
                                 committed, align)
                achieved_certainty = cert
                think = None
                stalled = 0
            else:
                stalled += 1

            # stop thinking once the answer is settled: high certainty and no
            # further resolution for two steps. (No point thinking in circles.)
            if cert > 0.8 and stalled >= 2:
                break
            if t == self.max_steps:
                break

            # collapse update: h = h + delta - force  (delta=0 = identity start)
            div = 1.0 - align[target]
            force = self.strength * div * _unit(h - ANCHORS[target])
            h = h + 0.0 - force
            n = np.linalg.norm(h)
            if n > 10.0:
                h = h * (10.0 / (n + 1e-9))

        # final flush: commit any leftover unresolved mass into a closing SPEAK
        if stream.unresolved_mass > 1e-6:
            committed = max(self._aligns(h), key=self._aligns(h).get)
            self._mint_speak(stream, 1.0, self.max_steps, committed,
                             self._aligns(h))
        # drop empty trailing THINK if it never spoke
        stream.containers = [c for c in stream.containers if c.size > 0]
        return stream


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #

def _demo(premise: str, hypothesis: str) -> None:
    stream = DynamicCollapse().run(premise, hypothesis)
    print("=" * 78)
    print(f"PREMISE    : {premise}")
    print(f"HYPOTHESIS : {hypothesis}")
    print(f"shape      : {stream.pattern()}      "
          f"(T = think tokens, S = speak tokens — sizes chosen at runtime)")
    print("-" * 78)
    for c in stream.containers:
        print("   ", c)
    print("-" * 78)
    print("   ", stream.speak_text())
    print(f"    conserved:  unresolved {stream.unresolved_mass:.3f} + "
          f"resolved {stream.resolved_mass:.3f} = "
          f"{stream.unresolved_mass + stream.resolved_mass:.3f}   "
          f"(error {stream.conservation_error():.1e})")
    print()


def main() -> None:
    print("\nONE CONSERVED TAPE OF DYNAMICALLY-SIZED THINK / SPEAK CONTAINERS")
    print("thinking and speaking interleave; container sizes are decided at runtime\n")
    _demo("a man is playing a guitar on stage",
          "a person is performing music")
    _demo("a dog is running in the park",
          "the animal is sleeping indoors")
    _demo("a woman is buying vegetables at a market",
          "she is preparing dinner tonight")


if __name__ == "__main__":
    main()
