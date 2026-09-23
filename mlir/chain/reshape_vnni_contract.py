#!/usr/bin/env python3
"""Drop unit extents on 6D VNNI contracts so packed-dot can match.

LLVM 22's vector_contract_to_packed_type_dot_product only matches
vector<1x1x2xbf16> x vector<1x16x2xbf16> -> vector<1x16xf32>.
After packing MR/NR, vectorize emits an extra two unit dims:
vector<1x1x1x2xbf16> x vector<1x1x16x2xbf16> -> vector<1x1x1x16xf32>.
This rewrite is shape_cast only — not a ukernel.
"""
import re
import sys

OFFICIAL = (
    "vector.contract {indexing_maps = ["
    "affine_map<(d4, d1, d2, d3) -> (d1, d3, d4)>, "
    "affine_map<(d4, d1, d2, d3) -> (d3, d2, d4)>, "
    "affine_map<(d4, d1, d2, d3) -> (d1, d2)>], "
    'iterator_types = ["reduction", "parallel", "parallel", "reduction"], '
    "kind = #vector.kind<add>}"
)

pat = re.compile(
    r"%(\w+) = vector\.contract \{[^}]+\} "
    r"%(\w+), %(\w+), %(\w+) : "
    r"vector<1x1x1x2xbf16>, vector<1x1x16x2xbf16> into vector<1x1x1x16xf32>"
)

def rewrite(src: str) -> str:
    n = 0
    def repl(m):
        nonlocal n
        n += 1
        res, a, b, c = m.group(1), m.group(2), m.group(3), m.group(4)
        sa, sb, sc, tmp = f"sca{n}", f"scb{n}", f"scc{n}", f"sct{n}"
        return (
            f"%{sa} = vector.shape_cast %{a} : vector<1x1x1x2xbf16> to vector<1x1x2xbf16>\n"
            f"          %{sb} = vector.shape_cast %{b} : vector<1x1x16x2xbf16> to vector<1x16x2xbf16>\n"
            f"          %{sc} = vector.shape_cast %{c} : vector<1x1x1x16xf32> to vector<1x16xf32>\n"
            f"          %{tmp} = {OFFICIAL} %{sa}, %{sb}, %{sc} : "
            f"vector<1x1x2xbf16>, vector<1x16x2xbf16> into vector<1x16xf32>\n"
            f"          %{res} = vector.shape_cast %{tmp} : vector<1x16xf32> to vector<1x1x1x16xf32>"
        )
    out, count = pat.subn(repl, src)
    print(f"rewrote {count} contracts", file=sys.stderr)
    return out

if __name__ == "__main__":
    sys.stdout.write(rewrite(sys.stdin.read()))
