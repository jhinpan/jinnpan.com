---
title: "FlyDSL notes — BasisAttr, the layer beneath Layout"
description: "My FlyDSL source reading collapsed the layout algebra into five words. This is the patch — Fly_Basis, BasisAttr, what they are, why layouts need them, and where to start when your mentor hands you the 'complete the BasisAttr surface' task."
date: 2026-05-26
tags: ["flydsl", "MLIR", "AMD", "layout", "kernel-optimization"]
category: "Technical"
lang: "en"
---

After [the FlyDSL source reading](/blog/source-reading-006-flydsl/) shipped, my mentor told me to "use what you've learned about layout to round out basisType." That is when it hit me — I could explain the layout algebra to someone else, but I had not actually understood what sits *underneath* it. This is the patch.

## Three intuitions about layout I had skipped

The § M1 section of the source reading collapsed layout into five words: shape, stride, layout, divide, slice. The "function from a coord-tuple to a linear index" framing is correct, as far as it goes. To work *on* the layout system instead of just *with* it, three more intuitions matter:

**(a) Shape and stride are two independent degrees of freedom.** Shape decides the domain; stride decides "how far one step along each mode moves in memory." The same `shape = (4, 8)` can be paired with `(1, 4)` (column-major), `(8, 1)` (row-major), `(0, 1)` (row broadcast), or `(1E0, 1E1)` — that last one is BasisAttr territory.

**(b) Layout is not data — it is an address-generating function.** You never write `A[i, j]` in a FlyDSL kernel. You see `logical_divide` / `partition_S` / `slice`. That is because layout is a first-class MLIR type — `!fly.layout<(8,16):(1,8)>` is real, and stage 3 of the pass pipeline (`fly-layout-lowering`) is what folds a symbolic layout into concrete `i*1 + j*8` address math.

**(c) The layout algebra is a function-composition algebra.** `composition(A, B)(x) = A(B(x))`; `logical_divide(A, tiler)` decomposes A into "intra-tile layout × inter-tile grid layout"; `make_layout_tv(thr, val)` constructs a `(thread_id, value_id) → tile coord` mapping. Every operation reshapes the function itself; nothing moves data.

## What BasisAttr actually is

The docs are quiet on it. The definition lives in `include/flydsl/Dialect/Fly/IR/FlyAttrDefs.td:112`:

```cpp
def Fly_BasisAttr : Fly_Attr<"Basis", "basis", [...]> {
  let parameters = (ins
    Fly_IntAttr:$value,
    ArrayRefParameter<"int32_t">:$modes
  );
}
```

The print format is `value E mode0 E mode1 ...`, so you see `1E0`, `1E1`, `2E0E1`. The `E` comes from the standard basis vectors e_0, e_1.

**Expansion** (see `intTupleBasis2Tuple` in `lib/Dialect/Fly/Utils/IntTupleUtils.cpp:745`):

- `1E0` → `(1)` — value 1 at position 0 of a 1-tuple
- `1E1` → `(0, 1)` — value 1 at position 1 of a 2-tuple
- `1E2` → `(0, 0, 1)` — value 1 at position 2 of a 3-tuple
- `2E0E1` → `((0, 2))` — nested tuple, value 2 at path `[0][1]`

So `BasisAttr` is **a sparse representation of a "standard basis vector"**: it compactly encodes a (possibly nested) `IntTuple` that is zero everywhere except for `value` at a specified `modes` path.

## Why layouts need it

A normal layout's stride is scalar — it maps a coordinate to a **flat int**: `(i, j)` → `i + 4*j`. Sometimes you need the stride to preserve *structure*, mapping coord to coord-tuple instead.

The cleanest example is identity layout. `make_identity_layout((M, N))` produces stride `(1E0, 1E1)`; the full type is `!fly.layout<(4, 8) : (1E0, 1E1)>` (`tests/mlir/LayoutAlgebra/construction.mlir:89`). Feed `(i, j)` to this layout and you get `(i, j)` back, not a flat int. That is only expressible with basis strides.

BasisAttr shows up in `make_identity_layout`, `right_inverse`, `coord_swizzle` — anywhere a layout needs to preserve the coordinate structure. `tests/mlir/LayoutAlgebra/coord_swizzle.mlir` is essentially wall-to-wall `(1E0, 1E1, 1E2)`.

Two things at the type level that are easy to confuse:

- `Fly_BasisAttr` — the MLIR attribute, lives inside `IntTupleAttr` leaves
- `Fly_Basis` — the MLIR type (`FlyTypeDefs.td:10`), wrapping the attribute as a first-class type

The "basisType" my mentor was talking about is the second one strictly speaking, but in practice you touch both.

## What "completing BasisAttr" can mean

Putting the current BasisAttr operator surface next to IntAttr makes the gap obvious:

| Op | IntAttr | BasisAttr |
|---|---|---|
| `+`, `-` | ✓ | ✗ |
| `*` | ✓ | ✓ (only against IntAttr) |
| `/`, `%` | ✓ | ✗ |
| `<`, `<=`, `>`, `>=` | ✓ | ✗ |
| `intMin`, `intMax` | ✓ | ✗ |
| `intShapeDiv` | ✓ | ✗ |
| `isStaticValue(v)` | ✓ | ✗ (only `isStatic`) |

Worse, `IntTupleBuilder<IntTupleAttr>::add` at `lib/Dialect/Fly/Utils/IntTupleUtils.cpp:105-121` calls `intTupleBasis2Tuple` the moment it sees a basis input — expanding it back to a dense tuple before doing arithmetic. The resulting IR ends up fatter than it needs to be.

Four candidate directions:

1. **Operator-surface completion** — add `+`, `-`, `/`, `%`, `<`, and friends to `lib/Dialect/Fly/Utils/IntUtils.cpp:277-304`. Smallest change, parallel IntAttr implementations to copy, easy to unit-test.
2. **Avoid unnecessary dense expansion** — basis-aware fast paths in `IntTupleBuilder` so the layout algebra stops degrading on basis values.
3. **`Fly_Basis` type API** — the type currently exposes only `depth()` and `isStatic()` (`FlyTypeDefs.td:10-25`); promoting `getValue()` / `getModes()` to the type level removes boilerplate at callsites.
4. **Basis-aware simplification in layout algebra** — `composition` / `divide` / `product` should trivialize on basis strides where possible.

I am starting with (1) because its boundary is the cleanest and it has obvious targets in `tests/mlir/LayoutAlgebra/` for lit tests. (2) is what actually improves the IR but is only worthwhile after (1).

## Coda

Before writing this I thought I understood layout. Writing it showed me that explaining an abstraction to someone else is not the same skill as changing its implementation. The second requires walking down one floor — into the things the abstraction pretends do not exist. BasisAttr is one of those floors.

The full FlyDSL system-level reading is still at [/sources/flydsl.html](/sources/flydsl.html).
