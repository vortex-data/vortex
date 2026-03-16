# Correctness Proof: FSST LIKE DFA Matching

This document proves that the DFA-based LIKE evaluation in
`encodings/fsst/src/compute/like.rs` is a correct, semantics-preserving
transform. We build up from the simplest possible correct algorithm, then
show that each optimization is an equivalence-preserving transformation.

---

## Part I: The Naive DFA on Plain Bytes

Before touching FSST at all, we establish correctness of the underlying
matching automata on ordinary (decompressed) byte strings.

### 1. Prefix Matching: A Trivial DFA

**Problem.** Given a literal prefix `P = p_0 p_1 ... p_{L-1}` and a byte
string `D = d_0 d_1 ... d_{n-1}`, determine whether `D` starts with `P`.

**Naive DFA.** States `{0, 1, ..., L, FAIL}`:
- State `s` (0 ≤ s < L) means "the first `s` bytes matched `P[0..s]`".
- State `L` = ACCEPT: the entire prefix has been matched.
- State FAIL: a mismatch occurred.

Transitions:
```
δ(s, b) :=
    if s == ACCEPT:  return ACCEPT      // sticky: P% matches any suffix
    if s == FAIL:    return FAIL        // sticky: no recovery for prefix
    if b == P[s]:    return s + 1       // match continues
    else:            return FAIL        // mismatch
```

**Claim.** After feeding `d_0, ..., d_{n-1}` through this DFA starting from
state 0, the DFA is in state ACCEPT iff `D` starts with `P`.

**Proof.** By induction on `i` (number of bytes consumed):

- *Base:* State is 0. No bytes consumed. Invariant: "the first 0 bytes match
  P[0..0]" is vacuously true. ✓

- *Step:* Suppose after consuming `d_0 ... d_{i-1}` we are in state `s`.
  - If `s < L` and `d_i == P[s]`: we move to `s+1`. Invariant: "the first
    `s+1` bytes match P[0..s+1]" holds because the first `s` matched (IH)
    and `d_i == P[s]`. ✓
  - If `s < L` and `d_i ≠ P[s]`: we move to FAIL. The prefix doesn't match
    at position `s`, and since prefix matching is strictly sequential (no
    fallback), this is correct. ✓
  - ACCEPT and FAIL are absorbing: once entered, they never change. ✓

After all bytes: state is ACCEPT iff we successfully matched all `L` bytes
of `P`, i.e., `D` starts with `P`. ∎

This is the simplest correct prefix matcher. `FsstPrefixDfa` is built on
exactly this DFA, lifted to operate on FSST codes (Part II).

---

### 2. Substring Matching: The KMP Automaton

**Problem.** Given a needle `N = n_0 ... n_{L-1}` and a byte string `D`,
determine whether `D` contains `N` as a substring.

The naive approach (check every starting position) is O(n·L). The
Knuth-Morris-Pratt (KMP) automaton solves this in O(n) by never re-scanning
bytes.

#### 2.1 KMP Failure Function

**Definition.** For needle `N`, the failure function `failure[i]` is the
length of the longest **proper** prefix of `N[0..=i]` that is also a suffix
of `N[0..=i]`.

**Example.** Needle `"abcabd"`:
```
i:        0  1  2  3  4  5
N[i]:     a  b  c  a  b  d
failure:  0  0  0  1  2  0
```
At `i=4`, the longest proper prefix of `"abcab"` that is also a suffix is
`"ab"` (length 2).

**Algorithm** (`kmp_failure_table`, lines 961-974):
```
failure = [0; L]
k = 0
for i in 1..L:
    while k > 0 and N[k] ≠ N[i]:
        k = failure[k-1]
    if N[k] == N[i]:
        k += 1
    failure[i] = k
```

**Lemma 1 (KMP failure function correctness).** This is the textbook KMP
algorithm (CLRS Section 32.4). The loop invariant is: `k` equals the length
of the longest proper prefix of `N[0..i]` that is also a suffix. The `while`
loop follows the failure chain to find the longest extension, and the `if`
extends it by one character when possible. This is a well-known correct
algorithm. ∎

#### 2.2 KMP as a DFA: Byte Transition Table

The standard KMP uses a `failure` array and processes bytes with an inner
`while` loop. We can **precompute** this into a full transition table —
turning KMP from a "pushdown" style into a pure DFA with O(1) per byte.

**Definition.** States `{0, 1, ..., L}`:
- State `s` means "the last `s` bytes of input match `N[0..s]`".
- State `L` = ACCEPT: needle found (sticky — stays forever).

**Transition function:**
```
δ_KMP(s, b) :=
    if s == L: return L                  // sticky accept
    t = s
    while t > 0 and N[t] ≠ b:
        t = failure[t-1]
    if N[t] == b: return t + 1
    else: return 0
```

**Precomputation** (`kmp_byte_transitions`, lines 932-958): Evaluates
`δ_KMP(s, b)` for all `(s, b) ∈ {0..L} × {0..255}` and stores the result
in a flat table `T[s][b]`.

**Lemma 2 (Byte transition table correctness).** `T[s][b] = δ_KMP(s, b)`
for all states `s` and bytes `b`.

**Proof.** The code directly evaluates the KMP transition formula:
1. Accept state: `T[L][b] = L` for all `b`. ✓
2. Other states: follows the failure chain from `s` until `N[t] == b`
   (advance to `t+1`) or `t = 0` (stay at 0 or advance to 1). This is
   `δ_KMP` by definition. By Lemma 1, `failure` is correct. ✓ ∎

**Lemma 3 (KMP DFA correctness).** After feeding a byte string
`D = d_0 ... d_{n-1}` through the transition table starting from state 0,
the DFA reaches state `L` iff `D` contains `N` as a substring.

**Proof.** This is the fundamental KMP correctness theorem (Knuth, Morris,
Pratt 1977). The state invariant is: after consuming `d_0 ... d_{i-1}`, the
DFA is in state `s` where `s` is the length of the longest suffix of
`d_0...d_{i-1}` that matches a prefix of `N`. The DFA reaches `L` iff some
suffix of length `L` matches `N[0..L] = N`, i.e., `N` appears as a
substring. The accept state is sticky, so once reached it is never lost. ∎

**This is the baseline correct algorithm.** Everything that follows is about
making it work on FSST codes instead of raw bytes, without decompressing.

---

## Part II: Lifting the DFA to FSST Codes

### 3. FSST Encoding Model

An FSST symbol table consists of:
- **Symbols** `sym[0], ..., sym[K-1]`, each a byte string of length 1-8.
- A distinguished **escape code** `ESC` (= `0xFF`).

A valid FSST code sequence `C = c_0, c_1, ...` is decoded as:
```
decompress(C) :=
    output = []
    i = 0
    while i < |C|:
        if C[i] == ESC:
            output.append(C[i+1])       // literal single byte
            i += 2
        else:
            output.append(sym[C[i]])    // 1-8 byte symbol
            i += 1
    return output
```

**Key insight.** Each code byte deterministically maps to a fixed sequence
of decompressed bytes. This means we can precompute what each code byte
*does to the DFA state* and then run the DFA on the code stream directly.

### 4. Symbol Transition Lifting

**Definition.** For each DFA state `s` and symbol code `c` (where
`sym[c] = b_0 b_1 ... b_{k-1}`), define:

```
sym_trans(s, c) := δ*_KMP(s, sym[c])
                 = δ_KMP(... δ_KMP(δ_KMP(s, b_0), b_1) ..., b_{k-1})
```

i.e., the state reached by feeding the symbol's bytes sequentially through
the KMP automaton.

**Construction** (e.g., lines 500-519):
```rust
let mut s = state;
for &b in &sym[code][..sym_len] {
    if s == accept_state { break; }
    s = byte_table[s][b];
}
sym_trans[state][code] = s;
```

**Lemma 4 (Symbol lifting correctness).** `sym_trans(s, c)` equals the KMP
state after processing the decompressed bytes of symbol `c` starting from
state `s`.

**Proof.** The loop iterates over each byte of `sym[c]` and applies the byte
transition table. By Lemma 2, `byte_table[s][b] = δ_KMP(s, b)`. The loop
computes the iterated composition. The early exit when `s == accept_state`
is safe because accept is sticky: `δ_KMP(L, b) = L` for all `b`, so the
remaining bytes cannot change the state. ∎

### 5. Escape Handling — Sentinel Approach

**Problem.** When code byte `ESC` appears, the next byte is a raw literal,
not a symbol code. The DFA must handle this two-byte sequence.

**Solution.** Define a **sentinel state** `S_esc` (a state value outside the
normal range). Build a "fused" transition table over all 256 byte values:

```
fused_trans(s, c) :=
    if c < K:     sym_trans(s, c)     // symbol code
    if c == ESC:  S_esc               // escape sentinel
    else:         (undefined — can't appear in valid FSST)
```

Plus a separate escape transition table:
```
esc_trans(s, b) := δ_KMP(s, b)       // raw byte transition
```

**The `matches()` loop:**
```
state = 0
for each code byte c in codes:
    next = fused_trans(state, c)
    if next == S_esc:
        b = next code byte            // the literal byte after ESC
        state = esc_trans(state, b)
    else:
        state = next
return state == ACCEPT
```

**Lemma 5 (Sentinel escape correctness).** The sentinel-based DFA correctly
simulates decompression followed by byte-level KMP evaluation.

**Proof.** We prove by induction that after processing each "logical unit"
(a symbol code, or an ESC+literal pair), the DFA state equals the KMP state
after processing the corresponding decompressed bytes.

*Case 1: code `c` is a symbol.* `next = sym_trans(s, c)`. By Lemma 4, this
equals the KMP state after processing the symbol's bytes. Decompression
would emit those same bytes. ✓

*Case 2: code is `ESC` followed by literal `b`.* The DFA reads `ESC`, gets
`S_esc`, then reads `b` and applies `esc_trans(s, b) = δ_KMP(s, b)`.
Decompression would emit the single byte `b`. The DFA applies exactly one
byte transition `δ_KMP(s, b)`. ✓

By induction, the final state matches. ∎

**This is the "simple" FSST DFA.** Structs `ShiftDfa`, `FsstPrefixDfa`, and
`FusedDfa` all use this sentinel approach. They differ only in how the
transition table is stored in memory (an optimization, covered in Part III).

### 6. Prefix DFA on FSST Codes

The prefix DFA (Section 1) is even simpler to lift because prefix matching
has no KMP-style fallback — there's only forward progress or failure.

**Construction** (lines 241-281): For each state `s` and symbol `c`:
- Compare `sym[c][0..cmp]` with `P[s..s+cmp]` where `cmp = min(|sym[c]|, |P|-s)`.
- If they match: next state is `s + cmp` (or ACCEPT if `s + cmp ≥ |P|`).
- If they don't: next state is FAIL.

**Lemma 6 (Prefix symbol lifting).** This correctly computes the prefix DFA
transition for the symbol's decompressed bytes.

**Proof.** Prefix matching is strictly sequential: at state `s`, we need the
next bytes to be exactly `P[s], P[s+1], ...`. A multi-byte symbol either
matches the corresponding segment of the prefix or it doesn't. There is no
partial-match recovery (unlike substring search). The comparison
`sym[0..cmp] == P[s..s+cmp]` checks exactly the right bytes.

If the symbol is longer than the remaining prefix (`|sym[c]| > |P| - s`),
matching only the first `|P| - s` bytes is correct because `P%` matches any
suffix after the prefix. ✓

Escape handling uses the same sentinel approach (Lemma 5), but with the
simpler prefix transition: `esc_trans(s, b) = s+1` if `b == P[s]`, else
FAIL. ∎

### 7. Simulation Theorem (Baseline)

**Theorem 1 (Baseline correctness).** For any valid FSST code sequence `C`
and supported pattern (prefix or contains), the sentinel-based FSST DFA
returns `true` iff the LIKE predicate holds on `decompress(C)`.

**Proof.** Combining the pieces:
1. The underlying matching DFA (prefix or KMP) is correct on plaintext
   (Sections 1, 2). ✓
2. Symbol transitions faithfully simulate feeding the symbol's bytes through
   the DFA (Lemma 4 for contains, Lemma 6 for prefix). ✓
3. Escape handling correctly processes literal bytes (Lemma 5). ✓
4. By induction on the code sequence, the FSST DFA's state after all codes
   equals the plaintext DFA's state after all decompressed bytes. ✓

Therefore: `dfa.matches(codes) = like(decompress(codes), pattern)`. ∎

---

## Part III: Optimizations (Equivalence-Preserving Transforms)

Each optimization below transforms **how** the transition table is stored
or looked up, without changing **what** it computes. We prove each is
semantics-preserving.

### 8. Optimization 1: Shift-Packing Transitions into `u64`

**Used by:** `ShiftDfa`, `FsstPrefixDfa`, `BranchlessShiftDfa`.

**Idea.** When the number of states `S` is small (S ≤ 16), each state
value fits in 4 bits. We can pack all state transitions for a given input
byte into a single `u64`:

**Construction:**
```
for each byte c in 0..255:
    transitions[c] = 0
    for each state s in 0..S-1:
        transitions[c] |= (next_state(s, c) as u64) << (s * 4)
```

**Retrieval:**
```
next = (transitions[c] >> (state * 4)) & 0xF
```

**Lemma 7 (Shift-packing correctness).** The retrieval produces the same
value as `next_state(s, c)` for all states `s` and bytes `c`.

**Proof.**

1. **No overflow:** Each `next_state(s, c)` satisfies
   `0 ≤ next_state(s, c) < 2^4 = 16`. This is enforced by the state count
   constraints:
   - Prefix DFA: `|P| + 2 ≤ 16` (checked at line 63)
   - ShiftDfa: `|N| + 2 ≤ 16` (checked at line 754)
   - BranchlessShiftDfa: `2|N| + 1 ≤ 16` (checked at line 438)

2. **No overlap:** Value for state `s` occupies bits `[4s, 4s+3]`. Value
   for state `s'` occupies bits `[4s', 4s'+3]`. Since `s ≠ s'`, these
   ranges are disjoint. The OR in construction does not interfere across
   states.

3. **Correct extraction:** `(transitions[c] >> (4s)) & 0xF` shifts the
   value for state `s` to the lowest 4 bits and masks off everything else.
   This recovers exactly `next_state(s, c)`.

This is a pure representation change — the function computed is identical.∎

### 9. Optimization 2: Escape Folding into the State Space

**Used by:** `BranchlessShiftDfa`, `FlatBranchlessDfa`.

**Idea.** The sentinel approach requires a branch in the inner loop (check
if `next == S_esc`). We can eliminate this branch by **doubling the state
space** to encode "I just saw ESC" as a state rather than a sentinel.

**State space:** For needle length `N`:
- States `0, ..., N-1`: normal KMP match-progress states.
- State `N`: ACCEPT (sticky).
- States `N+1, ..., 2N`: escape states. State `s + N + 1` means "was in
  normal state `s`, just consumed an ESC byte."

Total: `2N + 1` states.

**Transitions:**
```
For normal state s (0 ≤ s < N):
    folded(s, ESC)            = s + N + 1       // enter escape state
    folded(s, c < K)          = sym_trans(s, c)  // symbol transition
For accept state N:
    folded(N, any)            = N               // sticky
For escape state s+N+1 (0 ≤ s < N):
    folded(s+N+1, b)          = δ_KMP(s, b)     // byte-level KMP transition
```

**The branchless `matches()` loop:**
```
state = 0
for each byte c in codes:
    state = folded(state, c)
return state == ACCEPT
```

No branches. Every code byte consumes exactly one table lookup.

**Lemma 8 (Escape folding correctness).** The folded DFA produces the same
final accept/reject as the sentinel DFA.

**Proof.** Define a projection from folded states to logical KMP states:
```
proj(q) :=
    if 0 ≤ q ≤ N:   q           // normal or accept
    if N < q ≤ 2N:  q - N - 1   // escape state → the pre-escape KMP state
```

We prove by induction that after processing each pair of code bytes that
constitute a "logical unit" in the sentinel DFA, the projected state of the
folded DFA matches.

*Sub-case A: Normal state `s`, code `c ≠ ESC`.*
Folded: `state → sym_trans(s, c)`. This is a normal state.
Sentinel: `state → sym_trans(s, c)`.
Same result. ✓

*Sub-case B: Normal state `s`, code `ESC`, then literal `b`.*
Folded (two steps):
1. `state → s + N + 1` (escape state; `proj = s`)
2. `state → δ_KMP(s, b)` (normal state)

Sentinel (one logical step):
- Reads ESC, gets sentinel, reads `b`, applies `esc_trans(s, b) = δ_KMP(s, b)`.

Both end in state `δ_KMP(s, b)`. ✓

*Sub-case C: Accept.*
Both DFAs stay in ACCEPT regardless of input. ✓

The folded DFA processes the same code stream and reaches the same logical
state after every escape-pair boundary. The final accept check is:
`state == N` in both cases. ∎

### 10. Optimization 3: Flat `u8` Table (Alternative to Shift-Packing)

**Used by:** `FlatBranchlessDfa`, `FusedDfa`.

**Idea.** When there are too many states for 4-bit shift-packing (>16), use
a direct 2D array `transitions[state * 256 + byte] -> u8`.

**Lemma 9 (Flat table correctness).** `transitions[s * 256 + c]` stores
`next_state(s, c)` during construction and retrieves it during lookup.

**Proof.** This is a direct array store and load. The index `s * 256 + c`
is unique for each `(s, c)` pair (since `0 ≤ c < 256`). No packing,
composition, or reinterpretation occurs. ∎

### 11. Optimization 4: Equivalence Classes

**Used by:** `BranchlessShiftDfa`.

**Idea.** Most of the 256 possible code byte values produce identical
transitions for all states. We can group them into **equivalence classes**
and reduce the first-level lookup from 256 entries to ~6-10.

**Definition.** Bytes `a` and `b` are equivalent iff
`transitions_1b[a] == transitions_1b[b]` (their packed `u64` transition
values are bitwise identical).

**Construction** (lines 447-461): For each byte, find or create a class with
the same packed `u64`. Store the mapping in `eq_class[byte] -> class_id`.

**Lemma 10 (Equivalence class correctness).** For any state `s` and any
byte `c`:
```
extract(class_rep[eq_class[c]], s) = extract(transitions_1b[c], s)
```

where `class_rep[id]` is the packed `u64` of any representative of class `id`.

**Proof.** By definition of the equivalence relation, all bytes in the same
class have bitwise identical packed `u64`. Therefore the representative's
`u64` is literally the same value as `transitions_1b[c]`. Extracting any
state's transition from it yields the same result. ∎

### 12. Optimization 5: Hierarchical Composition (2-byte and 4-byte)

**Used by:** `BranchlessShiftDfa`.

**Idea.** Instead of applying one state transition per code byte, precompute
the **composed** transition for 2 bytes, then compose pairs of 2-byte
transitions into 4-byte transitions. This processes 4 code bytes per loop
iteration.

**Key mathematical property.** State transition functions can be composed:
if `f(s)` is the transition for input `a` and `g(s)` is the transition for
input `b`, then `g(f(s))` is the transition for input `ab`. Since the state
space is finite and small, the composed function can be represented in the
same packed `u64` format.

#### 12.1 Pair Composition

**Construction** (lines 561-581): For each pair of equivalence classes
`(c0, c1)`:
```
for each state s in 0..total_states:
    mid   = extract(class_rep[c0], s)    // state after first byte
    final = extract(class_rep[c1], mid)  // state after second byte
    packed |= final << (s * BITS)
```

Deduplicate results into a palette `palette_2b`.

**Lemma 11 (Pair composition correctness).** For any starting state `s`:
```
extract(palette_2b[pair_compose[c0 * n + c1]], s)
    = extract(class_rep[c1], extract(class_rep[c0], s))
```

**Proof.** The construction directly computes `g(f(s))` for each `s` and
packs the result. By Lemma 7 (shift-packing), the packed representation is
faithful. By Lemma 10 (equivalence classes), using class representatives is
correct. The deduplication into a palette is purely a space optimization —
identical packed `u64` values are shared, but the value retrieved is the
same. ∎

#### 12.2 Four-Byte Composition

**Construction** (lines 585-601): For each pair of palette entries `(p0, p1)`:
```
for each state s:
    mid   = extract(palette_2b[p0], s)
    final = extract(palette_2b[p1], mid)
    packed |= final << (s * BITS)
```

**Lemma 12 (Four-byte composition correctness).** For any starting state `s`:
```
extract(compose_4b[p0 * m + p1], s) =
    δ(δ(δ(δ(s, c0), c1), c2), c3)
```

where `(c0, c1)` maps to palette index `p0` and `(c2, c3)` maps to `p1`.

**Proof.** By Lemma 11, `palette_2b[p0]` correctly computes the 2-byte
composed transition for `(c0, c1)`, and `palette_2b[p1]` for `(c2, c3)`.
The 4-byte composition applies them in sequence: first `p0`, then `p1`.
This is function composition, which is associative:
```
(δ_{c3} ∘ δ_{c2}) ∘ (δ_{c1} ∘ δ_{c0}) = δ_{c3} ∘ δ_{c2} ∘ δ_{c1} ∘ δ_{c0}
```
The packed representation is faithful by Lemma 7. ∎

#### 12.3 The Full Lookup Chain

The `finish_tail` method (lines 605-638) processes codes in chunks of 4:
```
ec0 = eq_class[c0]      // Lemma 10: lossless
ec1 = eq_class[c1]
ec2 = eq_class[c2]
ec3 = eq_class[c3]
p0 = pair_compose[ec0 * n_classes + ec1]    // Lemma 11: correct 2-byte composition
p1 = pair_compose[ec2 * n_classes + ec3]
packed = compose_4b[p0 * n_palette + p1]    // Lemma 12: correct 4-byte composition
state = extract(packed, state)              // Lemma 7: correct extraction
```

Each step is proven correct. The chain computes
`δ(δ(δ(δ(state, c0), c1), c2), c3)` — the same result as processing all
four bytes individually. ✓

Remainder bytes (1-3 trailing) use `palette_2b` (Lemma 11) and
`transitions_1b` (Lemma 7) directly. ✓ ∎

---

## Part IV: Putting It All Together

### 13. DFA Variant Map

The code selects a DFA implementation based on needle length. Each variant
combines a subset of the optimizations proven above:

| Variant | Needle len | Escape handling | Storage | Composition | Correctness |
|---|---|---|---|---|---|
| `FsstPrefixDfa` | prefix ≤ 14 | Sentinel (Lemma 5) | Shift-packed (Lemma 7) | None | Lemma 6 + 5 + 7 |
| `BranchlessShiftDfa` | 1-7 | Folded (Lemma 8) | Shift-packed (Lemma 7) | 4-byte (Lemma 12) | Lemma 4 + 8 + 7 + 10-12 |
| `FlatBranchlessDfa` | 8-14 | Folded (Lemma 8) | Flat u8 (Lemma 9) | None | Lemma 4 + 8 + 9 |
| `ShiftDfa` | 8-14 | Sentinel (Lemma 5) | Shift-packed (Lemma 7) | None | Lemma 4 + 5 + 7 |
| `FusedDfa` | 15+ | Sentinel (Lemma 5) | Flat u8 (Lemma 9) | None | Lemma 4 + 5 + 9 |

Every variant is a composition of independently proven correct
transformations applied to the same baseline algorithm (Part I → Part II →
Part III). ✓

### 14. Bit-Packing Scan

**Lemma 13.** `dfa_scan_to_bitbuf` correctly packs per-string match results
into a `BitBuffer`.

**Proof.** The function processes strings in groups of 64:
```rust
for each group of 64 strings:
    word = 0
    for bit in 0..64:
        codes = all_bytes[off[base+bit] .. off[base+bit+1]]
        match_result = matcher(codes) XOR negated
        word |= (match_result as u64) << bit
    words.push(word)
```

- Each string's code bytes are correctly sliced using the offsets array
  (standard VarBin layout).
- XOR with `negated` implements `NOT LIKE`.
- Bit `i` in the word corresponds to string `base + i` (standard BitBuffer
  layout).
- The remainder loop handles the final `n % 64` strings identically. ∎

### 15. Main Theorem

**Theorem (End-to-End Correctness).** For any valid FSSTArray `A`, constant
pattern `P` of supported form (`prefix%` or `%needle%`), and `LikeOptions`
with `case_insensitive = false`:

```
FSST::like(A, P, opts) = canonical_like(decompress(A), P, opts)
```

**Proof.**

1. **Pattern classification** (`LikeKind::parse`): Correctly identifies
   `prefix%` and `%needle%` patterns by direct syntactic check. Returns
   `None` (falls back to decompression) for anything else. ✓

2. **DFA construction**: Uses the baseline DFA (Part I), lifted to FSST
   codes (Part II), with storage optimizations (Part III). Each step is
   proven correct by the corresponding lemma. ✓

3. **Scanning** (Lemma 13): Correctly applies the DFA to each string's
   code bytes and packs results. ✓

4. **Validity**: Null handling is delegated to the validity mask from the
   codes array, unioned with the pattern's nullability. This is correct
   because FSST delegates validity to its codes child array, and a null
   pattern produces null output (standard SQL LIKE semantics). ✓

Therefore the output `BoolArray` matches what decompression + standard LIKE
would produce, for every row. ∎

---

## 16. Preconditions

The following preconditions are required and enforced:

| Precondition | Where Enforced | Why Needed |
|---|---|---|
| `prefix.len() + 2 ≤ 16` | Line 63 (early return) | Shift-packing needs all states in 4 bits |
| `needle.len() ≤ 7` for `BranchlessShiftDfa` | `debug_assert` line 438 | `2N+1 ≤ 16` for 4-bit packing with escape folding |
| `needle.len() ≤ 14` for `FlatBranchlessDfa` | `debug_assert` line 664 | `2N+1 ≤ 29` fits in `u8`; table fits L1 |
| `needle.len() ≤ 14` for `ShiftDfa` | `debug_assert` line 754 | `N+2 ≤ 16` for 4-bit packing with sentinel |
| `case_insensitive = false` | Line 36 (early return) | DFA operates on exact byte values |
| Pattern has no `_` wildcards | Line 183 (early return) | DFA cannot match single-character wildcards |
| Symbol codes < `K` | FSST encoding invariant | Fused table only has valid entries for codes < K |
| `ESC` is not a valid symbol code | FSST encoding invariant | Sentinel/escape handling assumes ESC is distinguishable |

---

## 17. Summary: The Chain of Equivalences

```
Standard LIKE on plaintext
    ≡  Prefix DFA / KMP automaton on raw bytes          (Sections 1-2)
    ≡  Symbol-lifted DFA on FSST codes + sentinel ESC   (Sections 4-6)
    ≡  Shift-packed u64 representation                  (Section 8, Lemma 7)
    ≡  Escape-folded state space (branchless)            (Section 9, Lemma 8)
    ≡  Flat u8 table (for large state counts)            (Section 10, Lemma 9)
    ≡  Equivalence-class compression                     (Section 11, Lemma 10)
    ≡  Hierarchical 2-byte / 4-byte composition          (Section 12, Lemmas 11-12)
```

Each `≡` is a proven equivalence-preserving transformation. The composition
of all applicable transformations (selected per DFA variant) gives an
end-to-end correct implementation. ∎
