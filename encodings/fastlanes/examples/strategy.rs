// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Per-(T, W) strategy analysis for FastLanes packed-block eq-with-constant.
//!
//! For each (input type, packed bit-width) pair this tool computes:
//!   - The packed-block layout (LANES, rows, packed bytes per block)
//!   - The "natural SIMD granularity" for direct element-wise cmpeq
//!   - The structural cost (boundary-row count, elements per SIMD lane,
//!     load count per block)
//!   - The recommended kernel family from this bench suite, plus an
//!     estimated peak throughput on AVX-512 hardware.
//!
//! Run with: `cargo run --release --example strategy -p vortex-fastlanes`

#![expect(clippy::print_stdout, reason = "this is a CLI analysis tool")]

const TYPES: &[(&str, usize)] = &[("u8", 8), ("u16", 16), ("u32", 32), ("u64", 64)];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Strategy {
    /// W = T: just compare the whole word; one cmpeq per element.
    Identity,
    /// Per-W-bit field XOR with broadcast mask; for W=1 + appropriate
    /// constant the answer is the bits themselves (no compare needed).
    BitXor,
    /// W ∈ {8, 16, 32, 64}: direct vpcmpeqb/vpcmpeqw/vpcmpeqd/vpcmpeqq on
    /// the packed buffer, treating each byte/word/dword/qword as one
    /// element. ~`64/W` elements per zmm cmpeq op.
    NaturalCmpeq(usize), // granularity in bits
    /// W < natural-granularity but W divides T evenly: XOR + smear + extract
    /// (~log2(W) shifts) within each lane, then horizontal compress.
    /// Multiple elements per SIMD lane.
    SmearExtract,
    /// W doesn't divide T cleanly (or boundary-heavy widths): the v4
    /// row-major splat-cmp algorithm — one row at a time, per-row
    /// vpternlogd + vpcmpeqd → kmask. ~32 rows × 2 chunks per block.
    RowMajorSplat,
}

impl Strategy {
    fn name(&self) -> &'static str {
        match self {
            Strategy::Identity => "identity",
            Strategy::BitXor => "bit-xor",
            Strategy::NaturalCmpeq(8) => "v6_byte (vpcmpeqb)",
            Strategy::NaturalCmpeq(16) => "v6_word (vpcmpeqw)",
            Strategy::NaturalCmpeq(32) => "v6_dword (vpcmpeqd)",
            Strategy::NaturalCmpeq(64) => "v6_qword (vpcmpeqq)",
            Strategy::NaturalCmpeq(_) => "v6_other",
            Strategy::SmearExtract => "v6_smear",
            Strategy::RowMajorSplat => "v4 (row-major splat)",
        }
    }
}

/// Returns true iff `W` divides `T` evenly (every row aligns to a packed-word boundary).
fn divides(w: usize, t: usize) -> bool {
    t % w == 0
}

/// Number of rows per block where the W-bit field spans a packed-word boundary.
fn boundary_rows(w: usize, t: usize) -> usize {
    (0..t).filter(|&row| (row * w) % t + w > t).count()
}

/// Pick a recommended strategy for (T = `t_bits`, W).
fn recommend(t_bits: usize, w: usize) -> Strategy {
    if w == t_bits {
        return Strategy::Identity;
    }
    if w == 1 {
        return Strategy::BitXor;
    }
    // Natural granularity: W ∈ {8, 16, 32, 64} (matches a SIMD element size).
    if matches!(w, 8 | 16 | 32 | 64) && w <= t_bits {
        return Strategy::NaturalCmpeq(w);
    }
    // Sub-natural-but-divides: W ∈ {2, 4} packs multiple per byte.
    if divides(w, 8) {
        return Strategy::SmearExtract;
    }
    // Everything else: row-major splat (v4 style).
    Strategy::RowMajorSplat
}

/// Rough peak-throughput estimate (Gitem/s) on Sapphire Rapids-class AVX-512.
/// Anchored to measured u32 numbers from the bench suite, extrapolated by
/// "elements per zmm SIMD op" for other types and strategies.
fn estimate_peak_gitem_s(t_bits: usize, w: usize, strat: Strategy) -> f64 {
    let lanes = 1024 / t_bits;
    // Measured anchors for u32 (T=32):
    //   v6_w8  → 20 Gitem/s    (1 byte per element, 64 elem/zmm cmpeq)
    //   v6_w16 → 24 Gitem/s    (1 word per element, 32 elem/zmm cmpeq)
    //   v4 W=1..16 (divides): 22-24 Gitem/s
    //   v4 W∈{3,5,7,11,...} (non-div): 10-16 Gitem/s
    //   v4 W∈{17,23,29}: 8-11 Gitem/s
    let boundary_frac = boundary_rows(w, t_bits) as f64 / t_bits as f64;
    match strat {
        Strategy::Identity => 50.0, // pure copy + cmpeq; ~memory-bound
        Strategy::BitXor => 60.0,   // pure bit op; very fast
        Strategy::NaturalCmpeq(g) => {
            // 64/g elements per zmm cmpeq. Higher elem/zmm = higher Gitem/s.
            // u32 anchor: g=8 → 20, g=16 → 24, g=32 → 22. Scale by (64/g)/2.
            let base = match g {
                8 => 20.0,
                16 => 24.0,
                32 => 22.0,
                64 => 16.0,
                _ => 18.0,
            };
            base * (lanes as f64 / 32.0).sqrt() // scales mildly with LANES
        }
        Strategy::SmearExtract => {
            // Need to extract `8/w` bits per byte. Cost: ~log2(w) shifts + mask + extract.
            // Estimate: ~30% slower than NaturalCmpeq(8).
            14.0 * (lanes as f64 / 128.0).sqrt()
        }
        Strategy::RowMajorSplat => {
            // v4 throughput drops with boundary fraction.
            let base_no_bdry = 24.0;
            let bdry_floor = 9.0;
            base_no_bdry - (base_no_bdry - bdry_floor) * boundary_frac
        }
    }
}

fn print_table_for_type(name: &str, t_bits: usize) {
    println!("\n## {name} (T = {t_bits} bits, LANES = {lanes})\n",
             lanes = 1024 / t_bits);
    println!("| W  | divides T? | boundary rows | elems/zmm cmpeq | strategy              | est. Gitem/s |");
    println!("|---:|:----------:|:-------------:|:---------------:|:----------------------|-------------:|");

    for w in 1..=t_bits {
        let divs = divides(w, t_bits);
        let bdry = boundary_rows(w, t_bits);
        let strat = recommend(t_bits, w);
        let est = estimate_peak_gitem_s(t_bits, w, strat);
        let elems_per_zmm = match strat {
            Strategy::NaturalCmpeq(g) => format!("{}", 512 / g),
            Strategy::Identity => format!("{}", 512 / t_bits),
            Strategy::BitXor => "512".to_string(),
            Strategy::SmearExtract => format!("{}", 64 * (8 / w)),
            Strategy::RowMajorSplat => format!("{}", 512 / t_bits),
        };
        println!(
            "| {:>2} | {:^10} | {:^13} | {:^15} | {:<21} | {:>10.1} |",
            w,
            if divs { "yes" } else { "no" },
            bdry,
            elems_per_zmm,
            strat.name(),
            est,
        );
    }
}

fn print_summary() {
    println!("\n## Summary: when to use each kernel\n");
    println!("- **Identity (W == T)**: full-width type; just compare directly. Maximum throughput.");
    println!("- **BitXor (W = 1)**: result bits are derived directly from packed bits by XOR with a broadcast constant. ~60 Gitem/s.");
    println!("- **v6_byte (W = 8)**: vpcmpeqb against broadcast(C), 64 elements per zmm op. Per-row PEXT extraction.");
    println!("- **v6_word (W = 16)**: vpcmpeqw, 32 elements per zmm op.");
    println!("- **v6_dword (W = 32)**: vpcmpeqd, 16 elements per zmm op.");
    println!("- **v6_qword (W = 64)**: vpcmpeqq, 8 elements per zmm op.");
    println!("- **SmearExtract (W ∈ {{2, 4}})**: XOR + nibble/2-bit-field smear + horizontal extract.");
    println!("- **RowMajorSplat (other W)**: v4 algorithm — best for non-divides and high-boundary widths.");
}

fn print_potential_optimizations() {
    println!("\n## Potential further optimizations (not yet implemented)\n");
    println!("1. **W=1 BitXor path**: trivially equals `packed_array ^ broadcast(C)` plus the FastLanes");
    println!("   bit-transpose. ~60 Gitem/s achievable vs v4's 24 Gitem/s.");
    println!();
    println!("2. **SmearExtract for W=2, W=4**: e.g., for W=4 on u32 each byte holds 2 nibbles. XOR with");
    println!("   broadcast(C|C<<4), then per-nibble OR-smear, then PEXT 2 bits per byte. Saves vs v4 because");
    println!("   we process 128 elements per zmm op (vs 32 in v4's per-row pattern).");
    println!();
    println!("3. **Shared loads across boundary rows**: at non-divides W (e.g., W=12), adjacent");
    println!("   rows often touch the same pair of packed words. Hoist loads so each curr_word is");
    println!("   loaded once per block, not once per boundary row. Could cut load count by ~3x at W=12.");
    println!();
    println!("4. **Wider types (u8, u16) with AVX-512**: u8 has LANES=128 → 4 chunks of 32 u8 per zmm");
    println!("   for v4-style row processing. u16 has LANES=64 → 2 chunks. Throughput should scale");
    println!("   roughly with sqrt(LANES) due to better pipelining.");
    println!();
    println!("5. **AVX-512 VBMI bit-permute**: vpmultishiftqb / vpermt2b could collapse the smear + extract");
    println!("   into 1-2 SIMD ops for W ∈ {{2, 4}} instead of scalar PEXT per zmm.");
}

fn main() {
    println!("# FastLanes packed-block eq-with-constant: per-(T, W) strategy analysis");
    println!();
    println!("All rows are for a single 1024-element packed FastLanes block.");
    println!("Block size in bytes = 32 * W / 8 = 4W bytes for u32 inputs, scaled by sizeof(T).");
    println!("Numbers are estimates; measured anchors at u32 from the `bitpack_constant` bench suite.");

    for &(name, t_bits) in TYPES {
        print_table_for_type(name, t_bits);
    }

    print_summary();
    print_potential_optimizations();

    println!("\n## FastLanes layout cheatsheet\n");
    println!("- Block size: always 1024 elements regardless of T.");
    println!("- LANES = 1024 / T = number of independent SIMD-friendly columns.");
    println!("- Per lane: 1024 / LANES = T rows of one W-bit element each.");
    println!("- Packed storage: W * LANES T-bit words per block (32 * W for T=32).");
    println!("- Logical index of element at (row, lane): index(r, l) = FL_ORDER[r/8]*16 + (r%8)*128 + l");
    println!("  (for T=32 and the standard FL_ORDER permutation [0, 4, 2, 6, 1, 5, 3, 7]).");
    println!("- Boundary row: shift + W > T, where shift = (row * W) % T.");
    println!("  Boundary rows touch two packed words instead of one; doubles load count for that row.");
}
