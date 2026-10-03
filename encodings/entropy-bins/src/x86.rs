// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! AVX-512 kernels. Every kernel produces exactly what the portable decoder in `decode.rs` does.
//!
//! Vector loads deliberately read past a block's words and offsets; the array's data buffer ends
//! with [`TAIL_PADDING`](crate::coder::TAIL_PADDING) zero bytes, which covers every over-read.

// Lane values are bit fields and table indices: the narrowing and sign casts are intended.
// Single-letter names follow the ANS literature: x is a coder state, s the state log.
#![allow(
    clippy::many_single_char_names,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]

use std::arch::asm;
use std::arch::x86_64::*;
use std::sync::LazyLock;

use crate::coder::IdTable;
use crate::coder::LANES;
use crate::coder::MAX_LAG;
use crate::decode::BlockView;
use crate::decode::ChunkDecoder;
use crate::decode::OutInt;

static HAS_AVX512: LazyLock<bool> = LazyLock::new(|| {
    is_x86_feature_detected!("avx512f")
        && is_x86_feature_detected!("avx512bw")
        && is_x86_feature_detected!("avx512vl")
        && is_x86_feature_detected!("avx512dq")
        && is_x86_feature_detected!("avx512cd")
        && is_x86_feature_detected!("avx512vbmi")
        && is_x86_feature_detected!("avx512vbmi2")
});

thread_local! {
    static FORCE_SCALAR: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Whether the AVX-512 kernels may run on this CPU (and are not disabled for testing).
pub(crate) fn has_avx512() -> bool {
    *HAS_AVX512 && !FORCE_SCALAR.with(|f| f.get())
}

/// Disable the AVX-512 kernels on this thread (tests compare them with the portable decoder).
#[cfg(test)]
pub(crate) fn set_force_scalar(v: bool) {
    FORCE_SCALAR.with(|f| f.set(v));
}

// Variable shifts as inline asm: LLVM otherwise guards them with compares for counts >= the lane
// width, which cannot occur here and cost a shuffle-port uop each.
#[inline]
#[target_feature(enable = "avx512f")]
fn sllv32(a: __m512i, c: __m512i) -> __m512i {
    let r;
    // SAFETY: a register-only AVX-512F instruction with no memory or flag effects.
    unsafe {
        asm!("vpsllvd {r}, {a}, {c}", r = lateout(zmm_reg) r, a = in(zmm_reg) a, c = in(zmm_reg) c, options(pure, nomem, nostack));
    }
    r
}

#[inline]
#[target_feature(enable = "avx512f")]
fn srlv32(a: __m512i, c: __m512i) -> __m512i {
    let r;
    // SAFETY: a register-only AVX-512F instruction with no memory or flag effects.
    unsafe {
        asm!("vpsrlvd {r}, {a}, {c}", r = lateout(zmm_reg) r, a = in(zmm_reg) a, c = in(zmm_reg) c, options(pure, nomem, nostack));
    }
    r
}

/// Decode the bin ids of `V` blocks in lockstep (all with the same length), each into a buffer
/// of at least the block length rounded up to a multiple of 16.
///
/// # Safety
///
/// The CPU must support the features checked by [`has_avx512`]; every `outs[v]` must point to
/// at least that many writable bytes; every block must be non-uniform and come from a data
/// buffer with tail padding.
pub(crate) unsafe fn ids16<const V: usize>(
    t: &IdTable,
    blocks: [&BlockView<'_>; V],
    outs: [*mut u8; V],
    limit: usize,
) {
    // SAFETY: forwarded from the caller.
    unsafe {
        match t.r {
            8 => ids16_kernel::<V, 8>(t, blocks, outs, limit),
            5 => ids16_kernel::<V, 5>(t, blocks, outs, limit),
            4 => ids16_kernel::<V, 4>(t, blocks, outs, limit),
            3 => ids16_kernel::<V, 3>(t, blocks, outs, limit),
            _ => ids16_kernel::<V, 2>(t, blocks, outs, limit),
        }
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512dq,avx512cd,avx512vbmi,avx512vbmi2")]
unsafe fn ids16_kernel<const V: usize, const R: usize>(
    t: &IdTable,
    blocks: [&BlockView<'_>; V],
    outs: [*mut u8; V],
    limit: usize,
) {
    // SAFETY: the tables are 128-byte arrays; word loads stay within the padded data buffer and
    // stores within the caller's block-sized buffers (see the function's safety section).
    unsafe {
        let ld = |a: &[u8; 128]| {
            (
                _mm512_loadu_si512(a.as_ptr().cast()),
                _mm512_loadu_si512(a.as_ptr().add(64).cast()),
            )
        };
        let (s0, s1) = ld(&t.sym_tab);
        let (x0, x1) = ld(&t.xs_tab);
        let ff = _mm512_set1_epi32(0xff);
        let ones = _mm512_set1_epi32(-1);
        let c16 = _mm512_set1_epi32(16);
        let kk = _mm512_set1_epi32(31 - t.s as i32);
        let ll = _mm512_set1_epi32(1i32 << t.s);
        let mut p = [std::ptr::null::<u16>(); V];
        let mut buf = [_mm512_setzero_si512(); V];
        let mut avail = [_mm512_setzero_si512(); V];
        let mut x = [ll; V];
        let mut stop = [ll; V];
        for v in 0..V {
            p[v] = blocks[v].words.as_ptr().cast();
            x[v] = _mm512_add_epi32(
                _mm512_cvtepu8_epi32(_mm_loadu_si128(blocks[v].states.as_ptr().cast())),
                ll,
            );
            stop[v] = _mm512_cvtepu8_epi32(_mm_loadu_si128(blocks[v].stop.as_ptr().cast()));
        }
        let steps = blocks[0].n.min(limit).div_ceil(LANES);
        let mut step = 0;
        while step < steps {
            let rv = _mm512_set1_epi32((step / R) as i32);
            for v in 0..V {
                let live = _mm512_cmplt_epu32_mask(rv, stop[v]);
                let need = _mm512_mask_cmplt_epu32_mask(live, avail[v], c16);
                let words = _mm512_cvtepu16_epi32(_mm256_loadu_si256(p[v].cast()));
                let nw = _mm512_maskz_expand_epi32(need, words);
                buf[v] = _mm512_or_si512(buf[v], sllv32(nw, avail[v]));
                avail[v] = _mm512_mask_add_epi32(avail[v], need, avail[v], c16);
                p[v] = p[v].add(need.count_ones() as usize);
            }
            let end = (step + R).min(steps);
            while step < end {
                for v in 0..V {
                    let xv = x[v];
                    let sym = _mm512_permutex2var_epi8(s0, xv, s1);
                    let xs = _mm512_and_si512(_mm512_permutex2var_epi8(x0, xv, x1), ff);
                    let nb = _mm512_sub_epi32(_mm512_lzcnt_epi32(xs), kk);
                    let bits = _mm512_andnot_si512(sllv32(ones, nb), buf[v]);
                    buf[v] = srlv32(buf[v], nb);
                    avail[v] = _mm512_sub_epi32(avail[v], nb);
                    x[v] = _mm512_or_si512(sllv32(xs, nb), bits);
                    _mm_storeu_si128(outs[v].add(step * LANES).cast(), _mm512_cvtepi32_epi8(sym));
                }
                step += 1;
            }
        }
    }
}

/// Merge a decoded block's ids with its offsets into `out`, if a vector kernel applies.
/// Returns `false` when the caller must use the portable merge. `seeds` is as for
/// [`merge_scalar`](crate::decode::merge_scalar).
pub(crate) fn merge<T: OutInt>(
    d: &ChunkDecoder,
    b: &BlockView<'_>,
    ids: &[u8],
    out: &mut [T],
    seeds: &[u64],
) -> bool {
    let nb = d.widths.len();
    let n = b.n;
    if out.len() < n || ids.len() < n.div_ceil(LANES) * LANES + 16 || seeds.len() > MAX_LAG {
        return false;
    }
    let o = out.as_mut_ptr().cast::<u8>();
    macro_rules! go_out {
        ($kernel:ident, $tb:literal, $prefix:literal) => {
            match T::BYTES {
                // SAFETY: AVX-512 is available (checked by the caller), `out` holds `n` values,
                // `ids` holds the block's ids plus slack, and offsets come from a padded buffer.
                1 => unsafe { $kernel::<$tb, 1, $prefix>(d, ids, b.offsets, n, o, seeds) },
                2 => unsafe { $kernel::<$tb, 2, $prefix>(d, ids, b.offsets, n, o, seeds) },
                4 => unsafe { $kernel::<$tb, 4, $prefix>(d, ids, b.offsets, n, o, seeds) },
                _ => unsafe { $kernel::<$tb, 8, $prefix>(d, ids, b.offsets, n, o, seeds) },
            }
        };
    }
    macro_rules! go {
        ($kernel:ident, $tb:literal) => {
            if seeds.is_empty() {
                go_out!($kernel, $tb, false)
            } else {
                go_out!($kernel, $tb, true)
            }
        };
    }
    if T::BYTES <= 4 && nb <= 32 && d.max_width <= 31 {
        if nb <= 16 {
            go!(merge16, 0);
        } else {
            go!(merge16, 1);
        }
        return true;
    }
    if nb <= 64 && d.max_width <= 62 {
        if nb <= 16 {
            go!(merge8, 0);
        } else if nb <= 32 {
            go!(merge8, 1);
        } else {
            go!(merge8, 2);
        }
        return true;
    }
    false
}

/// Index vectors and masks for an in-register lag-`k` prefix sum over `L` lanes: step `t` adds
/// lane `j - (k << t)` into lane `j`, and the carry broadcasts each lane's last earlier value in
/// its residue class (`L - k + j % k`) from the previous vector.
struct LagPlan<const L: usize> {
    shift_idx: [[u32; L]; 4],
    shift_mask: [u32; 4],
    carry_idx: [u32; L],
    seeds: [u64; L],
    seed_mask: u32,
}

impl<const L: usize> LagPlan<L> {
    fn new(seeds: &[u64]) -> Self {
        let lag = seeds.len().max(1);
        let mut plan = Self {
            shift_idx: [[0; L]; 4],
            shift_mask: [0; 4],
            carry_idx: [0; L],
            seeds: [0; L],
            seed_mask: (1u32 << seeds.len()) - 1,
        };
        plan.seeds[..seeds.len()].copy_from_slice(seeds);
        for t in 0..4 {
            let s = lag << t;
            if s < L {
                for j in s..L {
                    plan.shift_idx[t][j] = (j - s) as u32;
                }
                plan.shift_mask[t] = ((1u32 << L) - 1) & !((1u32 << s) - 1);
            }
        }
        for j in 0..L {
            plan.carry_idx[j] = (L - lag + j % lag) as u32;
        }
        plan
    }
}

/// 8 values per step in u64 lanes: widths from a byte table, exclusive bit positions from one
/// `vpsadbw` over lane-masked widths, offsets from a 64-byte window with a funnel shift.
/// Requires <= 64 bins and widths <= 62 (8 offsets then fit one window).
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512dq,avx512cd,avx512vbmi,avx512vbmi2")]
unsafe fn merge8<const TB: u8, const OUT: usize, const PREFIX: bool>(
    d: &ChunkDecoder,
    ids: &[u8],
    offsets: &[u8],
    n: usize,
    out: *mut u8,
    seeds: &[u64],
) {
    // SAFETY: table loads read fixed-size arrays; id and offset loads stay within the caller's
    // padded buffers; stores are masked to `n` values.
    unsafe {
        let wtab = _mm512_loadu_si512(d.wtab.as_ptr().cast());
        let lt: [__m512i; 4] =
            std::array::from_fn(|r| _mm512_loadu_si512(d.tl64.as_ptr().add(8 * r).cast()));
        let mt: [__m512i; 2] =
            std::array::from_fn(|r| _mm512_loadu_si512(d.mask64.as_ptr().add(8 * r).cast()));
        let lmask = _mm512_set_epi64(
            0x00ff_ffff_ffff_ffff,
            0x0000_ffff_ffff_ffff,
            0x0000_00ff_ffff_ffff,
            0x0000_0000_ffff_ffff,
            0x0000_0000_00ff_ffff,
            0x0000_0000_0000_ffff,
            0x0000_0000_0000_00ff,
            0,
        );
        let shifts = _mm512_set_epi64(56, 48, 40, 32, 24, 16, 8, 0);
        let zero = _mm512_setzero_si512();
        let ones = _mm512_set1_epi64(-1);
        let ff = _mm512_set1_epi64(0xff);
        let seven = _mm512_set1_epi64(7);
        let one = _mm512_set1_epi64(1);
        let gp = d.tl.as_ptr().cast::<i64>();
        let bp = offsets.as_ptr();
        let ip = ids.as_ptr();
        let plan = LagPlan::<8>::new(seeds);
        let widen = |a: &[u32; 8]| _mm512_cvtepu32_epi64(_mm256_loadu_si256(a.as_ptr().cast()));
        let sh: [__m512i; 3] = std::array::from_fn(|t| widen(&plan.shift_idx[t]));
        let shm: [u8; 3] = std::array::from_fn(|t| plan.shift_mask[t] as u8);
        let cidx = widen(&plan.carry_idx);
        let seedv = _mm512_loadu_si512(plan.seeds.as_ptr().cast());
        let mut carry = zero;
        let mut basev = zero;
        let mut i = 0;
        while i < n {
            let b = _mm512_set1_epi64(std::ptr::read_unaligned(ip.add(i).cast::<i64>()));
            let bw = _mm512_permutexvar_epi8(b, wtab);
            let ex = _mm512_sad_epu8(_mm512_and_si512(bw, lmask), zero);
            let tot = _mm512_sad_epu8(bw, zero);
            let base = _mm_cvtsi128_si64(_mm512_castsi512_si128(basev)) as usize;
            let win = _mm512_loadu_si512(bp.add(base >> 3).cast());
            let bit = _mm512_add_epi64(_mm512_and_si512(basev, seven), ex);
            let q = _mm512_srli_epi64::<6>(bit);
            let lo = _mm512_permutexvar_epi64(q, win);
            let hi = _mm512_permutexvar_epi64(_mm512_add_epi64(q, one), win);
            let r = _mm512_shrdv_epi64(lo, hi, bit);
            let idq = _mm512_srlv_epi64(b, shifts);
            let r = if TB == 0 {
                _mm512_and_si512(r, _mm512_permutex2var_epi64(mt[0], idq, mt[1]))
            } else {
                let w = _mm512_and_si512(_mm512_srlv_epi64(bw, shifts), ff);
                _mm512_andnot_si512(_mm512_sllv_epi64(ones, w), r)
            };
            let lv = match TB {
                0 => _mm512_permutex2var_epi64(lt[0], idq, lt[1]),
                1 => {
                    let a = _mm512_permutex2var_epi64(lt[0], idq, lt[1]);
                    let h = _mm512_permutex2var_epi64(lt[2], idq, lt[3]);
                    _mm512_mask_blend_epi64(
                        _mm512_test_epi64_mask(idq, _mm512_set1_epi64(16)),
                        a,
                        h,
                    )
                }
                _ => _mm512_i64gather_epi64::<8>(_mm512_and_si512(idq, ff), gp),
            };
            let mut o = _mm512_add_epi64(lv, r);
            if PREFIX {
                // The block's first `lag` values are its seeds; then an in-register lag prefix
                // sum plus the previous vector's carries.
                if i == 0 {
                    o = _mm512_mask_blend_epi64(plan.seed_mask as u8, o, seedv);
                }
                for t in 0..3 {
                    o = _mm512_add_epi64(o, _mm512_maskz_permutexvar_epi64(shm[t], sh[t], o));
                }
                o = _mm512_add_epi64(o, carry);
                carry = _mm512_permutexvar_epi64(cidx, o);
            }
            let km: u8 = if i + 8 <= n {
                0xff
            } else {
                ((1u32 << (n - i)) - 1) as u8
            };
            let dst = out.add(i * OUT);
            match OUT {
                8 => _mm512_mask_storeu_epi64(dst.cast(), km, o),
                4 => _mm512_mask_cvtepi64_storeu_epi32(dst.cast(), km, o),
                2 => _mm512_mask_cvtepi64_storeu_epi16(dst.cast(), km, o),
                _ => _mm512_mask_cvtepi64_storeu_epi8(dst.cast(), km, o),
            }
            basev = _mm512_add_epi64(basev, tot);
            i += 8;
        }
    }
}

/// 16 values per step in u32 lanes for outputs of at most 32 bits; arithmetic wraps mod 2^32,
/// which is exact after truncation. Requires <= 32 bins and widths <= 31.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512dq,avx512cd,avx512vbmi,avx512vbmi2")]
unsafe fn merge16<const TB: u8, const OUT: usize, const PREFIX: bool>(
    d: &ChunkDecoder,
    ids: &[u8],
    offsets: &[u8],
    n: usize,
    out: *mut u8,
    seeds: &[u64],
) {
    // SAFETY: as for `merge8`.
    unsafe {
        let lt0 = _mm512_loadu_si512(d.tl32.as_ptr().cast());
        let lt1 = _mm512_loadu_si512(d.tl32.as_ptr().add(16).cast());
        let wt0 = _mm512_loadu_si512(d.w32.as_ptr().cast());
        let wt1 = _mm512_loadu_si512(d.w32.as_ptr().add(16).cast());
        let zero = _mm512_setzero_si512();
        let ones = _mm512_set1_epi32(-1);
        let seven = _mm512_set1_epi32(7);
        let one = _mm512_set1_epi32(1);
        let last = _mm512_set1_epi32(15);
        let bp = offsets.as_ptr();
        let ip = ids.as_ptr();
        let plan = LagPlan::<16>::new(seeds);
        let sh: [__m512i; 4] =
            std::array::from_fn(|t| _mm512_loadu_si512(plan.shift_idx[t].as_ptr().cast()));
        let shm: [u16; 4] = std::array::from_fn(|t| plan.shift_mask[t] as u16);
        let cidx = _mm512_loadu_si512(plan.carry_idx.as_ptr().cast());
        let seedv = _mm512_cvtepi64_epi32(_mm512_loadu_si512(plan.seeds.as_ptr().cast()));
        let seedv = _mm512_castsi256_si512(seedv);
        let mut carry = zero;
        let mut basev = zero;
        let mut i = 0;
        while i < n {
            let id = _mm512_cvtepu8_epi32(_mm_loadu_si128(ip.add(i).cast()));
            let (w, lv) = if TB == 0 {
                (
                    _mm512_permutexvar_epi32(id, wt0),
                    _mm512_permutexvar_epi32(id, lt0),
                )
            } else {
                (
                    _mm512_permutex2var_epi32(wt0, id, wt1),
                    _mm512_permutex2var_epi32(lt0, id, lt1),
                )
            };
            let mut incl = w;
            incl = _mm512_add_epi32(incl, _mm512_alignr_epi32::<15>(incl, zero));
            incl = _mm512_add_epi32(incl, _mm512_alignr_epi32::<14>(incl, zero));
            incl = _mm512_add_epi32(incl, _mm512_alignr_epi32::<12>(incl, zero));
            incl = _mm512_add_epi32(incl, _mm512_alignr_epi32::<8>(incl, zero));
            let ex = _mm512_sub_epi32(incl, w);
            let base = _mm_cvtsi128_si32(_mm512_castsi512_si128(basev)) as u32 as usize;
            let win = _mm512_loadu_si512(bp.add(base >> 3).cast());
            let bit = _mm512_add_epi32(_mm512_and_si512(basev, seven), ex);
            let q = _mm512_srli_epi32::<5>(bit);
            let lo = _mm512_permutexvar_epi32(q, win);
            let hi = _mm512_permutexvar_epi32(_mm512_add_epi32(q, one), win);
            let r = _mm512_shrdv_epi32(lo, hi, bit);
            let r = _mm512_andnot_si512(sllv32(ones, w), r);
            let mut o = _mm512_add_epi32(lv, r);
            if PREFIX {
                if i == 0 {
                    o = _mm512_mask_blend_epi32(plan.seed_mask as u16, o, seedv);
                }
                for t in 0..4 {
                    o = _mm512_add_epi32(o, _mm512_maskz_permutexvar_epi32(shm[t], sh[t], o));
                }
                o = _mm512_add_epi32(o, carry);
                carry = _mm512_permutexvar_epi32(cidx, o);
            }
            let km: u16 = if i + 16 <= n {
                0xffff
            } else {
                ((1u32 << (n - i)) - 1) as u16
            };
            let dst = out.add(i * OUT);
            match OUT {
                4 => _mm512_mask_storeu_epi32(dst.cast(), km, o),
                2 => _mm512_mask_cvtepi32_storeu_epi16(dst.cast(), km, o),
                _ => _mm512_mask_cvtepi32_storeu_epi8(dst.cast(), km, o),
            }
            basev = _mm512_add_epi32(basev, _mm512_permutexvar_epi32(last, incl));
            i += 16;
        }
    }
}

/// Write one answer bit per id from the per-bin `class` table (0 false, 1 true, 2 straddles),
/// 64 ids per word of `out`. Returns `false` as soon as an id's bin straddles.
///
/// # Safety
///
/// The CPU must support the features checked by [`has_avx512`]; `class` must hold 64 entries
/// and every id must be below 64; `out` must hold a word per 64 ids.
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi")]
pub(crate) unsafe fn classify_ids(ids: &[u8], class: &[u8], out: &mut [u64]) -> bool {
    // SAFETY: loads are masked to `ids` and read 64 bytes of `class`; stores index `out` checked.
    unsafe {
        let table = _mm512_loadu_si512(class.as_ptr().cast());
        let one = _mm512_set1_epi8(1);
        let two = _mm512_set1_epi8(2);
        let n = ids.len();
        let mut i = 0;
        while i < n {
            let k: u64 = if i + 64 <= n {
                u64::MAX
            } else {
                (1u64 << (n - i)) - 1
            };
            let id = _mm512_maskz_loadu_epi8(k, ids.as_ptr().add(i).cast());
            let c = _mm512_permutexvar_epi8(id, table);
            if _mm512_mask_cmpeq_epi8_mask(k, c, two) != 0 {
                return false;
            }
            out[i / 64] = _mm512_mask_cmpeq_epi8_mask(k, c, one);
            i += 64;
        }
        true
    }
}
