// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
// SPDX-FileCopyrightText: 2026 Marcin Żukowski
// SPDX-FileComment: Derived from pivco-huffman at commit 16b48fcf at https://github.com/MarcinZukowski/pivco-huffman

//! SIMD bottom-up wavelet-tree merge, flat-subtree unpack/pack, popcount, and top-down encoder
//! partition kernels adapted from Marcin Żukowski's Apache-2.0 `pivco-huffman` (`src/prims/`):
//! - `aarch64` NEON: 64B/iter `vcntq` popcount (`vaddlvq_u16` widened sum), 1-iter
//!   software-pipelined `SABD` two-table `vqtbl2q` merges with 8B residue, table-free D=1
//!   `vqtbl1q` `merge_cst_cst`, pair-gather SIMD flat unpack (`D in 2..=6`), converging-pyramid
//!   SIMD flat pack (`D in 2..=6`), 64-wide `masks64v_ge_neon` encoder partition, and 64-byte
//!   `pack_64_bytes01_neon`.
//! - `x86_64` runtime-dispatched (cached once via `LazyLock`):
//!   * `AVX2` / `SSE4.1` (`SSSE3`): two-table `PSHUFB`-complement + OR merges (`32B/step` on AVX2,
//!     `16B/step` on SSSE3), table-free `pblendvb` `merge_cst_cst`, `ryg` `pmullw` multiply-as-shift
//!     flat unpack (`D in 2..=6`), AVX2 `vpmaddubsw` flat pack, `movemask`+`pshufb` partition, and
//!     AVX2 `pack_64_bytes01_avx2`.
//! - Portable scalar fallbacks on all other targets.

#![expect(clippy::cast_possible_truncation)]

/// Trailing readable slack bytes required behind child merge buffers so 16-byte NEON/SSE/AVX2
/// vector loads never read out of bounds or drop out of the fast loop.
pub(crate) const MERGE_OVERREAD: usize = 16;

#[cfg(target_arch = "aarch64")]
#[repr(align(16))]
pub(super) struct AlignedI8Table(pub(super) [[i8; 16]; 256]);

#[cfg(any(target_arch = "x86_64", test))]
#[repr(align(16))]
pub(crate) struct AlignedU8Table16(pub(crate) [[u8; 16]; 256]);

#[repr(align(16))]
pub(crate) struct AlignedU8Table8(pub(crate) [[u8; 8]; 256]);

/// Low-byte shuffle table for NEON 2-source 16-byte merge (`src = {R16, L16}`), folded with
/// [`G_MERGE_SHUF1`] via `vabdq_s8(|shuf0 - shuf1|)`.
#[cfg(target_arch = "aarch64")]
pub(super) static G_MERGE_SHUF0: AlignedI8Table = build_merge_shuf0();

/// High-byte shuffle table for NEON 2-source 16-byte merge (`src = {R16, L16}`).
#[cfg(target_arch = "aarch64")]
pub(super) static G_MERGE_SHUF1: AlignedI8Table = build_merge_shuf1();

/// Low-byte shuffle table for `x86_64` SSE4.1/AVX2 PSHUFB-complement + OR merge (`x86.h:120`):
/// bytes `0..8` hold `r_idx` (for bit=1) or `255 - l_idx` (for bit=0, MSB=1 zeroes in `pshufb`);
/// bytes `8..16` replicate `popcnt(m_lo)`.
#[cfg(any(target_arch = "x86_64", test))]
pub(crate) static X86_MERGE_SHUF0: AlignedU8Table16 = build_x86_merge_shuf0();

/// High-byte shuffle table for `x86_64` SSE4.1/AVX2 PSHUFB-complement + OR merge (`x86.h:121`):
/// bytes `0..8` hold `r_idx` (for bit=1) or `247 - l_idx` (for bit=0); adding `popcnt(m_lo)` yields
/// `pop + r_idx` for bit=1 and `255 - ((8 - pop) + l_idx)` for bit=0 so `^ 0xFF` indexes `L16`.
#[cfg(any(target_arch = "x86_64", test))]
pub(crate) static X86_MERGE_SHUF1: AlignedU8Table8 = build_x86_merge_shuf1();

/// Expands 1 byte (`8` bits) into `8` mask bytes (`0xFF` if bit is `1`, `0x00` if `0`).
pub(super) static MERGE_BITS_TAB: AlignedU8Table8 = build_merge_bits_tab();

/// Compresses the indices `0..8` where `(mask >> j) & 1 == 0` (left partition).
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
pub(super) static COMPRESS_LEFT_TAB: AlignedU8Table8 = build_compress_tab(false);

/// Compresses the indices `0..8` where `(mask >> j) & 1 == 1` (right partition).
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
pub(super) static COMPRESS_RIGHT_TAB: AlignedU8Table8 = build_compress_tab(true);

/// Precomputed byte popcount table (`0..=8`), avoiding GPR->SIMD round-trips on `aarch64`.
pub(super) static POPCNT8: [u8; 256] = build_popcnt8();

#[cfg(target_arch = "aarch64")]
const fn build_merge_shuf0() -> AlignedI8Table {
    let mut tab = [[0i8; 16]; 256];
    let mut mask = 0usize;
    while mask < 256 {
        let mut pop = 0i8;
        let mut bit = 0usize;
        while bit < 8 {
            if ((mask >> bit) & 1) != 0 {
                tab[mask][bit] = pop;
                pop += 1;
            } else {
                tab[mask][bit] = -16 - (bit as i8) + pop;
            }
            bit += 1;
        }
        let mut hi = 8usize;
        while hi < 16 {
            tab[mask][hi] = pop;
            hi += 1;
        }
        mask += 1;
    }
    AlignedI8Table(tab)
}

#[cfg(target_arch = "aarch64")]
const fn build_merge_shuf1() -> AlignedI8Table {
    let mut tab = [[0i8; 16]; 256];
    let mut mask = 0usize;
    while mask < 256 {
        let mut pop = 0i8;
        let mut bit = 0usize;
        while bit < 8 {
            if ((mask >> bit) & 1) != 0 {
                tab[mask][bit + 8] = -pop;
                pop += 1;
            } else {
                let val = -16 - (bit as i8) + pop;
                tab[mask][bit + 8] = 8 - val;
            }
            bit += 1;
        }
        mask += 1;
    }
    AlignedI8Table(tab)
}

#[cfg(any(target_arch = "x86_64", test))]
const fn build_x86_merge_shuf0() -> AlignedU8Table16 {
    let mut tab = [[0u8; 16]; 256];
    let mut mask = 0usize;
    while mask < 256 {
        let mut rset = 0u8;
        let mut rclr = 0u8;
        let mut bit = 0usize;
        while bit < 8 {
            if ((mask >> bit) & 1) != 0 {
                tab[mask][bit] = rset;
                rset += 1;
            } else {
                tab[mask][bit] = 255u8 - rclr;
                rclr += 1;
            }
            bit += 1;
        }
        let mut hi = 8usize;
        while hi < 16 {
            tab[mask][hi] = rset;
            hi += 1;
        }
        mask += 1;
    }
    AlignedU8Table16(tab)
}

#[cfg(any(target_arch = "x86_64", test))]
const fn build_x86_merge_shuf1() -> AlignedU8Table8 {
    let mut tab = [[0u8; 8]; 256];
    let mut mask = 0usize;
    while mask < 256 {
        let mut rset = 0u8;
        let mut rclr = 0u8;
        let mut bit = 0usize;
        while bit < 8 {
            if ((mask >> bit) & 1) != 0 {
                tab[mask][bit] = rset;
                rset += 1;
            } else {
                tab[mask][bit] = 247u8 - rclr;
                rclr += 1;
            }
            bit += 1;
        }
        mask += 1;
    }
    AlignedU8Table8(tab)
}

const fn build_merge_bits_tab() -> AlignedU8Table8 {
    let mut tab = [[0u8; 8]; 256];
    let mut mask = 0usize;
    while mask < 256 {
        let mut bit = 0usize;
        while bit < 8 {
            tab[mask][bit] = if ((mask >> bit) & 1) != 0 { 0xFF } else { 0x00 };
            bit += 1;
        }
        mask += 1;
    }
    AlignedU8Table8(tab)
}

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
const fn build_compress_tab(want_one: bool) -> AlignedU8Table8 {
    let mut tab = [[0u8; 8]; 256];
    let mut mask = 0usize;
    while mask < 256 {
        let mut out_idx = 0usize;
        let mut bit = 0usize;
        while bit < 8 {
            let is_one = ((mask >> bit) & 1) != 0;
            if is_one == want_one {
                tab[mask][out_idx] = bit as u8;
                out_idx += 1;
            }
            bit += 1;
        }
        mask += 1;
    }
    AlignedU8Table8(tab)
}

const fn build_popcnt8() -> [u8; 256] {
    let mut tab = [0u8; 256];
    let mut mask = 0usize;
    while mask < 256 {
        tab[mask] = (mask as u8).count_ones() as u8;
        mask += 1;
    }
    tab
}

/// Counts set bits in the first `k` bits of `bm`.
#[inline]
pub(crate) fn popcount_bits(bm: &[u8], k: usize) -> usize {
    #[cfg(target_arch = "aarch64")]
    {
        // `popcount_bits_neon` accumulates into 8 `u16` lanes (each <= 65,535 -> safe up to
        // 524,280 bits) and widens horizontally via `vaddlvq_u16` (`u32`).
        if (128..=524_280).contains(&k) && bm.len() >= k.div_ceil(8) {
            // SAFETY: `aarch64` always provides NEON; `bm` has at least `ceil(k/8)` bytes.
            return unsafe { neon::popcount_bits_neon(bm, k) };
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if k >= 64 && bm.len() >= k.div_ceil(8) && x86::features().popcnt {
            // SAFETY: Runtime CPUID verified `popcnt`.
            return unsafe { x86::popcount_bits_x86(bm, k) };
        }
    }
    popcount_bits_scalar(bm, k)
}

#[inline]
pub(crate) fn popcount_bits_scalar(bm: &[u8], k: usize) -> usize {
    let full_bytes = k >> 3;
    let rem_bits = k & 7;
    let mut sum = 0usize;
    let (u64_chunks, tail) = bm[..full_bytes].as_chunks::<8>();
    for &chunk in u64_chunks {
        sum += u64::from_le_bytes(chunk).count_ones() as usize;
    }
    for &byte in tail {
        sum += usize::from(POPCNT8[usize::from(byte)]);
    }
    if rem_bits > 0 {
        let mask = (1u8 << rem_bits) - 1;
        sum += usize::from(POPCNT8[usize::from(bm[full_bytes] & mask)]);
    }
    sum
}

/// Raw-pointer inner dispatcher for 2-way merge (`left` and `right` may alias `out`'s tail
/// strictly at or after the write cursor, i.e., `side_offset + consumed >= idx`).
///
/// # Safety
/// - `out_ptr` must be valid for writes of `k` bytes.
/// - `l_base` must be valid for reads of `l_len` bytes, and `l_len >= k - popcount(bm, k)`.
/// - `r_base` must be valid for reads of `r_len` bytes, and `r_len >= popcount(bm, k)`.
/// - `bm.len() * 8 >= k`.
/// - If either `l_base` or `r_base` aliases `out_ptr`, it must be derived from `out_ptr` at an
///   offset `>= k_other` so writes at `idx` never overwrite unread bytes.
#[inline]
unsafe fn merge_vec_vec_raw(
    bm: &[u8],
    k: usize,
    l_base: *const u8,
    l_len: usize,
    r_base: *const u8,
    r_len: usize,
    out_ptr: *mut u8,
) {
    #[cfg(target_arch = "aarch64")]
    {
        if l_len >= 16 && r_len >= 16 {
            // SAFETY: Preconditions guarantee valid ranges and `aarch64` provides NEON.
            unsafe { neon::merge_vec_vec_neon(bm, k, l_base, l_len, r_base, r_len, out_ptr) };
            return;
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if l_len >= 16 && r_len >= 16 {
            let feats = x86::features();
            if feats.avx2 && feats.popcnt {
                // SAFETY: Runtime CPUID verified `avx2+popcnt`.
                unsafe { x86::merge_vec_vec_avx2(bm, k, l_base, l_len, r_base, r_len, out_ptr) };
                return;
            }
            if feats.ssse3 && feats.popcnt {
                // SAFETY: Runtime CPUID verified `ssse3+popcnt`.
                unsafe { x86::merge_vec_vec_ssse3(bm, k, l_base, l_len, r_base, r_len, out_ptr) };
                return;
            }
        }
    }
    // SAFETY: Preconditions guarantee all reads/writes stay in bounds.
    unsafe { merge_vec_vec_tail_raw(bm, 0, k, l_base, 0, r_base, 0, out_ptr) };
}

#[inline]
#[expect(clippy::too_many_arguments)]
pub(super) unsafe fn merge_vec_vec_tail_raw(
    bm: &[u8],
    mut idx: usize,
    k: usize,
    l_base: *const u8,
    mut lc: usize,
    r_base: *const u8,
    mut rc: usize,
    out_ptr: *mut u8,
) {
    while idx < k {
        let bit = (bm[idx >> 3] >> (idx & 7)) & 1;
        if bit != 0 {
            // SAFETY: Caller guarantees `rc < r_len` and `idx < k`.
            unsafe { *out_ptr.add(idx) = *r_base.add(rc) };
            rc += 1;
        } else {
            // SAFETY: Caller guarantees `lc < l_len` and `idx < k`.
            unsafe { *out_ptr.add(idx) = *l_base.add(lc) };
            lc += 1;
        }
        idx += 1;
    }
}

/// Bottom-up 2-way merge where both `left` and `right` are disjoint decoded symbol buffers.
#[cfg(test)]
#[inline]
pub(crate) fn merge_vec_vec(bm: &[u8], k: usize, left: &[u8], right: &[u8], out: &mut [u8]) {
    let out = &mut out[..k];
    assert!(bm.len() * 8 >= k, "bitmap too short for k={k}");
    // SAFETY: Disjoint slices; `out` has length `k`; callers guarantee `left`/`right` cover counts.
    unsafe {
        merge_vec_vec_raw(
            bm,
            k,
            left.as_ptr(),
            left.len(),
            right.as_ptr(),
            right.len(),
            out.as_mut_ptr(),
        );
    }
}

/// In-place 2-way merge where `left` resides at `buf[k_right..]` (`k_right == popcount(bm, k)`)
/// and `right` is a separate buffer, writing the merged `k` symbols to `buf[..k]`.
#[inline]
pub(crate) fn merge_vec_vec_in_place_left(
    bm: &[u8],
    k: usize,
    k_right: usize,
    right: &[u8],
    buf: &mut [u8],
) {
    assert!(bm.len() * 8 >= k, "bitmap too short for k={k}");
    assert!(
        k_right <= k && buf.len() >= k,
        "in-place left buffer too short"
    );
    let l_len = buf.len() - k_right;
    let out_ptr = buf.as_mut_ptr();
    // SAFETY: `l_base` is derived from `out_ptr` at offset `k_right`, so at every step
    // `k_right + lc >= idx` and writes to `out_ptr[..idx]` never overtake reads from `l_base`.
    unsafe {
        let l_base = out_ptr.add(k_right).cast_const();
        merge_vec_vec_raw(bm, k, l_base, l_len, right.as_ptr(), right.len(), out_ptr);
    }
}

/// In-place 2-way merge where `right` resides at `buf[k_left..]` (`k_left == k - popcount(bm, k)`)
/// and `left` is a separate buffer, writing the merged `k` symbols to `buf[..k]`.
#[inline]
pub(crate) fn merge_vec_vec_in_place_right(
    bm: &[u8],
    k: usize,
    left: &[u8],
    k_left: usize,
    buf: &mut [u8],
) {
    assert!(bm.len() * 8 >= k, "bitmap too short for k={k}");
    assert!(
        k_left <= k && buf.len() >= k,
        "in-place right buffer too short"
    );
    let r_len = buf.len() - k_left;
    let out_ptr = buf.as_mut_ptr();
    // SAFETY: `r_base` is derived from `out_ptr` at offset `k_left`, so at every step
    // `k_left + rc >= idx` and writes to `out_ptr[..idx]` never overtake reads from `r_base`.
    unsafe {
        let r_base = out_ptr.add(k_left).cast_const();
        merge_vec_vec_raw(bm, k, left.as_ptr(), left.len(), r_base, r_len, out_ptr);
    }
}

/// Raw-pointer inner dispatcher for constant-plus-vector merge:
/// - `INVERT == false`: `0 -> cst_sym`, `1 -> vec_buf` (`merge_cst_vec`)
/// - `INVERT == true`:  `0 -> vec_buf`, `1 -> cst_sym` (`merge_vec_cst`)
///
/// # Safety
/// Same bounds and in-place tail-aliasing invariants as [`merge_vec_vec_raw`].
#[inline]
unsafe fn merge_cst_side_raw<const INVERT: bool>(
    bm: &[u8],
    k: usize,
    cst_sym: u8,
    v_base: *const u8,
    v_len: usize,
    out_ptr: *mut u8,
) {
    #[cfg(target_arch = "aarch64")]
    {
        if v_len >= 16 {
            // SAFETY: Preconditions guarantee valid ranges and `aarch64` provides NEON.
            unsafe { neon::merge_cst_side_neon::<INVERT>(bm, k, cst_sym, v_base, v_len, out_ptr) };
            return;
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if v_len >= 16 {
            let feats = x86::features();
            if feats.avx2 && feats.popcnt {
                // SAFETY: Runtime CPUID verified `avx2+popcnt`.
                unsafe {
                    x86::merge_cst_side_avx2::<INVERT>(bm, k, cst_sym, v_base, v_len, out_ptr);
                }
                return;
            }
            if feats.ssse3 && feats.popcnt {
                // SAFETY: Runtime CPUID verified `ssse3+popcnt`.
                unsafe {
                    x86::merge_cst_side_ssse3::<INVERT>(bm, k, cst_sym, v_base, v_len, out_ptr);
                }
                return;
            }
        }
    }
    // SAFETY: Preconditions guarantee all reads/writes stay in bounds.
    unsafe { merge_cst_side_tail_raw::<INVERT>(bm, 0, k, cst_sym, v_base, 0, out_ptr) };
}

#[inline]
pub(super) unsafe fn merge_cst_side_tail_raw<const INVERT: bool>(
    bm: &[u8],
    mut idx: usize,
    k: usize,
    cst_sym: u8,
    v_base: *const u8,
    mut vc: usize,
    out_ptr: *mut u8,
) {
    while idx < k {
        let bit = ((bm[idx >> 3] >> (idx & 7)) & 1) != 0;
        let take_vec = bit ^ INVERT;
        if take_vec {
            // SAFETY: Caller guarantees `vc < v_len` and `idx < k`.
            unsafe { *out_ptr.add(idx) = *v_base.add(vc) };
            vc += 1;
        } else {
            // SAFETY: Caller guarantees `idx < k`.
            unsafe { *out_ptr.add(idx) = cst_sym };
        }
        idx += 1;
    }
}

/// Bottom-up merge where `left` is a constant leaf symbol `left_sym` and `right` is a decoded buffer.
#[cfg(test)]
#[inline]
pub(crate) fn merge_cst_vec(bm: &[u8], k: usize, left_sym: u8, right: &[u8], out: &mut [u8]) {
    let out = &mut out[..k];
    assert!(bm.len() * 8 >= k, "bitmap too short for k={k}");
    // SAFETY: Disjoint slices and `out` has length `k`.
    unsafe {
        merge_cst_side_raw::<false>(
            bm,
            k,
            left_sym,
            right.as_ptr(),
            right.len(),
            out.as_mut_ptr(),
        );
    }
}

/// In-place `merge_cst_vec` where `right` resides at `buf[k_left..]` (`k_left == k - k_right`).
#[inline]
pub(crate) fn merge_cst_vec_in_place(
    bm: &[u8],
    k: usize,
    left_sym: u8,
    k_left: usize,
    buf: &mut [u8],
) {
    assert!(bm.len() * 8 >= k, "bitmap too short for k={k}");
    assert!(
        k_left <= k && buf.len() >= k,
        "in-place cst_vec buffer too short"
    );
    let v_len = buf.len() - k_left;
    let out_ptr = buf.as_mut_ptr();
    // SAFETY: `v_base` is derived from `out_ptr` at offset `k_left`; `k_left + rc >= idx`.
    unsafe {
        let v_base = out_ptr.add(k_left).cast_const();
        merge_cst_side_raw::<false>(bm, k, left_sym, v_base, v_len, out_ptr);
    }
}

/// Bottom-up merge where `left` is a decoded buffer and `right` is a constant leaf symbol `right_sym`.
#[cfg(test)]
#[inline]
pub(crate) fn merge_vec_cst(bm: &[u8], k: usize, left: &[u8], right_sym: u8, out: &mut [u8]) {
    let out = &mut out[..k];
    assert!(bm.len() * 8 >= k, "bitmap too short for k={k}");
    // SAFETY: Disjoint slices and `out` has length `k`.
    unsafe {
        merge_cst_side_raw::<true>(
            bm,
            k,
            right_sym,
            left.as_ptr(),
            left.len(),
            out.as_mut_ptr(),
        );
    }
}

/// In-place `merge_vec_cst` where `left` resides at `buf[k_right..]`.
#[inline]
pub(crate) fn merge_vec_cst_in_place(
    bm: &[u8],
    k: usize,
    k_right: usize,
    right_sym: u8,
    buf: &mut [u8],
) {
    assert!(bm.len() * 8 >= k, "bitmap too short for k={k}");
    assert!(
        k_right <= k && buf.len() >= k,
        "in-place vec_cst buffer too short"
    );
    let v_len = buf.len() - k_right;
    let out_ptr = buf.as_mut_ptr();
    // SAFETY: `v_base` is derived from `out_ptr` at offset `k_right`; `k_right + lc >= idx`.
    unsafe {
        let v_base = out_ptr.add(k_right).cast_const();
        merge_cst_side_raw::<true>(bm, k, right_sym, v_base, v_len, out_ptr);
    }
}

/// Bottom-up merge where both children are leaf symbols (`0 -> left_sym`, `1 -> right_sym`).
#[inline]
pub(crate) fn merge_cst_cst(bm: &[u8], k: usize, left_sym: u8, right_sym: u8, out: &mut [u8]) {
    let out = &mut out[..k];
    assert!(bm.len() * 8 >= k, "bitmap too short for k={k}");
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: `aarch64` always provides NEON; `bm` covers all `k` bits and `out.len() == k`.
        unsafe { neon::merge_cst_cst_neon(bm, k, left_sym, right_sym, out) };
        return;
    }
    #[cfg(target_arch = "x86_64")]
    {
        let feats = x86::features();
        if feats.avx2 {
            // SAFETY: Runtime CPUID verified `avx2` and `out.len() == k`.
            unsafe { x86::merge_cst_cst_avx2(bm, k, left_sym, right_sym, out) };
            return;
        }
        if feats.sse41 {
            // SAFETY: Runtime CPUID verified `sse4.1` and `out.len() == k`.
            unsafe { x86::merge_cst_cst_sse41(bm, k, left_sym, right_sym, out) };
            return;
        }
    }
    #[allow(unreachable_code)]
    merge_cst_cst_scalar(bm, k, left_sym, right_sym, out);
}

#[inline]
pub(crate) fn merge_cst_cst_scalar(
    bm: &[u8],
    k: usize,
    left_sym: u8,
    right_sym: u8,
    out: &mut [u8],
) {
    let delta = left_sym ^ right_sym;
    let mut pos = 0usize;
    while pos + 8 <= k {
        let row = MERGE_BITS_TAB.0[usize::from(bm[pos >> 3])];
        for bit in 0..8 {
            out[pos + bit] = left_sym ^ (delta & row[bit]);
        }
        pos += 8;
    }
    while pos < k {
        let mb = (bm[pos >> 3] >> (pos & 7)) & 1;
        out[pos] = if mb != 0 { right_sym } else { left_sym };
        pos += 1;
    }
}

#[cfg(test)]
#[inline]
pub(crate) fn merge_vec_vec_scalar(bm: &[u8], k: usize, left: &[u8], right: &[u8], out: &mut [u8]) {
    let out = &mut out[..k];
    // SAFETY: Disjoint slices and `out.len() == k`.
    unsafe {
        merge_vec_vec_tail_raw(
            bm,
            0,
            k,
            left.as_ptr(),
            0,
            right.as_ptr(),
            0,
            out.as_mut_ptr(),
        );
    }
}

#[cfg(test)]
#[inline]
pub(crate) fn merge_cst_vec_scalar(
    bm: &[u8],
    k: usize,
    left_sym: u8,
    right: &[u8],
    out: &mut [u8],
) {
    let out = &mut out[..k];
    // SAFETY: Disjoint slices and `out.len() == k`.
    unsafe {
        merge_cst_side_tail_raw::<false>(bm, 0, k, left_sym, right.as_ptr(), 0, out.as_mut_ptr());
    }
}

#[cfg(test)]
#[inline]
pub(crate) fn merge_vec_cst_scalar(
    bm: &[u8],
    k: usize,
    left: &[u8],
    right_sym: u8,
    out: &mut [u8],
) {
    let out = &mut out[..k];
    // SAFETY: Disjoint slices and `out.len() == k`.
    unsafe {
        merge_cst_side_tail_raw::<true>(bm, 0, k, right_sym, left.as_ptr(), 0, out.as_mut_ptr());
    }
}

/// Unpacks `n` packed `d`-bit codes (`d in 2..=6`) from `packed` through `c2s[0..(1<<d)]` into `out[..n]`.
#[inline]
pub(crate) fn unpack_flat_codes(d: u8, n: usize, packed: &[u8], c2s: &[u8; 64], out: &mut [u8]) {
    let out = &mut out[..n];
    debug_assert!((2..=6).contains(&d));
    #[cfg(target_arch = "aarch64")]
    {
        if packed.len() >= (n * usize::from(d)).div_ceil(8) {
            match d {
                2 if n >= 16 => {
                    // SAFETY: `aarch64` provides NEON, `out.len() == n`, `packed` covers `n` codes.
                    unsafe { neon::unpack_flat_d2_neon(n, packed, c2s, out) };
                    return;
                }
                3 if n >= 8 => {
                    // SAFETY: `aarch64` provides NEON, `out.len() == n`, `packed` covers `n` codes.
                    unsafe { neon::unpack_flat_d3_neon(n, packed, c2s, out) };
                    return;
                }
                4 if n >= 16 => {
                    // SAFETY: `aarch64` provides NEON, `out.len() == n`, `packed` covers `n` codes.
                    unsafe { neon::unpack_flat_d4_neon(n, packed, c2s, out) };
                    return;
                }
                5 if n >= 8 => {
                    // SAFETY: `aarch64` provides NEON, `out.len() == n`, `packed` covers `n` codes.
                    unsafe { neon::unpack_flat_d5_neon(n, packed, c2s, out) };
                    return;
                }
                6 if n >= 8 => {
                    // SAFETY: `aarch64` provides NEON, `out.len() == n`, `packed` covers `n` codes.
                    unsafe { neon::unpack_flat_d6_neon(n, packed, c2s, out) };
                    return;
                }
                _ => {}
            }
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if packed.len() >= (n * usize::from(d)).div_ceil(8) {
            let feats = x86::features();
            if d == 2 && n >= 16 && feats.avx2 {
                // SAFETY: Runtime CPUID verified `avx2` and `out.len() == n`.
                unsafe { x86::unpack_flat_d2_avx2(n, packed, c2s, out) };
                return;
            }
            if d == 4 && n >= 64 && feats.avx2 {
                // SAFETY: Runtime CPUID verified `avx2` and `out.len() == n`.
                unsafe { x86::unpack_flat_d4_avx2(n, packed, c2s, out) };
                return;
            }
            if feats.ssse3 {
                match d {
                    2 if n >= 64 => {
                        // SAFETY: Runtime CPUID verified `ssse3` and `out.len() == n`.
                        unsafe { x86::unpack_flat_d2_ssse3(n, packed, c2s, out) };
                        return;
                    }
                    3 if n >= 16 => {
                        // SAFETY: Runtime CPUID verified `ssse3` and `out.len() == n`.
                        unsafe { x86::unpack_flat_d3_ssse3(n, packed, c2s, out) };
                        return;
                    }
                    4 if n >= 16 => {
                        // SAFETY: Runtime CPUID verified `ssse3` and `out.len() == n`.
                        unsafe { x86::unpack_flat_d4_ssse3(n, packed, c2s, out) };
                        return;
                    }
                    5 if n >= 16 => {
                        // SAFETY: Runtime CPUID verified `ssse3` and `out.len() == n`.
                        unsafe { x86::unpack_flat_d5_ssse3(n, packed, c2s, out) };
                        return;
                    }
                    6 if n >= 16 => {
                        // SAFETY: Runtime CPUID verified `ssse3` and `out.len() == n`.
                        unsafe { x86::unpack_flat_d6_ssse3(n, packed, c2s, out) };
                        return;
                    }
                    _ => {}
                }
            }
        }
    }

    unpack_flat_codes_scalar(d, n, packed, c2s, out);
}

#[inline]
pub(crate) fn unpack_flat_codes_scalar(
    depth: u8,
    len: usize,
    packed: &[u8],
    c2s: &[u8; 64],
    out: &mut [u8],
) {
    unpack_flat_tail_scalar(usize::from(depth), 0, len, packed, c2s, out);
}

#[inline]
pub(super) fn unpack_flat_tail_scalar(
    depth: usize,
    mut idx: usize,
    len: usize,
    packed: &[u8],
    c2s: &[u8; 64],
    out: &mut [u8],
) {
    match depth {
        2 => {
            while idx + 4 <= len {
                let byte = packed[idx >> 2];
                out[idx] = c2s[usize::from(byte & 3)];
                out[idx + 1] = c2s[usize::from((byte >> 2) & 3)];
                out[idx + 2] = c2s[usize::from((byte >> 4) & 3)];
                out[idx + 3] = c2s[usize::from((byte >> 6) & 3)];
                idx += 4;
            }
            if idx < len {
                let byte = packed[idx >> 2];
                for rem in 0..(len - idx) {
                    out[idx + rem] = c2s[usize::from((byte >> (rem * 2)) & 3)];
                }
            }
        }
        4 => {
            while idx + 2 <= len {
                let byte = packed[idx >> 1];
                out[idx] = c2s[usize::from(byte & 0x0F)];
                out[idx + 1] = c2s[usize::from(byte >> 4)];
                idx += 2;
            }
            if idx < len {
                out[idx] = c2s[usize::from(packed[idx >> 1] & 0x0F)];
            }
        }
        _ => {
            let mask = (1u64 << depth) - 1;
            while idx + 8 <= len && ((idx * depth) >> 3) + 8 <= packed.len() {
                let bp = idx * depth;
                let byte_off = bp >> 3;
                let chunk: [u8; 8] = [
                    packed[byte_off],
                    packed[byte_off + 1],
                    packed[byte_off + 2],
                    packed[byte_off + 3],
                    packed[byte_off + 4],
                    packed[byte_off + 5],
                    packed[byte_off + 6],
                    packed[byte_off + 7],
                ];
                let word = u64::from_le_bytes(chunk) >> (bp & 7);
                for slot in 0..8 {
                    let code = ((word >> (slot * depth)) & mask) as usize;
                    out[idx + slot] = c2s[code];
                }
                idx += 8;
            }
            let mask32 = mask as u32;
            while idx < len {
                let bp = idx * depth;
                let byte_idx = bp >> 3;
                let bit_off = bp & 7;
                let mut val = u32::from(packed[byte_idx]);
                if bit_off + depth > 8 && byte_idx + 1 < packed.len() {
                    val |= u32::from(packed[byte_idx + 1]) << 8;
                }
                if bit_off + depth > 16 && byte_idx + 2 < packed.len() {
                    val |= u32::from(packed[byte_idx + 2]) << 16;
                }
                out[idx] = c2s[((val >> bit_off) & mask32) as usize];
                idx += 1;
            }
        }
    }
}

/// Top-down SIMD partition for encoder: splits `ranks[..n]` by `rank >= pivot_rank`, writing the
/// `ceil(n/8)`-byte bitmap into `bm_out`, left ranks (`< pivot_rank`) into `left_out`, and right
/// ranks (`>= pivot_rank`) into `right_out`. Returns `k_right`.
#[inline]
pub(crate) fn partition_ranks_by_pivot(
    ranks: &[u8],
    pivot_rank: u8,
    bm_out: &mut [u8],
    left_out: &mut [u8],
    right_out: &mut [u8],
) -> usize {
    let len = ranks.len();
    #[cfg(target_arch = "aarch64")]
    {
        if len >= 16
            && bm_out.len() >= len.div_ceil(8)
            && left_out.len() >= len + 8
            && right_out.len() >= len + 8
        {
            // SAFETY: `aarch64` provides NEON and output slices have >= 8 bytes trailing slack.
            return unsafe {
                neon::partition_ranks_neon(ranks, pivot_rank, bm_out, left_out, right_out)
            };
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if len >= 16
            && bm_out.len() >= len.div_ceil(8)
            && left_out.len() >= len + 8
            && right_out.len() >= len + 8
            && x86::features().ssse3
        {
            // SAFETY: Runtime CPUID verified `ssse3`.
            return unsafe {
                x86::partition_ranks_ssse3(ranks, pivot_rank, bm_out, left_out, right_out)
            };
        }
    }
    partition_ranks_scalar(ranks, pivot_rank, bm_out, left_out, right_out)
}

#[inline]
pub(crate) fn partition_ranks_scalar(
    ranks: &[u8],
    pivot_rank: u8,
    bm_out: &mut [u8],
    left_out: &mut [u8],
    right_out: &mut [u8],
) -> usize {
    let len = ranks.len();
    let nbytes = len.div_ceil(8);
    bm_out[..nbytes].fill(0);
    let mut lc = 0usize;
    let mut rc = 0usize;
    for (idx, &rank) in ranks.iter().enumerate() {
        if rank >= pivot_rank {
            bm_out[idx >> 3] |= 1u8 << (idx & 7);
            right_out[rc] = rank;
            rc += 1;
        } else {
            left_out[lc] = rank;
            lc += 1;
        }
    }
    rc
}

/// Packs `n` codes (`ranks[i] - base_rank`, each in `0..(1 << depth)`) into `ceil(n * depth / 8)` bytes.
pub(crate) fn pack_flat_codes(depth: u8, ranks: &[u8], base_rank: u8, dst: &mut Vec<u8>) {
    debug_assert!((2..=6).contains(&depth));
    debug_assert!(
        ranks
            .iter()
            .all(|&r| r.wrapping_sub(base_rank) < (1u8 << depth)),
        "pack_flat_codes rank out of range for depth={depth}, base_rank={base_rank}"
    );
    let len = ranks.len();
    let depth_usize = usize::from(depth);
    let start_len = dst.len();
    let nbytes = (len * depth_usize).div_ceil(8);
    dst.resize(start_len + nbytes, 0);
    let out = &mut dst[start_len..];

    let mut done = 0usize;
    #[cfg(target_arch = "aarch64")]
    {
        if len >= 16 {
            // SAFETY: `aarch64` provides NEON and all SIMD stores stay within `out[..nbytes]`.
            done = unsafe { neon::pack_flat_codes_neon(depth, ranks, base_rank, out) };
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if len >= 32 && x86::features().avx2 {
            // SAFETY: Runtime CPUID verified `avx2`.
            done = unsafe { x86::pack_flat_codes_avx2(depth, ranks, base_rank, out) };
        }
    }
    if done < len {
        pack_flat_codes_tail_scalar(depth, ranks, done, base_rank, out);
    }
}

#[cfg(test)]
#[inline]
pub(crate) fn pack_flat_codes_scalar(depth: u8, ranks: &[u8], base_rank: u8, dst: &mut Vec<u8>) {
    let depth_usize = usize::from(depth);
    let start_len = dst.len();
    let nbytes = (ranks.len() * depth_usize).div_ceil(8);
    dst.resize(start_len + nbytes, 0);
    pack_flat_codes_tail_scalar(depth, ranks, 0, base_rank, &mut dst[start_len..]);
}

#[inline]
fn pack_flat_codes_tail_scalar(
    depth: u8,
    ranks: &[u8],
    start: usize,
    base_rank: u8,
    out: &mut [u8],
) {
    let len = ranks.len();
    let depth_usize = usize::from(depth);
    let bit_pos = start * depth_usize;
    let mut byte_idx = bit_pos >> 3;
    let mut bits = bit_pos & 7;
    let mut acc = if bits > 0 {
        u64::from(out[byte_idx]) & ((1u64 << bits) - 1)
    } else {
        0u64
    };
    let code_mask = (1u64 << depth_usize) - 1;
    for &rank in &ranks[start..len] {
        let code = u64::from(rank.wrapping_sub(base_rank)) & code_mask;
        acc |= code << bits;
        bits += depth_usize;
        while bits >= 8 {
            out[byte_idx] = acc as u8;
            acc >>= 8;
            bits -= 8;
            byte_idx += 1;
        }
    }
    if bits > 0 {
        out[byte_idx] = acc as u8;
    }
}

/// Packs 64 `0/1` bytes (each byte must be strictly `0` or `1`) into one `u64` bitmask in `O(1)`
/// SIMD / SWAR instructions.
#[inline]
pub(crate) fn pack_64_bytes01(chunk64: &[u8]) -> u64 {
    let chunk64 = &chunk64[..64];
    debug_assert!(chunk64.iter().all(|&b| b <= 1));
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: `aarch64` provides NEON and `chunk64.len() == 64`.
        return unsafe { neon::pack_64_bytes01_neon(chunk64) };
    }
    #[cfg(target_arch = "x86_64")]
    {
        if x86::features().avx2 {
            // SAFETY: Runtime CPUID verified `avx2` and `chunk64.len() == 64`.
            return unsafe { x86::pack_64_bytes01_avx2(chunk64) };
        }
    }
    #[allow(unreachable_code)]
    pack_64_bytes01_scalar(chunk64)
}

#[inline]
pub(crate) fn pack_64_bytes01_scalar(chunk64: &[u8]) -> u64 {
    let mut out = 0u64;
    for blk in 0..8 {
        let base = blk * 8;
        let word = u64::from_le_bytes([
            chunk64[base],
            chunk64[base + 1],
            chunk64[base + 2],
            chunk64[base + 3],
            chunk64[base + 4],
            chunk64[base + 5],
            chunk64[base + 6],
            chunk64[base + 7],
        ]);
        // Each byte is `0` or `1`: bit `8*k` * `1 << (56 - 7*k)` lands at bit `56 + k`,
        // with strictly zero cross-lane carry into bits `56..64`.
        let byte_bits = ((word & 0x0101_0101_0101_0101).wrapping_mul(0x0102_0408_1020_4080)) >> 56;
        out |= byte_bits << (blk * 8);
    }
    out
}

#[cfg(target_arch = "aarch64")]
mod neon;
#[cfg(target_arch = "x86_64")]
mod x86;
