// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The per-token copy primitives.
//!
//! Decode throughput is dominated by a fixed 16-byte over-copy per token. A
//! single 16-byte value copy ([`copy16`]) lowers to one native 128-bit store on
//! every target (`movups` on x86 at the SSE2 baseline, `str q` on AArch64), so a
//! hand-written SIMD intrinsic buys nothing over it.
//!
//! [`copy_token_bytes`] is the exact-length companion, for the one place that must
//! not over-store: the no-padding tail of
//! [`try_decode_into`](super::try_decode_into), where fewer than `MAX_TOKEN_SIZE`
//! output bytes remain. It copies exactly `len` bytes with at most two overlapping
//! power-of-two stores — no `memcpy` call — so a short final stretch, or an entire
//! small decode that never enters the over-copy fast path (e.g. a single short row
//! through `scalar_at`), stays branch-light.

/// Over-copy a full 16-byte token chunk from `src` to `dst`.
///
/// Callers advance the output cursor by the token's true length; the extra bytes
/// are overwritten by the next token, and the trailing
/// [`DECODE_PADDING`](super::DECODE_PADDING) room absorbs the last token's
/// over-store.
///
/// # Safety
/// `src` must be readable for 16 bytes and `dst` writable for 16 bytes.
#[inline(always)]
pub(crate) unsafe fn copy16(src: *const u8, dst: *mut u8) {
    // SAFETY: caller guarantees 16 readable/writable bytes. A 16-byte value copy
    // lowers to a single 128-bit `movups` (x86) / `str q` (AArch64).
    unsafe {
        dst.cast::<[u8; 16]>()
            .write_unaligned(src.cast::<[u8; 16]>().read_unaligned());
    }
}

/// Copy exactly `len` bytes (`len <= MAX_TOKEN_SIZE`) from `src` to `dst` — the
/// exact-length companion to [`copy16`], for callers that must not over-store.
///
/// For a non-power-of-two `len` it writes the first and last chunk of the next
/// lower power of two; the overlapping middle bytes are written twice but never
/// outside `[dst, dst + len)`, and reads stay within `[src, src + len)`. So it
/// needs no source or destination slack and lowers to at most two unaligned stores
/// rather than a `memcpy` call.
///
/// # Safety
/// `src` and `dst` must be valid for `len` bytes.
#[inline(always)]
pub(crate) unsafe fn copy_token_bytes(src: *const u8, dst: *mut u8, len: usize) {
    // SAFETY: every arm reads and writes only within `[ptr, ptr + len)` (the
    // overlapping-power-of-two scheme), which the caller guarantees is valid.
    unsafe {
        match len {
            0 => {}
            1 => dst.write(src.read()),
            2 | 3 => {
                dst.cast::<u16>()
                    .write_unaligned(src.cast::<u16>().read_unaligned());
                dst.add(len - 2)
                    .cast::<u16>()
                    .write_unaligned(src.add(len - 2).cast::<u16>().read_unaligned());
            }
            4..=7 => {
                dst.cast::<u32>()
                    .write_unaligned(src.cast::<u32>().read_unaligned());
                dst.add(len - 4)
                    .cast::<u32>()
                    .write_unaligned(src.add(len - 4).cast::<u32>().read_unaligned());
            }
            8..=15 => {
                dst.cast::<u64>()
                    .write_unaligned(src.cast::<u64>().read_unaligned());
                dst.add(len - 8)
                    .cast::<u64>()
                    .write_unaligned(src.add(len - 8).cast::<u64>().read_unaligned());
            }
            16 => copy16(src, dst),
            // Unreachable for a dictionary token (`len <= MAX_TOKEN_SIZE`); kept so
            // the primitive is total.
            _ => std::ptr::copy_nonoverlapping(src, dst, len),
        }
    }
}
