// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Variable-length value body encoder: 32-byte blocks with continuation/length markers.

use super::*;

/// Encode a non-empty variable-length byte slice into `out` in 32-byte blocks with
/// continuation/length markers. Returns the number of bytes written. Empty values are
/// encoded by the caller as a single sentinel byte and never reach this function.
///
/// For the ascending path the hot loop is a `copy_nonoverlapping` of 32 bytes per block
/// plus one stamped continuation byte. For the descending path it reads a u64 at a time and
/// XORs with `0xFF`, giving LLVM a vectorizable inner loop.
pub(super) fn encode_non_empty_varlen_body(
    bytes: &[u8],
    out: &mut [u8],
    descending: bool,
) -> VortexResult<u32> {
    debug_assert!(!bytes.is_empty());
    let len = bytes.len();
    let full_blocks = len / VARLEN_BLOCK_SIZE;
    let partial = len % VARLEN_BLOCK_SIZE;
    let (full_to_write, partial_block_len) = if partial == 0 {
        // Length is an exact multiple of 32: emit (full_blocks - 1) full blocks with the
        // 0xFF continuation marker, then a final block whose continuation byte is 32.
        (full_blocks - 1, VARLEN_BLOCK_SIZE)
    } else {
        (full_blocks, partial)
    };
    let total = (full_to_write + 1) * VARLEN_BLOCK_TOTAL;
    // The caller reserved this slot from `encoded_size_for_non_empty_varlen`, which already
    // verified the byte total fits `u32`; re-check here so the conversion never panics.
    let total_u32 =
        u32::try_from(total).map_err(|_| vortex_err!("encoded varlen size overflows u32"))?;
    debug_assert!(out.len() >= total);
    // The final block's continuation byte encodes its content length (1..=32).
    let len_byte =
        u8::try_from(partial_block_len).vortex_expect("varlen final block length (1..=32) fits u8");

    // SAFETY: `out` has at least `total` bytes — the caller sizes every varlen slot via
    // `encoded_size_for_non_empty_varlen` (which equals `1 + total`, the extra byte being the
    // leading sentinel that the caller wrote and that is not part of `out`). `bytes` is valid
    // for `len` reads, and every pointer advance below stays within `[0, total)` for `dst`
    // and `[0, len)` for `src`.
    unsafe {
        let mut src = bytes.as_ptr();
        let mut dst = out.as_mut_ptr();

        if !descending {
            // Ascending fast path: each full block is a 32-byte memcpy + a single 0xFF stamp.
            for _ in 0..full_to_write {
                std::ptr::copy_nonoverlapping(src, dst, VARLEN_BLOCK_SIZE);
                *dst.add(VARLEN_BLOCK_SIZE) = 0xFF;
                src = src.add(VARLEN_BLOCK_SIZE);
                dst = dst.add(VARLEN_BLOCK_TOTAL);
            }
            // Final block: copy the partial data, zero-pad the tail, write the length byte.
            std::ptr::copy_nonoverlapping(src, dst, partial_block_len);
            std::ptr::write_bytes(
                dst.add(partial_block_len),
                0,
                VARLEN_BLOCK_SIZE - partial_block_len,
            );
            *dst.add(VARLEN_BLOCK_SIZE) = len_byte;
        } else {
            // Descending: invert every value byte. A u64-stride XOR gives LLVM a vectorizable
            // inner loop; the tail handles the partial block byte-wise.
            for _ in 0..full_to_write {
                xor_copy_block(src, dst);
                *dst.add(VARLEN_BLOCK_SIZE) = 0x00; // descending counterpart of 0xFF
                src = src.add(VARLEN_BLOCK_SIZE);
                dst = dst.add(VARLEN_BLOCK_TOTAL);
            }
            for i in 0..partial_block_len {
                *dst.add(i) = *src.add(i) ^ 0xFF;
            }
            std::ptr::write_bytes(
                dst.add(partial_block_len),
                0xFF, // 0x00 XOR 0xFF
                VARLEN_BLOCK_SIZE - partial_block_len,
            );
            *dst.add(VARLEN_BLOCK_SIZE) = len_byte ^ 0xFF;
        }
    }
    Ok(total_u32)
}

/// Copy 32 bytes from `src` to `dst`, XORing each with `0xFF`. LLVM auto-vectorizes the
/// four u64-wide iterations into SIMD on x86.
///
/// # Safety
/// `src` must be valid for 32 reads, `dst` valid for 32 writes, and the regions must not
/// overlap.
#[inline(always)]
unsafe fn xor_copy_block(src: *const u8, dst: *mut u8) {
    // Four u64 lanes of 8 bytes each = 32 bytes total.
    for i in 0..4 {
        let off = i * 8;
        // SAFETY: the caller guarantees src/dst are valid for the full 32-byte block.
        let v = unsafe { std::ptr::read_unaligned(src.add(off) as *const u64) };
        unsafe { std::ptr::write_unaligned(dst.add(off) as *mut u64, v ^ u64::MAX) };
    }
}
