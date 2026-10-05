// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::arch::aarch64::uint8x16x2_t;
use std::arch::aarch64::uint8x16x3_t;
use std::arch::aarch64::vaddq_u8;
use std::arch::aarch64::vdupq_n_u8;
use std::arch::aarch64::vld1q_u8;
use std::arch::aarch64::vld1q_u8_x4;
use std::arch::aarch64::vmaxq_u8;
use std::arch::aarch64::vmaxvq_u8;
use std::arch::aarch64::vmulq_u8;
use std::arch::aarch64::vorrq_u8;
use std::arch::aarch64::vqtbl1q_u8;
use std::arch::aarch64::vqtbl2q_u8;
use std::arch::aarch64::vqtbl3q_u8;
use std::arch::aarch64::vqtbl4q_u8;
use std::arch::aarch64::vst1q_u8;
use std::arch::aarch64::vsubq_u8;

/// Returns the number of initialized records and maximum observed code.
///
/// # Safety
///
/// Requires NEON, a value width of 1, 2, 4, or 8 bytes, and output capacity for `indices.len()`
/// records. `TABLE_BYTES` must be 16, 32, 48, 64, 128, 192, or 256 and cover every dictionary byte.
/// The caller must check the returned maximum and handle the scalar tail before exposing output.
pub(super) unsafe fn take_vectors<T, const TABLE_BYTES: usize>(
    table: &[u8; 256],
    indices: &[u8],
    output: *mut u8,
) -> (usize, u8) {
    let width = size_of::<T>();
    let records_per_vector = 16 / width;
    let repeat = std::array::from_fn::<_, 16, _>(|i| (i / width) as u8);
    let byte_offsets = std::array::from_fn::<_, 16, _>(|i| (i % width) as u8);
    // SAFETY: All table and mask loads are within fully initialized arrays. The loop bound
    // covers four input loads and output stores; each store writes 16 / width complete records.
    unsafe {
        let first = vld1q_u8_x4(table.as_ptr());
        let second = vld1q_u8_x4(table.as_ptr().add(64));
        let third = vld1q_u8_x4(table.as_ptr().add(128));
        let fourth = vld1q_u8_x4(table.as_ptr().add(192));
        let repeat = vld1q_u8(repeat.as_ptr());
        let byte_offsets = vld1q_u8(byte_offsets.as_ptr());
        let widths = vdupq_n_u8(width as u8);
        // Independent accumulators avoid a dependency chain through every lookup.
        let mut max_codes = [vdupq_n_u8(0); 4];
        let mut offset = 0;
        while offset + 3 * records_per_vector + 16 <= indices.len() {
            for maximum in &mut max_codes {
                let codes = vld1q_u8(indices.as_ptr().add(offset));
                *maximum = vmaxq_u8(*maximum, codes);
                let byte_indices = if width == 1 {
                    codes
                } else {
                    vaddq_u8(vmulq_u8(vqtbl1q_u8(codes, repeat), widths), byte_offsets)
                };
                let mut taken = match TABLE_BYTES {
                    16 => vqtbl1q_u8(first.0, byte_indices),
                    32 => vqtbl2q_u8(uint8x16x2_t(first.0, first.1), byte_indices),
                    48 => vqtbl3q_u8(uint8x16x3_t(first.0, first.1, first.2), byte_indices),
                    _ => vqtbl4q_u8(first, byte_indices),
                };
                // TBL returns zero outside its bank. Wrapping subtraction selects exactly one
                // 64-byte bank for each byte index, so the partial lookups can be ORed together.
                if TABLE_BYTES > 64 {
                    taken = vorrq_u8(
                        taken,
                        vqtbl4q_u8(second, vsubq_u8(byte_indices, vdupq_n_u8(64))),
                    );
                }
                if TABLE_BYTES > 128 {
                    taken = vorrq_u8(
                        taken,
                        vqtbl4q_u8(third, vsubq_u8(byte_indices, vdupq_n_u8(128))),
                    );
                }
                if TABLE_BYTES > 192 {
                    taken = vorrq_u8(
                        taken,
                        vqtbl4q_u8(fourth, vsubq_u8(byte_indices, vdupq_n_u8(192))),
                    );
                }
                vst1q_u8(output.add(offset * width), taken);
                offset += records_per_vector;
            }
        }
        let maximum = vmaxq_u8(
            vmaxq_u8(max_codes[0], max_codes[1]),
            vmaxq_u8(max_codes[2], max_codes[3]),
        );
        (offset, vmaxvq_u8(maximum))
    }
}
