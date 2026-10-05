// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Separate byte tables avoid repeating every multi-byte lookup for each output byte.

use std::arch::aarch64::uint16x8x4_t;
use std::arch::aarch64::uint8x16x2_t;
use std::arch::aarch64::uint8x16x4_t;
use std::arch::aarch64::vdupq_n_u8;
use std::arch::aarch64::vld1q_u8;
use std::arch::aarch64::vld1q_u8_x4;
use std::arch::aarch64::vmaxq_u8;
use std::arch::aarch64::vmaxvq_u8;
use std::arch::aarch64::vorrq_u8;
use std::arch::aarch64::vqtbl1q_u8;
use std::arch::aarch64::vqtbl2q_u8;
use std::arch::aarch64::vqtbl4q_u8;
use std::arch::aarch64::vreinterpretq_u16_u8;
use std::arch::aarch64::vst2q_u8;
use std::arch::aarch64::vst4q_u16;
use std::arch::aarch64::vst4q_u8;
use std::arch::aarch64::vsubq_u8;
use std::arch::aarch64::vzip1q_u8;
use std::arch::aarch64::vzip2q_u8;

/// Returns the initialized record count and maximum observed code.
///
/// # Safety
///
/// Requires NEON, W of 2, 4, or 8, and 1..=128 complete initialized records in `values`.
/// Output must have capacity for `indices.len()` records. The caller must check the returned
/// maximum and initialize the scalar tail before exposing output.
pub(super) unsafe fn take_vectors<const W: usize>(
    values: &[u8],
    indices: &[u8],
    output: *mut u8,
) -> (usize, u8) {
    // SAFETY: The caller bounds the dictionary to 128 records and supplies enough output space.
    unsafe {
        match values.len() / W {
            1..=16 => take_inner::<W, 16>(values, indices, output),
            17..=32 => take_inner::<W, 32>(values, indices, output),
            33..=64 => take_inner::<W, 64>(values, indices, output),
            _ => take_inner::<W, 128>(values, indices, output),
        }
    }
}

unsafe fn take_inner<const W: usize, const N: usize>(
    values: &[u8],
    indices: &[u8],
    output: *mut u8,
) -> (usize, u8) {
    let mut planes = [[0u8; 128]; W];
    for (i, value) in values.chunks_exact(W).enumerate() {
        for j in 0..W {
            planes[j][i] = value[j];
        }
    }
    // SAFETY: Plane loads are within initialized arrays. Each iteration reads 16 codes and
    // writes exactly 16 complete records within the caller-provided output capacity.
    unsafe {
        let first = std::array::from_fn::<_, W, _>(|j| vld1q_u8_x4(planes[j].as_ptr()));
        let second = std::array::from_fn::<_, W, _>(|j| vld1q_u8_x4(planes[j].as_ptr().add(64)));
        let mut maximum = vdupq_n_u8(0);
        let mut offset = 0;
        while offset + 16 <= indices.len() {
            let code = vld1q_u8(indices.as_ptr().add(offset));
            maximum = vmaxq_u8(maximum, code);
            let taken = std::array::from_fn::<_, W, _>(|j| {
                let a = first[j];
                let mut v = match N {
                    16 => vqtbl1q_u8(a.0, code),
                    32 => vqtbl2q_u8(uint8x16x2_t(a.0, a.1), code),
                    _ => vqtbl4q_u8(a, code),
                };
                if N > 64 {
                    v = vorrq_u8(v, vqtbl4q_u8(second[j], vsubq_u8(code, vdupq_n_u8(64))));
                }
                v
            });
            let ptr = output.add(offset * W);
            if W == 2 {
                vst2q_u8(ptr, uint8x16x2_t(taken[0], taken[1]));
            } else if W == 4 {
                vst4q_u8(ptr, uint8x16x4_t(taken[0], taken[1], taken[2], taken[3]));
            } else if W == 8 {
                let low = std::array::from_fn::<_, 4, _>(|j| {
                    vreinterpretq_u16_u8(vzip1q_u8(taken[j * 2], taken[j * 2 + 1]))
                });
                let high = std::array::from_fn::<_, 4, _>(|j| {
                    vreinterpretq_u16_u8(vzip2q_u8(taken[j * 2], taken[j * 2 + 1]))
                });
                vst4q_u16(ptr.cast(), uint16x8x4_t(low[0], low[1], low[2], low[3]));
                vst4q_u16(
                    ptr.add(64).cast(),
                    uint16x8x4_t(high[0], high[1], high[2], high[3]),
                );
            }
            offset += 16;
        }
        (offset, vmaxvq_u8(maximum))
    }
}
