// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Scalar arithmetic kernels: correctness references and non-SIMD baselines.

use crate::layout::SplitI128;
use crate::layout::SplitI256;

/// Array-of-structs i128 add (the layout Arrow uses). The compiler lowers each
/// `i128` add to a 64-bit `add` + `adc`; it cannot lane-parallelize across
/// elements because the carry crosses the two halves of one value.
pub fn add_i128_aos(a: &[i128], b: &[i128], out: &mut [i128]) {
    for i in 0..a.len() {
        out[i] = a[i].wrapping_add(b[i]);
    }
}

/// Struct-of-arrays i128 add, scalar. Same math as AVX-512 but one lane at a
/// time, so it isolates the win from layout vs the win from SIMD.
pub fn add_i128_soa(a: &SplitI128, b: &SplitI128, out: &mut SplitI128) {
    for i in 0..a.len() {
        let (lo, carry) = a.lo[i].overflowing_add(b.lo[i]);
        out.lo[i] = lo;
        out.hi[i] = a.hi[i].wrapping_add(b.hi[i]).wrapping_add(u64::from(carry));
    }
}

/// Struct-of-arrays i128 subtract, scalar.
pub fn sub_i128_soa(a: &SplitI128, b: &SplitI128, out: &mut SplitI128) {
    for i in 0..a.len() {
        let (lo, borrow) = a.lo[i].overflowing_sub(b.lo[i]);
        out.lo[i] = lo;
        out.hi[i] = a.hi[i]
            .wrapping_sub(b.hi[i])
            .wrapping_sub(u64::from(borrow));
    }
}

/// "Small decimal" fast path: assume the value fits in the low limb, so add is
/// just a 64-bit add with no carry into `hi`. This is what makes small decimals
/// cheap once the high limb is known to be a constant (zero / sign word).
pub fn add_i128_lo_only(a: &SplitI128, b: &SplitI128, out: &mut SplitI128) {
    for i in 0..a.lo.len() {
        out.lo[i] = a.lo[i].wrapping_add(b.lo[i]);
    }
}

/// Array-of-structs i256 add baseline using the limb math directly on
/// reassembled values (mirrors what an interleaved kernel must do per element).
pub fn add_i256_soa(a: &SplitI256, b: &SplitI256, out: &mut SplitI256) {
    let n = a.len();
    for i in 0..n {
        let mut carry = 0u64;
        for k in 0..4 {
            let (s1, c1) = a.limbs[k][i].overflowing_add(b.limbs[k][i]);
            let (s2, c2) = s1.overflowing_add(carry);
            out.limbs[k][i] = s2;
            carry = u64::from(c1 || c2);
        }
    }
}
