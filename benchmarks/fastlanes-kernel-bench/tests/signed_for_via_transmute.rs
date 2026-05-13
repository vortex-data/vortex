// SPDX-FileCopyrightText: Copyright the Vortex contributors
// SPDX-License-Identifier: Apache-2.0
//
//! Proves that a single *unsigned* FoR kernel (`FoR::for_pack` /
//! `FoR::unfor_pack` for `u8`/`u16`/`u32`/`u64`) is sufficient to handle
//! frame-of-reference encoding of signed integers, in both directions of the
//! delta (values strictly above the reference and values strictly below).
//!
//! No `i*`-typed kernel is needed. We never touch any signed-arithmetic
//! instruction in the hot path -- the unsigned kernel is called as-is and we
//! only `transmute` the slice/reference at the boundary.
//!
//! Why this works:
//!
//! * Bit-packing is shift-and-mask; the bit pattern produced is invariant
//!   under signed vs unsigned reinterpretation of the operand.
//! * `wrapping_add` / `wrapping_sub` on a `T`-bit two's-complement integer
//!   produce identical bit patterns regardless of whether the operands are
//!   treated as `iT` or `uT`. So FoR's `value - reference` (encode) and
//!   `packed + reference` (decode) round-trip the bit pattern losslessly.
//! * The conventional FoR rule "reference = min(values)" makes every delta
//!   non-negative *as a signed quantity*, which is the same bit pattern as
//!   a small non-negative unsigned quantity, so the W low bits stored by
//!   `pack` reconstruct the original signed value when `wrapping_add(ref)`
//!   is applied.

use fastlanes_kernel_bench::FoR;

/// Treat `&[i32; 1024]` as `&[u32; 1024]` without copying. Equivalent to what
/// `vortex_array::PrimitiveArray::reinterpret_cast(ptype.to_unsigned())` does
/// in the production Vortex bitpacking path.
fn as_u32_array(s: &[i32; 1024]) -> &[u32; 1024] {
    // SAFETY: i32 and u32 have identical layout (size 4, align 4).
    unsafe { &*(s as *const [i32; 1024] as *const [u32; 1024]) }
}

fn as_i32_array_mut(s: &mut [u32; 1024]) -> &mut [i32; 1024] {
    // SAFETY: u32 and i32 have identical layout.
    unsafe { &mut *(s as *mut [u32; 1024] as *mut [i32; 1024]) }
}

/// Round-trip a signed i32 dataset that straddles zero through the unsigned
/// FoR kernel. Reference is `min(values)` so all signed deltas are
/// non-negative -- the standard practice and the only thing the W-bit
/// zero-extending unpack supports correctly.
#[test]
fn i32_round_trip_through_unsigned_kernel() {
    const W: usize = 11; // need ceil(log2(max - min + 1)) bits, range is 1024 here
    const B: usize = 1024 * W / 32;

    // Values from -500 to +523 inclusive (exactly 1024 distinct values).
    let mut values: [i32; 1024] = [0; 1024];
    for (i, v) in values.iter_mut().enumerate() {
        *v = (i as i32) - 500;
    }
    let reference_i32: i32 = -500;

    // Encode using the unsigned kernel via transmute. Note: bitwise this is
    // the *same operation* as if we'd written an i32 kernel, because pack is
    // pure shift+mask and `wrapping_sub` is sign-agnostic.
    let mut packed = [0u32; B];
    <u32 as FoR>::for_pack::<W, B>(
        as_u32_array(&values),
        reference_i32 as u32,
        &mut packed,
    );

    // Decode.
    let mut decoded = [0u32; 1024];
    <u32 as FoR>::unfor_pack::<W, B>(&packed, reference_i32 as u32, &mut decoded);
    let decoded_i32 = as_i32_array_mut(&mut decoded);

    assert_eq!(decoded_i32, &values, "i32 round-trip failed");
}

/// Same idea but for i64, and with a positive reference (values entirely
/// above zero but their deltas relative to `reference` go in both directions
/// of zero IF we chose a non-min reference -- here we still use min, which is
/// the only way to keep all deltas non-negative).
#[test]
fn i64_round_trip_through_unsigned_kernel() {
    const W: usize = 10;
    const B: usize = 1024 * W / 64;

    let mut values: [i64; 1024] = [0; 1024];
    for (i, v) in values.iter_mut().enumerate() {
        *v = 1_000_000 + (i as i64);
    }
    let reference_i64: i64 = 1_000_000;

    let mut packed = [0u64; B];

    // SAFETY: i64 and u64 have identical layout.
    let values_u64: &[u64; 1024] =
        unsafe { &*(&values as *const [i64; 1024] as *const [u64; 1024]) };
    <u64 as FoR>::for_pack::<W, B>(values_u64, reference_i64 as u64, &mut packed);

    let mut decoded = [0u64; 1024];
    <u64 as FoR>::unfor_pack::<W, B>(&packed, reference_i64 as u64, &mut decoded);

    // SAFETY: u64 and i64 have identical layout.
    let decoded_i64: &[i64; 1024] =
        unsafe { &*(&decoded as *const [u64; 1024] as *const [i64; 1024]) };
    assert_eq!(decoded_i64, &values, "i64 round-trip failed");
}

/// Round-trip an i32 dataset that uses a `reference` that produces NEGATIVE
/// deltas for some values (i.e. `value < reference`). With reference = min,
/// all deltas are non-negative -- so this test deliberately uses a
/// non-canonical reference and shows that the round trip still works
/// *bit-exactly* because of `wrapping_sub` / `wrapping_add` symmetry, even
/// though some intermediate "packed" values are huge unsigned numbers.
///
/// The catch: the W chosen must be wide enough that the wrapped delta still
/// fits modulo `2^W`. Here we choose W=32 (full width) so all deltas survive.
#[test]
fn i32_with_arbitrary_reference_round_trips_when_w_is_full_width() {
    const W: usize = 32;
    const B: usize = 1024 * W / 32; // = 1024

    let mut values: [i32; 1024] = [0; 1024];
    for (i, v) in values.iter_mut().enumerate() {
        // Half positive, half negative relative to `reference` below.
        *v = (i as i32) - 200;
    }
    let reference_i32: i32 = 0; // NOT min -- deltas straddle zero.

    let mut packed = [0u32; B];
    <u32 as FoR>::for_pack::<W, B>(
        as_u32_array(&values),
        reference_i32 as u32,
        &mut packed,
    );

    let mut decoded = [0u32; 1024];
    <u32 as FoR>::unfor_pack::<W, B>(&packed, reference_i32 as u32, &mut decoded);
    let decoded_i32 = as_i32_array_mut(&mut decoded);

    assert_eq!(
        decoded_i32, &values,
        "i32 round-trip with bidirectional deltas failed"
    );
}
