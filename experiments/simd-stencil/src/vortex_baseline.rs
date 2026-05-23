// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Real Vortex decode baselines.
//!
//! These build genuine Vortex compressed arrays over the *same* synthetic data
//! the prototype encodes, then decode them through Vortex's production
//! array-by-array `execute` path. This anchors the prototype's numbers to the
//! shipping engine (the in-crate "materialized" strategy is the controlled,
//! same-kernel model of this same path).
//!
//! Two flavours of baseline per stack:
//! - `*_same_stack` / `build_b_full_same_stack`: the *same* encoding the
//!   prototype decodes, hand-built from Vortex's public constructors, so it is a
//!   fair head-to-head against the fused/stitched pipeline.
//! - `build_a` / `build_b` / `build_b_core_shallow` / `build_c_regular`: what
//!   Vortex's own compressor / encoders pick ("regular" Vortex) — typically a
//!   shallower encoding that decodes faster but compresses worse.

use vortex_alp::ALP;
use vortex_alp::ALPArrayExt;
use vortex_alp::alp_encode;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::LEGACY_SESSION;
use vortex_array::RecursiveCanonical;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::Nullability;
use vortex_array::scalar::Scalar;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_fastlanes::Delta;
use vortex_fastlanes::FoR;
use vortex_fastlanes::bitpack_compress::bitpack_to_best_bit_width;
use vortex_runend::RunEnd;

/// Build a real Vortex Delta array over the `u32` column, with its deltas left
/// uncompressed (what Vortex's own compressor chose). This decodes in fewer
/// passes than the explicit `delta(bitpacking)` stack — see `build_a_same_stack`.
pub fn build_a(values: &[u32]) -> ArrayRef {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let prim = PrimitiveArray::new(Buffer::copy_from(values), Validity::NonNullable);
    Delta::try_from_primitive_array(&prim, &mut ctx)
        .expect("delta encode")
        .into_array()
}

/// Build a genuine Vortex `delta(bitpacking)` array: a `DeltaArray` whose
/// `deltas` child is a `BitPackedArray`. This is the *same* stack the prototype
/// decodes, so Vortex's array-by-array `execute` (which materialises the
/// unpacked deltas, then undeltas) is a fair head-to-head against the fused
/// stencil pipeline.
pub fn build_a_same_stack(values: &[u32]) -> ArrayRef {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let prim = PrimitiveArray::new(Buffer::copy_from(values), Validity::NonNullable);
    let delta = Delta::try_from_primitive_array(&prim, &mut ctx)
        .expect("delta encode")
        .into_array();
    let len = delta.len();
    let children = delta.children();
    let bases = children[0].clone();
    let deltas = children[1]
        .clone()
        .execute::<PrimitiveArray>(&mut ctx)
        .expect("deltas to primitive");
    let packed = bitpack_to_best_bit_width(&deltas, &mut ctx).expect("bitpack deltas");
    Delta::try_new(bases, packed.into_array(), 0, len)
        .expect("delta(bitpacking)")
        .into_array()
}

/// Build a genuine Vortex `delta(ffor(bitpacking))` array over `i64` digits:
/// `DeltaArray` -> `FoRArray` -> `BitPackedArray`, decoded by Vortex's per-layer
/// `execute`.
fn deep_delta_for_bitpack(digits: &[i64]) -> ArrayRef {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let prim = PrimitiveArray::new(Buffer::copy_from(digits), Validity::NonNullable);
    let delta = Delta::try_from_primitive_array(&prim, &mut ctx)
        .expect("delta encode")
        .into_array();
    let len = delta.len();
    let children = delta.children();
    let bases = children[0].clone();
    let dvals = children[1]
        .clone()
        .execute::<PrimitiveArray>(&mut ctx)
        .expect("deltas to primitive");

    // FoR: shift by the signed minimum so the residuals bit-pack tightly.
    let min = dvals.as_slice::<i64>().iter().copied().min().unwrap_or(0);
    let resid: Vec<i64> = dvals.as_slice::<i64>().iter().map(|&d| d - min).collect();
    let resid = PrimitiveArray::new(Buffer::copy_from(&resid), Validity::NonNullable);
    let packed = bitpack_to_best_bit_width(&resid, &mut ctx).expect("bitpack residuals");
    let for_ = FoR::try_new(
        packed.into_array(),
        Scalar::primitive(min, Nullability::NonNullable),
    )
    .expect("ffor(bitpacking)");
    Delta::try_new(bases, for_.into_array(), 0, len)
        .expect("delta(ffor(bitpacking))")
        .into_array()
}

/// Same-stack integer core of stack B: `delta(ffor(bitpacking))` over `i64`.
pub fn build_b_core_same_stack(digits: &[i64]) -> ArrayRef {
    deep_delta_for_bitpack(digits)
}

/// Regular-Vortex equivalent of the stack-B integer core: Vortex's own Delta
/// encoding of the digits (deltas left uncompressed, like `build_a`).
pub fn build_b_core_shallow(digits: &[i64]) -> ArrayRef {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let prim = PrimitiveArray::new(Buffer::copy_from(digits), Validity::NonNullable);
    Delta::try_from_primitive_array(&prim, &mut ctx)
        .expect("delta encode")
        .into_array()
}

/// Build a genuine full `alp(delta(ffor(bitpacking)))` Vortex array: take the
/// exponents, patches and digits from Vortex's own `alp_encode`, then replace
/// the (raw) encoded child with the deep `delta(ffor(bitpacking))` cascade over
/// the same digits. Decodes identically to regular ALP, but over the deep stack.
pub fn build_b_full_same_stack(values: &[f64]) -> ArrayRef {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let prim = PrimitiveArray::new(Buffer::copy_from(values), Validity::NonNullable);
    let alp = alp_encode(prim.as_view(), None, &mut ctx).expect("alp encode");
    let exponents = alp.exponents();
    let patches = alp.patches();
    let alp_arr = alp.into_array();
    let digits = alp_arr.children()[0]
        .clone()
        .execute::<PrimitiveArray>(&mut ctx)
        .expect("alp digits to primitive");
    let deep = deep_delta_for_bitpack(digits.as_slice::<i64>());
    ALP::try_new(deep, exponents, patches)
        .expect("alp(delta(ffor(bitpacking)))")
        .into_array()
}

/// Build a real Vortex ALP array over the `f64` column ("regular" Vortex: ALP
/// over an uncompressed/bit-packed encoded child, as the compressor picks).
pub fn build_b(values: &[f64]) -> ArrayRef {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let prim = PrimitiveArray::new(Buffer::copy_from(values), Validity::NonNullable);
    alp_encode(prim.as_view(), None, &mut ctx)
        .expect("alp encode")
        .into_array()
}

/// Regular-Vortex equivalent of stack C: Vortex's RunEnd encoding of the logical
/// `f64` column, decoded via `execute`.
pub fn build_c_regular(values: &[f64]) -> ArrayRef {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let prim = PrimitiveArray::new(Buffer::copy_from(values), Validity::NonNullable);
    RunEnd::encode(prim.into_array(), &mut ctx)
        .expect("runend encode")
        .into_array()
}

/// Decode a Vortex array to a canonical primitive array via `execute`.
pub fn decode(array: &ArrayRef) -> PrimitiveArray {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    array
        .clone()
        .execute::<PrimitiveArray>(&mut ctx)
        .expect("execute")
}

/// Decode every layer through Vortex and finish with a single `execute` to the
/// recursive canonical form — the idiomatic end-to-end Vortex scan output.
pub fn decode_canonical(array: &ArrayRef) -> RecursiveCanonical {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    array
        .clone()
        .execute::<RecursiveCanonical>(&mut ctx)
        .expect("execute canonical")
}

/// Decode a stack-A array to a `u32` vector.
pub fn decode_a(array: &ArrayRef) -> Vec<u32> {
    decode(array).as_slice::<u32>().to_vec()
}

/// Decode a stack-B array to an `f64` vector.
pub fn decode_b(array: &ArrayRef) -> Vec<f64> {
    decode(array).as_slice::<f64>().to_vec()
}
