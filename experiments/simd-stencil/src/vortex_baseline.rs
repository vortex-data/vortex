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
//! Note: Vortex's public constructors don't expose a hand-built
//! `alp(delta(ffor(bitpacking)))` cascade, so stack A is decoded as Vortex's
//! Delta encoding and stack B as Vortex's ALP encoding of the identical inputs.

use vortex_alp::alp_encode;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::LEGACY_SESSION;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_fastlanes::Delta;

/// Build a real Vortex Delta array over the `u32` column.
pub fn build_a(values: &[u32]) -> ArrayRef {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let prim = PrimitiveArray::new(Buffer::copy_from(values), Validity::NonNullable);
    Delta::try_from_primitive_array(&prim, &mut ctx)
        .expect("delta encode")
        .into_array()
}

/// Build a real Vortex ALP array over the `f64` column.
pub fn build_b(values: &[f64]) -> ArrayRef {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let prim = PrimitiveArray::new(Buffer::copy_from(values), Validity::NonNullable);
    alp_encode(prim.as_view(), None, &mut ctx)
        .expect("alp encode")
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

/// Decode a stack-A array to a `u32` vector.
pub fn decode_a(array: &ArrayRef) -> Vec<u32> {
    decode(array).as_slice::<u32>().to_vec()
}

/// Decode a stack-B array to an `f64` vector.
pub fn decode_b(array: &ArrayRef) -> Vec<f64> {
    decode(array).as_slice::<f64>().to_vec()
}
