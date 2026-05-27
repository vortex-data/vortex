// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Checked `u32 + u32` over two nullable columns: `vortex_array::compute::checked_add`
//! (built on the autovectorizable `vortex_buffer::lane_ops_indexed` kernels) vs
//! `arrow_arith::numeric::add`. Both compute the validity union and a checked add; arrow's
//! checked path is scalar (per-element `?`), the lane kernel vectorizes.

#![expect(clippy::unwrap_used, clippy::clone_on_ref_ptr)]

use std::sync::Arc;

use arrow_array::Datum;
use arrow_array::UInt32Array;
use arrow_buffer::NullBuffer;
use arrow_buffer::ScalarBuffer;
use divan::Bencher;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vortex_array::LEGACY_SESSION;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::compute::checked_add;
use vortex_array::validity::Validity;
use vortex_buffer::BitBuffer;
use vortex_buffer::Buffer;

fn main() {
    divan::main();
}

const SIZES: &[usize] = &[4_096, 65_536, 1_048_576];
const LHS_VALID_RATE: f64 = 0.7;
const RHS_VALID_RATE: f64 = 0.8;

struct Fixture {
    vortex_lhs: PrimitiveArray,
    vortex_rhs: PrimitiveArray,
    arrow_lhs: Arc<UInt32Array>,
    arrow_rhs: Arc<UInt32Array>,
}

fn fixture(n: usize) -> Fixture {
    let mut vr = StdRng::seed_from_u64(0);
    let mut wr = StdRng::seed_from_u64(1);
    let mut lvr = StdRng::seed_from_u64(2);
    let mut rvr = StdRng::seed_from_u64(3);

    let raw_lhs: Vec<u32> = (0..n)
        .map(|_| vr.random_range(0..u16::MAX as u32))
        .collect();
    let raw_rhs: Vec<u32> = (0..n)
        .map(|_| wr.random_range(0..u16::MAX as u32))
        .collect();
    let lhs_valid: Vec<bool> = (0..n).map(|_| lvr.random_bool(LHS_VALID_RATE)).collect();
    let rhs_valid: Vec<bool> = (0..n).map(|_| rvr.random_bool(RHS_VALID_RATE)).collect();

    let vortex_lhs = PrimitiveArray::new(
        Buffer::from(raw_lhs.clone()),
        Validity::from(BitBuffer::from(lhs_valid.clone())),
    );
    let vortex_rhs = PrimitiveArray::new(
        Buffer::from(raw_rhs.clone()),
        Validity::from(BitBuffer::from(rhs_valid.clone())),
    );

    let arrow_lhs = Arc::new(UInt32Array::new(
        ScalarBuffer::from(raw_lhs),
        Some(NullBuffer::from(lhs_valid)),
    ));
    let arrow_rhs = Arc::new(UInt32Array::new(
        ScalarBuffer::from(raw_rhs),
        Some(NullBuffer::from(rhs_valid)),
    ));

    Fixture {
        vortex_lhs,
        vortex_rhs,
        arrow_lhs,
        arrow_rhs,
    }
}

#[divan::bench(args = SIZES)]
fn vortex_checked_add(bencher: Bencher, n: usize) {
    let f = fixture(n);
    bencher
        .with_inputs(|| {
            (
                f.vortex_lhs.clone(),
                f.vortex_rhs.clone(),
                LEGACY_SESSION.create_execution_ctx(),
            )
        })
        .bench_refs(|(lhs, rhs, ctx)| checked_add(lhs, rhs, ctx).unwrap());
}

#[divan::bench(args = SIZES)]
fn arrow_add(bencher: Bencher, n: usize) {
    let f = fixture(n);
    bencher
        .with_inputs(|| (f.arrow_lhs.clone(), f.arrow_rhs.clone()))
        .bench_refs(|(lhs, rhs)| {
            arrow_arith::numeric::add(lhs.as_ref() as &dyn Datum, rhs.as_ref() as &dyn Datum)
                .unwrap()
        });
}
