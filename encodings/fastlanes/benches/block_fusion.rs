// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Block fusion in the executor against level-wise execution, on the same arrays.
//!
//! Each tree is executed to canonical twice: `fused` with block fusion on, `levelwise` with it
//! off. `for_stack` with depth 1 is FoR over BitPacked alone, which fuses nothing and so measures
//! the noise between the two.

#![expect(clippy::unwrap_used)]
#![allow(clippy::cast_possible_truncation)]

use std::sync::LazyLock;

use divan::Bencher;
use mimalloc::MiMalloc;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::scalar::Scalar;
use vortex_fastlanes::FoR;
use vortex_fastlanes::bitpack_compress::bitpack_encode;
use vortex_session::VortexSession;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    divan::main();
}

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty();
    vortex_fastlanes::initialize(&session);
    session
});

/// A scan split, whose intermediates stay in L2, and a large array, whose do not.
const LENS: [usize; 2] = [64 * 1024, 4 * 1024 * 1024];

fn bench_execute(bencher: Bencher, array: &ArrayRef, fused: bool) {
    bencher
        .with_inputs(|| {
            (
                array.clone(),
                SESSION.create_execution_ctx().with_block_fusion(fused),
            )
        })
        .bench_values(|(array, mut ctx)| array.execute::<PrimitiveArray>(&mut ctx).unwrap());
}

/// `depth` FoR levels over a BitPacked leaf at width 10.
fn for_stack(len: usize, depth: usize) -> ArrayRef {
    let mut ctx = SESSION.create_execution_ctx();
    let values = PrimitiveArray::from_iter((0..len as u64).map(|i| ((i * 7) % 1000) as u32));
    let mut array = bitpack_encode(&values, 10, None, &mut ctx)
        .unwrap()
        .into_array();
    for level in 0..depth {
        array = FoR::try_new(array, Scalar::from(1_000u32 + level as u32))
            .unwrap()
            .into_array();
    }
    array
}

/// `depth` FoR levels over an uncompressed leaf.
fn for_over_primitive(len: usize, depth: usize) -> ArrayRef {
    let mut array =
        PrimitiveArray::from_iter((0..len as u64).map(|i| ((i * 7) % 1000) as u32)).into_array();
    for level in 0..depth {
        array = FoR::try_new(array, Scalar::from(1_000u32 + level as u32))
            .unwrap()
            .into_array();
    }
    array
}

const DEPTHS: &[usize] = &[1, 2, 3, 4, 8];

#[divan::bench(consts = LENS, args = DEPTHS)]
fn for_stack_fused<const N: usize>(bencher: Bencher, depth: usize) {
    bench_execute(bencher, &for_stack(N, depth), true);
}

#[divan::bench(consts = LENS, args = DEPTHS)]
fn for_stack_levelwise<const N: usize>(bencher: Bencher, depth: usize) {
    bench_execute(bencher, &for_stack(N, depth), false);
}

/// A slice of FoR over BitPacked: FoR waits on the slice, which resolves into a BitPacked leaf
/// that is fused with FoR's op instead of FoR's own fused kernel.
const SLICE_STARTS: &[usize] = &[512, 517];

#[divan::bench(consts = LENS, args = SLICE_STARTS)]
fn sliced_for_bp_fused<const N: usize>(bencher: Bencher, start: usize) {
    bench_execute(
        bencher,
        &for_stack(N + 1024, 1).slice(start..N + start).unwrap(),
        true,
    );
}

#[divan::bench(consts = LENS, args = SLICE_STARTS)]
fn sliced_for_bp_levelwise<const N: usize>(bencher: Bencher, start: usize) {
    bench_execute(
        bencher,
        &for_stack(N + 1024, 1).slice(start..N + start).unwrap(),
        false,
    );
}

const PRIMITIVE_DEPTHS: &[usize] = &[2, 4];

#[divan::bench(consts = LENS, args = PRIMITIVE_DEPTHS)]
fn for_over_primitive_fused<const N: usize>(bencher: Bencher, depth: usize) {
    bench_execute(bencher, &for_over_primitive(N, depth), true);
}

#[divan::bench(consts = LENS, args = PRIMITIVE_DEPTHS)]
fn for_over_primitive_levelwise<const N: usize>(bencher: Bencher, depth: usize) {
    bench_execute(bencher, &for_over_primitive(N, depth), false);
}
