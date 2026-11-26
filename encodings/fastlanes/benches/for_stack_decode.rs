// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Executing a stack of FoR levels over BitPacked level by level, where each FoR writes a
//! full-length buffer, and with block fusion, which decodes the stack one block at a time.
//!
//! Run with `cargo bench -p vortex-fastlanes --bench for_stack_decode`.

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
use vortex_array::session::ArraySession;
use vortex_fastlanes::FoR;
use vortex_fastlanes::bitpack_compress::bitpack_encode;
use vortex_session::VortexSession;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    LazyLock::force(&SESSION);
    divan::main();
}

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty().with::<ArraySession>();
    vortex_fastlanes::initialize(&session);
    session
});

/// A scan split's rows, whose intermediates fit in L2.
const LEN: usize = 64 * 1024;

/// FoR levels over the BitPacked leaf.
const DEPTH: usize = 4;

/// `depth` FoR levels over values bit-packed at width 10.
fn for_stack(depth: usize) -> ArrayRef {
    let mut ctx = SESSION.create_execution_ctx();
    let values = PrimitiveArray::from_iter((0..LEN as u32).map(|i| (i * 7) % 1000));
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

#[divan::bench]
fn for_stack_execute(bencher: Bencher) {
    let array = for_stack(DEPTH);
    bencher
        .with_inputs(|| (array.clone(), SESSION.create_execution_ctx()))
        .bench_values(|(array, mut ctx)| array.execute::<PrimitiveArray>(&mut ctx).unwrap());
}

#[divan::bench]
fn for_stack_fused(bencher: Bencher) {
    let array = for_stack(DEPTH);
    bencher
        .with_inputs(|| {
            let ctx = SESSION.create_execution_ctx().with_block_fusion(true);
            (array.clone(), ctx)
        })
        .bench_values(|(array, mut ctx)| array.execute::<PrimitiveArray>(&mut ctx).unwrap());
}
