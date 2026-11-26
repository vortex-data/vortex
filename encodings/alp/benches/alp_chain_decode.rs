// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Executing ALP over FoR over BitPacked, the layout the compressor writes for decimal floats, level
//! by level, where FoR writes a full-length buffer of integers that ALP then decodes, and with block
//! fusion, which unpacks, adds the reference and decodes one block at a time.
//!
//! Run with `cargo bench -p vortex-alp --bench alp_chain_decode`.

#![expect(clippy::unwrap_used)]
#![allow(clippy::cast_possible_truncation)]

use std::f64::consts::PI;
use std::sync::LazyLock;

use divan::Bencher;
use mimalloc::MiMalloc;
use vortex_alp::ALP;
use vortex_alp::ALPArrayExt;
use vortex_alp::ALPArraySlotsExt;
use vortex_alp::alp_encode;
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
    vortex_alp::initialize(&session);
    session
});

/// A scan split's rows, whose intermediates fit in L2.
const LEN: usize = 64 * 1024;

/// Values of which every 997th is not representable, so ALP patches it.
fn alp_for_bp() -> ArrayRef {
    let mut ctx = SESSION.create_execution_ctx();
    let values = PrimitiveArray::from_iter((0..LEN).map(|i| {
        if i % 997 == 0 {
            PI * i as f64
        } else {
            100.0 + (i % 1000) as f64 * 0.25
        }
    }));
    let alp = alp_encode(values.as_view(), None, &mut ctx).unwrap();
    let encoded = alp
        .encoded()
        .clone()
        .execute::<PrimitiveArray>(&mut ctx)
        .unwrap();
    let min = encoded.as_slice::<i64>().iter().copied().min().unwrap();
    let shifted = PrimitiveArray::from_iter(encoded.as_slice::<i64>().iter().map(|&e| e - min));
    let max = shifted.as_slice::<i64>().iter().copied().max().unwrap();
    let bp = bitpack_encode(&shifted, (64 - max.leading_zeros()) as u8, None, &mut ctx).unwrap();
    let for_bp = FoR::try_new(bp.into_array(), Scalar::from(min))
        .unwrap()
        .into_array();
    ALP::try_new(for_bp, alp.exponents(), alp.patches())
        .unwrap()
        .into_array()
}

#[divan::bench]
fn alp_for_bp_execute(bencher: Bencher) {
    let array = alp_for_bp();
    bencher
        .with_inputs(|| (array.clone(), SESSION.create_execution_ctx()))
        .bench_values(|(array, mut ctx)| array.execute::<PrimitiveArray>(&mut ctx).unwrap());
}

#[divan::bench]
fn alp_for_bp_fused(bencher: Bencher) {
    let array = alp_for_bp();
    bencher
        .with_inputs(|| {
            let ctx = SESSION.create_execution_ctx().with_block_fusion(true);
            (array.clone(), ctx)
        })
        .bench_values(|(array, mut ctx)| array.execute::<PrimitiveArray>(&mut ctx).unwrap());
}
