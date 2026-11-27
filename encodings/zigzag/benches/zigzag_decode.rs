// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Decode 64 Ki signed sensor residuals represented as ZigZag over nine-bit packed integers.

#![expect(clippy::unwrap_used)]

use std::sync::LazyLock;

use divan::Bencher;
use mimalloc::MiMalloc;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::session::ArraySession;
#[cfg(not(codspeed))]
use vortex_array::test_harness::fusion::fusion_trace;
use vortex_fastlanes::bitpack_compress::bitpack_encode;
use vortex_pco::Pco;
use vortex_session::VortexSession;
use vortex_zigzag::ZigZag;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty().with::<ArraySession>();
    vortex_fastlanes::initialize(&session);
    vortex_zigzag::initialize(&session);
    vortex_pco::initialize(&session);
    session
});

fn main() {
    LazyLock::force(&SESSION);
    divan::main();
}

#[divan::bench(args = [false, true])]
fn packed_sensor_residuals(bencher: Bencher, fused: bool) {
    let encoded =
        PrimitiveArray::from_iter((0..65_536u32).map(|i| i.wrapping_mul(2_654_435_761) >> 23));
    let packed = bitpack_encode(&encoded, 9, None, &mut SESSION.create_execution_ctx()).unwrap();
    let array = ZigZag::try_new(packed.into_array()).unwrap().into_array();
    bencher
        .with_inputs(|| {
            (
                array.clone(),
                SESSION.create_execution_ctx().with_block_fusion(fused),
            )
        })
        .bench_values(|(array, mut ctx)| array.execute::<PrimitiveArray>(&mut ctx).unwrap());
}

/// The compact compressor's integer-runs golden fixture, with its standard 8 Ki Pco pages.
#[divan::bench(args = [false, true])]
fn pco_integer_runs(bencher: Bencher, fused: bool) {
    bench_pco_runs(bencher, fused, false);
}

/// Integer run data with missing measurements.
#[divan::bench(args = [false, true])]
fn pco_nullable_runs(bencher: Bencher, fused: bool) {
    bench_pco_runs(bencher, fused, true);
}

fn bench_pco_runs(bencher: Bencher, fused: bool, nullable: bool) {
    let mut rng = StdRng::seed_from_u64(103);
    let mut values = Vec::with_capacity(16_384);
    while values.len() < 16_384 {
        let value = rng.random_range(-50_000..50_000i32);
        let encoded = ((value << 1) ^ (value >> 31)).cast_unsigned();
        let run = rng.random_range(8..25);
        values.extend(std::iter::repeat_n(encoded, run));
    }
    values.truncate(16_384);
    let encoded = if nullable {
        PrimitiveArray::from_option_iter(
            values
                .into_iter()
                .enumerate()
                .map(|(i, v)| (i % 11 != 0).then_some(v)),
        )
    } else {
        PrimitiveArray::from_iter(values)
    };
    let pco = Pco::from_primitive(
        encoded.as_view(),
        8,
        8192,
        &mut SESSION.create_execution_ctx(),
    )
    .unwrap();
    let array = ZigZag::try_new(pco.into_array()).unwrap().into_array();
    #[cfg(not(codspeed))]
    {
        let trace = fusion_trace(array.clone(), &SESSION).unwrap();
        assert_eq!(
            trace.contains("fuse start array=vortex.pco"),
            nullable,
            "{trace}"
        );
    }
    bencher
        .with_inputs(|| {
            (
                array.clone(),
                SESSION.create_execution_ctx().with_block_fusion(fused),
            )
        })
        .bench_values(|(array, mut ctx)| array.execute::<PrimitiveArray>(&mut ctx).unwrap());
}
