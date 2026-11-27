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
use vortex_alp::Exponents;
use vortex_alp::alp_encode;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::Patched;
use vortex_array::arrays::PiecewiseSequenceArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::PType;
use vortex_array::scalar::Scalar;
use vortex_array::session::ArraySession;
use vortex_array::validity::Validity;
use vortex_fastlanes::BitPacked;
use vortex_fastlanes::BitPackedArrayExt;
use vortex_fastlanes::Delta;
use vortex_fastlanes::DeltaArraySlotsExt;
use vortex_fastlanes::FoR;
use vortex_fastlanes::bitpack_compress::bitpack_encode;
use vortex_fastlanes::bitpack_compress::bitpack_encode_blocked;
use vortex_session::VortexSession;
use vortex_sparse::Sparse;

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
    vortex_sparse::initialize(&session);
    session
});

/// A scan split's rows, whose intermediates fit in L2.
const LEN: usize = 64 * 1024;

/// Values of which every 997th is not representable, so ALP patches it.
fn alp_for_bp(blocked: bool) -> ArrayRef {
    let mut ctx = SESSION.create_execution_ctx();
    let values = PrimitiveArray::from_iter((0..LEN).map(|i| {
        if i % 997 == 0 {
            PI * i as f64
        } else {
            100.0
                + (i % if blocked && (i / 1024) % 2 == 0 {
                    16
                } else {
                    1000
                }) as f64
                    * 0.25
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
    let widths: Vec<u8> = shifted
        .as_slice::<i64>()
        .chunks(1024)
        .map(|block| (64 - block.iter().copied().max().unwrap().leading_zeros()) as u8)
        .collect();
    let bp = if blocked {
        bitpack_encode_blocked(&shifted, &widths, Some(0), &mut ctx).unwrap()
    } else {
        bitpack_encode(&shifted, *widths.iter().max().unwrap(), None, &mut ctx).unwrap()
    };
    let for_bp = FoR::try_new(bp.into_array(), Scalar::from(min))
        .unwrap()
        .into_array();
    ALP::try_new(for_bp, alp.exponents(), alp.patches())
        .unwrap()
        .into_array()
}

#[divan::bench]
fn alp_for_bp_execute(bencher: Bencher) {
    let array = alp_for_bp(false);
    bencher
        .with_inputs(|| {
            (
                array.clone(),
                SESSION.create_execution_ctx().with_block_fusion(false),
            )
        })
        .bench_values(|(array, mut ctx)| array.execute::<PrimitiveArray>(&mut ctx).unwrap());
}

#[divan::bench]
fn alp_for_bp_fused(bencher: Bencher) {
    let array = alp_for_bp(false);
    bencher
        .with_inputs(|| {
            let ctx = SESSION.create_execution_ctx().with_block_fusion(true);
            (array.clone(), ctx)
        })
        .bench_values(|(array, mut ctx)| array.execute::<PrimitiveArray>(&mut ctx).unwrap());
}

#[divan::bench(args = [false, true])]
fn alp_blocked_for_bp(bencher: Bencher, fused: bool) {
    run(bencher, alp_for_bp(true), fused, "fastlanes.for");
}

fn run(bencher: Bencher, array: ArrayRef, fused: bool, source: &str) {
    let _ = source;
    #[cfg(not(codspeed))]
    {
        let trace =
            vortex_array::test_harness::fusion::fusion_trace(array.clone(), &SESSION).unwrap();
        assert!(
            trace.contains(&format!("fuse start array={source}")),
            "{trace}"
        );
        assert!(trace.contains("fuse extend parent=vortex.alp"), "{trace}");
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

#[divan::bench(args = [false, true])]
fn alp_packed_delta_counter(bencher: Bencher, fused: bool) {
    let mut ctx = SESSION.create_execution_ctx();
    // A meter reading in eighths of a unit with occasional two-step increases.
    let values =
        PrimitiveArray::from_iter((0..LEN).map(|i| 100_000.0 + (i + i / 11) as f64 * 0.125));
    let alp = alp_encode(values.as_view(), None, &mut ctx).unwrap();
    let counts = alp
        .encoded()
        .clone()
        .execute::<PrimitiveArray>(&mut ctx)
        .unwrap();
    let delta = Delta::try_from_primitive_array(&counts, &mut ctx).unwrap();
    let residuals = delta
        .deltas()
        .clone()
        .execute::<PrimitiveArray>(&mut ctx)
        .unwrap();
    let bits = (64
        - residuals
            .as_slice::<i64>()
            .iter()
            .max()
            .unwrap()
            .leading_zeros()) as u8;
    let packed = bitpack_encode(&residuals, bits, None, &mut ctx).unwrap();
    let delta = Delta::try_new(
        delta.bases().clone(),
        packed.into_array(),
        delta.validity().unwrap(),
        0,
        LEN,
    )
    .unwrap();
    let array = ALP::try_new(delta.into_array(), alp.exponents(), alp.patches())
        .unwrap()
        .into_array();
    run(bencher, array, fused, "fastlanes.delta");
}

/// Repeated invoice amounts share a dictionary of integer cents under the ALP decoder.
#[divan::bench(args = [false, true])]
fn alp_dictionary_prices(bencher: Bencher, fusion: bool) {
    let amount = |index: u16| 100_i64 + i64::from(index) * 73;
    let code = |row: u32| (row.wrapping_mul(2_654_435_761) >> 23) as u16;
    let codes = PrimitiveArray::from_iter((0..LEN as u32).map(code));
    let table = PrimitiveArray::from_iter((0..512u16).map(amount));
    let dict = DictArray::try_new(codes.into_array(), table.into_array()).unwrap();
    let array = ALP::try_new(dict.into_array(), Exponents { e: 2, f: 0 }, None)
        .unwrap()
        .into_array();
    run(bencher, array, fusion, "vortex.dict");
}

/// Dictionary prices sorted into runs of variable length, using the codes from RunEnd::to_dict.
#[divan::bench(args = [false, true])]
fn alp_dictionary_runs(bencher: Bencher, fusion: bool) {
    let runs = 4096u64;
    let lengths = (0..runs).map(|run| 1 + (run * 7) % 31);
    let rows = lengths.clone().sum::<u64>() as usize;
    let codes = PiecewiseSequenceArray::try_new(
        PrimitiveArray::from_iter(0..runs).into_array(),
        PrimitiveArray::from_iter(lengths).into_array(),
        ConstantArray::new(0u64, runs as usize).into_array(),
        rows,
    )
    .unwrap();
    let table = PrimitiveArray::from_iter((0..runs).map(|run| (100 + run * 73) as i64));
    let dict = DictArray::try_new(codes.into_array(), table.into_array()).unwrap();
    let array = ALP::try_new(dict.into_array(), Exponents { e: 2, f: 0 }, None)
        .unwrap()
        .into_array();
    run(bencher, array, fusion, "vortex.dict");
}

fn bench_integer_source(bencher: Bencher, encoded: ArrayRef, fused: bool) {
    let source = encoded.encoding_id().to_string();
    let array = ALP::try_new(encoded, Exponents { e: 2, f: 0 }, None)
        .unwrap()
        .into_array();
    run(bencher, array, fused, &source);
}

/// A baseline sensor reading with a short-lived load spike in one row per hundred.
#[divan::bench(args = [false, true])]
fn alp_sparse_load(bencher: Bencher, fused: bool) {
    let rows = u32::try_from(LEN).unwrap();
    let indices = PrimitiveArray::from_iter((0..rows).step_by(100));
    let readings = PrimitiveArray::from_iter((0..indices.len()).map(|i| 2100 + (i % 31) as i64));
    let sparse = Sparse::try_new(
        indices.into_array(),
        readings.into_array(),
        LEN,
        Scalar::from(2000i64),
        &mut SESSION.create_execution_ctx(),
    )
    .unwrap();
    bench_integer_source(bencher, sparse.into_array(), fused);
}

/// Nine-bit readings with intermittent larger readings, scaled to hundredths.
#[divan::bench(args = [false, true])]
fn alp_patched_packed_readings(bencher: Bencher, fusion: bool) {
    bench_patched_readings(bencher, fusion, true);
}

#[divan::bench(args = [false, true])]
fn alp_native_packed_readings(bencher: Bencher, fusion: bool) {
    bench_patched_readings(bencher, fusion, false);
}

fn bench_patched_readings(bencher: Bencher, fusion: bool, wrapper: bool) {
    let mut ctx = SESSION.create_execution_ctx();
    let rows = 65_536u32;
    let reading = |row: u32| {
        let base = i64::from(row.wrapping_mul(2_654_435_761) >> 23);
        if row.is_multiple_of(97) {
            base + 4096
        } else {
            base
        }
    };
    let values = PrimitiveArray::from_iter((0..rows).map(reading));
    let native = bitpack_encode(&values, 9, None, &mut ctx).unwrap();
    let patches = native.patches().unwrap();
    let packed = if wrapper {
        let inner = BitPacked::try_new(
            native.packed().clone(),
            PType::I64,
            Validity::NonNullable,
            None,
            9,
            native.len(),
            native.offset(),
        )
        .unwrap();
        Patched::from_array_and_patches(inner.into_array(), &patches, &mut ctx)
            .unwrap()
            .into_array()
    } else {
        native.into_array()
    };
    let patched = packed.slice(37..rows as usize - 21).unwrap();
    let array = ALP::try_new(patched, Exponents { e: 2, f: 0 }, None)
        .unwrap()
        .into_array();
    #[cfg(not(codspeed))]
    {
        let trace =
            vortex_array::test_harness::fusion::fusion_trace(array.clone(), &SESSION).unwrap();
        assert_eq!(
            trace.contains("fuse absorb parent=vortex.patched"),
            wrapper,
            "{trace}"
        );
        assert!(
            trace.contains("fuse start array=fastlanes.bitpacked"),
            "{trace}"
        );
        assert!(trace.contains("fuse extend parent=vortex.alp"), "{trace}");
    }
    bencher
        .with_inputs(|| {
            (
                array.clone(),
                SESSION.create_execution_ctx().with_block_fusion(fusion),
            )
        })
        .bench_values(|(array, mut ctx)| array.execute::<PrimitiveArray>(&mut ctx).unwrap());
}
