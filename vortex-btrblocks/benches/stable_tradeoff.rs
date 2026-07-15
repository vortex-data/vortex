// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Targeted size-versus-execution benchmark for stable encodings.

#![expect(clippy::unwrap_used)]

use std::sync::Arc;
use std::sync::LazyLock;

use divan::Bencher;
use divan::counter::ItemsCount;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::scalar_fn::ScalarFnFactoryExt;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::scalar_fn::fns::like::Like;
use vortex_array::scalar_fn::fns::like::LikeOptions;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_btrblocks::ArrayAndStats;
use vortex_btrblocks::BtrBlocksCompressorBuilder;
use vortex_btrblocks::Cost;
use vortex_btrblocks::CostModel;
use vortex_btrblocks::ScanCost;
use vortex_btrblocks::SchemeExt;
use vortex_btrblocks::SchemeId;
use vortex_btrblocks::SizeCost;
use vortex_btrblocks::schemes::integer::BitPackingScheme;
use vortex_btrblocks::schemes::integer::RunEndScheme;
use vortex_btrblocks::schemes::integer::SparseScheme;
use vortex_btrblocks::schemes::string::FSSTScheme;
use vortex_btrblocks::schemes::string::StringDictScheme;
use vortex_buffer::Buffer;
use vortex_compressor::cost::Candidate;
use vortex_mask::Mask;
use vortex_session::VortexSession;

const LEN: usize = 1 << 18;
const UNIQUE: usize = LEN / 2;
const INTEGER_LEN: usize = 1 << 20;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = vortex_array::array_session();
    vortex_fastlanes::initialize(&session);
    vortex_fsst::initialize(&session);
    session
});

static INPUT: LazyLock<ArrayRef> = LazyLock::new(|| {
    VarBinViewArray::from_iter_str((0..LEN).map(|index| {
        format!(
            "tenant-{:06}-event-checkout-completed-region-us-east",
            index % UNIQUE
        )
    }))
    .into_array()
});

static SIZE_WINNER: LazyLock<ArrayRef> = LazyLock::new(|| {
    BtrBlocksCompressorBuilder::default()
        .with_cost_model(Arc::new(SizeCost))
        .build()
        .compress(&INPUT, &mut SESSION.create_execution_ctx())
        .unwrap()
});

static FSST: LazyLock<ArrayRef> = LazyLock::new(|| compress_with_root(&INPUT, FSSTScheme.id()));
static DICT: LazyLock<ArrayRef> =
    LazyLock::new(|| compress_with_root(&INPUT, StringDictScheme.id()));

static INTEGER_INPUT: LazyLock<ArrayRef> = LazyLock::new(|| {
    let values: Buffer<u32> = (0..INTEGER_LEN)
        .map(|index| {
            if index.is_multiple_of(20) {
                u32::try_from(index).unwrap()
            } else {
                0
            }
        })
        .collect();
    values.into_array()
});

static INTEGER_SIZE_WINNER: LazyLock<ArrayRef> = LazyLock::new(|| {
    BtrBlocksCompressorBuilder::default()
        .with_cost_model(Arc::new(SizeCost))
        .build()
        .compress(&INTEGER_INPUT, &mut SESSION.create_execution_ctx())
        .unwrap()
});

static SPARSE: LazyLock<ArrayRef> =
    LazyLock::new(|| compress_with_root(&INTEGER_INPUT, SparseScheme.id()));
static BITPACKED: LazyLock<ArrayRef> =
    LazyLock::new(|| compress_with_root(&INTEGER_INPUT, BitPackingScheme.id()));
static INTEGER_FILTER: LazyLock<Mask> =
    LazyLock::new(|| Mask::from_iter((0..INTEGER_LEN).map(|index| index.is_multiple_of(10))));

static RUN_INPUT: LazyLock<ArrayRef> = LazyLock::new(|| {
    let values: Buffer<u32> = (0..INTEGER_LEN)
        .map(|index| u32::try_from(index / 16).unwrap())
        .collect();
    values.into_array()
});

static RUN_SIZE_WINNER: LazyLock<ArrayRef> = LazyLock::new(|| {
    BtrBlocksCompressorBuilder::default()
        .with_cost_model(Arc::new(SizeCost))
        .build()
        .compress(&RUN_INPUT, &mut SESSION.create_execution_ctx())
        .unwrap()
});

static RUNEND: LazyLock<ArrayRef> =
    LazyLock::new(|| compress_with_root(&RUN_INPUT, RunEndScheme.id()));
static RUN_BITPACKED: LazyLock<ArrayRef> =
    LazyLock::new(|| compress_with_root(&RUN_INPUT, BitPackingScheme.id()));
static RUN_SCAN_COST: LazyLock<ArrayRef> = LazyLock::new(|| {
    BtrBlocksCompressorBuilder::default()
        .with_cost_model(Arc::new(ScanCost::default()))
        .build()
        .compress(&RUN_INPUT, &mut SESSION.create_execution_ctx())
        .unwrap()
});

/// Forces one root family while preserving normal size-based descendant selection.
#[derive(Debug)]
struct ForceRootCost {
    root: SchemeId,
}

impl CostModel for ForceRootCost {
    fn cost(&self, candidate: &Candidate<'_>) -> Option<Cost> {
        if candidate.cascade().is_empty() && candidate.scheme_id() != self.root {
            return None;
        }
        SizeCost.cost(candidate)
    }

    fn canonical_cost(&self, data: &ArrayAndStats, n_values: u64) -> Cost {
        SizeCost.canonical_cost(data, n_values)
    }
}

fn compress_with_root(input: &ArrayRef, root: SchemeId) -> ArrayRef {
    BtrBlocksCompressorBuilder::default()
        .with_cost_model(Arc::new(ForceRootCost { root }))
        .build()
        .compress(input, &mut SESSION.create_execution_ctx())
        .unwrap()
}

fn bench_decode(bencher: Bencher, array: &'static LazyLock<ArrayRef>) {
    bencher
        .counter(ItemsCount::new(LEN))
        .with_inputs(|| {
            (
                LazyLock::force(array).clone(),
                SESSION.create_execution_ctx(),
            )
        })
        .bench_refs(|(array, ctx)| {
            divan::black_box(array.clone().execute::<VarBinViewArray>(ctx).unwrap())
        });
}

fn bench_eq(bencher: Bencher, array: &'static LazyLock<ArrayRef>) {
    bencher
        .counter(ItemsCount::new(LEN))
        .with_inputs(|| {
            let lhs = LazyLock::force(array).clone();
            let rhs =
                ConstantArray::new("tenant-131071-event-checkout-completed-region-us-east", LEN)
                    .into_array();
            (lhs, rhs, SESSION.create_execution_ctx())
        })
        .bench_refs(|(lhs, rhs, ctx)| {
            divan::black_box(
                lhs.clone()
                    .binary(rhs.clone(), Operator::Eq)
                    .unwrap()
                    .execute::<BoolArray>(ctx)
                    .unwrap(),
            )
        });
}

fn bench_like(bencher: Bencher, array: &'static LazyLock<ArrayRef>) {
    bencher
        .counter(ItemsCount::new(LEN))
        .with_inputs(|| {
            let lhs = LazyLock::force(array).clone();
            let pattern = ConstantArray::new("%checkout-completed%", LEN).into_array();
            (lhs, pattern, SESSION.create_execution_ctx())
        })
        .bench_refs(|(lhs, pattern, ctx)| {
            divan::black_box(
                Like.try_new_array(LEN, LikeOptions::default(), [lhs.clone(), pattern.clone()])
                    .unwrap()
                    .into_array()
                    .execute::<Canonical>(ctx)
                    .unwrap(),
            )
        });
}

fn bench_integer_decode(bencher: Bencher, array: &'static LazyLock<ArrayRef>) {
    bencher
        .counter(ItemsCount::new(INTEGER_LEN))
        .with_inputs(|| {
            (
                LazyLock::force(array).clone(),
                SESSION.create_execution_ctx(),
            )
        })
        .bench_refs(|(array, ctx)| {
            divan::black_box(array.clone().execute::<PrimitiveArray>(ctx).unwrap())
        });
}

fn bench_integer_eq(bencher: Bencher, array: &'static LazyLock<ArrayRef>) {
    bencher
        .counter(ItemsCount::new(INTEGER_LEN))
        .with_inputs(|| {
            let lhs = LazyLock::force(array).clone();
            let rhs = ConstantArray::new(0u32, INTEGER_LEN).into_array();
            (lhs, rhs, SESSION.create_execution_ctx())
        })
        .bench_refs(|(lhs, rhs, ctx)| {
            divan::black_box(
                lhs.clone()
                    .binary(rhs.clone(), Operator::Eq)
                    .unwrap()
                    .execute::<BoolArray>(ctx)
                    .unwrap(),
            )
        });
}

fn bench_integer_filter(bencher: Bencher, array: &'static LazyLock<ArrayRef>) {
    bencher
        .counter(ItemsCount::new(INTEGER_LEN))
        .with_inputs(|| {
            (
                LazyLock::force(array).clone(),
                LazyLock::force(&INTEGER_FILTER).clone(),
                SESSION.create_execution_ctx(),
            )
        })
        .bench_refs(|(array, mask, ctx)| {
            divan::black_box(
                array
                    .clone()
                    .filter(mask.clone())
                    .unwrap()
                    .execute::<Canonical>(ctx)
                    .unwrap(),
            )
        });
}

#[divan::bench]
fn fsst_decode(bencher: Bencher) {
    bench_decode(bencher, &FSST);
}

#[divan::bench]
fn dict_decode(bencher: Bencher) {
    bench_decode(bencher, &DICT);
}

#[divan::bench]
fn fsst_eq(bencher: Bencher) {
    bench_eq(bencher, &FSST);
}

#[divan::bench]
fn dict_eq(bencher: Bencher) {
    bench_eq(bencher, &DICT);
}

#[divan::bench]
fn fsst_like(bencher: Bencher) {
    bench_like(bencher, &FSST);
}

#[divan::bench]
fn dict_like(bencher: Bencher) {
    bench_like(bencher, &DICT);
}

#[divan::bench]
fn sparse_decode(bencher: Bencher) {
    bench_integer_decode(bencher, &SPARSE);
}

#[divan::bench]
fn bitpacked_decode(bencher: Bencher) {
    bench_integer_decode(bencher, &BITPACKED);
}

#[divan::bench]
fn sparse_eq_fill(bencher: Bencher) {
    bench_integer_eq(bencher, &SPARSE);
}

#[divan::bench]
fn bitpacked_eq_fill(bencher: Bencher) {
    bench_integer_eq(bencher, &BITPACKED);
}

#[divan::bench]
fn sparse_filter_10pct(bencher: Bencher) {
    bench_integer_filter(bencher, &SPARSE);
}

#[divan::bench]
fn bitpacked_filter_10pct(bencher: Bencher) {
    bench_integer_filter(bencher, &BITPACKED);
}

#[divan::bench]
fn runend_decode(bencher: Bencher) {
    bench_integer_decode(bencher, &RUNEND);
}

#[divan::bench]
fn run_bitpacked_decode(bencher: Bencher) {
    bench_integer_decode(bencher, &RUN_BITPACKED);
}

#[divan::bench]
fn runend_eq(bencher: Bencher) {
    bench_integer_eq(bencher, &RUNEND);
}

#[divan::bench]
fn run_bitpacked_eq(bencher: Bencher) {
    bench_integer_eq(bencher, &RUN_BITPACKED);
}

#[divan::bench]
fn runend_filter_10pct(bencher: Bencher) {
    bench_integer_filter(bencher, &RUNEND);
}

#[divan::bench]
fn run_bitpacked_filter_10pct(bencher: Bencher) {
    bench_integer_filter(bencher, &RUN_BITPACKED);
}

#[divan::bench]
fn run_scan_cost_decode(bencher: Bencher) {
    bench_integer_decode(bencher, &RUN_SCAN_COST);
}

#[divan::bench]
fn run_scan_cost_eq(bencher: Bencher) {
    bench_integer_eq(bencher, &RUN_SCAN_COST);
}

#[divan::bench]
fn run_scan_cost_filter_10pct(bencher: Bencher) {
    bench_integer_filter(bencher, &RUN_SCAN_COST);
}

fn main() {
    let size_winner = LazyLock::force(&SIZE_WINNER);
    let fsst = LazyLock::force(&FSST);
    let dict = LazyLock::force(&DICT);
    println!(
        "size winner: encoding={} bytes={}; fsst={} dict={} dict/fsst={:.3}",
        size_winner.encoding_id(),
        size_winner.nbytes(),
        fsst.nbytes(),
        dict.nbytes(),
        dict.nbytes() as f64 / fsst.nbytes() as f64,
    );
    let integer_size_winner = LazyLock::force(&INTEGER_SIZE_WINNER);
    let sparse = LazyLock::force(&SPARSE);
    let bitpacked = LazyLock::force(&BITPACKED);
    println!(
        "integer size winner: encoding={} bytes={}; sparse={} bitpacked={} bitpacked/sparse={:.3}",
        integer_size_winner.encoding_id(),
        integer_size_winner.nbytes(),
        sparse.nbytes(),
        bitpacked.nbytes(),
        bitpacked.nbytes() as f64 / sparse.nbytes() as f64,
    );
    let run_size_winner = LazyLock::force(&RUN_SIZE_WINNER);
    let runend = LazyLock::force(&RUNEND);
    let run_bitpacked = LazyLock::force(&RUN_BITPACKED);
    let run_scan_cost = LazyLock::force(&RUN_SCAN_COST);
    println!(
        "run size winner: encoding={} bytes={}; runend={} bitpacked={} bitpacked/runend={:.3}",
        run_size_winner.encoding_id(),
        run_size_winner.nbytes(),
        runend.nbytes(),
        run_bitpacked.nbytes(),
        run_bitpacked.nbytes() as f64 / runend.nbytes() as f64,
    );
    println!(
        "run scan-cost alternative: encoding={} bytes={}",
        run_scan_cost.encoding_id(),
        run_scan_cost.nbytes(),
    );
    divan::main();
}
