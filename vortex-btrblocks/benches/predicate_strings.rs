// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! String size-versus-predicate-operation calibration benchmark.

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
use vortex_btrblocks::SchemeExt;
use vortex_btrblocks::SchemeId;
use vortex_btrblocks::SizeCost;
use vortex_btrblocks::schemes::string::FSSTScheme;
use vortex_btrblocks::schemes::string::StringDictScheme;
use vortex_compressor::cost::Candidate;
use vortex_session::VortexSession;

const LEN: usize = 1 << 18;
const UNIQUE: usize = 3;

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

static CANONICAL: LazyLock<ArrayRef> = LazyLock::new(|| LazyLock::force(&INPUT).clone());

static SIZE_WINNER: LazyLock<ArrayRef> = LazyLock::new(|| {
    BtrBlocksCompressorBuilder::default()
        .build()
        .compress(&INPUT, &mut SESSION.create_execution_ctx())
        .unwrap()
});

static LIKE_WINNER: LazyLock<ArrayRef> = LazyLock::new(|| {
    BtrBlocksCompressorBuilder::default()
        .with_like_strings()
        .build()
        .compress(&INPUT, &mut SESSION.create_execution_ctx())
        .unwrap()
});

static FSST: LazyLock<ArrayRef> = LazyLock::new(|| compress_with_root(&INPUT, FSSTScheme.id()));
static DICT: LazyLock<ArrayRef> =
    LazyLock::new(|| compress_with_root(&INPUT, StringDictScheme.id()));

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

#[divan::bench]
fn canonical_decode(bencher: Bencher) {
    bench_decode(bencher, &CANONICAL);
}

#[divan::bench]
fn canonical_eq(bencher: Bencher) {
    bench_eq(bencher, &CANONICAL);
}

#[divan::bench]
fn canonical_like(bencher: Bencher) {
    bench_like(bencher, &CANONICAL);
}

#[divan::bench]
fn fsst_decode(bencher: Bencher) {
    bench_decode(bencher, &FSST);
}

#[divan::bench]
fn fsst_eq(bencher: Bencher) {
    bench_eq(bencher, &FSST);
}

#[divan::bench]
fn fsst_like(bencher: Bencher) {
    bench_like(bencher, &FSST);
}

#[divan::bench]
fn dict_decode(bencher: Bencher) {
    bench_decode(bencher, &DICT);
}

#[divan::bench]
fn dict_eq(bencher: Bencher) {
    bench_eq(bencher, &DICT);
}

#[divan::bench]
fn dict_like(bencher: Bencher) {
    bench_like(bencher, &DICT);
}

#[divan::bench]
fn like_winner_decode(bencher: Bencher) {
    bench_decode(bencher, &LIKE_WINNER);
}

#[divan::bench]
fn like_winner_eq(bencher: Bencher) {
    bench_eq(bencher, &LIKE_WINNER);
}

#[divan::bench]
fn like_winner_like(bencher: Bencher) {
    bench_like(bencher, &LIKE_WINNER);
}

fn print_selection_sweep() {
    const PROBE_LEN: usize = 1 << 14;

    for unique in [3, 32, 256, 2_048, 8_192] {
        let input = VarBinViewArray::from_iter_str((0..PROBE_LEN).map(|index| {
            format!(
                "tenant-{:06}-event-checkout-completed-region-us-east",
                index % unique
            )
        }))
        .into_array();
        let size = BtrBlocksCompressorBuilder::default()
            .build()
            .compress(&input, &mut SESSION.create_execution_ctx())
            .unwrap();
        let like = BtrBlocksCompressorBuilder::default()
            .with_like_strings()
            .build()
            .compress(&input, &mut SESSION.create_execution_ctx())
            .unwrap();
        println!(
            "unique={unique:5}: size={} bytes={:7}; like={} bytes={:7}",
            size.encoding_id(),
            size.nbytes(),
            like.encoding_id(),
            like.nbytes(),
        );
    }
}

fn main() {
    print_selection_sweep();
    let input = LazyLock::force(&INPUT);
    let size_winner = LazyLock::force(&SIZE_WINNER);
    let like_winner = LazyLock::force(&LIKE_WINNER);
    let fsst = LazyLock::force(&FSST);
    let dict = LazyLock::force(&DICT);
    println!(
        "input={} bytes; size winner={} bytes={}; LIKE winner={} bytes={}; fsst={} dict={}",
        input.nbytes(),
        size_winner.encoding_id(),
        size_winner.nbytes(),
        like_winner.encoding_id(),
        like_winner.nbytes(),
        fsst.nbytes(),
        dict.nbytes(),
    );
    divan::main();
}
