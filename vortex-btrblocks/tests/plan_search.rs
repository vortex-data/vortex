// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Plan recording and replay, and the exhaustive and genetic plan searches, over the built-in
//! schemes.

#![cfg(test)]

use std::sync::LazyLock;

use rstest::rstest;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::Dict;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::dict::DictArraySlotsExt;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_btrblocks::BtrBlocksCompressor;
use vortex_btrblocks::BtrBlocksCompressorBuilder;
use vortex_btrblocks::SchemeExt;
use vortex_btrblocks::schemes::integer::IntDictScheme;
use vortex_btrblocks::schemes::integer::SequenceScheme;
use vortex_compressor::plan::Plan;
use vortex_compressor::search::Candidate;
use vortex_compressor::search::CostModel;
use vortex_compressor::search::ExhaustiveSearch;
use vortex_compressor::search::GeneticSearch;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

static SESSION: LazyLock<VortexSession> = LazyLock::new(vortex_array::array_session);

const LEN: u16 = 4096;

fn compressor() -> BtrBlocksCompressor {
    BtrBlocksCompressorBuilder::from_session(&SESSION)
        .unrestricted()
        .build()
}

/// Integers in long runs over a small domain.
fn runs() -> ArrayRef {
    PrimitiveArray::from_iter((0..LEN).map(|i| i32::from((i / 64) % 7) + 1_000)).into_array()
}

/// Increasing integers with a little noise, some of them null.
fn nullable_trend() -> ArrayRef {
    PrimitiveArray::from_option_iter(
        (0..LEN)
            .map(i64::from)
            .map(|i| (i % 13 != 0).then_some(1_000_000 + 3 * i + (i * 7) % 5)),
    )
    .into_array()
}

/// Decimal-like doubles.
fn prices() -> ArrayRef {
    PrimitiveArray::from_iter((0..LEN).map(|i| f64::from(u32::from(i) * 37 % 1000) / 100.0))
        .into_array()
}

/// Low-cardinality strings.
fn categories() -> ArrayRef {
    const NAMES: [&str; 5] = ["alpha", "bravo", "charlie", "delta", "echo"];
    VarBinViewArray::from_iter(
        (0..LEN)
            .map(usize::from)
            .map(|i| Some(NAMES[(i * i) % NAMES.len()])),
        DType::Utf8(Nullability::NonNullable),
    )
    .into_array()
}

/// Checks that `candidate`'s plan replays to an array of the measured size that decompresses to
/// `input`.
#[track_caller]
fn assert_replays(
    compressor: &BtrBlocksCompressor,
    input: &ArrayRef,
    candidate: &Candidate,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let replayed = compressor.compress_with_plan(input, &candidate.plan, &mut ctx)?;
    assert_eq!(replayed.nbytes(), candidate.measurement.nbytes);
    assert_arrays_eq!(&replayed, input, &mut ctx);
    Ok(())
}

#[rstest]
#[case::runs(runs())]
#[case::nullable_trend(nullable_trend())]
#[case::prices(prices())]
#[case::categories(categories())]
fn recorded_plan_replays_to_the_same_array(#[case] input: ArrayRef) -> VortexResult<()> {
    let compressor = compressor();
    let mut ctx = SESSION.create_execution_ctx();

    let (compressed, plan) = compressor.compress_recording_plan(&input, &mut ctx)?;
    assert_eq!(
        compressed.nbytes(),
        compressor.compress(&input, &mut ctx)?.nbytes()
    );
    assert!(plan.scheme_id().is_some(), "expected a scheme, got {plan}");

    let replayed = compressor.compress_with_plan(&input, &plan, &mut ctx)?;
    assert_eq!(replayed.encoding_id(), compressed.encoding_id());
    assert_eq!(replayed.nbytes(), compressed.nbytes());
    assert_arrays_eq!(&replayed, &input, &mut ctx);
    Ok(())
}

#[test]
fn adaptive_plan_matches_estimate_based_compression() -> VortexResult<()> {
    let compressor = compressor();
    let mut ctx = SESSION.create_execution_ctx();
    let input = runs();

    let adaptive = compressor.compress_with_plan(&input, &Plan::Adaptive, &mut ctx)?;
    assert_eq!(
        adaptive.nbytes(),
        compressor.compress(&input, &mut ctx)?.nbytes()
    );
    Ok(())
}

#[test]
fn plan_forces_the_cascade() -> VortexResult<()> {
    let compressor = compressor();
    let mut ctx = SESSION.create_execution_ctx();
    let input = runs();

    let plan = Plan::scheme(IntDictScheme.id(), vec![Plan::Canonical, Plan::Canonical]);
    let compressed = compressor.compress_with_plan(&input, &plan, &mut ctx)?;
    let dict = compressed
        .as_opt::<Dict>()
        .expect("the plan asks for a dictionary");
    assert!(dict.codes().is::<Primitive>());
    assert!(dict.values().is::<Primitive>());
    assert_arrays_eq!(&compressed, &input, &mut ctx);

    let candidate = compressor.measure_plan(&input, &plan, 1, &mut ctx)?;
    assert_eq!(
        candidate.plan.to_string(),
        "vortex.int.dict(canonical, canonical)"
    );
    assert_eq!(candidate.measurement.nbytes, compressed.nbytes());
    Ok(())
}

#[test]
fn inapplicable_plan_falls_back_to_adaptive_selection() -> VortexResult<()> {
    let compressor = compressor();
    let mut ctx = SESSION.create_execution_ctx();
    let input = runs();

    // The runs are not an arithmetic sequence, so the sequence scheme fails on them.
    let plan = Plan::scheme(SequenceScheme.id(), Vec::<Plan>::new());
    let candidate = compressor.measure_plan(&input, &plan, 1, &mut ctx)?;
    assert_ne!(candidate.plan, plan);
    assert_eq!(
        candidate.measurement.nbytes,
        compressor.compress(&input, &mut ctx)?.nbytes()
    );
    assert_replays(&compressor, &input, &candidate)
}

#[rstest]
#[case::runs(runs())]
#[case::nullable_trend(nullable_trend())]
#[case::prices(prices())]
#[case::categories(categories())]
fn exhaustive_size_search_is_no_larger_than_estimate(#[case] input: ArrayRef) -> VortexResult<()> {
    let compressor = compressor();
    let mut ctx = SESSION.create_execution_ctx();

    let best = ExhaustiveSearch::new(CostModel::size())
        .with_iterations(1)
        .run(&compressor, &input, &mut ctx)?;
    let estimated = compressor.compress(&input, &mut ctx)?;
    assert!(
        best.measurement.nbytes <= estimated.nbytes(),
        "exhaustive plan {} has {} bytes, estimate-based compression {}",
        best.plan,
        best.measurement.nbytes,
        estimated.nbytes(),
    );
    assert_replays(&compressor, &input, &best)
}

#[test]
fn exhaustive_read_time_search_replays() -> VortexResult<()> {
    let compressor = compressor();
    let mut ctx = SESSION.create_execution_ctx();
    let input = nullable_trend();

    let best = ExhaustiveSearch::new(CostModel::read_time(1e9).with_reads_per_write(10.0))
        .with_iterations(1)
        .run(&compressor, &input, &mut ctx)?;
    assert_replays(&compressor, &input, &best)
}

#[rstest]
#[case::runs(runs())]
#[case::categories(categories())]
fn genetic_search_returns_a_replayable_pareto_front(#[case] input: ArrayRef) -> VortexResult<()> {
    let compressor = compressor();
    let mut ctx = SESSION.create_execution_ctx();

    let exhaustive = ExhaustiveSearch::new(CostModel::size())
        .with_iterations(1)
        .run(&compressor, &input, &mut ctx)?;
    let search = GeneticSearch {
        population: 8,
        generations: 4,
        iterations: 1,
        ..GeneticSearch::default()
    }
    .with_initial_plans([exhaustive.plan.clone()]);
    let front = search.run(&compressor, &input, &mut ctx)?;

    assert!(!front.is_empty());
    for (i, a) in front.iter().enumerate() {
        for b in &front[i + 1..] {
            assert!(!a.measurement.dominates(&b.measurement));
            assert!(!b.measurement.dominates(&a.measurement));
        }
        assert_replays(&compressor, &input, a)?;
    }

    // The smallest plan seen is never dropped, and the seeds include the exhaustive optimum.
    assert!(front[0].measurement.nbytes <= exhaustive.measurement.nbytes);
    assert!(
        front
            .windows(2)
            .all(|w| w[0].measurement.nbytes <= w[1].measurement.nbytes)
    );
    Ok(())
}

#[test]
fn search_rejects_nested_arrays() -> VortexResult<()> {
    let compressor = compressor();
    let mut ctx = SESSION.create_execution_ctx();
    let input = StructArray::from_fields(&[("runs", runs())])?.into_array();

    assert!(
        ExhaustiveSearch::new(CostModel::size())
            .run(&compressor, &input, &mut ctx)
            .is_err()
    );
    assert!(
        GeneticSearch::default()
            .run(&compressor, &input, &mut ctx)
            .is_err()
    );
    Ok(())
}
