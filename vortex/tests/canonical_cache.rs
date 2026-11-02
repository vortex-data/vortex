// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The `execute_until` canonical cache on Vortex Compact data.
//!
//! The conjuncts of one filter unit read the same column. Compact encodings such as Zstd and
//! Pco have no compute kernels, so every conjunct decodes the column to canonical. The scalar
//! functions of a unit execute their inputs iteratively, so the whole unit runs in one
//! `execute_until`, whose canonical cache lets the first conjunct's decode serve the others.

#![cfg(feature = "zstd")]
#![allow(clippy::tests_outside_test_module)]

use std::ops::BitAnd;
use std::time::Duration;
use std::time::Instant;

use rstest::rstest;
use vortex::VortexSessionDefault;
use vortex::array::AnyCanonical;
use vortex::array::ArrayRef;
use vortex::array::IntoArray;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::VarBinViewArray;
use vortex::array::builtins::ArrayBuiltins;
use vortex::array::test_harness::CanonicalCacheStats;
use vortex::array::test_harness::canonical_cache_stats;
use vortex::compressor::BtrBlocksCompressorBuilder;
use vortex::error::VortexResult;
use vortex::expr::Expression;
use vortex::expr::gt;
use vortex::expr::like;
use vortex::expr::lit;
use vortex::expr::lt;
use vortex::expr::not_like;
use vortex::expr::root;
use vortex::mask::Mask;
use vortex_session::VortexSession;

const ROWS: usize = 256 * 1024;
const RUNS: usize = 5;

fn strings() -> ArrayRef {
    VarBinViewArray::from_iter_str((0..ROWS as u64).map(|i| {
        format!(
            "user-{}-{}",
            i.wrapping_mul(2_654_435_761) % 1_000_003,
            i % 13
        )
    }))
    .into_array()
}

fn integers() -> ArrayRef {
    PrimitiveArray::from_iter((0..ROWS as u64).map(|i| i.wrapping_mul(2_654_435_761) % 1_000_003))
        .into_array()
}

/// Conjuncts over one string column, as JOB's `mc.note LIKE … AND mc.note NOT LIKE …`.
fn string_conjuncts() -> [Expression; 2] {
    [like(root(), lit("user-1%")), not_like(root(), lit("%-12"))]
}

/// Conjuncts over one integer column that `find_between` does not merge.
fn integer_conjuncts() -> [Expression; 2] {
    [gt(root(), lit(1_000u64)), lt(root(), lit(900_000u64))]
}

fn compact(session: &VortexSession, array: &ArrayRef) -> VortexResult<ArrayRef> {
    BtrBlocksCompressorBuilder::from_session(session)
        .with_compact()
        .build()
        .compress(array, &mut session.create_execution_ctx())
}

/// Evaluates `filter` over `array` to a mask, as the flat reader evaluates a filter unit.
fn evaluate(session: &VortexSession, array: &ArrayRef, filter: &Expression) -> VortexResult<Mask> {
    let bound = filter.bind(array.dtype())?;
    array
        .clone()
        .apply_bound(&bound)?
        .fill_null(false)?
        .execute::<Mask>(&mut session.create_execution_ctx())
}

/// The canonical cache activity on this thread while `f` runs.
fn cache_activity<T>(f: impl FnOnce() -> T) -> (T, CanonicalCacheStats) {
    let before = canonical_cache_stats();
    let result = f();
    let after = canonical_cache_stats();
    let delta = CanonicalCacheStats {
        hits: after.hits - before.hits,
        inserts: after.inserts - before.inserts,
        saved_bytes: after.saved_bytes - before.saved_bytes,
    };
    (result, delta)
}

/// The fastest of [`RUNS`] runs of `f`.
fn fastest(mut f: impl FnMut() -> VortexResult<()>) -> VortexResult<Duration> {
    let mut fastest = Duration::MAX;
    for _ in 0..RUNS {
        let start = Instant::now();
        f()?;
        fastest = fastest.min(start.elapsed());
    }
    Ok(fastest)
}

#[rstest]
#[case::compact_strings(strings(), string_conjuncts())]
#[case::compact_integers(integers(), integer_conjuncts())]
fn filter_unit_decodes_compact_column_once(
    #[case] array: ArrayRef,
    #[case] conjuncts: [Expression; 2],
) -> VortexResult<()> {
    let session = VortexSession::default();
    let compressed = compact(&session, &array)?;
    assert!(
        !compressed.is::<AnyCanonical>(),
        "the compact compressor left the column canonical"
    );
    let unit = vortex::expr::and_collect(conjuncts.iter().cloned()).expect("two conjuncts");

    // One unit, executed once: the second conjunct reuses the first one's decode.
    let (actual, unit_stats) = cache_activity(|| evaluate(&session, &compressed, &unit));
    assert_eq!(unit_stats.hits, 1, "{unit_stats:?}");
    assert!(unit_stats.saved_bytes > 0, "{unit_stats:?}");

    // Conjuncts executed one by one share nothing, so each decodes the column.
    let (separate, separate_stats) = cache_activity(|| {
        conjuncts
            .iter()
            .map(|conjunct| evaluate(&session, &compressed, conjunct))
            .collect::<VortexResult<Vec<_>>>()
    });
    assert_eq!(separate_stats.hits, 0, "{separate_stats:?}");
    let expected = separate?
        .into_iter()
        .reduce(|lhs, rhs| lhs.bitand(&rhs))
        .expect("two conjuncts");
    assert_eq!(actual?, expected);

    let as_unit = fastest(|| evaluate(&session, &compressed, &unit).map(drop))?;
    let one_by_one = fastest(|| {
        for conjunct in &conjuncts {
            evaluate(&session, &compressed, conjunct)?;
        }
        Ok(())
    })?;
    let millis = |duration: Duration| duration.as_secs_f64() * 1e3;
    eprintln!(
        "{} ({}): {} hit, {} canonical bytes reused; fastest of {RUNS}: {:.1} ms as a unit, \
         {:.1} ms one conjunct at a time, {:.1} ms saved",
        compressed.encoding_id(),
        compressed.dtype(),
        unit_stats.hits,
        unit_stats.saved_bytes,
        millis(as_unit),
        millis(one_by_one),
        millis(one_by_one.saturating_sub(as_unit)),
    );
    Ok(())
}
