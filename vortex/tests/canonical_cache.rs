// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The canonical cache on Vortex Compact data.
//!
//! The conjuncts of one filter unit read the same column. Compact encodings such as Zstd have
//! no compute kernels, so every conjunct decodes the column to canonical first. With the
//! canonical cache, the first conjunct's decode is reused by the others.

#![cfg(feature = "zstd")]
#![allow(clippy::tests_outside_test_module)]

use std::time::Duration;
use std::time::Instant;

use rstest::rstest;
use vortex::VortexSessionDefault;
use vortex::array::AnyCanonical;
use vortex::array::ArrayRef;
use vortex::array::ExecutionCtx;
use vortex::array::IntoArray;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::VarBinViewArray;
use vortex::array::builtins::ArrayBuiltins;
use vortex::compressor::BtrBlocksCompressorBuilder;
use vortex::error::VortexResult;
use vortex::expr::Expression;
use vortex::expr::and;
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
fn string_unit() -> Expression {
    and(like(root(), lit("user-1%")), not_like(root(), lit("%-12")))
}

/// Conjuncts over one integer column that `find_between` does not merge.
fn integer_unit() -> Expression {
    and(gt(root(), lit(1_000u64)), lt(root(), lit(900_000u64)))
}

fn compact(session: &VortexSession, array: &ArrayRef) -> VortexResult<ArrayRef> {
    BtrBlocksCompressorBuilder::from_session(session)
        .with_compact()
        .build()
        .compress(array, &mut session.create_execution_ctx())
}

fn evaluate(array: &ArrayRef, unit: &Expression, ctx: &mut ExecutionCtx) -> VortexResult<Mask> {
    let bound = unit.bind(array.dtype())?;
    array
        .clone()
        .apply_bound(&bound)?
        .fill_null(false)?
        .execute::<Mask>(ctx)
}

/// The fastest of [`RUNS`] evaluations, each in a fresh context.
fn fastest(
    array: &ArrayRef,
    unit: &Expression,
    ctx: impl Fn() -> ExecutionCtx,
) -> VortexResult<Duration> {
    let mut fastest = Duration::MAX;
    for _ in 0..RUNS {
        let mut ctx = ctx();
        let start = Instant::now();
        evaluate(array, unit, &mut ctx)?;
        fastest = fastest.min(start.elapsed());
    }
    Ok(fastest)
}

#[rstest]
#[case::compact_strings(strings(), string_unit())]
#[case::compact_integers(integers(), integer_unit())]
fn filter_unit_decodes_compact_column_once(
    #[case] array: ArrayRef,
    #[case] unit: Expression,
) -> VortexResult<()> {
    let session = VortexSession::default();
    let compressed = compact(&session, &array)?;
    assert!(
        !compressed.is::<AnyCanonical>(),
        "the compact compressor left the column canonical"
    );

    let mut uncached = session.create_execution_ctx();
    let expected = evaluate(&compressed, &unit, &mut uncached)?;
    assert_eq!(uncached.canonical_cache_stats(), None);

    let mut cached = session.create_execution_ctx().with_canonical_cache();
    let actual = evaluate(&compressed, &unit, &mut cached)?;
    assert_eq!(actual, expected);

    let stats = cached
        .canonical_cache_stats()
        .expect("the canonical cache is enabled");
    // The second conjunct reuses the first one's decode of the column.
    assert!(stats.hits >= 1, "no canonical cache hit: {stats:?}");
    assert!(stats.saved_bytes > 0, "the hits saved no bytes: {stats:?}");

    let without = fastest(&compressed, &unit, || session.create_execution_ctx())?;
    let with = fastest(&compressed, &unit, || {
        session.create_execution_ctx().with_canonical_cache()
    })?;
    let millis = |duration: Duration| duration.as_secs_f64() * 1e3;
    eprintln!(
        "{} ({}): {} hits, {} canonical bytes reused; fastest of {RUNS}: {:.1} ms uncached, \
         {:.1} ms cached, {:.1} ms saved",
        compressed.encoding_id(),
        compressed.dtype(),
        stats.hits,
        stats.saved_bytes,
        millis(without),
        millis(with),
        millis(without.saturating_sub(with)),
    );
    Ok(())
}
