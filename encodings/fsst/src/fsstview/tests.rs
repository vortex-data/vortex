// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use rstest::rstest;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::LEGACY_SESSION;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::dict::TakeExecute;
use vortex_array::assert_arrays_eq;
use vortex_array::compute::conformance::consistency::test_array_consistency;
use vortex_array::compute::conformance::filter::test_filter_conformance;
use vortex_array::compute::conformance::take::test_take_conformance;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::FSSTArray;
use crate::FSSTView;
use crate::FSSTViewArray;
use crate::FsstViewCompaction;
use crate::canonicalize_fsstview_to_varbin;
use crate::canonicalize_fsstview_with;
use crate::fsst_compress;
use crate::fsst_filter_to_view;
use crate::fsst_take_to_view;
use crate::fsst_train_compressor;
use crate::fsstview_from_fsst;

fn make_fsstview(
    strings: &[Option<&str>],
    nullability: Nullability,
    ctx: &mut ExecutionCtx,
) -> FSSTViewArray {
    let varbin = VarBinArray::from_iter(strings.iter().copied(), DType::Utf8(nullability));
    let compressor = fsst_train_compressor(&varbin);
    let fsst = fsst_compress(&varbin, varbin.len(), varbin.dtype(), &compressor, ctx);
    fsstview_from_fsst(&fsst, ctx).expect("fsstview_from_fsst")
}

const SAMPLE: [Option<&str>; 6] = [
    Some("hello world"),
    Some("testing fsst compression"),
    Some("hello world"),
    Some("another string here"),
    Some("the quick brown fox"),
    Some("hello world"),
];

const SAMPLE_NULLABLE: [Option<&str>; 6] = [
    Some("hello world"),
    None,
    Some("testing fsst compression"),
    Some("another string here"),
    None,
    Some("the quick brown fox"),
];

#[test]
fn canonicalizes_to_same_values() -> VortexResult<()> {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let view = make_fsstview(&SAMPLE, Nullability::NonNullable, &mut ctx);
    let array = view.into_array();
    assert!(array.is::<FSSTView>());

    let canonical = array.execute::<VarBinViewArray>(&mut ctx)?;
    let expected = VarBinArray::from_iter(
        SAMPLE.iter().copied(),
        DType::Utf8(Nullability::NonNullable),
    )
    .into_array()
    .execute::<VarBinViewArray>(&mut ctx)?;
    assert_arrays_eq!(canonical.into_array(), expected.into_array());
    Ok(())
}

#[test]
fn scalar_at_decodes_each_element() -> VortexResult<()> {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let view = make_fsstview(&SAMPLE, Nullability::NonNullable, &mut ctx);
    let array = view.into_array();

    for (i, expected) in SAMPLE.iter().enumerate() {
        let scalar = array.execute_scalar(i, &mut ctx)?;
        let value = scalar.as_utf8().value().expect("non-null");
        assert_eq!(value.as_str(), expected.unwrap());
    }
    Ok(())
}

#[rstest]
#[case(&SAMPLE, Nullability::NonNullable)]
#[case(&SAMPLE_NULLABLE, Nullability::Nullable)]
fn filter_conformance(#[case] strings: &[Option<&str>], #[case] nullability: Nullability) {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let view = make_fsstview(strings, nullability, &mut ctx);
    test_filter_conformance(&view.into_array());
}

#[rstest]
#[case(&SAMPLE, Nullability::NonNullable)]
#[case(&SAMPLE_NULLABLE, Nullability::Nullable)]
fn take_conformance(#[case] strings: &[Option<&str>], #[case] nullability: Nullability) {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let view = make_fsstview(strings, nullability, &mut ctx);
    test_take_conformance(&view.into_array());
}

#[rstest]
#[case(&SAMPLE, Nullability::NonNullable)]
#[case(&SAMPLE_NULLABLE, Nullability::Nullable)]
fn consistency(#[case] strings: &[Option<&str>], #[case] nullability: Nullability) {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let view = make_fsstview(strings, nullability, &mut ctx);
    test_array_consistency(&view.into_array());
}

fn make_fsst(
    strings: &[Option<&str>],
    nullability: Nullability,
    ctx: &mut ExecutionCtx,
) -> FSSTArray {
    let varbin = VarBinArray::from_iter(strings.iter().copied(), DType::Utf8(nullability));
    let compressor = fsst_train_compressor(&varbin);
    fsst_compress(&varbin, varbin.len(), varbin.dtype(), &compressor, ctx)
}

/// `fsst_filter_to_view` must agree with filtering the canonical VarBin, and must not touch the
/// codes bytes (the produced view shares the original heap).
#[test]
fn fsst_filter_to_view_matches_canonical() -> VortexResult<()> {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let fsst = make_fsst(&SAMPLE_NULLABLE, Nullability::Nullable, &mut ctx);
    let mask = Mask::from_iter([true, false, true, false, true, true]);

    let view = fsst_filter_to_view(&fsst, &mask, &mut ctx)?;
    let result = view.into_array().execute::<VarBinViewArray>(&mut ctx)?;

    let expected = VarBinArray::from_iter(
        SAMPLE_NULLABLE.iter().copied(),
        DType::Utf8(Nullability::Nullable),
    )
    .into_array()
    .filter(mask)?
    .execute::<VarBinViewArray>(&mut ctx)?;
    assert_arrays_eq!(result.into_array(), expected.into_array());
    Ok(())
}

#[test]
fn fsst_take_to_view_matches_canonical() -> VortexResult<()> {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let fsst = make_fsst(&SAMPLE, Nullability::NonNullable, &mut ctx);
    let indices = PrimitiveArray::from_iter([5u64, 0, 0, 3, 1]).into_array();

    let view = fsst_take_to_view(&fsst, &indices, &mut ctx)?;
    let result = view.into_array().execute::<VarBinViewArray>(&mut ctx)?;

    let expected = VarBinArray::from_iter(
        SAMPLE.iter().copied(),
        DType::Utf8(Nullability::NonNullable),
    )
    .into_array()
    .take(indices)?
    .execute::<VarBinViewArray>(&mut ctx)?;
    assert_arrays_eq!(result.into_array(), expected.into_array());
    Ok(())
}

/// All explicit compaction strategies must produce identical canonical output, both for a
/// contiguous (sliced) view and a scattered (taken) one.
#[rstest]
#[case(FsstViewCompaction::Auto)]
#[case(FsstViewCompaction::Direct)]
#[case(FsstViewCompaction::GatherBulk)]
#[case(FsstViewCompaction::RunDecode)]
fn compaction_strategies_agree(#[case] strategy: FsstViewCompaction) -> VortexResult<()> {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let fsst = make_fsst(&SAMPLE, Nullability::NonNullable, &mut ctx);

    // Scattered view via a take (reorders + duplicates -> non-contiguous codes).
    let indices = PrimitiveArray::from_iter([5u64, 0, 0, 3, 1, 2]).into_array();
    let scattered = fsst_take_to_view(&fsst, &indices, &mut ctx)?;
    let got = canonicalize_fsstview_with(scattered.as_view(), strategy, &mut ctx)?;
    let expected = VarBinArray::from_iter(
        SAMPLE.iter().copied(),
        DType::Utf8(Nullability::NonNullable),
    )
    .into_array()
    .take(indices)?
    .execute::<VarBinViewArray>(&mut ctx)?;
    assert_arrays_eq!(got, expected.into_array());

    // Contiguous view (untouched) — exercises the Direct fast path.
    let contiguous = fsstview_from_fsst(&fsst, &mut ctx)?;
    let got = canonicalize_fsstview_with(contiguous.as_view(), strategy, &mut ctx)?;
    let expected = VarBinArray::from_iter(
        SAMPLE.iter().copied(),
        DType::Utf8(Nullability::NonNullable),
    )
    .into_array()
    .execute::<VarBinViewArray>(&mut ctx)?;
    assert_arrays_eq!(got, expected.into_array());
    Ok(())
}

/// Adversarial coverage: a filter that punches gaps into the heap (so survivors form multiple
/// runs), then a shuffle take (reorders runs, forcing `GatherBulk`), over nullable data. Every
/// strategy must still agree with the canonical result.
#[rstest]
#[case(FsstViewCompaction::Auto)]
#[case(FsstViewCompaction::GatherBulk)]
fn gaps_and_shuffle_agree(#[case] strategy: FsstViewCompaction) -> VortexResult<()> {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    // 12 distinct-ish strings, nullable.
    let strings: Vec<Option<&str>> = vec![
        Some("alpha"),
        None,
        Some("bravo bravo"),
        Some("charlie"),
        Some("delta delta delta"),
        None,
        Some("echo"),
        Some("foxtrot foxtrot"),
        Some("golf"),
        Some("hotel hotel hotel"),
        None,
        Some("india"),
    ];
    let fsst = make_fsst(&strings, Nullability::Nullable, &mut ctx);

    // Filter to keep a gapped subset (drops 1,2,5,8,10 -> remaining survivors aren't all adjacent).
    let keep = [
        true, false, false, true, true, false, true, true, false, true, false, true,
    ];
    let mask = Mask::from_iter(keep);
    let filtered = fsst_filter_to_view(&fsst, &mask, &mut ctx)?;

    // Then a shuffle+dup take over the filtered length (7 survivors).
    let indices = PrimitiveArray::from_iter([6u64, 0, 3, 3, 5, 1, 2, 4]).into_array();
    let view = <FSSTView as TakeExecute>::take(filtered.as_view(), &indices, &mut ctx)?
        .unwrap()
        .try_downcast::<FSSTView>()
        .ok()
        .unwrap();

    let got = canonicalize_fsstview_with(view.as_view(), strategy, &mut ctx)?;

    let expected =
        VarBinArray::from_iter(strings.iter().copied(), DType::Utf8(Nullability::Nullable))
            .into_array()
            .filter(mask)?
            .take(indices)?
            .execute::<VarBinViewArray>(&mut ctx)?;
    assert_arrays_eq!(got, expected.into_array());
    Ok(())
}

/// `RunDecode` ("export all in place") must agree with the canonical result on a *monotonic*
/// gapped view (a filter, which keeps offsets increasing). Covers nulls, empty strings, and a
/// trailing run, across the strategies that accept monotonic input.
#[rstest]
#[case(FsstViewCompaction::Auto)]
#[case(FsstViewCompaction::RunDecode)]
#[case(FsstViewCompaction::GatherBulk)]
#[case(FsstViewCompaction::Direct)]
fn run_decode_monotonic_filter(#[case] strategy: FsstViewCompaction) -> VortexResult<()> {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let strings: Vec<Option<&str>> = vec![
        Some("alpha"),
        Some(""),
        None,
        Some("bravo bravo"),
        Some("charlie"),
        None,
        Some("delta delta delta"),
        Some("echo"),
        Some(""),
        Some("foxtrot foxtrot"),
        Some("golf golf"),
    ];
    let fsst = make_fsst(&strings, Nullability::Nullable, &mut ctx);
    // Keep a gapped-but-ordered subset (multiple runs, including an adjacent pair and a trailing
    // run) so RunDecode exercises >1 run and the GatherBulk fallback is also valid.
    let keep = [
        true, true, false, true, false, false, true, true, true, false, true,
    ];
    let mask = Mask::from_iter(keep);
    let view = fsst_filter_to_view(&fsst, &mask, &mut ctx)?;

    let got = canonicalize_fsstview_with(view.as_view(), strategy, &mut ctx)?;
    let expected =
        VarBinArray::from_iter(strings.iter().copied(), DType::Utf8(Nullability::Nullable))
            .into_array()
            .filter(mask)?
            .execute::<VarBinViewArray>(&mut ctx)?;
    assert_arrays_eq!(got, expected.into_array());
    Ok(())
}

/// The VarBin exporter must agree with the canonical VarBin filter, across the export strategies,
/// for a gapped filter over nullable data.
#[rstest]
#[case(FsstViewCompaction::Auto)]
#[case(FsstViewCompaction::GatherBulk)]
#[case(FsstViewCompaction::RunDecode)]
fn varbin_export_matches_canonical(#[case] strategy: FsstViewCompaction) -> VortexResult<()> {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let strings: Vec<Option<&str>> = vec![
        Some("alpha"),
        None,
        Some("bravo bravo"),
        Some("charlie"),
        Some("delta delta delta"),
        None,
        Some("echo"),
        Some("foxtrot foxtrot"),
    ];
    let fsst = make_fsst(&strings, Nullability::Nullable, &mut ctx);
    let keep = [true, false, true, true, false, false, true, true];
    let mask = Mask::from_iter(keep);
    let view = fsst_filter_to_view(&fsst, &mask, &mut ctx)?;

    let got = canonicalize_fsstview_to_varbin(view.as_view(), strategy, &mut ctx)?;
    // Compare as VarBinView so the offsets-vs-views layout difference doesn't matter.
    let got_view = got.execute::<VarBinViewArray>(&mut ctx)?;

    let expected =
        VarBinArray::from_iter(strings.iter().copied(), DType::Utf8(Nullability::Nullable))
            .into_array()
            .filter(mask)?
            .execute::<VarBinViewArray>(&mut ctx)?;
    assert_arrays_eq!(got_view.into_array(), expected.into_array());
    Ok(())
}
