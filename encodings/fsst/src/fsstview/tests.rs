// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use rstest::rstest;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::LEGACY_SESSION;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::VarBinArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::assert_arrays_eq;
use vortex_array::compute::conformance::consistency::test_array_consistency;
use vortex_array::compute::conformance::filter::test_filter_conformance;
use vortex_array::compute::conformance::take::test_take_conformance;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::FSSTView;
use crate::FSSTViewArray;
use crate::fsst_compress;
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
fn filter_matches_canonical() -> VortexResult<()> {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let view = make_fsstview(&SAMPLE_NULLABLE, Nullability::Nullable, &mut ctx);

    let mask = Mask::from_iter([true, false, true, false, true, true]);

    // The filtered FSSTView reuses the original byte heap untouched.
    let filtered = view.into_array().filter(mask.clone())?;
    let result = filtered.execute::<VarBinViewArray>(&mut ctx)?;

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
fn take_matches_canonical() -> VortexResult<()> {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let view = make_fsstview(&SAMPLE, Nullability::NonNullable, &mut ctx);

    // Reorders and duplicates, which is fine for offsets+sizes addressing.
    let indices = vortex_array::arrays::PrimitiveArray::from_iter([5u64, 0, 0, 3, 1]).into_array();

    let taken = view.into_array().take(indices.clone())?;
    let result = taken.execute::<VarBinViewArray>(&mut ctx)?;

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

#[test]
fn slice_matches_canonical() -> VortexResult<()> {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let view = make_fsstview(&SAMPLE, Nullability::NonNullable, &mut ctx);

    let sliced = view.into_array().slice(1..4)?;
    let result = sliced.execute::<VarBinViewArray>(&mut ctx)?;

    let expected = VarBinArray::from_iter(
        SAMPLE.iter().copied(),
        DType::Utf8(Nullability::NonNullable),
    )
    .into_array()
    .slice(1..4)?
    .execute::<VarBinViewArray>(&mut ctx)?;

    assert_arrays_eq!(result.into_array(), expected.into_array());
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
