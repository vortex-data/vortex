// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`PatchesV2`] must be able to stand in for the patch set inside a constant-fill
//! [`Sparse`] array, for every dtype `Sparse` supports.
//!
//! Each case builds a `Sparse` array from patch values of one dtype, converts those patches
//! through [`PatchesV2`] and back, and asserts the rebuilt array is identical. The container
//! addresses patches by position only, so this pins down that nothing in the chunk-local
//! addressing is dtype-specific.

#![allow(clippy::cast_possible_truncation, clippy::tests_outside_test_module)]

use std::sync::LazyLock;

use rstest::rstest;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::DecimalArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::NullArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::DType;
use vortex_array::dtype::DecimalDType;
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::Nullability::Nullable;
use vortex_array::dtype::PType;
use vortex_array::dtype::StructFields;
use vortex_array::patches::Patches;
use vortex_array::patches_v2::PatchesV2;
use vortex_array::patches_v2::force_patches_v2_scatter;
use vortex_array::patches_v2::patches_v2_scatter_count;
use vortex_array::scalar::Scalar;
use vortex_array::validity::Validity;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::VortexSession;
use vortex_sparse::Sparse;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = vortex_array::array_session();
    vortex_sparse::initialize(&session);
    session
});

/// Rows carrying a patch, spread across chunk boundaries of a 3000-row array.
const POSITIONS: [u64; 4] = [3, 1023, 1024, 2050];
const ARRAY_LEN: usize = 3000;

fn primitive_values<T: NativePType>(values: [T; 4]) -> ArrayRef {
    PrimitiveArray::from_option_iter(values.map(Some)).into_array()
}

/// Four patch values of the named dtype, always nullable so `Scalar::null` is a legal fill.
fn values_for(case: &str) -> VortexResult<ArrayRef> {
    Ok(match case {
        "null" => NullArray::new(4).into_array(),
        "bool" => BoolArray::from_iter([Some(true), None, Some(false), Some(true)]).into_array(),
        "u8" => primitive_values([1u8, 2, 3, 4]),
        "i16" => primitive_values([-1i16, 2, -3, 4]),
        "u32" => primitive_values([1u32, 2, 3, 4]),
        "i64" => primitive_values([-1i64, 2, -3, 4]),
        "f32" => primitive_values([1.5f32, -2.5, 3.5, f32::NAN]),
        "f64" => primitive_values([1.5f64, -2.5, 3.5, f64::INFINITY]),
        "decimal" => DecimalArray::new(
            buffer![100i128, 200, 300, 4000],
            DecimalDType::new(3, 2),
            Validity::from_iter([true, true, true, false]),
        )
        .into_array(),
        "utf8_varbin" => VarBinArray::from_iter(
            [
                Some("a"),
                None,
                Some("ccc"),
                Some("a string too long to inline"),
            ],
            DType::Utf8(Nullable),
        )
        .into_array(),
        "utf8_varbinview" => VarBinViewArray::from_iter(
            [
                Some("a"),
                None,
                Some("ccc"),
                Some("a string too long to inline"),
            ],
            DType::Utf8(Nullable),
        )
        .into_array(),
        "binary" => VarBinArray::from_iter(
            [
                Some(vec![1u8, 2]),
                None,
                Some(vec![3u8]),
                Some(vec![4u8; 40]),
            ],
            DType::Binary(Nullable),
        )
        .into_array(),
        "list" => ListArray::try_new(
            buffer![1i32, 2, 3, 4, 5, 6].into_array(),
            buffer![0u32, 1, 3, 3, 6].into_array(),
            Validity::AllValid,
        )?
        .into_array(),
        "fixed_size_list" => FixedSizeListArray::try_new(
            buffer![1i32, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12].into_array(),
            3,
            Validity::AllValid,
            4,
        )?
        .into_array(),
        "struct" => StructArray::try_new_with_dtype(
            vec![
                primitive_values([1i32, 2, 3, 4]),
                VarBinViewArray::from_iter(
                    [Some("x"), Some("y"), None, Some("z")],
                    DType::Utf8(Nullable),
                )
                .into_array(),
            ],
            StructFields::new(
                FieldNames::from_iter(["a", "b"]),
                vec![
                    DType::Primitive(PType::I32, Nullable),
                    DType::Utf8(Nullable),
                ],
            ),
            4,
            Validity::AllValid,
        )?
        .into_array(),
        other => vortex_bail!("unknown case {other}"),
    })
}

/// Build the equivalent chunk-local patch set for `values` at [`POSITIONS`].
fn build(values: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<(Patches, PatchesV2)> {
    let indices = buffer![POSITIONS[0], POSITIONS[1], POSITIONS[2], POSITIONS[3]].into_array();
    let patches = Patches::new(ARRAY_LEN, 0, indices, values.clone(), None)?;
    let v2 = PatchesV2::from_patches(&patches, ctx)?;
    Ok((patches, v2))
}

/// A `Sparse` array rebuilt through `PatchesV2` is identical to the original, for every dtype.
#[rstest]
#[case::null("null")]
#[case::bool("bool")]
#[case::u8("u8")]
#[case::i16("i16")]
#[case::u32("u32")]
#[case::i64("i64")]
#[case::f32("f32")]
#[case::f64("f64")]
#[case::decimal("decimal")]
#[case::utf8_varbin("utf8_varbin")]
#[case::utf8_varbinview("utf8_varbinview")]
#[case::binary("binary")]
#[case::list("list")]
#[case::fixed_size_list("fixed_size_list")]
#[case::struct_("struct")]
fn stands_in_for_constant_fill_sparse(#[case] case: &str) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let values = values_for(case)?;
    let fill = Scalar::null(values.dtype().clone());
    let (patches, v2) = build(&values, &mut ctx)?;

    assert_eq!(
        v2.dtype(),
        values.dtype(),
        "{case}: dtype is carried through"
    );
    assert_eq!(v2.num_patches(), POSITIONS.len());

    // Positions survive the round trip through chunk-local addressing.
    let original = Sparse::try_new_from_patches(patches, fill.clone())?.into_array();
    let rebuilt =
        Sparse::try_new_from_patches(v2.to_patches(&mut ctx)?, fill.clone())?.into_array();
    assert_arrays_eq!(original, rebuilt, &mut ctx);

    // Scalar lookups match the canonicalized array at every patched and unpatched row.
    let canonical = original.execute::<Canonical>(&mut ctx)?.into_array();
    for (ordinal, &position) in POSITIONS.iter().enumerate() {
        let position = position as usize;
        let patched = v2
            .get_patched(position, &mut ctx)?
            .unwrap_or_else(|| panic!("{case}: row {position} is patched"));
        assert_eq!(
            patched,
            canonical.execute_scalar(position, &mut ctx)?,
            "{case}: patch value at row {position}"
        );
        assert_eq!(patched, values.execute_scalar(ordinal, &mut ctx)?);
        // The next row, when it is not itself patched, falls back to the fill value.
        let next = position + 1;
        if !POSITIONS.contains(&(next as u64)) {
            assert_eq!(v2.get_patched(next, &mut ctx)?, None, "{case}: row {next}");
            assert_eq!(canonical.execute_scalar(next, &mut ctx)?, fill);
        }
    }
    Ok(())
}

/// Slicing the patch set matches slicing the equivalent `Sparse` array, for every dtype.
#[rstest]
#[case::bool("bool")]
#[case::i64("i64")]
#[case::f64("f64")]
#[case::decimal("decimal")]
#[case::utf8_varbinview("utf8_varbinview")]
#[case::binary("binary")]
#[case::list("list")]
#[case::fixed_size_list("fixed_size_list")]
#[case::struct_("struct")]
fn slices_like_sparse(#[case] case: &str) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let values = values_for(case)?;
    let fill = Scalar::null(values.dtype().clone());
    let (patches, v2) = build(&values, &mut ctx)?;

    // 1023..2051 drops the patch at row 3, keeps the other three, and starts unaligned in the
    // middle of the first chunk.
    let range = 1023..2051;
    let sliced = v2
        .slice(range.clone(), &mut ctx)?
        .expect("patches remain in slice");
    assert_eq!(sliced.num_patches(), 3);
    assert_eq!(sliced.offset(), 1023);
    assert_eq!(sliced.array_len(), range.len());

    let original = Sparse::try_new_from_patches(patches, fill.clone())?
        .into_array()
        .slice(range)?;
    let rebuilt = Sparse::try_new_from_patches(sliced.to_patches(&mut ctx)?, fill)?.into_array();
    assert_arrays_eq!(original, rebuilt, &mut ctx);
    Ok(())
}

/// Canonicalizing a constant-fill `Sparse` array through the chunk-local scatter produces the
/// same array as the default patch path.
///
/// The scatter only applies where the types match: a non-nullable primitive result, where there
/// is no validity to patch and canonicalization is purely a value scatter over the fill.
///
/// The dtypes are covered in one test rather than as `rstest` cases because the scatter switch is
/// process-global, so cases toggling it in parallel would race.
#[test]
fn scatter_matches_default_sparse_execution() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    // Non-nullable values and a non-null fill are what put this on the chunk-local path.
    let cases: Vec<(&str, ArrayRef)> = vec![
        (
            "u8",
            PrimitiveArray::new(buffer![1u8, 2, 3, 4], Validity::NonNullable).into_array(),
        ),
        (
            "i16",
            PrimitiveArray::new(buffer![-1i16, 2, -3, 4], Validity::NonNullable).into_array(),
        ),
        (
            "u32",
            PrimitiveArray::new(buffer![1u32, 2, 3, 4], Validity::NonNullable).into_array(),
        ),
        (
            "i64",
            PrimitiveArray::new(buffer![-1i64, 2, -3, 4], Validity::NonNullable).into_array(),
        ),
        (
            "f32",
            PrimitiveArray::new(buffer![1.5f32, -2.5, 3.5, 4.5], Validity::NonNullable)
                .into_array(),
        ),
        (
            "f64",
            PrimitiveArray::new(buffer![1.5f64, -2.5, 3.5, 4.5], Validity::NonNullable)
                .into_array(),
        ),
    ];

    for (case, values) in cases {
        let fill = values.execute_scalar(0, &mut ctx)?;
        let indices = buffer![POSITIONS[0], POSITIONS[1], POSITIONS[2], POSITIONS[3]].into_array();
        let sparse = Sparse::try_new(indices, values, ARRAY_LEN, fill)?.into_array();

        force_patches_v2_scatter(false);
        let default_path = sparse.clone().execute::<Canonical>(&mut ctx)?.into_array();

        // Guard against this going vacuous: the scatter count must move, or the flag silently
        // did nothing and both sides would be the same code path.
        let before = patches_v2_scatter_count();
        force_patches_v2_scatter(true);
        let chunk_local_path = sparse.execute::<Canonical>(&mut ctx)?.into_array();
        force_patches_v2_scatter(false);
        assert!(
            patches_v2_scatter_count() > before,
            "{case}: expected canonicalization to take the chunk-local scatter"
        );

        assert_arrays_eq!(default_path, chunk_local_path, &mut ctx);
    }
    Ok(())
}
