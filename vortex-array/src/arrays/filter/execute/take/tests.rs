// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use crate::IntoArray;
use crate::RecursiveCanonical;
use crate::arrays::BoolArray;
use crate::arrays::DecimalArray;
use crate::arrays::DictArray;
use crate::arrays::FilterArray;
use crate::arrays::FixedSizeListArray;
use crate::arrays::ListArray;
use crate::arrays::Primitive;
use crate::arrays::PrimitiveArray;
use crate::arrays::StructArray;
use crate::arrays::VarBinViewArray;
use crate::assert_arrays_eq;
use crate::dtype::DecimalDType;
use crate::dtype::FieldNames;
use crate::executor::ExecutionCtx;
use crate::validity::Validity;

#[test]
fn test_take_execute_kernel_maps_indices_through_filter() -> VortexResult<()> {
    let filter = FilterArray::new(
        PrimitiveArray::from_option_iter([Some(10i32), Some(20), Some(30), Some(40), None])
            .into_array(),
        Mask::from_iter([true, false, true, true, false]),
    )
    .into_array();
    let parent = DictArray::try_new(
        PrimitiveArray::new(
            buffer![2u64, 100, 0],
            Validity::Array(BoolArray::from_iter([true, false, true]).into_array()),
        )
        .into_array(),
        filter.clone(),
    )?
    .into_array();
    let mut ctx = ExecutionCtx::new(VortexSession::empty());

    let result = filter
        .execute_parent(&parent, 1, &mut ctx)?
        .expect("filter child should execute its take parent");

    assert_arrays_eq!(
        result.execute::<RecursiveCanonical>(&mut ctx)?.0,
        PrimitiveArray::from_option_iter([Some(40i32), None, Some(10)]).into_array()
    );
    Ok(())
}

#[test]
fn test_take_execute_kernel_nullable_fast_path_maps_indices_through_filter() -> VortexResult<()> {
    let filter = FilterArray::new(
        buffer![10i32, 20, 30, 40, 50].into_array(),
        Mask::from_slices(5, vec![(1, 4)]),
    )
    .into_array();
    let parent = DictArray::try_new(
        PrimitiveArray::new(
            buffer![2u64, 100, 0],
            Validity::Array(BoolArray::from_iter([true, false, true]).into_array()),
        )
        .into_array(),
        filter.clone(),
    )?
    .into_array();
    let mut ctx = ExecutionCtx::new(VortexSession::empty());

    let result = filter
        .execute_parent(&parent, 1, &mut ctx)?
        .expect("filter child should execute its take parent");

    assert!(result.as_opt::<Primitive>().is_some());
    assert_arrays_eq!(
        result.execute::<RecursiveCanonical>(&mut ctx)?.0,
        PrimitiveArray::from_option_iter([Some(40i32), None, Some(20)]).into_array()
    );
    Ok(())
}

#[test]
fn test_take_execute_kernel_fast_path_maps_indices_through_filter() -> VortexResult<()> {
    let filter = FilterArray::new(
        buffer![10i32, 20, 30, 40, 50, 60].into_array(),
        Mask::from_indices(6, vec![1, 3, 4, 5]),
    )
    .into_array();
    let parent = DictArray::try_new(buffer![2u64, 0, 3].into_array(), filter.clone())?.into_array();
    let mut ctx = ExecutionCtx::new(VortexSession::empty());

    let result = filter
        .execute_parent(&parent, 1, &mut ctx)?
        .expect("filter child should execute its take parent");

    assert!(result.as_opt::<Primitive>().is_some());
    assert_arrays_eq!(
        result.execute::<RecursiveCanonical>(&mut ctx)?.0,
        PrimitiveArray::from_iter([50i32, 20, 60]).into_array()
    );
    Ok(())
}

fn assert_take_execute_rejects_out_of_bounds_rank(
    child: crate::ArrayRef,
    filter_mask: Mask,
    codes: crate::ArrayRef,
) -> VortexResult<()> {
    let filter = FilterArray::new(child, filter_mask).into_array();
    let parent = DictArray::try_new(codes, filter.clone())?.into_array();
    let mut ctx = ExecutionCtx::new(VortexSession::empty());

    if let Err(err) = filter.execute_parent(&parent, 1, &mut ctx) {
        assert!(
            err.to_string().contains("out of bounds"),
            "unexpected error: {err}"
        );
        return Ok(());
    }

    panic!("out-of-bounds rank should fail");
}

#[test]
fn test_take_execute_kernel_rejects_contiguous_sequential_rank_past_filter_len() -> VortexResult<()>
{
    assert_take_execute_rejects_out_of_bounds_rank(
        buffer![10i32, 20, 30, 40, 50].into_array(),
        Mask::from_slices(5, vec![(1, 4)]),
        buffer![0u64, 1, 2, 3].into_array(),
    )
}

#[test]
fn test_take_execute_kernel_rejects_random_mask_rank_past_filter_len() -> VortexResult<()> {
    assert_take_execute_rejects_out_of_bounds_rank(
        buffer![10i32, 20, 30, 40, 50].into_array(),
        Mask::from_indices(5, vec![1, 3, 4]),
        buffer![2u64, 3].into_array(),
    )
}

#[test]
fn test_take_execute_kernel_rejects_non_contiguous_sequential_rank_past_filter_len()
-> VortexResult<()> {
    assert_take_execute_rejects_out_of_bounds_rank(
        ListArray::try_new(
            buffer![10u32, 11, 20, 30, 31, 32, 40, 50, 51].into_array(),
            buffer![0u32, 2, 3, 6, 7, 9].into_array(),
            Validity::NonNullable,
        )?
        .into_array(),
        Mask::from_indices(5, vec![0, 2, 4]),
        buffer![0u64, 1, 2, 3].into_array(),
    )
}

#[test]
fn test_take_execute_kernel_rejects_translated_rank_past_filter_len() -> VortexResult<()> {
    assert_take_execute_rejects_out_of_bounds_rank(
        ListArray::try_new(
            buffer![10u32, 11, 20, 30, 31, 32, 40, 50, 51].into_array(),
            buffer![0u32, 2, 3, 6, 7, 9].into_array(),
            Validity::NonNullable,
        )?
        .into_array(),
        Mask::from_indices(5, vec![0, 2, 4]),
        buffer![0u64, 3].into_array(),
    )
}

#[test]
fn test_take_execute_kernel_handles_empty_sequential_take() -> VortexResult<()> {
    let filter = FilterArray::new(
        ListArray::try_new(
            buffer![10u32, 11, 20, 30, 31, 32, 40, 50, 51].into_array(),
            buffer![0u32, 2, 3, 6, 7, 9].into_array(),
            Validity::NonNullable,
        )?
        .into_array(),
        Mask::from_indices(5, vec![0, 2, 4]),
    )
    .into_array();
    let parent = DictArray::try_new(
        PrimitiveArray::from_iter(std::iter::empty::<u64>()).into_array(),
        filter.clone(),
    )?
    .into_array();
    let mut ctx = ExecutionCtx::new(VortexSession::empty());

    let result = filter
        .execute_parent(&parent, 1, &mut ctx)?
        .expect("filter child should execute its take parent");

    assert_arrays_eq!(
        result.execute::<RecursiveCanonical>(&mut ctx)?.0,
        ListArray::try_new(
            PrimitiveArray::from_iter(std::iter::empty::<u32>()).into_array(),
            buffer![0u32].into_array(),
            Validity::NonNullable,
        )?
        .into_array()
    );
    Ok(())
}

fn assert_take_execute_maps_child_dtype(
    child: crate::ArrayRef,
    expected: crate::ArrayRef,
) -> VortexResult<()> {
    let filter =
        FilterArray::new(child, Mask::from_iter([true, false, true, true, false])).into_array();
    let parent = DictArray::try_new(buffer![2u64, 0, 1].into_array(), filter.clone())?.into_array();
    let mut ctx = ExecutionCtx::new(VortexSession::empty());

    let result = filter
        .execute_parent(&parent, 1, &mut ctx)?
        .expect("filter child should execute its take parent");

    assert_arrays_eq!(result.execute::<RecursiveCanonical>(&mut ctx)?.0, expected);
    Ok(())
}

#[test]
fn test_take_execute_kernel_skips_bool_filter_child() -> VortexResult<()> {
    let filter = FilterArray::new(
        BoolArray::from_iter([true, false, true, true, false]).into_array(),
        Mask::from_iter([true, false, true, true, false]),
    )
    .into_array();
    let parent = DictArray::try_new(buffer![2u64, 0, 1].into_array(), filter.clone())?.into_array();
    let mut ctx = ExecutionCtx::new(VortexSession::empty());

    let result = filter.execute_parent(&parent, 1, &mut ctx)?;

    assert!(result.is_none());
    Ok(())
}

fn execute_primitive_take(
    filter_mask: Mask,
    take_len: usize,
) -> VortexResult<Option<crate::ArrayRef>> {
    let child_len = filter_mask.len();
    let filtered_len = filter_mask.true_count();
    let child_len_u32 = u32::try_from(child_len)?;
    let filter = FilterArray::new(
        PrimitiveArray::from_iter(0..child_len_u32).into_array(),
        filter_mask,
    )
    .into_array();
    let indices = PrimitiveArray::from_iter((0..take_len).map(|idx| (idx % filtered_len) as u64));
    let parent = DictArray::try_new(indices.into_array(), filter.clone())?.into_array();
    let mut ctx = ExecutionCtx::new(VortexSession::empty());

    filter.execute_parent(&parent, 1, &mut ctx)
}

#[test]
fn test_take_execute_kernel_materializes_large_full_take() -> VortexResult<()> {
    let filtered_len = super::BIG_TAKE_FALLBACK_LEN;
    let result = execute_primitive_take(
        Mask::from_indices(filtered_len * 2, (0..filtered_len).map(|idx| idx * 2)),
        filtered_len,
    )?;

    assert!(result.is_none());
    Ok(())
}

#[test]
fn test_take_execute_kernel_materializes_huge_fixed_width_fanout() -> VortexResult<()> {
    let filtered_len = super::BIG_TAKE_FALLBACK_MIN_FIXED_WIDTH_TAKE_LEN
        / super::BIG_TAKE_FALLBACK_MIN_FIXED_WIDTH_RATIO;
    let result = execute_primitive_take(
        Mask::from_indices(filtered_len * 2, (0..filtered_len).map(|idx| idx * 2)),
        super::BIG_TAKE_FALLBACK_MIN_FIXED_WIDTH_TAKE_LEN,
    )?;

    assert!(result.is_none());
    Ok(())
}

#[test]
fn test_take_execute_kernel_keeps_moderate_fixed_width_fanout_fused() -> VortexResult<()> {
    let filtered_len = super::BIG_TAKE_FALLBACK_MIN_FIXED_WIDTH_TAKE_LEN
        / super::BIG_TAKE_FALLBACK_MIN_FIXED_WIDTH_RATIO;
    let result = execute_primitive_take(
        Mask::from_indices(filtered_len * 2, (0..filtered_len).map(|idx| idx * 2)),
        filtered_len * (super::BIG_TAKE_FALLBACK_MIN_FIXED_WIDTH_RATIO - 1),
    )?;

    assert!(result.is_some());
    Ok(())
}

#[test]
fn test_take_execute_kernel_materializes_large_contiguous_full_take() -> VortexResult<()> {
    let filtered_len = super::BIG_TAKE_FALLBACK_LEN;
    let result = execute_primitive_take(
        Mask::from_slices(
            filtered_len * 2,
            vec![(filtered_len / 2, filtered_len / 2 + filtered_len)],
        ),
        filtered_len,
    )?;

    assert!(result.is_none());
    Ok(())
}

#[test]
fn test_take_execute_kernel_handles_nullable_primitive_filter_child() -> VortexResult<()> {
    let filter = FilterArray::new(
        PrimitiveArray::from_option_iter([Some(10i32), Some(20), None, Some(40), Some(50)])
            .into_array(),
        Mask::from_iter([true, false, true, true, false]),
    )
    .into_array();
    let parent = DictArray::try_new(buffer![2u64, 0, 1].into_array(), filter.clone())?.into_array();
    let mut ctx = ExecutionCtx::new(VortexSession::empty());

    let result = filter.execute_parent(&parent, 1, &mut ctx)?;

    assert_arrays_eq!(
        result
            .expect("filter child should execute its take parent")
            .execute::<RecursiveCanonical>(&mut ctx)?
            .0,
        PrimitiveArray::from_option_iter([Some(40i32), Some(10), None]).into_array()
    );
    Ok(())
}

#[test]
fn test_take_execute_kernel_handles_nullable_decimal_filter_child() -> VortexResult<()> {
    let decimal_dtype = DecimalDType::new(19, 2);
    let filter = FilterArray::new(
        DecimalArray::from_option_iter(
            [Some(100i128), Some(200), None, Some(400), Some(500)],
            decimal_dtype,
        )
        .into_array(),
        Mask::from_iter([true, false, true, true, false]),
    )
    .into_array();
    let parent = DictArray::try_new(buffer![2u64, 0, 1].into_array(), filter.clone())?.into_array();
    let mut ctx = ExecutionCtx::new(VortexSession::empty());

    let result = filter.execute_parent(&parent, 1, &mut ctx)?;

    assert_arrays_eq!(
        result
            .expect("filter child should execute its take parent")
            .execute::<RecursiveCanonical>(&mut ctx)?
            .0,
        DecimalArray::from_option_iter([Some(400i128), Some(100), None], decimal_dtype)
            .into_array()
    );
    Ok(())
}

#[test]
fn test_take_execute_kernel_handles_decimal_filter_child() -> VortexResult<()> {
    let decimal_dtype = DecimalDType::new(19, 2);

    assert_take_execute_maps_child_dtype(
        DecimalArray::new(
            buffer![100i128, 200, 300, 400, 500],
            decimal_dtype,
            Validity::NonNullable,
        )
        .into_array(),
        DecimalArray::new(
            buffer![400i128, 100, 300],
            decimal_dtype,
            Validity::NonNullable,
        )
        .into_array(),
    )
}

#[test]
fn test_take_execute_kernel_handles_fixed_size_list_filter_child() -> VortexResult<()> {
    assert_take_execute_maps_child_dtype(
        FixedSizeListArray::new(
            buffer![10u32, 11, 20, 21, 30, 31, 40, 41, 50, 51].into_array(),
            2,
            Validity::NonNullable,
            5,
        )
        .into_array(),
        FixedSizeListArray::new(
            buffer![40u32, 41, 10, 11, 30, 31].into_array(),
            2,
            Validity::NonNullable,
            3,
        )
        .into_array(),
    )
}

#[test]
fn test_take_execute_kernel_handles_list_filter_child() -> VortexResult<()> {
    assert_take_execute_maps_child_dtype(
        ListArray::try_new(
            buffer![10u32, 11, 20, 30, 31, 32, 40, 50, 51].into_array(),
            buffer![0u32, 2, 3, 6, 7, 9].into_array(),
            Validity::NonNullable,
        )?
        .into_array(),
        ListArray::try_new(
            buffer![40u32, 10, 11, 30, 31, 32].into_array(),
            buffer![0u32, 1, 3, 6].into_array(),
            Validity::NonNullable,
        )?
        .into_array(),
    )
}

#[test]
fn test_take_execute_kernel_handles_string_filter_child() -> VortexResult<()> {
    assert_take_execute_maps_child_dtype(
        VarBinViewArray::from_iter_str(["a", "b", "c", "d", "e"]).into_array(),
        VarBinViewArray::from_iter_str(["d", "a", "c"]).into_array(),
    )
}

#[test]
fn test_take_execute_kernel_handles_struct_filter_child() -> VortexResult<()> {
    assert_take_execute_maps_child_dtype(
        StructArray::try_new(
            FieldNames::from(["id", "value"]),
            vec![
                buffer![10u32, 20, 30, 40, 50].into_array(),
                buffer![100u64, 200, 300, 400, 500].into_array(),
            ],
            5,
            Validity::NonNullable,
        )?
        .into_array(),
        StructArray::try_new(
            FieldNames::from(["id", "value"]),
            vec![
                buffer![40u32, 10, 30].into_array(),
                buffer![400u64, 100, 300].into_array(),
            ],
            3,
            Validity::NonNullable,
        )?
        .into_array(),
    )
}
