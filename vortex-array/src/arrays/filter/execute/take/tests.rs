// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::LazyLock;

use rstest::rstest;
use vortex_buffer::Alignment;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_buffer::buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::arrays::BoolArray;
use crate::arrays::DecimalArray;
use crate::arrays::Dict;
use crate::arrays::DictArray;
use crate::arrays::Filter;
use crate::arrays::FilterArray;
use crate::arrays::FixedSizeListArray;
use crate::arrays::ListArray;
use crate::arrays::Primitive;
use crate::arrays::PrimitiveArray;
use crate::arrays::StructArray;
use crate::arrays::VarBinViewArray;
use crate::arrays::decimal::DecimalArrayExt;
use crate::arrays::dict::TakeExecuteAdaptor;
use crate::assert_arrays_eq;
use crate::buffer::BufferHandle;
use crate::dtype::BigCast;
use crate::dtype::DecimalDType;
use crate::dtype::DecimalType;
use crate::dtype::FieldNames;
use crate::dtype::half::f16;
use crate::dtype::i256;
use crate::executor::ExecutionCtx;
use crate::kernel::ExecuteParentKernel;
use crate::match_each_decimal_value_type;
use crate::validity::Validity;

static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

fn execute_parent(
    child: &crate::ArrayRef,
    parent: &crate::ArrayRef,
    child_idx: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<crate::ArrayRef>> {
    TakeExecuteAdaptor(Filter).execute_parent(
        child.as_::<Filter>(),
        parent.as_::<Dict>(),
        child_idx,
        ctx,
    )
}

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
    let mut ctx = SESSION.create_execution_ctx();

    let result = execute_parent(&filter, &parent, 1, &mut ctx)?
        .expect("filter child should execute its take parent");

    assert_arrays_eq!(
        result,
        PrimitiveArray::from_option_iter([Some(40i32), None, Some(10)]).into_array(),
        &mut ctx
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
    let mut ctx = SESSION.create_execution_ctx();

    let result = execute_parent(&filter, &parent, 1, &mut ctx)?
        .expect("filter child should execute its take parent");

    assert!(result.as_opt::<Primitive>().is_some());
    assert_arrays_eq!(
        result,
        PrimitiveArray::from_option_iter([Some(40i32), None, Some(20)]).into_array(),
        &mut ctx
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
    let mut ctx = SESSION.create_execution_ctx();

    let result = execute_parent(&filter, &parent, 1, &mut ctx)?
        .expect("filter child should execute its take parent");

    assert!(result.as_opt::<Primitive>().is_some());
    assert_arrays_eq!(
        result,
        PrimitiveArray::from_iter([50i32, 20, 60]).into_array(),
        &mut ctx
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
    let mut ctx = SESSION.create_execution_ctx();

    if let Err(err) = execute_parent(&filter, &parent, 1, &mut ctx) {
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
    let mut ctx = SESSION.create_execution_ctx();
    assert_take_execute_rejects_out_of_bounds_rank(
        ListArray::try_new(
            buffer![10u32, 11, 20, 30, 31, 32, 40, 50, 51].into_array(),
            buffer![0u32, 2, 3, 6, 7, 9].into_array(),
            Validity::NonNullable,
            &mut ctx,
        )?
        .into_array(),
        Mask::from_indices(5, vec![0, 2, 4]),
        buffer![0u64, 1, 2, 3].into_array(),
    )
}

#[test]
fn test_take_execute_kernel_rejects_translated_rank_past_filter_len() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    assert_take_execute_rejects_out_of_bounds_rank(
        ListArray::try_new(
            buffer![10u32, 11, 20, 30, 31, 32, 40, 50, 51].into_array(),
            buffer![0u32, 2, 3, 6, 7, 9].into_array(),
            Validity::NonNullable,
            &mut ctx,
        )?
        .into_array(),
        Mask::from_indices(5, vec![0, 2, 4]),
        buffer![0u64, 3].into_array(),
    )
}

#[test]
fn test_take_execute_kernel_handles_empty_sequential_take() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let filter = FilterArray::new(
        ListArray::try_new(
            buffer![10u32, 11, 20, 30, 31, 32, 40, 50, 51].into_array(),
            buffer![0u32, 2, 3, 6, 7, 9].into_array(),
            Validity::NonNullable,
            &mut ctx,
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
    let mut ctx = SESSION.create_execution_ctx();

    let result = execute_parent(&filter, &parent, 1, &mut ctx)?
        .expect("filter child should execute its take parent");

    assert_arrays_eq!(
        result,
        ListArray::try_new(
            PrimitiveArray::from_iter(std::iter::empty::<u32>()).into_array(),
            buffer![0u32].into_array(),
            Validity::NonNullable,
            &mut ctx,
        )?
        .into_array(),
        &mut ctx
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
    let mut ctx = SESSION.create_execution_ctx();

    let result = execute_parent(&filter, &parent, 1, &mut ctx)?
        .expect("filter child should execute its take parent");

    assert_arrays_eq!(result, expected, &mut ctx);
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
    let mut ctx = SESSION.create_execution_ctx();

    let result = execute_parent(&filter, &parent, 1, &mut ctx)?;

    assert!(result.is_none());
    Ok(())
}

fn execute_primitive_take(
    filter_mask: Mask,
    take_len: usize,
) -> VortexResult<Option<crate::ArrayRef>> {
    let selected: Vec<_> = filter_mask
        .iter()
        .enumerate()
        .filter_map(|(i, keep)| keep.then_some(u32::try_from(i).unwrap()))
        .collect();
    let expected = PrimitiveArray::from_iter((0..take_len).map(|i| selected[i % selected.len()]));
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
    let mut ctx = SESSION.create_execution_ctx();

    let result = execute_parent(&filter, &parent, 1, &mut ctx)?;
    let actual = result.as_ref().unwrap_or(&parent);
    assert_arrays_eq!(actual, expected, &mut ctx);
    Ok(result)
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
    let mut ctx = SESSION.create_execution_ctx();

    let result = execute_parent(&filter, &parent, 1, &mut ctx)?;

    assert_arrays_eq!(
        result.expect("filter child should execute its take parent"),
        PrimitiveArray::from_option_iter([Some(40i32), Some(10), None]).into_array(),
        &mut ctx
    );
    Ok(())
}

#[test]
fn test_take_execute_kernel_preserves_nullable_all_valid_fixed_width_child() -> VortexResult<()> {
    let filter = FilterArray::new(
        PrimitiveArray::new(buffer![10i32, 20, 30], Validity::AllValid).into_array(),
        Mask::new_true(3),
    )
    .into_array();
    let parent = DictArray::try_new(buffer![0u64, 1].into_array(), filter.clone())?.into_array();
    let mut ctx = SESSION.create_execution_ctx();

    let result = execute_parent(&filter, &parent, 1, &mut ctx)?
        .expect("filter child should execute its take parent");

    assert_eq!(result.dtype(), parent.dtype());
    assert_arrays_eq!(
        result,
        PrimitiveArray::new(buffer![10i32, 20], Validity::AllValid).into_array(),
        &mut ctx
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
    let mut ctx = SESSION.create_execution_ctx();

    let result = execute_parent(&filter, &parent, 1, &mut ctx)?;

    assert_arrays_eq!(
        result.expect("filter child should execute its take parent"),
        DecimalArray::from_option_iter([Some(400i128), Some(100), None], decimal_dtype)
            .into_array(),
        &mut ctx
    );
    Ok(())
}

/// Each case has values of a different width, which the take copies as raw bytes.
#[rstest]
#[case::f16(
    PrimitiveArray::from_iter([1.0f32, 2.0, 3.0, 4.0, 5.0].map(f16::from_f32)).into_array(),
    PrimitiveArray::from_option_iter([Some(f16::from_f32(4.0)), None, Some(f16::from_f32(1.0))])
        .into_array(),
)]
#[case::f32(
    PrimitiveArray::from_iter([1.5f32, 2.5, 3.5, 4.5, 5.5]).into_array(),
    PrimitiveArray::from_option_iter([Some(4.5f32), None, Some(1.5)]).into_array(),
)]
#[case::f64(
    PrimitiveArray::from_iter([1.5f64, 2.5, 3.5, 4.5, 5.5]).into_array(),
    PrimitiveArray::from_option_iter([Some(4.5f64), None, Some(1.5)]).into_array(),
)]
#[case::decimal_i8(
    DecimalArray::from_iter([10i8, 20, 30, 40, 50], DecimalDType::new(2, 1)).into_array(),
    DecimalArray::from_option_iter([Some(40i8), None, Some(10)], DecimalDType::new(2, 1))
        .into_array(),
)]
#[case::decimal_i256(
    DecimalArray::from_iter(
        [10i128, 20, 30, 40, 50].map(i256::from_i128),
        DecimalDType::new(40, 1),
    )
    .into_array(),
    DecimalArray::from_option_iter(
        [Some(i256::from_i128(40)), None, Some(i256::from_i128(10))],
        DecimalDType::new(40, 1),
    )
    .into_array(),
)]
fn test_take_execute_kernel_copies_every_fixed_width(
    #[case] child: crate::ArrayRef,
    #[case] expected: crate::ArrayRef,
) -> VortexResult<()> {
    let filter =
        FilterArray::new(child, Mask::from_iter([true, false, true, true, false])).into_array();
    let codes = PrimitiveArray::new(
        buffer![2u64, 100, 0],
        Validity::Array(BoolArray::from_iter([true, false, true]).into_array()),
    );
    let parent = DictArray::try_new(codes.into_array(), filter.clone())?.into_array();
    let mut ctx = SESSION.create_execution_ctx();

    let result = execute_parent(&filter, &parent, 1, &mut ctx)?;

    assert_arrays_eq!(
        result.expect("filter child should execute its take parent"),
        expected,
        &mut ctx
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

#[rstest]
fn decimal_contiguous_take_preserves_storage_and_shares_buffer(
    #[values(
        DecimalType::I8,
        DecimalType::I16,
        DecimalType::I32,
        DecimalType::I64,
        DecimalType::I128,
        DecimalType::I256
    )]
    values_type: DecimalType,
) -> VortexResult<()> {
    let dtype = DecimalDType::new(2, 0);
    let (child, expected) = match_each_decimal_value_type!(values_type, |T| {
        let values = Buffer::from_iter([1i8, 2, 3, 4, 5].map(|value| {
            <T as BigCast>::from(value).vortex_expect("small values fit every decimal type")
        }));
        let alignment = Alignment::of::<T>();
        let mut bytes =
            BufferMut::<u8>::with_capacity(values.as_bytes().len() + alignment.as_usize());

        // Offset by the native alignment to exercise storage that may not align to its byte width.
        bytes.extend_from_slice(&vec![0; alignment.as_usize()]);
        bytes.extend_from_slice(values.as_bytes());
        let bytes = bytes
            .freeze()
            .slice_with_alignment(alignment.as_usize().., alignment);
        let child = DecimalArray::try_new_handle(
            BufferHandle::new_host(bytes),
            values_type,
            dtype,
            Validity::NonNullable,
        )?;
        let expected = DecimalArray::from_iter(
            [3i8, 4].map(|value| {
                <T as BigCast>::from(value).vortex_expect("small values fit every decimal type")
            }),
            dtype,
        );

        (child, expected)
    });
    let original_ptr = child.buffer_handle().as_host_opt().unwrap().as_ptr();
    let filter =
        FilterArray::new(child.into_array(), Mask::from_slices(5, vec![(1, 5)])).into_array();
    let parent = DictArray::try_new(buffer![1u64, 2].into_array(), filter.clone())?.into_array();
    let mut ctx = SESSION.create_execution_ctx();
    let taken = execute_parent(&filter, &parent, 1, &mut ctx)?
        .vortex_expect("decimal filter should execute its take parent")
        .execute::<DecimalArray>(&mut ctx)?;

    assert_eq!(taken.values_type(), values_type);
    assert_eq!(
        taken.buffer_handle().as_host_opt().unwrap().as_ptr(),
        original_ptr.wrapping_add(2 * values_type.byte_width()),
    );
    assert_arrays_eq!(taken, expected, &mut ctx);
    Ok(())
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
    let mut ctx = SESSION.create_execution_ctx();
    assert_take_execute_maps_child_dtype(
        ListArray::try_new(
            buffer![10u32, 11, 20, 30, 31, 32, 40, 50, 51].into_array(),
            buffer![0u32, 2, 3, 6, 7, 9].into_array(),
            Validity::NonNullable,
            &mut ctx,
        )?
        .into_array(),
        ListArray::try_new(
            buffer![40u32, 10, 11, 30, 31, 32].into_array(),
            buffer![0u32, 1, 3, 6].into_array(),
            Validity::NonNullable,
            &mut ctx,
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
fn test_take_execute_kernel_preserves_nullable_indices_dtype_fast_path() -> VortexResult<()> {
    let filter = FilterArray::new(
        VarBinViewArray::from_iter_str(["a", "b", "c"]).into_array(),
        Mask::new_true(3),
    )
    .into_array();
    let parent = DictArray::try_new(
        PrimitiveArray::new(buffer![0u64, 1], Validity::AllValid).into_array(),
        filter.clone(),
    )?
    .into_array();
    let mut ctx = SESSION.create_execution_ctx();

    let result = execute_parent(&filter, &parent, 1, &mut ctx)?
        .expect("filter child should execute its nullable take parent");

    assert_eq!(result.dtype(), parent.dtype());
    assert_arrays_eq!(
        result,
        VarBinViewArray::from_iter_nullable_str([Some("a"), Some("b")]).into_array(),
        &mut ctx
    );
    Ok(())
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
