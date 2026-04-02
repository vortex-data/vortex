// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use num_traits::AsPrimitive;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_mask::AllOr;
use vortex_mask::Mask;
use vortex_mask::MaskIter;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::PrimitiveArray;
use crate::arrays::VarBin;
use crate::arrays::VarBinArray;
use crate::arrays::filter::FilterKernel;
use crate::arrays::varbin::builder::VarBinBuilder;
use crate::dtype::DType;
use crate::dtype::IntegerPType;
use crate::match_each_integer_ptype;
use crate::validity::Validity;

impl FilterKernel for VarBin {
    fn filter(
        array: ArrayView<'_, VarBin>,
        mask: &Mask,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        filter_select_var_bin(array, mask, ctx).map(|a| Some(a.into_array()))
    }
}

fn filter_select_var_bin(
    arr: ArrayView<'_, VarBin>,
    mask: &Mask,
    ctx: &mut ExecutionCtx,
) -> VortexResult<VarBinArray> {
    match mask
        .values()
        .vortex_expect("AllTrue and AllFalse are handled by filter fn")
        .threshold_iter(0.5)
    {
        MaskIter::Indices(indices) => {
            filter_select_var_bin_by_index(arr, indices, mask.true_count(), ctx)
        }
        MaskIter::Slices(slices) => {
            filter_select_var_bin_by_slice(arr, slices, mask.true_count(), ctx)
        }
    }
}

fn filter_select_var_bin_by_slice(
    values: ArrayView<'_, VarBin>,
    mask_slices: &[(usize, usize)],
    selection_count: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<VarBinArray> {
    let offsets = values.offsets().clone().execute::<PrimitiveArray>(ctx)?;
    match_each_integer_ptype!(offsets.ptype(), |O| {
        filter_select_var_bin_by_slice_primitive_offset(
            values.dtype().clone(),
            offsets.as_slice::<O>(),
            values.bytes().as_slice(),
            mask_slices,
            values.validity_mask(),
            selection_count,
        )
    })
}

fn filter_select_var_bin_by_slice_primitive_offset<O>(
    dtype: DType,
    offsets: &[O],
    data: &[u8],
    mask_slices: &[(usize, usize)],
    logical_validity: Mask,
    selection_count: usize,
) -> VortexResult<VarBinArray>
where
    O: IntegerPType,
    usize: AsPrimitive<O>,
{
    let mut builder = VarBinBuilder::<O>::with_capacity(selection_count);
    match logical_validity.bit_buffer() {
        AllOr::All => {
            mask_slices.iter().for_each(|(start, end)| {
                update_non_nullable_slice(data, offsets, &mut builder, *start, *end)
            });
        }
        AllOr::None => {
            builder.append_n_nulls(selection_count);
        }
        AllOr::Some(validity) => {
            for (start, end) in mask_slices.iter().copied() {
                let null_sl = validity.slice(start..end);
                if null_sl.true_count() == null_sl.len() {
                    update_non_nullable_slice(data, offsets, &mut builder, start, end)
                } else {
                    for (idx, valid) in null_sl.iter().enumerate() {
                        if valid {
                            let s = offsets[idx + start].to_usize().ok_or_else(|| {
                                vortex_err!(
                                    "Failed to convert offset to usize: {}",
                                    offsets[idx + start]
                                )
                            })?;
                            let e = offsets[idx + start + 1].to_usize().ok_or_else(|| {
                                vortex_err!(
                                    "Failed to convert offset to usize: {}",
                                    offsets[idx + start + 1]
                                )
                            })?;
                            builder.append_value(&data[s..e])
                        } else {
                            builder.append_null()
                        }
                    }
                }
            }
        }
    }
    Ok(builder.finish(dtype))
}

fn update_non_nullable_slice<O>(
    data: &[u8],
    offsets: &[O],
    builder: &mut VarBinBuilder<O>,
    start: usize,
    end: usize,
) where
    O: IntegerPType,
    usize: AsPrimitive<O>,
{
    let new_data = {
        let offset_start = offsets[start].to_usize().unwrap_or_else(|| {
            vortex_panic!("Failed to convert offset to usize: {}", offsets[start])
        });
        let offset_end = offsets[end].to_usize().unwrap_or_else(|| {
            vortex_panic!("Failed to convert offset to usize: {}", offsets[end])
        });
        &data[offset_start..offset_end]
    };
    let new_offsets = offsets[start..end + 1]
        .iter()
        .map(|o| *o - offsets[start])
        .dropping(1);
    builder.append_values(new_data, new_offsets, end - start)
}

fn filter_select_var_bin_by_index(
    values: ArrayView<'_, VarBin>,
    mask_indices: &[usize],
    selection_count: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<VarBinArray> {
    let offsets = values.offsets().clone().execute::<PrimitiveArray>(ctx)?;
    match_each_integer_ptype!(offsets.ptype(), |O| {
        filter_select_var_bin_by_index_primitive_offset(
            values.dtype().clone(),
            offsets.as_slice::<O>(),
            values.bytes().as_slice(),
            mask_indices,
            values.validity(),
            selection_count,
        )
    })
}

fn filter_select_var_bin_by_index_primitive_offset<O: IntegerPType>(
    dtype: DType,
    offsets: &[O],
    data: &[u8],
    mask_indices: &[usize],
    // TODO(ngates): pass LogicalValidity instead
    validity: Validity,
    selection_count: usize,
) -> VortexResult<VarBinArray> {
    let mut builder = VarBinBuilder::<O>::with_capacity(selection_count);
    for idx in mask_indices.iter().copied() {
        if validity.is_valid(idx)? {
            let (start, end) = (
                offsets[idx].to_usize().ok_or_else(|| {
                    vortex_err!("Failed to convert offset to usize: {}", offsets[idx])
                })?,
                offsets[idx + 1].to_usize().ok_or_else(|| {
                    vortex_err!("Failed to convert offset to usize: {}", offsets[idx + 1])
                })?,
            );
            builder.append_value(&data[start..end])
        } else {
            builder.append_null()
        }
    }
    Ok(builder.finish(dtype))
}

#[cfg(test)]
mod test {
    use vortex_buffer::ByteBuffer;
    use vortex_buffer::buffer;

    use crate::IntoArray;
    use crate::LEGACY_SESSION;
    use crate::VortexSessionExecute;
    use crate::arrays::BoolArray;
    use crate::arrays::VarBinArray;
    use crate::arrays::varbin::compute::filter::filter_select_var_bin_by_index;
    use crate::arrays::varbin::compute::filter::filter_select_var_bin_by_slice;
    use crate::assert_arrays_eq;
    use crate::compute::conformance::filter::test_filter_conformance;
    use crate::dtype::DType;
    use crate::dtype::Nullability::NonNullable;
    use crate::dtype::Nullability::Nullable;
    use crate::validity::Validity;

    #[test]
    fn filter_var_bin_test() {
        let arr = VarBinArray::from_vec(
            vec![
                b"hello".as_slice(),
                b"world".as_slice(),
                b"filter".as_slice(),
            ],
            DType::Utf8(NonNullable),
        );
        let arr = arr.as_view();
        let buf = filter_select_var_bin_by_index(
            arr,
            &[0, 2],
            2,
            &mut LEGACY_SESSION.create_execution_ctx(),
        )
        .unwrap();

        assert_arrays_eq!(buf, VarBinArray::from(vec!["hello", "filter"]));
    }

    #[test]
    fn filter_var_bin_slice_test() {
        let arr = VarBinArray::from_vec(
            vec![
                b"hello".as_slice(),
                b"world".as_slice(),
                b"filter".as_slice(),
                b"filter2".as_slice(),
                b"filter3".as_slice(),
            ],
            DType::Utf8(NonNullable),
        );

        let arr = arr.as_view();
        let buf = filter_select_var_bin_by_slice(
            arr,
            &[(0, 1), (2, 3), (4, 5)],
            3,
            &mut LEGACY_SESSION.create_execution_ctx(),
        )
        .unwrap();

        assert_arrays_eq!(buf, VarBinArray::from(vec!["hello", "filter", "filter3"]));
    }

    #[test]
    fn filter_var_bin_slice_null() {
        let bytes = [
            b"one".as_slice(),
            b"two".as_slice(),
            b"three".as_slice(),
            b"four".as_slice(),
            b"five".as_slice(),
            b"six".as_slice(),
        ]
        .into_iter()
        .flat_map(|x| x.iter().cloned())
        .collect::<ByteBuffer>();

        let offsets = buffer![0, 3, 6, 11, 15, 19, 22].into_array();
        let validity = Validity::Array(
            BoolArray::from_iter([true, false, true, true, true, true]).into_array(),
        );
        let arr = VarBinArray::try_new(offsets, bytes, DType::Utf8(Nullable), validity).unwrap();

        let arr = arr.as_view();
        let buf = filter_select_var_bin_by_slice(
            arr,
            &[(0, 3), (4, 6)],
            5,
            &mut LEGACY_SESSION.create_execution_ctx(),
        )
        .unwrap();

        assert_arrays_eq!(
            buf,
            VarBinArray::from(vec![
                Some("one"),
                None,
                Some("three"),
                Some("five"),
                Some("six")
            ])
        );
    }

    #[test]
    fn filter_varbin_nulls() {
        let bytes = [b"".as_slice(), b"two".as_slice(), b"two".as_slice()]
            .into_iter()
            .flat_map(|x| x.iter().cloned())
            .collect::<ByteBuffer>();

        let offsets = buffer![0, 0, 3, 6].into_array();
        let validity = Validity::Array(BoolArray::from_iter([false, true, true]).into_array());
        let arr = VarBinArray::try_new(offsets, bytes, DType::Utf8(Nullable), validity).unwrap();

        let arr = arr.as_view();
        let buf = filter_select_var_bin_by_slice(
            arr,
            &[(0, 1), (2, 3)],
            2,
            &mut LEGACY_SESSION.create_execution_ctx(),
        )
        .unwrap();

        assert_arrays_eq!(buf, VarBinArray::from(vec![None, Some("two")]));
    }

    #[test]
    fn filter_varbin_all_null() {
        let offsets = buffer![0, 0, 0, 0].into_array();
        let validity = Validity::Array(BoolArray::from_iter([false, false, false]).into_array());
        let arr = VarBinArray::try_new(
            offsets,
            ByteBuffer::empty(),
            DType::Utf8(Nullable),
            validity,
        )
        .unwrap();

        let arr = arr.as_view();
        let buf = filter_select_var_bin_by_slice(
            arr,
            &[(0, 1), (2, 3)],
            2,
            &mut LEGACY_SESSION.create_execution_ctx(),
        )
        .unwrap();

        assert_arrays_eq!(buf, VarBinArray::from(vec![None::<&str>, None]));
    }

    #[test]
    fn test_filter_var_bin_array() {
        let array = VarBinArray::from_vec(
            vec!["hello", "world", "filter", "good", "bye"],
            DType::Utf8(NonNullable),
        );
        test_filter_conformance(&array.into_array());

        let array = VarBinArray::from_iter(
            vec![Some("hello"), None, Some("filter"), Some("good"), None],
            DType::Utf8(Nullable),
        );
        test_filter_conformance(&array.into_array());
    }
}
