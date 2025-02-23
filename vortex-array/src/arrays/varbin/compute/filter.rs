use itertools::Itertools;
use num_traits::{AsPrimitive, PrimInt, Zero};
use vortex_dtype::{match_each_integer_ptype, DType, NativePType};
use vortex_error::{vortex_err, vortex_panic, VortexExpect, VortexResult};
use vortex_mask::{AllOr, Mask, MaskIter};

use crate::arrays::varbin::builder::VarBinBuilder;
use crate::arrays::varbin::VarBinArray;
use crate::arrays::VarBinEncoding;
use crate::compute::FilterFn;
use crate::validity::Validity;
use crate::variants::PrimitiveArrayTrait;
use crate::{Array, ArrayRef, IntoArray, ToCanonical};

impl FilterFn<&VarBinArray> for VarBinEncoding {
    fn filter(&self, array: &VarBinArray, mask: &Mask) -> VortexResult<ArrayRef> {
        filter_select_var_bin(array, mask).map(|a| a.into_array())
    }
}

fn filter_select_var_bin(arr: &VarBinArray, mask: &Mask) -> VortexResult<VarBinArray> {
    match mask
        .values()
        .vortex_expect("AllTrue and AllFalse are handled by filter fn")
        .threshold_iter(0.5)
    {
        MaskIter::Indices(indices) => {
            filter_select_var_bin_by_index(arr, indices, mask.true_count())
        }
        MaskIter::Slices(slices) => filter_select_var_bin_by_slice(arr, slices, mask.true_count()),
    }
}

fn filter_select_var_bin_by_slice(
    values: &VarBinArray,
    mask_slices: &[(usize, usize)],
    selection_count: usize,
) -> VortexResult<VarBinArray> {
    let offsets = values.offsets().to_primitive()?;
    match_each_integer_ptype!(offsets.ptype(), |$O| {
        filter_select_var_bin_by_slice_primitive_offset(
            values.dtype().clone(),
            offsets.as_slice::<$O>(),
            values.bytes().as_slice(),
            mask_slices,
            values.validity().clone(),
            selection_count
        )
    })
}

fn filter_select_var_bin_by_slice_primitive_offset<O>(
    dtype: DType,
    offsets: &[O],
    data: &[u8],
    mask_slices: &[(usize, usize)],
    validity: Validity,
    selection_count: usize,
) -> VortexResult<VarBinArray>
where
    O: NativePType + PrimInt + Zero,
    usize: AsPrimitive<O>,
{
    let logical_validity = validity.to_logical(offsets.len() - 1)?;
    let mut builder = VarBinBuilder::<O>::with_capacity(selection_count);
    match logical_validity.boolean_buffer() {
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
                let null_sl = validity.slice(start, end - start);
                if null_sl.count_set_bits() == null_sl.len() {
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
    O: NativePType + PrimInt + Zero + Copy,
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
    values: &VarBinArray,
    mask_indices: &[usize],
    selection_count: usize,
) -> VortexResult<VarBinArray> {
    let offsets = values.offsets().to_primitive()?;
    match_each_integer_ptype!(offsets.ptype(), |$O| {
        filter_select_var_bin_by_index_primitive_offset(
            values.dtype().clone(),
            offsets.as_slice::<$O>(),
            values.bytes().as_slice(),
            mask_indices,
            values.validity().clone(),
            selection_count
        )
    })
}

fn filter_select_var_bin_by_index_primitive_offset<O: NativePType + PrimInt>(
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
    use vortex_dtype::DType;
    use vortex_dtype::Nullability::{NonNullable, Nullable};
    use vortex_scalar::Scalar;

    use crate::array::Array;
    use crate::arrays::primitive::PrimitiveArray;
    use crate::arrays::varbin::compute::filter::{
        filter_select_var_bin_by_index, filter_select_var_bin_by_slice,
    };
    use crate::arrays::varbin::VarBinArray;
    use crate::arrays::BoolArray;
    use crate::compute::scalar_at;
    use crate::validity::Validity;
    use crate::IntoArray;

    fn nullable_scalar_str(s: &str) -> Scalar {
        Scalar::utf8(s.to_owned(), Nullable)
    }

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
        let buf = filter_select_var_bin_by_index(&arr, &[0, 2], 2).unwrap();

        assert_eq!(buf.len(), 2);
        assert_eq!(scalar_at(&buf, 0).unwrap(), "hello".into());
        assert_eq!(scalar_at(&buf, 1).unwrap(), "filter".into());
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

        let buf = filter_select_var_bin_by_slice(&arr, &[(0, 1), (2, 3), (4, 5)], 3).unwrap();

        assert_eq!(buf.len(), 3);
        assert_eq!(scalar_at(&buf, 0).unwrap(), "hello".into());
        assert_eq!(scalar_at(&buf, 1).unwrap(), "filter".into());
        assert_eq!(scalar_at(&buf, 2).unwrap(), "filter3".into());
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

        let offsets = PrimitiveArray::from_iter([0, 3, 6, 11, 15, 19, 22]).into_array();
        let validity = Validity::Array(
            BoolArray::from_iter([true, false, true, true, true, true]).into_array(),
        );
        let arr = VarBinArray::try_new(offsets, bytes, DType::Utf8(Nullable), validity).unwrap();

        let buf = filter_select_var_bin_by_slice(&arr, &[(0, 3), (4, 6)], 5).unwrap();

        let null = Scalar::null(DType::Utf8(Nullable));
        assert_eq!(buf.len(), 5);

        assert_eq!(scalar_at(&buf, 0).unwrap(), nullable_scalar_str("one"));
        assert_eq!(scalar_at(&buf, 1).unwrap(), null);
        assert_eq!(scalar_at(&buf, 2).unwrap(), nullable_scalar_str("three"));
        assert_eq!(scalar_at(&buf, 3).unwrap(), nullable_scalar_str("five"));
        assert_eq!(scalar_at(&buf, 4).unwrap(), nullable_scalar_str("six"));
    }

    #[test]
    fn filter_varbin_nulls() {
        let bytes = [b"".as_slice(), b"two".as_slice(), b"two".as_slice()]
            .into_iter()
            .flat_map(|x| x.iter().cloned())
            .collect::<ByteBuffer>();

        let offsets = PrimitiveArray::from_iter([0, 0, 3, 6]).into_array();
        let validity = Validity::Array(BoolArray::from_iter([false, true, true]).into_array());
        let arr = VarBinArray::try_new(offsets, bytes, DType::Utf8(Nullable), validity).unwrap();

        let buf = filter_select_var_bin_by_slice(&arr, &[(0, 1), (2, 3)], 2).unwrap();

        let null = Scalar::null(DType::Utf8(Nullable));
        assert_eq!(buf.len(), 2);

        assert_eq!(scalar_at(&buf, 0).unwrap(), null);
        assert_eq!(scalar_at(&buf, 1).unwrap(), nullable_scalar_str("two"));
    }

    #[test]
    fn filter_varbin_all_null() {
        let offsets = PrimitiveArray::from_iter([0, 0, 0, 0]).into_array();
        let validity = Validity::Array(BoolArray::from_iter([false, false, false]).into_array());
        let arr = VarBinArray::try_new(
            offsets,
            ByteBuffer::empty(),
            DType::Utf8(Nullable),
            validity,
        )
        .unwrap();

        let buf = filter_select_var_bin_by_slice(&arr, &[(0, 1), (2, 3)], 2).unwrap();

        let null = Scalar::null(DType::Utf8(Nullable));
        assert_eq!(buf.len(), 2);

        assert_eq!(scalar_at(&buf, 0).unwrap(), null);
        assert_eq!(scalar_at(&buf, 1).unwrap(), null);
    }
}
