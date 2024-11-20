use itertools::Itertools;
use num_traits::{AsPrimitive, Zero};
use vortex_dtype::{match_each_integer_ptype, DType, NativePType};
use vortex_error::{vortex_err, vortex_panic, VortexResult};

use crate::array::varbin::builder::VarBinBuilder;
use crate::array::varbin::VarBinArray;
use crate::array::VarBinEncoding;
use crate::compute::{FilterFn, FilterMask};
use crate::validity::Validity;
use crate::variants::PrimitiveArrayTrait;
use crate::{ArrayDType, ArrayData, IntoArrayData, IntoArrayVariant};

impl FilterFn<VarBinArray> for VarBinEncoding {
    fn filter(&self, array: &VarBinArray, mask: FilterMask) -> VortexResult<ArrayData> {
        filter_select_var_bin(array, mask).map(|a| a.into_array())
    }
}

fn filter_select_var_bin(arr: &VarBinArray, mask: FilterMask) -> VortexResult<VarBinArray> {
    let selection_count = mask.true_count();
    if selection_count * 2 > mask.len() {
        filter_select_var_bin_by_slice(arr, mask, selection_count)
    } else {
        filter_select_var_bin_by_index(arr, mask, selection_count)
    }
}

fn filter_select_var_bin_by_slice(
    values: &VarBinArray,
    mask: FilterMask,
    selection_count: usize,
) -> VortexResult<VarBinArray> {
    let offsets = values.offsets().into_primitive()?;
    match_each_integer_ptype!(offsets.ptype(), |$O| {
        filter_select_var_bin_by_slice_primitive_offset(
            values.dtype().clone(),
            offsets.maybe_null_slice::<$O>(),
            values.bytes().into_primitive()?.maybe_null_slice::<u8>(),
            mask,
            values.validity(),
            selection_count
        )
    })
}

#[allow(deprecated)]
fn filter_select_var_bin_by_slice_primitive_offset<O>(
    dtype: DType,
    offsets: &[O],
    data: &[u8],
    mask: FilterMask,
    validity: Validity,
    selection_count: usize,
) -> VortexResult<VarBinArray>
where
    O: NativePType + 'static + Zero,
    usize: AsPrimitive<O>,
{
    let logical_validity = validity.to_logical(offsets.len() - 1);
    if let Some(val) = logical_validity.to_null_buffer()? {
        let mut builder = VarBinBuilder::<O>::with_capacity(selection_count);

        for (start, end) in mask.iter_slices()? {
            let null_sl = val.slice(start, end - start);
            if null_sl.null_count() == 0 {
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
                        builder.push_value(&data[s..e])
                    } else {
                        builder.push_null()
                    }
                }
            }
        }

        return Ok(builder.finish(dtype));
    }

    let mut builder = VarBinBuilder::<O>::with_capacity(selection_count);

    mask.iter_slices()?.for_each(|(start, end)| {
        update_non_nullable_slice(data, offsets, &mut builder, start, end)
    });

    Ok(builder.finish(dtype))
}

fn update_non_nullable_slice<O>(
    data: &[u8],
    offsets: &[O],
    builder: &mut VarBinBuilder<O>,
    start: usize,
    end: usize,
) where
    O: NativePType + 'static + Zero + Copy,
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
    builder.push_values(new_data, new_offsets, end - start)
}

fn filter_select_var_bin_by_index(
    values: &VarBinArray,
    mask: FilterMask,
    selection_count: usize,
) -> VortexResult<VarBinArray> {
    let offsets = values.offsets().into_primitive()?;
    match_each_integer_ptype!(offsets.ptype(), |$O| {
        filter_select_var_bin_by_index_primitive_offset(
            values.dtype().clone(),
            offsets.maybe_null_slice::<$O>(),
            values.bytes().into_primitive()?.maybe_null_slice::<u8>(),
            mask,
            values.validity(),
            selection_count
        )
    })
}

#[allow(deprecated)]
fn filter_select_var_bin_by_index_primitive_offset<O: NativePType>(
    dtype: DType,
    offsets: &[O],
    data: &[u8],
    mask: FilterMask,
    validity: Validity,
    selection_count: usize,
) -> VortexResult<VarBinArray> {
    let mut builder = VarBinBuilder::<O>::with_capacity(selection_count);
    for idx in mask.iter_indices()? {
        if validity.is_valid(idx) {
            let (start, end) = (
                offsets[idx].to_usize().ok_or_else(|| {
                    vortex_err!("Failed to convert offset to usize: {}", offsets[idx])
                })?,
                offsets[idx + 1].to_usize().ok_or_else(|| {
                    vortex_err!("Failed to convert offset to usize: {}", offsets[idx + 1])
                })?,
            );
            builder.push(Some(&data[start..end]))
        } else {
            builder.push_null()
        }
    }
    Ok(builder.finish(dtype))
}

#[cfg(test)]
mod test {
    use itertools::Itertools;
    use vortex_dtype::DType;
    use vortex_dtype::Nullability::{NonNullable, Nullable};
    use vortex_scalar::Scalar;

    use crate::array::primitive::PrimitiveArray;
    use crate::array::varbin::compute::filter::{
        filter_select_var_bin_by_index, filter_select_var_bin_by_slice,
    };
    use crate::array::varbin::VarBinArray;
    use crate::array::BoolArray;
    use crate::compute::unary::scalar_at;
    use crate::compute::FilterMask;
    use crate::validity::Validity;
    use crate::ToArrayData;

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
        let filter = FilterMask::from_iter([true, false, true]);

        let buf = filter_select_var_bin_by_index(&arr, filter, 2)
            .unwrap()
            .to_array();

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
        let filter = FilterMask::from_iter([true, false, true, false, true]);

        let buf = filter_select_var_bin_by_slice(&arr, filter, 3)
            .unwrap()
            .to_array();

        assert_eq!(buf.len(), 3);
        assert_eq!(scalar_at(&buf, 0).unwrap(), "hello".into());
        assert_eq!(scalar_at(&buf, 1).unwrap(), "filter".into());
        assert_eq!(scalar_at(&buf, 2).unwrap(), "filter3".into());
    }

    #[test]
    fn filter_var_bin_slice_null_test() {
        let x = vec![
            b"one".as_slice(),
            b"two".as_slice(),
            b"three".as_slice(),
            b"four".as_slice(),
            b"five".as_slice(),
            b"six".as_slice(),
        ]
        .into_iter()
        .flat_map(|x| x.iter().cloned())
        .collect_vec();

        let bytes = PrimitiveArray::from(x).to_array();

        let offsets = PrimitiveArray::from(vec![0, 3, 6, 11, 15, 19, 22]).to_array();
        let validity =
            Validity::Array(BoolArray::from_iter([true, false, true, true, true, true]).to_array());
        let arr = VarBinArray::try_new(offsets, bytes, DType::Utf8(Nullable), validity).unwrap();
        let filter = FilterMask::from_iter([true, true, true, false, true, true]);

        let buf = filter_select_var_bin_by_slice(&arr, filter, 5)
            .unwrap()
            .to_array();

        let null = Scalar::null(DType::Utf8(Nullable));
        assert_eq!(buf.len(), 5);

        assert_eq!(scalar_at(&buf, 0).unwrap(), nullable_scalar_str("one"));
        assert_eq!(scalar_at(&buf, 1).unwrap(), null);
        assert_eq!(scalar_at(&buf, 2).unwrap(), nullable_scalar_str("three"));
        assert_eq!(scalar_at(&buf, 3).unwrap(), nullable_scalar_str("five"));
        assert_eq!(scalar_at(&buf, 4).unwrap(), nullable_scalar_str("six"));
    }
}
