// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use itertools::Itertools;
use num_traits::NumCast;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::ListViewArray;
use vortex_array::arrays::NullArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::varbinview::build_views::BinaryView;
use vortex_array::buffer::BufferHandle;
use vortex_array::builders::ArrayBuilder;
use vortex_array::builders::DecimalBuilder;
use vortex_array::builders::ListViewBuilder;
use vortex_array::builders::builder_with_capacity;
use vortex_array::dtype::DType;
use vortex_array::dtype::DecimalDType;
use vortex_array::dtype::DecimalType;
use vortex_array::dtype::IntegerPType;
use vortex_array::dtype::NativeDecimalType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::StructFields;
use vortex_array::match_each_decimal_value_type;
use vortex_array::match_each_integer_ptype;
use vortex_array::match_each_native_ptype;
use vortex_array::match_smallest_offset_type;
use vortex_array::patches::Patches;
use vortex_array::scalar::DecimalScalar;
use vortex_array::scalar::ListScalar;
use vortex_array::scalar::Scalar;
use vortex_array::scalar::StructScalar;
use vortex_array::validity::Validity;
use vortex_buffer::BitBuffer;
use vortex_buffer::Buffer;
use vortex_buffer::BufferString;
use vortex_buffer::ByteBuffer;
use vortex_buffer::buffer;
use vortex_buffer::buffer_mut;
use vortex_error::VortexError;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_panic;

use crate::ConstantArray;
use crate::SparseArray;
use crate::SparseData;
pub(super) fn execute_sparse(
    array: &SparseArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    if array.patches().num_patches() == 0 {
        return Ok(ConstantArray::new(array.fill_scalar().clone(), array.len()).into_array());
    }

    Ok(match array.dtype() {
        DType::Null => {
            assert!(array.fill_scalar().is_null());
            NullArray::new(array.len()).into_array()
        }
        DType::Bool(..) => {
            let resolved_patches = array.resolved_patches()?;
            execute_sparse_bools(&resolved_patches, array.fill_scalar(), ctx)?
        }
        DType::Primitive(ptype, ..) => {
            let resolved_patches = array.resolved_patches()?;
            match_each_native_ptype!(ptype, |P| {
                execute_sparse_primitives::<P>(&resolved_patches, array.fill_scalar(), ctx)?
            })
        }
        DType::Struct(struct_fields, ..) => execute_sparse_struct(
            struct_fields,
            array.fill_scalar().as_struct(),
            array.dtype(),
            array.patches(),
            array.len(),
            ctx,
        )?,
        DType::Decimal(decimal_dtype, nullability) => {
            let canonical_decimal_value_type =
                DecimalType::smallest_decimal_value_type(decimal_dtype);
            let fill_value = array.fill_scalar().as_decimal();
            match_each_decimal_value_type!(canonical_decimal_value_type, |D| {
                execute_sparse_decimal::<D>(
                    *decimal_dtype,
                    *nullability,
                    fill_value,
                    array.patches(),
                    array.len(),
                    ctx,
                )?
            })
        }
        dtype @ DType::Utf8(..) => {
            let fill_value = array.fill_scalar().as_utf8().value().cloned();
            let fill_value = fill_value.map(BufferString::into_inner);
            execute_varbin(array, dtype.clone(), fill_value, ctx)?
        }
        dtype @ DType::Binary(..) => {
            let fill_value = array.fill_scalar().as_binary().value().cloned();
            execute_varbin(array, dtype.clone(), fill_value, ctx)?
        }
        DType::List(values_dtype, nullability) => {
            execute_sparse_lists(array, values_dtype.clone(), *nullability, ctx)?
        }
        DType::FixedSizeList(.., nullability) => {
            execute_sparse_fixed_size_list(array, *nullability, ctx)?
        }
        DType::Extension(_ext_dtype) => todo!(),
        DType::Variant(_) => vortex_bail!("Sparse canonicalization does not support Variant"),
    })
}

#[expect(
    clippy::cognitive_complexity,
    reason = "complexity is from nested match_smallest_offset_type macro"
)]
fn execute_sparse_lists(
    array: &SparseArray,
    values_dtype: Arc<DType>,
    nullability: Nullability,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let resolved_patches = array.resolved_patches()?;

    let indices = resolved_patches
        .indices()
        .clone()
        .execute::<PrimitiveArray>(ctx)?;
    let values = resolved_patches
        .values()
        .clone()
        .execute::<ListViewArray>(ctx)?;
    let fill_value = array.fill_scalar().as_list();

    let n_filled = array.len() - resolved_patches.num_patches();
    let total_canonical_values = values.elements().len() + fill_value.len() * n_filled;

    let validity = Validity::from_mask(array.as_array().validity_mask()?, nullability);

    Ok(match_each_integer_ptype!(indices.ptype(), |I| {
        match_smallest_offset_type!(total_canonical_values, |O| {
            execute_sparse_lists_inner::<I, O>(
                indices.as_slice(),
                values,
                fill_value,
                values_dtype,
                array.len(),
                total_canonical_values,
                validity,
            )
        })
    }))
}

fn execute_sparse_lists_inner<I: IntegerPType, O: IntegerPType>(
    patch_indices: &[I],
    patch_values: ListViewArray,
    fill_value: ListScalar,
    values_dtype: Arc<DType>,
    len: usize,
    total_canonical_values: usize,
    validity: Validity,
) -> ArrayRef {
    // Create the builder with appropriate types. It is easy to just use the same type for both
    // `offsets` and `sizes` since we have no other constraints.
    let mut builder = ListViewBuilder::<O, O>::with_capacity(
        values_dtype,
        validity.nullability(),
        total_canonical_values,
        len,
    );

    let mut patch_idx = 0;

    // Loop over the patch indices and set them to the corresponding scalar values. For positions
    // that are not patched, use the fill value.
    for position in 0..len {
        let position_is_patched = patch_idx < patch_indices.len()
            && patch_indices[patch_idx]
                .to_usize()
                .vortex_expect("patch index must fit in usize")
                == position;

        if position_is_patched {
            // Set with the patch value.
            builder
                .append_value(
                    patch_values
                        .scalar_at(patch_idx)
                        .vortex_expect("scalar_at")
                        .as_list(),
                )
                .vortex_expect("Failed to append sparse value");
            patch_idx += 1;
        } else {
            // Set with the fill value.
            builder
                .append_value(fill_value.clone())
                .vortex_expect("Failed to append fill value");
        }
    }

    builder.finish()
}

/// Canonicalize a sparse [`FixedSizeListArray`] by expanding it into a dense representation.
fn execute_sparse_fixed_size_list(
    array: &SparseArray,
    nullability: Nullability,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let resolved_patches = array.resolved_patches()?;
    let indices = resolved_patches
        .indices()
        .clone()
        .execute::<PrimitiveArray>(ctx)?;
    let values = resolved_patches
        .values()
        .clone()
        .execute::<FixedSizeListArray>(ctx)?;
    let fill_value = array.fill_scalar().as_list();

    let validity = Validity::from_mask(array.as_array().validity_mask()?, nullability);

    Ok(match_each_integer_ptype!(indices.ptype(), |I| {
        execute_sparse_fixed_size_list_inner::<I>(
            indices.as_slice(),
            values,
            fill_value,
            array.len(),
            validity,
        )
        .into_array()
    }))
}

/// Build a canonical [`FixedSizeListArray`] from sparse patches by interleaving patch values with
/// fill values.
///
/// This algorithm walks through the sparse indices sequentially, filling gaps with the fill value's
/// elements (or defaults if null). Since all lists have the same size, we can directly append
/// elements without tracking offsets.
fn execute_sparse_fixed_size_list_inner<I: IntegerPType>(
    indices: &[I],
    values: FixedSizeListArray,
    fill_value: ListScalar,
    array_len: usize,
    validity: Validity,
) -> FixedSizeListArray {
    let list_size = values.list_size();
    let element_dtype = values.elements().dtype();
    let total_elements = array_len * list_size as usize;
    let mut builder = builder_with_capacity(element_dtype, total_elements);
    let fill_elements = fill_value.elements();

    let mut next_index = 0;
    let indices = indices
        .iter()
        .map(|x| (*x).to_usize().vortex_expect("index must fit in usize"));

    for (patch_idx, sparse_idx) in indices.enumerate() {
        // Fill gap before this patch with fill values.
        append_n_lists(
            &mut *builder,
            fill_elements.as_deref(),
            list_size,
            sparse_idx - next_index,
        );

        // Append the patch value, handling null patches by appending defaults.
        if values
            .validity()
            .is_valid(patch_idx)
            .vortex_expect("is_valid")
        {
            let patch_list = values
                .fixed_size_list_elements_at(patch_idx)
                .vortex_expect("fixed_size_list_elements_at");
            for i in 0..list_size as usize {
                builder
                    .append_scalar(&patch_list.scalar_at(i).vortex_expect("scalar_at"))
                    .vortex_expect("element dtype must match");
            }
        } else {
            builder.append_defaults(list_size as usize);
        }

        next_index = sparse_idx + 1;
    }

    // Fill remaining positions after last patch.
    append_n_lists(
        &mut *builder,
        fill_elements.as_deref(),
        list_size,
        array_len - next_index,
    );

    let elements = builder.finish();

    // SAFETY: elements.len() == array_len * list_size, validity length matches array_len.
    unsafe { FixedSizeListArray::new_unchecked(elements, list_size, validity, array_len) }
}

/// Append `count` copies of a fixed-size list to the builder.
///
/// If `fill_elements` is `Some`, appends those elements `count` times.
/// If `fill_elements` is `None` (null fill), appends `list_size` default elements `count` times.
fn append_n_lists(
    builder: &mut dyn ArrayBuilder,
    fill_elements: Option<&[Scalar]>,
    list_size: u32,
    count: usize,
) {
    for _ in 0..count {
        if let Some(fill_elems) = fill_elements {
            for elem in fill_elems {
                builder
                    .append_scalar(elem)
                    .vortex_expect("element dtype must match");
            }
        } else {
            builder.append_defaults(list_size as usize);
        }
    }
}

fn execute_sparse_bools(
    patches: &Patches,
    fill_value: &Scalar,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let (fill_bool, validity) = if fill_value.is_null() {
        (false, Validity::AllInvalid)
    } else {
        (
            fill_value
                .try_into()
                .vortex_expect("Fill value must convert to bool"),
            if patches.dtype().nullability() == Nullability::NonNullable {
                Validity::NonNullable
            } else {
                Validity::AllValid
            },
        )
    };

    let bools = BoolArray::new(BitBuffer::full(fill_bool, patches.array_len()), validity);

    Ok(bools.patch(patches, ctx)?.into_array())
}

fn execute_sparse_primitives<T: NativePType + for<'a> TryFrom<&'a Scalar, Error = VortexError>>(
    patches: &Patches,
    fill_value: &Scalar,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let (primitive_fill, validity) = if fill_value.is_null() {
        (T::default(), Validity::AllInvalid)
    } else {
        (
            fill_value
                .try_into()
                .vortex_expect("Fill value must convert to target T"),
            if patches.dtype().nullability() == Nullability::NonNullable {
                Validity::NonNullable
            } else {
                Validity::AllValid
            },
        )
    };

    let parray = PrimitiveArray::new(buffer![primitive_fill; patches.array_len()], validity);

    Ok(parray.patch(patches, ctx)?.into_array())
}

fn execute_sparse_struct(
    struct_fields: &StructFields,
    fill_struct: StructScalar,
    dtype: &DType,
    // Resolution is unnecessary b/c we're just pushing the patches into the fields.
    unresolved_patches: &Patches,
    len: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let (fill_values, top_level_fill_validity) = match fill_struct.fields_iter() {
        Some(fill_values) => (fill_values.collect::<Vec<_>>(), Validity::AllValid),
        None => (
            struct_fields
                .fields()
                .map(|f| Scalar::default_value(&f))
                .collect::<Vec<_>>(),
            Validity::AllInvalid,
        ),
    };
    let patch_values_as_struct = unresolved_patches
        .values()
        .clone()
        .execute::<StructArray>(ctx)?;
    let columns_patch_values = patch_values_as_struct.unmasked_fields();
    let names = patch_values_as_struct.names();
    let validity = if dtype.is_nullable() {
        top_level_fill_validity.patch(
            len,
            unresolved_patches.offset(),
            unresolved_patches.indices(),
            &Validity::from_mask(
                unresolved_patches
                    .values()
                    .validity_mask()
                    .vortex_expect("validity_mask"),
                Nullability::Nullable,
            ),
            ctx,
        )?
    } else {
        top_level_fill_validity
            .into_non_nullable(len)
            .unwrap_or_else(|| vortex_panic!("fill validity should match sparse array nullability"))
    };

    Ok(StructArray::try_from_iter_with_validity(
        names.iter().zip_eq(
            columns_patch_values
                .iter()
                .cloned()
                .zip_eq(fill_values)
                .map(|(patch_values, fill_value)| unsafe {
                    SparseData::new_unchecked(
                        unresolved_patches
                            .clone()
                            .map_values(|_| Ok(patch_values))
                            .vortex_expect("Replacing patch values"),
                        fill_value,
                    )
                }),
        ),
        validity,
    )?
    .into_array())
}

fn execute_sparse_decimal<D: NativeDecimalType>(
    decimal_dtype: DecimalDType,
    nullability: Nullability,
    fill_value: DecimalScalar,
    patches: &Patches,
    len: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let mut builder = DecimalBuilder::with_capacity::<D>(len, decimal_dtype, nullability);
    match fill_value.decimal_value() {
        Some(fill_value) => {
            let fill_value = fill_value
                .cast::<D>()
                .vortex_expect("unexpected value type");
            for _ in 0..len {
                builder.append_value(fill_value)
            }
        }
        None => {
            builder.append_nulls(len);
        }
    }
    let filled_array = builder.finish_into_decimal();
    let array = filled_array.into_data().patch(patches, ctx)?;
    Ok(array.into_array())
}

fn execute_varbin(
    array: &SparseArray,
    dtype: DType,
    fill_value: Option<ByteBuffer>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let patches = array.resolved_patches()?;
    let indices = patches.indices().clone().execute::<PrimitiveArray>(ctx)?;
    let values = patches.values().clone().execute::<VarBinViewArray>(ctx)?;
    let validity = Validity::from_mask(array.as_array().validity_mask()?, dtype.nullability());
    let len = array.len();

    Ok(match_each_integer_ptype!(indices.ptype(), |I| {
        let indices = indices.to_buffer::<I>();
        execute_varbin_inner::<I>(fill_value, indices, values, dtype, validity, len).into_array()
    }))
}

fn execute_varbin_inner<I: IntegerPType>(
    fill_value: Option<ByteBuffer>,
    indices: Buffer<I>,
    values: VarBinViewArray,
    dtype: DType,
    validity: Validity,
    len: usize,
) -> VarBinViewArray {
    assert_eq!(dtype.nullability(), validity.nullability());

    let n_patch_buffers = values.data_buffers().len();
    let mut buffers = values.data_buffers().to_vec();

    let fill = if let Some(buffer) = &fill_value {
        buffers.push(BufferHandle::new_host(buffer.clone()));
        BinaryView::make_view(
            buffer.as_ref(),
            u32::try_from(n_patch_buffers).vortex_expect("too many buffers"),
            0,
        )
    } else {
        // any <=12 character value will do
        BinaryView::make_view(&[], 0, 0)
    };

    let mut views = buffer_mut![fill; len];
    for (patch_index, &patch) in indices.into_iter().zip_eq(values.views().iter()) {
        let patch_index_usize = <usize as NumCast>::from(patch_index)
            .vortex_expect("var bin view indices must fit in usize");
        views[patch_index_usize] = patch;
    }

    let views = BufferHandle::new_host(views.freeze().into_byte_buffer());

    // SAFETY: views are constructed to maintain the invariants
    unsafe { VarBinViewArray::new_handle_unchecked(views, Arc::from(buffers), dtype, validity) }
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::ToCanonical;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::DecimalArray;
    use vortex_array::arrays::FixedSizeListArray;
    use vortex_array::arrays::ListArray;
    use vortex_array::arrays::ListViewArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::StructArray;
    use vortex_array::arrays::VarBinArray;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::arrow::IntoArrowArray as _;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::DecimalDType;
    use vortex_array::dtype::FieldNames;
    use vortex_array::dtype::Nullability::NonNullable;
    use vortex_array::dtype::Nullability::Nullable;
    use vortex_array::dtype::PType;
    use vortex_array::dtype::StructFields;
    use vortex_array::scalar::DecimalValue;
    use vortex_array::scalar::Scalar;
    use vortex_array::validity::Validity;
    use vortex_buffer::ByteBuffer;
    use vortex_buffer::buffer;
    use vortex_buffer::buffer_mut;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;
    use vortex_mask::Mask;

    use crate::Sparse;

    #[rstest]
    #[case(Some(true))]
    #[case(Some(false))]
    #[case(None)]
    fn test_sparse_bool(#[case] fill_value: Option<bool>) {
        let indices = buffer![0u64, 1, 7].into_array();
        let values = BoolArray::from_iter([Some(true), None, Some(false)]).into_array();
        let sparse_bools = Sparse::try_new(indices, values, 10, Scalar::from(fill_value)).unwrap();
        let actual = sparse_bools.as_array().to_bool();

        let expected = BoolArray::from_iter([
            Some(true),
            None,
            fill_value,
            fill_value,
            fill_value,
            fill_value,
            fill_value,
            Some(false),
            fill_value,
            fill_value,
        ]);

        assert_arrays_eq!(actual, expected);
    }

    #[rstest]
    #[case(Some(0i32))]
    #[case(Some(-1i32))]
    #[case(None)]
    fn test_sparse_primitive(#[case] fill_value: Option<i32>) {
        let indices = buffer![0u64, 1, 7].into_array();
        let values = PrimitiveArray::from_option_iter([Some(0i32), None, Some(1)]).into_array();
        let sparse_ints = Sparse::try_new(indices, values, 10, Scalar::from(fill_value)).unwrap();
        assert_eq!(*sparse_ints.dtype(), DType::Primitive(PType::I32, Nullable));

        let flat_ints = sparse_ints.as_array().to_primitive();
        let expected = PrimitiveArray::from_option_iter([
            Some(0i32),
            None,
            fill_value,
            fill_value,
            fill_value,
            fill_value,
            fill_value,
            Some(1),
            fill_value,
            fill_value,
        ]);

        assert_arrays_eq!(&flat_ints, &expected);
    }

    #[test]
    fn test_sparse_struct_valid_fill() {
        let field_names = FieldNames::from_iter(["a", "b"]);
        let field_types = vec![
            DType::Primitive(PType::I32, Nullable),
            DType::Primitive(PType::I32, Nullable),
        ];
        let struct_fields = StructFields::new(field_names, field_types);
        let struct_dtype = DType::Struct(struct_fields.clone(), Nullable);

        let indices = buffer![0u64, 1, 7, 8].into_array();
        let patch_values_a =
            PrimitiveArray::from_option_iter([Some(10i32), None, Some(20), Some(30)]).into_array();
        let patch_values_b =
            PrimitiveArray::from_option_iter([Some(1i32), Some(2), None, Some(3)]).into_array();
        let patch_values = StructArray::try_new_with_dtype(
            vec![patch_values_a, patch_values_b],
            struct_fields.clone(),
            4,
            Validity::Array(
                BoolArray::from_indices(4, vec![0, 1, 2], Validity::NonNullable).into_array(),
            ),
        )
        .unwrap()
        .into_array();

        let fill_scalar = Scalar::struct_(
            struct_dtype,
            vec![Scalar::from(Some(-10i32)), Scalar::from(Some(-1i32))],
        );
        let len = 10;
        let sparse_struct = Sparse::try_new(indices, patch_values, len, fill_scalar).unwrap();

        let expected_a = PrimitiveArray::from_option_iter((0..len).map(|i| {
            if i == 0 {
                Some(10)
            } else if i == 1 {
                None
            } else if i == 7 {
                Some(20)
            } else {
                Some(-10)
            }
        }));
        let expected_b = PrimitiveArray::from_option_iter((0..len).map(|i| {
            if i == 0 {
                Some(1i32)
            } else if i == 1 {
                Some(2)
            } else if i == 7 {
                None
            } else {
                Some(-1)
            }
        }));

        let expected = StructArray::try_new_with_dtype(
            vec![expected_a.into_array(), expected_b.into_array()],
            struct_fields,
            len,
            // NB: patch indices: [0, 1, 7, 8]; patch validity: [Valid, Valid, Valid, Invalid]; ergo 8 is Invalid.
            Validity::from_mask(Mask::from_excluded_indices(10, vec![8]), Nullable),
        )
        .unwrap()
        .into_array();

        let actual = sparse_struct.as_array().to_struct();
        assert_arrays_eq!(actual, expected);
    }

    #[test]
    fn test_sparse_struct_invalid_fill() {
        let field_names = FieldNames::from_iter(["a", "b"]);
        let field_types = vec![
            DType::Primitive(PType::I32, Nullable),
            DType::Primitive(PType::I32, Nullable),
        ];
        let struct_fields = StructFields::new(field_names, field_types);
        let struct_dtype = DType::Struct(struct_fields.clone(), Nullable);

        let indices = buffer![0u64, 1, 7, 8].into_array();
        let patch_values_a =
            PrimitiveArray::from_option_iter([Some(10i32), None, Some(20), Some(30)]).into_array();
        let patch_values_b =
            PrimitiveArray::from_option_iter([Some(1i32), Some(2), None, Some(3)]).into_array();
        let patch_values = StructArray::try_new_with_dtype(
            vec![patch_values_a, patch_values_b],
            struct_fields.clone(),
            4,
            Validity::Array(
                BoolArray::from_indices(4, vec![0, 1, 2], Validity::NonNullable).into_array(),
            ),
        )
        .unwrap()
        .into_array();

        let fill_scalar = Scalar::null(struct_dtype);
        let len = 10;
        let sparse_struct = Sparse::try_new(indices, patch_values, len, fill_scalar).unwrap();

        let expected_a = PrimitiveArray::from_option_iter((0..len).map(|i| {
            if i == 0 {
                Some(10)
            } else if i == 1 {
                None
            } else if i == 7 {
                Some(20)
            } else {
                Some(-10)
            }
        }));
        let expected_b = PrimitiveArray::from_option_iter((0..len).map(|i| {
            if i == 0 {
                Some(1i32)
            } else if i == 1 {
                Some(2)
            } else if i == 7 {
                None
            } else {
                Some(-1)
            }
        }));

        let expected = StructArray::try_new_with_dtype(
            vec![expected_a.into_array(), expected_b.into_array()],
            struct_fields,
            len,
            // NB: patch indices: [0, 1, 7, 8]; patch validity: [Valid, Valid, Valid, Invalid]; ergo 0, 1, 7 are valid.
            Validity::from_mask(Mask::from_indices(10, vec![0, 1, 7]), Nullable),
        )
        .unwrap()
        .into_array();

        let actual = sparse_struct.as_array().to_struct();
        assert_arrays_eq!(actual, expected);
    }

    #[test]
    fn test_sparse_decimal() {
        let indices = buffer![0u32, 1u32, 7u32, 8u32].into_array();
        let decimal_dtype = DecimalDType::new(3, 2);
        let patch_values = DecimalArray::new(
            buffer![100i128, 200i128, 300i128, 4000i128],
            decimal_dtype,
            Validity::from_iter([true, true, true, false]),
        )
        .into_array();
        let len = 10;
        let fill_scalar = Scalar::decimal(DecimalValue::I32(123), decimal_dtype, Nullable);
        let sparse_struct = Sparse::try_new(indices, patch_values, len, fill_scalar).unwrap();

        let expected = DecimalArray::new(
            buffer![100i128, 200, 123, 123, 123, 123, 123, 300, 4000, 123],
            decimal_dtype,
            // NB: patch indices: [0, 1, 7, 8]; patch validity: [Valid, Valid, Valid, Invalid]; ergo 0, 1, 7 are valid.
            Validity::from_mask(Mask::from_excluded_indices(10, vec![8]), Nullable),
        )
        .into_array()
        .into_arrow_preferred()
        .unwrap();

        let actual = sparse_struct
            .as_array()
            .to_decimal()
            .into_array()
            .into_arrow_preferred()
            .unwrap();

        assert_eq!(expected.data_type(), actual.data_type());
        assert_eq!(&expected, &actual);
    }

    #[test]
    fn test_sparse_utf8_varbinview_non_null_fill() {
        let strings = <VarBinViewArray as FromIterator<_>>::from_iter([
            Some("hello"),
            Some("goodbye"),
            Some("hello"),
            None,
            Some("bonjour"),
            Some("你好"),
            None,
        ])
        .into_array();

        let array = Sparse::try_new(
            buffer![0u16, 3, 4, 5, 7, 9, 10].into_array(),
            strings,
            12,
            Scalar::from(Some("123".to_owned())),
        )
        .unwrap();

        let actual = array.as_array().to_varbinview().into_array();
        let expected = <VarBinViewArray as FromIterator<_>>::from_iter([
            Some("hello"),
            Some("123"),
            Some("123"),
            Some("goodbye"),
            Some("hello"),
            None,
            Some("123"),
            Some("bonjour"),
            Some("123"),
            Some("你好"),
            None,
            Some("123"),
        ])
        .into_array();

        assert_arrays_eq!(actual, expected);
    }

    #[test]
    fn test_sparse_utf8_varbinview_null_fill() {
        let strings = <VarBinViewArray as FromIterator<_>>::from_iter([
            Some("hello"),
            Some("goodbye"),
            Some("hello"),
            None,
            Some("bonjour"),
            Some("你好"),
            None,
        ])
        .into_array();

        let array = Sparse::try_new(
            buffer![0u16, 3, 4, 5, 7, 9, 10].into_array(),
            strings,
            12,
            Scalar::null(DType::Utf8(Nullable)),
        )
        .unwrap();

        let actual = array.as_array().to_varbinview().into_array();
        let expected = <VarBinViewArray as FromIterator<_>>::from_iter([
            Some("hello"),
            None,
            None,
            Some("goodbye"),
            Some("hello"),
            None,
            None,
            Some("bonjour"),
            None,
            Some("你好"),
            None,
            None,
        ])
        .into_array();

        assert_arrays_eq!(actual, expected);
    }

    #[test]
    fn test_sparse_utf8_varbinview_non_nullable() {
        let strings =
            VarBinViewArray::from_iter_str(["hello", "goodbye", "hello", "bonjour", "你好"])
                .into_array();

        let array = Sparse::try_new(
            buffer![0u16, 3, 4, 5, 8].into_array(),
            strings,
            9,
            Scalar::from("123".to_owned()),
        )
        .unwrap();

        let actual = array.as_array().to_varbinview().into_array();
        let expected = VarBinViewArray::from_iter_str([
            "hello", "123", "123", "goodbye", "hello", "bonjour", "123", "123", "你好",
        ])
        .into_array();

        assert_arrays_eq!(actual, expected);
    }

    #[test]
    fn test_sparse_utf8_varbin_null_fill() {
        let strings = <VarBinArray as FromIterator<_>>::from_iter([
            Some("hello"),
            Some("goodbye"),
            Some("hello"),
            None,
            Some("bonjour"),
            Some("你好"),
            None,
        ])
        .into_array();

        let array = Sparse::try_new(
            buffer![0u16, 3, 4, 5, 7, 9, 10].into_array(),
            strings,
            12,
            Scalar::null(DType::Utf8(Nullable)),
        )
        .unwrap();

        let actual = array.as_array().to_varbinview().into_array();
        let expected = <VarBinViewArray as FromIterator<_>>::from_iter([
            Some("hello"),
            None,
            None,
            Some("goodbye"),
            Some("hello"),
            None,
            None,
            Some("bonjour"),
            None,
            Some("你好"),
            None,
            None,
        ])
        .into_array();

        assert_arrays_eq!(actual, expected);
    }

    #[test]
    fn test_sparse_binary_varbinview_non_null_fill() {
        let binaries = VarBinViewArray::from_iter_nullable_bin([
            Some(b"hello" as &[u8]),
            Some(b"goodbye"),
            Some(b"hello"),
            None,
            Some(b"\x00"),
            Some(b"\xE4\xBD\xA0\xE5\xA5\xBD"),
            None,
        ])
        .into_array();

        let array = Sparse::try_new(
            buffer![0u16, 3, 4, 5, 7, 9, 10].into_array(),
            binaries,
            12,
            Scalar::from(Some(ByteBuffer::from(b"123".to_vec()))),
        )
        .unwrap();

        let actual = array.as_array().to_varbinview().into_array();
        let expected = VarBinViewArray::from_iter_nullable_bin([
            Some(b"hello" as &[u8]),
            Some(b"123"),
            Some(b"123"),
            Some(b"goodbye"),
            Some(b"hello"),
            None,
            Some(b"123"),
            Some(b"\x00"),
            Some(b"123"),
            Some(b"\xE4\xBD\xA0\xE5\xA5\xBD"),
            None,
            Some(b"123"),
        ])
        .into_array();

        assert_arrays_eq!(actual, expected);
    }

    #[test]
    fn test_sparse_list_null_fill() -> VortexResult<()> {
        // Use ListViewArray consistently
        let elements = buffer![1i32, 2, 1, 2].into_array();
        // Create ListView with offsets and sizes
        // List 0: [1] at offset 0, size 1
        // List 1: [2] at offset 1, size 1
        // List 2: [1] at offset 2, size 1
        // List 3: [2] at offset 3, size 1
        let offsets = buffer![0u32, 1, 2, 3].into_array();
        let sizes = buffer![1u32, 1, 1, 1].into_array();
        let lists = unsafe {
            ListViewArray::new_unchecked(elements, offsets, sizes, Validity::AllValid)
                .with_zero_copy_to_list(true)
        }
        .into_array();

        let indices = buffer![0u8, 3u8, 4u8, 5u8].into_array();
        let fill_value = Scalar::null(lists.dtype().clone());
        let sparse = Sparse::try_new(indices, lists, 6, fill_value)
            .unwrap()
            .into_array();

        let actual = sparse.to_canonical()?.into_array();
        let result_listview = actual.to_listview();

        // Check the structure
        assert_eq!(result_listview.len(), 6);

        // Verify sizes: positions 0,3,4,5 have data, positions 1,2 are null
        assert_eq!(result_listview.size_at(0), 1); // [1]
        assert_eq!(result_listview.size_at(1), 0); // null
        assert_eq!(result_listview.size_at(2), 0); // null
        assert_eq!(result_listview.size_at(3), 1); // [2]
        assert_eq!(result_listview.size_at(4), 1); // [1]
        assert_eq!(result_listview.size_at(5), 1); // [2]

        // Verify actual values
        let elements_array = result_listview.elements().to_primitive();
        let elements_slice = elements_array.as_slice::<i32>();

        let list0_offset = result_listview.offset_at(0);
        assert_eq!(elements_slice[list0_offset], 1);

        let list3_offset = result_listview.offset_at(3);
        assert_eq!(elements_slice[list3_offset], 2);

        let list4_offset = result_listview.offset_at(4);
        assert_eq!(elements_slice[list4_offset], 1);

        let list5_offset = result_listview.offset_at(5);
        assert_eq!(elements_slice[list5_offset], 2);

        Ok(())
    }

    #[test]
    fn test_sparse_list_null_fill_sliced_sparse_values() {
        // Create ListViewArray with 8 elements forming 8 single-element lists
        let elements = buffer![1i32, 2, 1, 2, 1, 2, 1, 2].into_array();
        let offsets = buffer![0u32, 1, 2, 3, 4, 5, 6, 7].into_array();
        let sizes = buffer![1u32, 1, 1, 1, 1, 1, 1, 1].into_array();
        let lists = unsafe {
            ListViewArray::new_unchecked(elements, offsets, sizes, Validity::AllValid)
                .with_zero_copy_to_list(true)
        }
        .into_array();

        // Slice to get lists 2..6, which are: [1], [2], [1], [2]
        let lists = lists.slice(2..6).unwrap();

        let indices = buffer![0u8, 3u8, 4u8, 5u8].into_array();
        let fill_value = Scalar::null(lists.dtype().clone());
        let sparse = Sparse::try_new(indices, lists, 6, fill_value)
            .unwrap()
            .into_array();

        let actual = sparse.to_canonical().vortex_expect("no fail").into_array();
        let result_listview = actual.to_listview();

        // Check the structure
        assert_eq!(result_listview.len(), 6);

        // Verify sizes: positions 0,3,4,5 have data (from the sliced lists), positions 1,2 are null
        assert_eq!(result_listview.size_at(0), 1); // [1] - from slice index 0 (original index 2)
        assert_eq!(result_listview.size_at(1), 0); // null
        assert_eq!(result_listview.size_at(2), 0); // null
        assert_eq!(result_listview.size_at(3), 1); // [2] - from slice index 3 (original index 5)
        assert_eq!(result_listview.size_at(4), 1); // [1] - extra element beyond original slice
        assert_eq!(result_listview.size_at(5), 1); // [2] - extra element beyond original slice

        // Verify actual values
        let elements_array = result_listview.elements().to_primitive();
        let elements_slice = elements_array.as_slice::<i32>();

        let list0_offset = result_listview.offset_at(0);
        assert_eq!(elements_slice[list0_offset], 1);

        let list3_offset = result_listview.offset_at(3);
        assert_eq!(elements_slice[list3_offset], 2);
    }

    #[test]
    fn test_sparse_list_non_null_fill() -> VortexResult<()> {
        // Create ListViewArray with 4 single-element lists
        let elements = buffer![1i32, 2, 1, 2].into_array();
        let offsets = buffer![0u32, 1, 2, 3].into_array();
        let sizes = buffer![1u32, 1, 1, 1].into_array();
        let lists = unsafe {
            ListViewArray::new_unchecked(elements, offsets, sizes, Validity::AllValid)
                .with_zero_copy_to_list(true)
        }
        .into_array();

        let indices = buffer![0u8, 3u8, 4u8, 5u8].into_array();
        let fill_value = Scalar::from(Some(vec![5i32, 6, 7, 8]));
        let sparse = Sparse::try_new(indices, lists, 6, fill_value)
            .unwrap()
            .into_array();

        let actual = sparse.to_canonical()?.into_array();
        let result_listview = actual.to_listview();

        // Check the structure
        assert_eq!(result_listview.len(), 6);

        // Verify sizes: positions 0,3,4,5 have sparse data, positions 1,2 have fill values
        assert_eq!(result_listview.size_at(0), 1); // [1] from sparse
        assert_eq!(result_listview.size_at(1), 4); // [5,6,7,8] fill value
        assert_eq!(result_listview.size_at(2), 4); // [5,6,7,8] fill value
        assert_eq!(result_listview.size_at(3), 1); // [2] from sparse
        assert_eq!(result_listview.size_at(4), 1); // [1] from sparse
        assert_eq!(result_listview.size_at(5), 1); // [2] from sparse

        // Verify actual values
        let elements_array = result_listview.elements().to_primitive();
        let elements_slice = elements_array.as_slice::<i32>();

        // List 0: [1]
        let list0_offset = result_listview.offset_at(0) as usize;
        assert_eq!(elements_slice[list0_offset], 1);

        // List 1: [5,6,7,8]
        let list1_offset = result_listview.offset_at(1) as usize;
        let list1_size = result_listview.size_at(1) as usize;
        assert_eq!(
            &elements_slice[list1_offset..list1_offset + list1_size],
            &[5, 6, 7, 8]
        );

        // List 2: [5,6,7,8]
        let list2_offset = result_listview.offset_at(2) as usize;
        let list2_size = result_listview.size_at(2) as usize;
        assert_eq!(
            &elements_slice[list2_offset..list2_offset + list2_size],
            &[5, 6, 7, 8]
        );

        // List 3: [2]
        let list3_offset = result_listview.offset_at(3) as usize;
        assert_eq!(elements_slice[list3_offset], 2);

        // List 4: [1]
        let list4_offset = result_listview.offset_at(4) as usize;
        assert_eq!(elements_slice[list4_offset], 1);

        // List 5: [2]
        let list5_offset = result_listview.offset_at(5) as usize;
        assert_eq!(elements_slice[list5_offset], 2);
        Ok(())
    }

    #[test]
    fn test_sparse_binary_varbin_null_fill() {
        let strings = <VarBinArray as FromIterator<_>>::from_iter([
            Some(b"hello" as &[u8]),
            Some(b"goodbye"),
            Some(b"hello"),
            None,
            Some(b"\x00"),
            Some(b"\xE4\xBD\xA0\xE5\xA5\xBD"),
            None,
        ])
        .into_array();

        let array = Sparse::try_new(
            buffer![0u16, 3, 4, 5, 7, 9, 10].into_array(),
            strings,
            12,
            Scalar::null(DType::Binary(Nullable)),
        )
        .unwrap();

        let actual = array.as_array().to_varbinview().into_array();
        let expected = VarBinViewArray::from_iter_nullable_bin([
            Some(b"hello" as &[u8]),
            None,
            None,
            Some(b"goodbye"),
            Some(b"hello"),
            None,
            None,
            Some(b"\x00"),
            None,
            Some(b"\xE4\xBD\xA0\xE5\xA5\xBD"),
            None,
            None,
        ])
        .into_array();

        assert_arrays_eq!(actual, expected);
    }

    #[test]
    fn test_sparse_fixed_size_list_null_fill() -> VortexResult<()> {
        // Create a FixedSizeListArray with 3 lists of size 3.
        let elements = buffer![1i32, 2, 3, 4, 5, 6, 7, 8, 9].into_array();
        let fsl = FixedSizeListArray::try_new(elements, 3, Validity::AllValid, 3)
            .unwrap()
            .into_array();

        let indices = buffer![0u8, 2u8, 3u8].into_array();
        let fill_value = Scalar::null(DType::FixedSizeList(
            Arc::new(DType::Primitive(PType::I32, NonNullable)),
            3,
            Nullable,
        ));
        let sparse = Sparse::try_new(indices, fsl, 5, fill_value)
            .unwrap()
            .into_array();

        let actual = sparse.to_canonical()?.into_array();

        // Expected: [1,2,3], null, [4,5,6], [7,8,9], null.
        let expected_elements =
            buffer![1i32, 2, 3, 0, 0, 0, 4, 5, 6, 7, 8, 9, 0, 0, 0].into_array();
        let expected = FixedSizeListArray::try_new(
            expected_elements,
            3,
            Validity::Array(BoolArray::from_iter([true, false, true, true, false]).into_array()),
            5,
        )
        .unwrap()
        .into_array();

        assert_arrays_eq!(actual, expected);
        Ok(())
    }

    #[test]
    fn test_sparse_fixed_size_list_non_null_fill() -> VortexResult<()> {
        let elements = buffer![1i32, 2, 3, 4, 5, 6].into_array();
        let fsl = FixedSizeListArray::try_new(elements, 2, Validity::AllValid, 3)
            .unwrap()
            .into_array();

        let indices = buffer![0u8, 2u8, 4u8].into_array();
        let fill_value = Scalar::fixed_size_list(
            Arc::new(DType::Primitive(PType::I32, NonNullable)),
            vec![
                Scalar::primitive(99i32, NonNullable),
                Scalar::primitive(88i32, NonNullable),
            ],
            NonNullable,
        );
        let sparse = Sparse::try_new(indices, fsl, 6, fill_value)
            .unwrap()
            .into_array();

        let actual = sparse.to_canonical()?.into_array();

        // Expected: [1,2], [99,88], [3,4], [99,88], [5,6], [99,88].
        let expected_elements = buffer![1i32, 2, 99, 88, 3, 4, 99, 88, 5, 6, 99, 88].into_array();
        let expected = FixedSizeListArray::try_new(expected_elements, 2, Validity::NonNullable, 6)
            .unwrap()
            .into_array();

        assert_arrays_eq!(actual, expected);
        Ok(())
    }

    #[test]
    fn test_sparse_fixed_size_list_with_validity() -> VortexResult<()> {
        // Create FSL values with some nulls.
        let elements = buffer![10i32, 20, 30, 40, 50, 60].into_array();
        let fsl = FixedSizeListArray::try_new(
            elements,
            2,
            Validity::Array(BoolArray::from_iter([true, false, true]).into_array()),
            3,
        )
        .unwrap()
        .into_array();

        let indices = buffer![1u16, 3u16, 4u16].into_array();
        let fill_value = Scalar::fixed_size_list(
            Arc::new(DType::Primitive(PType::I32, NonNullable)),
            vec![
                Scalar::primitive(7i32, NonNullable),
                Scalar::primitive(8i32, NonNullable),
            ],
            Nullable,
        );
        let sparse = Sparse::try_new(indices, fsl, 6, fill_value)
            .unwrap()
            .into_array();

        let actual = sparse.to_canonical()?.into_array();

        // Expected validity: [true, true, true, false, true, true].
        // Expected elements: [7,8], [10,20], [7,8], [30,40], [50,60], [7,8].
        let expected_elements = buffer![7i32, 8, 10, 20, 7, 8, 30, 40, 50, 60, 7, 8].into_array();
        let expected = FixedSizeListArray::try_new(
            expected_elements,
            2,
            Validity::Array(
                BoolArray::from_iter([true, true, true, false, true, true]).into_array(),
            ),
            6,
        )
        .unwrap()
        .into_array();

        assert_arrays_eq!(actual, expected);
        Ok(())
    }

    #[test]
    fn test_sparse_fixed_size_list_truly_sparse() -> VortexResult<()> {
        // Test with a truly sparse array where most values are the fill value.
        // This demonstrates the compression benefit of sparse encoding.

        // Create patch values: only 3 distinct lists out of 100 total positions.
        let elements = buffer![10i32, 11, 20, 21, 30, 31].into_array();
        let fsl = FixedSizeListArray::try_new(elements, 2, Validity::AllValid, 3)
            .unwrap()
            .into_array();

        // Patches at positions 5, 50, and 95 out of 100.
        let indices = buffer![5u32, 50, 95].into_array();

        // Fill value [99, 99] will appear 97 times but stored only once.
        let fill_value = Scalar::fixed_size_list(
            Arc::new(DType::Primitive(PType::I32, NonNullable)),
            vec![
                Scalar::primitive(99i32, NonNullable),
                Scalar::primitive(99i32, NonNullable),
            ],
            NonNullable,
        );

        let sparse = Sparse::try_new(indices, fsl, 100, fill_value)
            .unwrap()
            .into_array();

        let actual = sparse.to_canonical()?.into_array();

        // Build expected: 97 copies of [99,99] with patches at positions 5, 50, 95.
        let mut expected_elements_vec = Vec::with_capacity(200);
        // Positions 0-4: fill values
        for _ in 0..5 {
            expected_elements_vec.extend([99i32, 99]);
        }
        // Position 5: first patch [10, 11]
        expected_elements_vec.extend([10, 11]);
        // Positions 6-49: fill values
        for _ in 6..50 {
            expected_elements_vec.extend([99, 99]);
        }
        // Position 50: second patch [20, 21]
        expected_elements_vec.extend([20, 21]);
        // Positions 51-94: fill values
        for _ in 51..95 {
            expected_elements_vec.extend([99, 99]);
        }
        // Position 95: third patch [30, 31]
        expected_elements_vec.extend([30, 31]);
        // Positions 96-99: fill values
        for _ in 96..100 {
            expected_elements_vec.extend([99, 99]);
        }
        let expected_elements = PrimitiveArray::from_iter(expected_elements_vec).into_array();
        let expected =
            FixedSizeListArray::try_new(expected_elements, 2, Validity::NonNullable, 100)
                .unwrap()
                .into_array();

        assert_arrays_eq!(actual, expected);
        Ok(())
    }

    #[test]
    fn test_sparse_fixed_size_list_single_element() -> VortexResult<()> {
        // Test with a single element FSL array.
        let elements = buffer![42i32, 43].into_array();
        let fsl = FixedSizeListArray::try_new(elements, 2, Validity::AllValid, 1)
            .unwrap()
            .into_array();

        let indices = buffer![0u32].into_array();
        let fill_value = Scalar::fixed_size_list(
            Arc::new(DType::Primitive(PType::I32, NonNullable)),
            vec![
                Scalar::primitive(1i32, NonNullable),
                Scalar::primitive(2i32, NonNullable),
            ],
            NonNullable,
        );
        let sparse = Sparse::try_new(indices, fsl, 1, fill_value)
            .unwrap()
            .into_array();

        let actual = sparse.to_canonical()?.into_array();

        // Expected: just [42, 43].
        let expected_elements = buffer![42i32, 43].into_array();
        let expected = FixedSizeListArray::try_new(expected_elements, 2, Validity::NonNullable, 1)
            .unwrap()
            .into_array();

        assert_arrays_eq!(actual, expected);
        Ok(())
    }

    #[test]
    fn test_sparse_list_grows_offset_type() -> VortexResult<()> {
        let elements = buffer![1i32, 2, 1, 2].into_array();
        let offsets = buffer![0u8, 1, 2, 3, 4].into_array();
        let lists = ListArray::try_new(elements, offsets, Validity::AllValid)
            .unwrap()
            .into_array();

        let indices = buffer![0u8, 1u8, 2u8, 3u8].into_array();
        let fill_value = Scalar::from(Some(vec![42i32; 252])); // 252 + 4 elements = 256 > u8::MAX
        let sparse = Sparse::try_new(indices, lists, 5, fill_value)
            .unwrap()
            .into_array();

        let actual = sparse.to_canonical()?.into_array();
        let mut expected_elements = buffer_mut![1, 2, 1, 2];
        expected_elements.extend(buffer![42i32; 252]);
        let expected = ListArray::try_new(
            expected_elements.freeze().into_array(),
            buffer![0u16, 1, 2, 3, 4, 256].into_array(),
            Validity::AllValid,
        )
        .unwrap()
        .into_array();

        assert_eq!(
            actual.to_listview().offsets().dtype(),
            &DType::Primitive(PType::U16, NonNullable)
        );
        assert_arrays_eq!(&actual, &expected);

        // Note that the preferred arrow list representation is `List` (not `ListView`).
        let arrow_dtype = expected.dtype().to_arrow_dtype().unwrap();
        let actual = actual.into_arrow(&arrow_dtype).unwrap();
        let expected = expected.into_arrow(&arrow_dtype).unwrap();

        assert_eq!(actual.data_type(), expected.data_type());
        Ok(())
    }

    #[test]
    fn test_sparse_listview_null_fill_with_gaps() -> VortexResult<()> {
        // This test specifically catches the bug where the old implementation
        // incorrectly tracked `last_valid_offset` as the START of the last list
        // instead of properly handling ListView's offset/size pairs.

        // Create a ListViewArray with non-trivial offsets and sizes.
        // Elements: [10, 11, 12, 20, 21, 30, 31, 32, 33]
        // List 0: elements[0..3] = [10, 11, 12]
        // List 1: elements[3..5] = [20, 21]
        // List 2: elements[5..9] = [30, 31, 32, 33]
        let elements = buffer![10i32, 11, 12, 20, 21, 30, 31, 32, 33].into_array();
        let offsets = buffer![0u32, 3, 5].into_array();
        let sizes = buffer![3u32, 2, 4].into_array();

        let list_view = unsafe {
            ListViewArray::new_unchecked(elements, offsets, sizes, Validity::AllValid)
                .with_zero_copy_to_list(true)
        };

        let list_dtype = list_view.dtype().clone();

        // Create sparse array with indices [1, 4, 7] and length 10
        // This means we have:
        // - Index 0: null
        // - Index 1: List 0 [10, 11, 12]
        // - Index 2-3: null
        // - Index 4: List 1 [20, 21]
        // - Index 5-6: null
        // - Index 7: List 2 [30, 31, 32, 33]
        // - Index 8-9: null
        let indices = buffer![1u8, 4, 7].into_array();
        let sparse = Sparse::try_new(
            indices,
            list_view.into_array(),
            10,
            Scalar::null(list_dtype),
        )
        .unwrap();

        // Convert to canonical form - this triggers the function we're testing
        let canonical = sparse.to_canonical()?.into_array();
        let result_listview = canonical.to_listview();

        // Verify the structure
        assert_eq!(result_listview.len(), 10);

        // Helper to get list values at an index
        let get_list_values = |idx: usize| -> Vec<i32> {
            let offset = result_listview.offset_at(idx);
            let size = result_listview.size_at(idx);
            if size == 0 {
                vec![] // null/empty list
            } else {
                let elements = result_listview.elements().to_primitive();
                let slice = elements.as_slice::<i32>();
                slice[offset..offset + size].to_vec()
            }
        };

        let empty: Vec<i32> = vec![];

        // Verify all list values
        assert_eq!(get_list_values(0), empty); // null
        assert_eq!(get_list_values(1), vec![10, 11, 12]); // sparse index 0
        assert_eq!(get_list_values(2), empty); // null
        assert_eq!(get_list_values(3), empty); // null
        assert_eq!(get_list_values(4), vec![20, 21]); // sparse index 1
        assert_eq!(get_list_values(5), empty); // null
        assert_eq!(get_list_values(6), empty); // null
        assert_eq!(get_list_values(7), vec![30, 31, 32, 33]); // sparse index 2
        assert_eq!(get_list_values(8), empty); // null
        assert_eq!(get_list_values(9), empty); // null
        Ok(())
    }

    #[test]
    fn test_sparse_listview_sliced_values_null_fill() -> VortexResult<()> {
        // This test uses sliced ListView values to ensure proper handling
        // of non-zero starting offsets in the source data.

        // Create a larger ListViewArray and then slice it
        // Original elements: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        let elements = buffer![0i32, 1, 2, 3, 4, 5, 6, 7, 8, 9].into_array();

        // Create 5 lists with different offsets and sizes
        // List 0: [0, 1] at offset 0
        // List 1: [2, 3, 4] at offset 2
        // List 2: [5] at offset 5
        // List 3: [6, 7] at offset 6
        // List 4: [8, 9] at offset 8
        let offsets = buffer![0u32, 2, 5, 6, 8].into_array();
        let sizes = buffer![2u32, 3, 1, 2, 2].into_array();

        let full_listview = unsafe {
            ListViewArray::new_unchecked(elements, offsets, sizes, Validity::AllValid)
                .with_zero_copy_to_list(true)
        }
        .into_array();

        // Slice to get lists 1, 2, 3 (indices 1..4)
        // This gives us lists with elements:
        // - Index 0: [2, 3, 4] (original list 1)
        // - Index 1: [5] (original list 2)
        // - Index 2: [6, 7] (original list 3)
        let sliced = full_listview.slice(1..4).unwrap();

        // Create sparse array with indices [0, 1] and length 5
        // Expected result:
        // - Index 0: [2, 3, 4] (from sliced[0])
        // - Index 1: [5] (from sliced[1])
        // - Index 2: null
        // - Index 3: null
        // - Index 4: null
        let indices = buffer![0u8, 1].into_array();
        // Extract only the values we need from the sliced array
        let values = sliced.slice(0..2).unwrap();
        let sparse =
            Sparse::try_new(indices, values, 5, Scalar::null(sliced.dtype().clone())).unwrap();

        let canonical = sparse.to_canonical()?.into_array();
        let result_listview = canonical.to_listview();

        assert_eq!(result_listview.len(), 5);

        // Helper to get list values at an index
        let get_list_values = |idx: usize| -> Vec<i32> {
            let offset = result_listview.offset_at(idx);
            let size = result_listview.size_at(idx);
            if size == 0 {
                vec![] // null/empty list
            } else {
                let elements = result_listview.elements().to_primitive();
                let slice = elements.as_slice::<i32>();
                slice[offset..offset + size].to_vec()
            }
        };

        let empty: Vec<i32> = vec![];

        // Verify all list values
        // Original slice had lists at indices 1,2,3 which were: [2,3,4], [5], [6,7]
        // We take indices 0 and 1 from the slice
        assert_eq!(get_list_values(0), vec![2, 3, 4]); // From slice index 0 (original list 1)
        assert_eq!(get_list_values(1), vec![5]); // From slice index 1 (original list 2)
        assert_eq!(get_list_values(2), empty); // null
        assert_eq!(get_list_values(3), empty); // null
        assert_eq!(get_list_values(4), empty); // null

        // The bug in the old implementation would have incorrectly used
        // offsets[sparse_index] for filling nulls, which would be wrong
        // when dealing with sliced arrays that have non-zero starting offsets.
        Ok(())
    }
}
