// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use itertools::Itertools;
use num_traits::NumCast;
use vortex_array::arrays::{
    BinaryView, BoolArray, BooleanBuffer, ConstantArray, ListArray, NullArray, OffsetPType,
    PrimitiveArray, StructArray, VarBinViewArray, smallest_storage_type,
};
use vortex_array::builders::{
    ArrayBuilder as _, ArrayBuilderExt, DecimalBuilder, ListBuilder, builder_with_capacity,
};
use vortex_array::patches::Patches;
use vortex_array::validity::Validity;
use vortex_array::vtable::CanonicalVTable;
use vortex_array::{Array, ArrayRef, Canonical, IntoArray as _, ToCanonical as _};
use vortex_buffer::{Buffer, BufferMut, BufferString, ByteBuffer, buffer, buffer_mut};
use vortex_dtype::{
    DType, DecimalDType, NativePType, Nullability, StructFields, match_each_integer_ptype,
    match_each_native_ptype,
};
use vortex_error::{VortexError, VortexExpect as _, VortexResult, vortex_err};
use vortex_scalar::{
    DecimalScalar, ListScalar, NativeDecimalType, Scalar, StructScalar,
    match_each_decimal_value_type,
};

use crate::{SparseArray, SparseVTable};

impl CanonicalVTable<SparseVTable> for SparseVTable {
    fn canonicalize(array: &SparseArray) -> VortexResult<Canonical> {
        if array.patches().num_patches() == 0 {
            return ConstantArray::new(array.fill_scalar().clone(), array.len()).to_canonical();
        }

        match array.dtype() {
            DType::Null => {
                assert!(array.fill_scalar().is_null());
                Ok(Canonical::Null(NullArray::new(array.len())))
            }
            DType::Bool(..) => {
                let resolved_patches = array.resolved_patches()?;
                canonicalize_sparse_bools(&resolved_patches, array.fill_scalar())
            }
            DType::Primitive(ptype, ..) => {
                let resolved_patches = array.resolved_patches()?;
                match_each_native_ptype!(ptype, |P| {
                    canonicalize_sparse_primitives::<P>(&resolved_patches, array.fill_scalar())
                })
            }
            DType::Struct(struct_fields, ..) => canonicalize_sparse_struct(
                struct_fields,
                array.fill_scalar().as_struct(),
                array.dtype(),
                array.patches(),
                array.len(),
            ),
            DType::Decimal(decimal_dtype, nullability) => {
                let canonical_decimal_value_type = smallest_storage_type(decimal_dtype);
                let fill_value = array.fill_scalar().as_decimal();
                match_each_decimal_value_type!(canonical_decimal_value_type, |D| {
                    canonicalize_sparse_decimal::<D>(
                        *decimal_dtype,
                        *nullability,
                        fill_value,
                        array.patches(),
                        array.len(),
                    )
                })
            }
            dtype @ DType::Utf8(..) => {
                let fill_value = array.fill_scalar().as_utf8().value();
                let fill_value = fill_value.map(BufferString::into_inner);
                canonicalize_varbin(array, dtype.clone(), fill_value)
            }
            dtype @ DType::Binary(..) => {
                let fill_value = array.fill_scalar().as_binary().value();
                canonicalize_varbin(array, dtype.clone(), fill_value)
            }
            DType::List(values_dtype, nullability) => {
                let resolved_patches = array.resolved_patches()?;
                canonicalize_sparse_lists(
                    array,
                    resolved_patches,
                    values_dtype.clone(),
                    *nullability,
                )
            }
            DType::Extension(_ext_dtype) => todo!(),
        }
    }
}

/// The elements of this [ListScalar] as an array or `None` if scalar is null.
fn list_scalar_to_elements_array(scalar: ListScalar) -> VortexResult<Option<ArrayRef>> {
    let Some(elements) = scalar.elements() else {
        return Ok(None);
    };

    let mut builder = builder_with_capacity(scalar.element_dtype(), scalar.len());
    for s in elements {
        builder.append_scalar(&s)?;
    }
    Ok(Some(builder.finish()))
}

/// Create a list-typed array containing one element, scalar, or `None` if scalar is null.
fn list_scalar_to_singleton_list_array(scalar: ListScalar) -> VortexResult<Option<ArrayRef>> {
    let nullability = scalar.dtype().nullability();
    let Some(elements) = list_scalar_to_elements_array(scalar)? else {
        return Ok(None);
    };

    let validity = match nullability {
        Nullability::NonNullable => Validity::NonNullable,
        Nullability::Nullable => Validity::AllValid,
    };

    let n = elements.len();
    ListArray::try_new(elements, buffer![0_u64, n as u64].into_array(), validity)
        .map(|x| Some(x.into_array()))
}

#[allow(clippy::cognitive_complexity)]
fn canonicalize_sparse_lists(
    array: &SparseArray,
    resolved_patches: Patches,
    values_dtype: Arc<DType>,
    nullability: Nullability,
) -> VortexResult<Canonical> {
    macro_rules! match_smallest_offset_type {
        ($n_elements:expr, | $offset_type:ident | $body:block) => {{
            let n_elements = $n_elements;
            if n_elements <= u8::MAX as usize {
                type $offset_type = u8;
                $body
            } else if n_elements <= u16::MAX as usize {
                type $offset_type = u16;
                $body
            } else if n_elements <= u32::MAX as usize {
                type $offset_type = u32;
                $body
            } else {
                assert!(u64::try_from(n_elements).is_ok());
                type $offset_type = u64;
                $body
            }
        }};
    }

    let indices = resolved_patches.indices().to_primitive()?;
    let values = resolved_patches.values().to_list()?;
    let fill_value = array.fill_scalar().as_list();

    let n_filled = array.len() - resolved_patches.num_patches();
    let total_canonical_values = values.elements().len() + fill_value.len() * n_filled;

    let validity = array
        .validity_mask()
        .map(|x| Validity::from_mask(x, nullability))?;

    match_each_integer_ptype!(indices.ptype(), |I| {
        match_smallest_offset_type!(total_canonical_values, |O| {
            canonicalize_sparse_lists_inner::<I, O>(
                indices.as_slice(),
                values,
                fill_value,
                values_dtype,
                array.len(),
                total_canonical_values,
                validity,
            )
        })
    })
}

fn canonicalize_sparse_lists_inner<I: NativePType, SmallestViableOffsetType: OffsetPType>(
    indices: &[I],
    values: ListArray,
    fill_value: ListScalar,
    values_dtype: Arc<DType>,
    len: usize,
    total_canonical_values: usize,
    validity: Validity,
) -> VortexResult<Canonical> {
    let Some(fill_value_array) = list_scalar_to_singleton_list_array(fill_value)? else {
        let sparse_list_elements = values.elements().clone();
        let sparse_list_offsets = values.offsets().to_primitive()?;
        match_each_integer_ptype!(sparse_list_offsets.ptype(), |SparseValuesOffsetType| {
            let sparse_list_offsets = sparse_list_offsets.as_slice::<SparseValuesOffsetType>();
            // If the values are a small slice of a large array, their offsets may not fit in
            // SmallestViableOffsetType. We avoid a copy by reusing values.elements(), but we
            // therefore must use the offset type of values.offsets().
            return canonicalize_sparse_lists_inner_with_null_fill_value(
                indices,
                sparse_list_elements,
                sparse_list_offsets,
                len,
                validity,
            );
        });
    };

    let mut builder = ListBuilder::<SmallestViableOffsetType>::with_values_and_index_capacity(
        values_dtype,
        validity.nullability(),
        total_canonical_values,
        len + 1,
    );
    let mut next_index = 0_usize;
    let enumerated_indices_usize = indices
        .iter()
        .map(|x| (*x).to_usize().vortex_expect("index must fit in usize"))
        .enumerate();
    for (patch_values_index, next_patched_index) in enumerated_indices_usize {
        for _ in next_index..next_patched_index {
            builder.extend_from_array(&fill_value_array)?;
        }
        builder.extend_from_array(&values.slice(patch_values_index, patch_values_index + 1)?)?;
        next_index = next_patched_index + 1;
    }

    for _ in next_index..len {
        builder.extend_from_array(&fill_value_array)?;
    }

    builder.finish().to_canonical()
}

fn canonicalize_sparse_lists_inner_with_null_fill_value<I: NativePType, O: OffsetPType>(
    indices: &[I],
    elements: ArrayRef,
    offsets: &[O],
    len: usize,
    validity: Validity,
) -> VortexResult<Canonical> {
    assert!(indices.len() < len + 1);
    let mut dense_offsets = BufferMut::with_capacity(len + 1);

    // We cannot use zero because elements may have leading junk values (e.g. the result of a slice).
    dense_offsets.push(offsets[0]);
    let mut dense_last_set_index = 0_usize;
    for (sparse_start_index, dense_start_index) in indices.iter().enumerate() {
        let sparse_end_index = sparse_start_index + 1;
        let dense_start_index = (*dense_start_index)
            .to_usize()
            .vortex_expect("index must fit in usize");
        let dense_end_index = dense_start_index + 1;

        for _ in (dense_last_set_index + 1)..dense_end_index {
            // For each null list, copy-forward the old index. These empty lists are masked by the validity.
            dense_offsets.push(dense_offsets[dense_last_set_index]);
        }
        dense_offsets.push(offsets[sparse_end_index]);
        dense_last_set_index = dense_end_index;
    }
    for _ in (dense_last_set_index + 1)..len {
        // For each null list, copy-forward the old index. These empty lists are masked by the validity.
        dense_offsets.push(dense_offsets[dense_last_set_index]);
    }
    let array = ListArray::try_new(elements, dense_offsets.into_array(), validity)?;
    Ok(Canonical::List(array))
}

fn canonicalize_sparse_bools(patches: &Patches, fill_value: &Scalar) -> VortexResult<Canonical> {
    let (fill_bool, validity) = if fill_value.is_null() {
        (false, Validity::AllInvalid)
    } else {
        (
            fill_value.try_into()?,
            if patches.dtype().nullability() == Nullability::NonNullable {
                Validity::NonNullable
            } else {
                Validity::AllValid
            },
        )
    };

    let bools = BoolArray::new(
        if fill_bool {
            BooleanBuffer::new_set(patches.array_len())
        } else {
            BooleanBuffer::new_unset(patches.array_len())
        },
        validity,
    );

    bools.patch(patches).map(Canonical::Bool)
}

fn canonicalize_sparse_primitives<
    T: NativePType + for<'a> TryFrom<&'a Scalar, Error = VortexError>,
>(
    patches: &Patches,
    fill_value: &Scalar,
) -> VortexResult<Canonical> {
    let (primitive_fill, validity) = if fill_value.is_null() {
        (T::default(), Validity::AllInvalid)
    } else {
        (
            fill_value.try_into()?,
            if patches.dtype().nullability() == Nullability::NonNullable {
                Validity::NonNullable
            } else {
                Validity::AllValid
            },
        )
    };

    let parray = PrimitiveArray::new(buffer![primitive_fill; patches.array_len()], validity);

    parray.patch(patches).map(Canonical::Primitive)
}

fn canonicalize_sparse_struct(
    struct_fields: &StructFields,
    fill_struct: StructScalar,
    dtype: &DType,
    // Resolution is unnecessary b/c we're just pushing the patches into the fields.
    unresolved_patches: &Patches,
    len: usize,
) -> VortexResult<Canonical> {
    let (fill_values, top_level_fill_validity) = match fill_struct.fields() {
        Some(fill_values) => (fill_values, Validity::AllValid),
        None => (
            struct_fields.fields().map(Scalar::default_value).collect(),
            Validity::AllInvalid,
        ),
    };
    let patch_values_as_struct = unresolved_patches.values().to_canonical()?.into_struct()?;
    let columns_patch_values = patch_values_as_struct.fields();
    let names = patch_values_as_struct.names();
    let validity = if dtype.is_nullable() {
        top_level_fill_validity.patch(
            len,
            unresolved_patches.offset(),
            unresolved_patches.indices(),
            &Validity::from_mask(
                unresolved_patches.values().validity_mask()?,
                Nullability::Nullable,
            ),
        )?
    } else {
        top_level_fill_validity
            .into_non_nullable()
            .ok_or_else(|| vortex_err!("fill validity should match sparse array nullability"))?
    };

    columns_patch_values
        .iter()
        .cloned()
        .zip_eq(fill_values.into_iter())
        .map(|(patch_values, fill_value)| -> VortexResult<_> {
            SparseArray::try_new_from_patches(
                unresolved_patches
                    .clone()
                    .map_values(|_| Ok(patch_values))?,
                fill_value,
            )
        })
        .process_results(|sparse_columns| {
            StructArray::try_from_iter_with_validity(names.iter().zip_eq(sparse_columns), validity)
                .map(Canonical::Struct)
        })?
}

fn canonicalize_sparse_decimal<D: NativeDecimalType>(
    decimal_dtype: DecimalDType,
    nullability: Nullability,
    fill_value: DecimalScalar,
    patches: &Patches,
    len: usize,
) -> VortexResult<Canonical> {
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
    let array = filled_array.patch(patches)?;
    Ok(Canonical::Decimal(array))
}

fn canonicalize_varbin(
    array: &SparseArray,
    dtype: DType,
    fill_value: Option<ByteBuffer>,
) -> VortexResult<Canonical> {
    let patches = array.resolved_patches()?;
    let indices = patches.indices().to_primitive()?;
    let values = patches.values().to_varbinview()?;
    let validity = array
        .validity_mask()
        .map(|x| Validity::from_mask(x, dtype.nullability()))?;
    let len = array.len();

    match_each_integer_ptype!(indices.ptype(), |I| {
        let indices = indices.buffer::<I>();
        canonicalize_varbin_inner::<I>(fill_value, indices, values, dtype, validity, len)
    })
}

fn canonicalize_varbin_inner<I: NativePType>(
    fill_value: Option<ByteBuffer>,
    indices: Buffer<I>,
    values: VarBinViewArray,
    dtype: DType,
    validity: Validity,
    len: usize,
) -> VortexResult<Canonical> {
    assert_eq!(dtype.nullability(), validity.nullability());

    let n_patch_buffers = values.buffers().len();
    let mut buffers = values.buffers().to_vec();

    let fill = if let Some(buffer) = &fill_value {
        buffers.push(buffer.clone());
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

    let array = VarBinViewArray::try_new(views.freeze(), Arc::from(buffers), dtype, validity)?;

    Ok(Canonical::VarBinView(array))
}

#[cfg(test)]
mod test {
    use rstest::rstest;
    use vortex_array::arrays::{
        BoolArray, BooleanBufferBuilder, DecimalArray, ListArray, PrimitiveArray, StructArray,
        VarBinArray, VarBinViewArray,
    };
    use vortex_array::arrow::IntoArrowArray as _;
    use vortex_array::validity::Validity;
    use vortex_array::vtable::ValidityHelper;
    use vortex_array::{IntoArray, ToCanonical};
    use vortex_buffer::{ByteBuffer, buffer, buffer_mut};
    use vortex_dtype::Nullability::{NonNullable, Nullable};
    use vortex_dtype::{DType, DecimalDType, FieldNames, PType, StructFields};
    use vortex_mask::Mask;
    use vortex_scalar::{DecimalValue, Scalar};

    use crate::SparseArray;
    use crate::canonical::{list_scalar_to_elements_array, list_scalar_to_singleton_list_array};

    #[rstest]
    #[case(Some(true))]
    #[case(Some(false))]
    #[case(None)]
    fn test_sparse_bool(#[case] fill_value: Option<bool>) {
        let indices = buffer![0u64, 1, 7].into_array();
        let values = bool_array_from_nullable_vec(vec![Some(true), None, Some(false)], fill_value)
            .into_array();
        let sparse_bools =
            SparseArray::try_new(indices, values, 10, Scalar::from(fill_value)).unwrap();
        assert_eq!(sparse_bools.dtype(), &DType::Bool(Nullable));

        let flat_bools = sparse_bools.to_bool().unwrap();
        let expected = bool_array_from_nullable_vec(
            vec![
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
            ],
            fill_value,
        );

        assert_eq!(flat_bools.boolean_buffer(), expected.boolean_buffer());
        assert_eq!(flat_bools.validity(), expected.validity());

        assert!(flat_bools.boolean_buffer().value(0));
        assert!(flat_bools.validity().is_valid(0).unwrap());
        assert_eq!(
            flat_bools.boolean_buffer().value(1),
            fill_value.unwrap_or_default()
        );
        assert!(!flat_bools.validity().is_valid(1).unwrap());
        assert_eq!(
            flat_bools.validity().is_valid(2).unwrap(),
            fill_value.is_some()
        );
        assert!(!flat_bools.boolean_buffer().value(7));
        assert!(flat_bools.validity().is_valid(7).unwrap());
    }

    fn bool_array_from_nullable_vec(
        bools: Vec<Option<bool>>,
        fill_value: Option<bool>,
    ) -> BoolArray {
        let mut buffer = BooleanBufferBuilder::new(bools.len());
        let mut validity = BooleanBufferBuilder::new(bools.len());
        for maybe_bool in bools {
            buffer.append(maybe_bool.unwrap_or_else(|| fill_value.unwrap_or_default()));
            validity.append(maybe_bool.is_some());
        }
        BoolArray::new(buffer.finish(), Validity::from(validity.finish()))
    }

    #[rstest]
    #[case(Some(0i32))]
    #[case(Some(-1i32))]
    #[case(None)]
    fn test_sparse_primitive(#[case] fill_value: Option<i32>) {
        let indices = buffer![0u64, 1, 7].into_array();
        let values = PrimitiveArray::from_option_iter([Some(0i32), None, Some(1)]).into_array();
        let sparse_ints =
            SparseArray::try_new(indices, values, 10, Scalar::from(fill_value)).unwrap();
        assert_eq!(*sparse_ints.dtype(), DType::Primitive(PType::I32, Nullable));

        let flat_ints = sparse_ints.to_primitive().unwrap();
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

        assert_eq!(flat_ints.byte_buffer(), expected.byte_buffer());
        assert_eq!(flat_ints.validity(), expected.validity());

        assert_eq!(flat_ints.as_slice::<i32>()[0], 0);
        assert!(flat_ints.validity().is_valid(0).unwrap());
        assert_eq!(flat_ints.as_slice::<i32>()[1], 0);
        assert!(!flat_ints.validity().is_valid(1).unwrap());
        assert_eq!(
            flat_ints.as_slice::<i32>()[2],
            fill_value.unwrap_or_default()
        );
        assert_eq!(
            flat_ints.validity().is_valid(2).unwrap(),
            fill_value.is_some()
        );
        assert_eq!(flat_ints.as_slice::<i32>()[7], 1);
        assert!(flat_ints.validity().is_valid(7).unwrap());
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
                BoolArray::from_indices(4, vec![0, 1, 2], Validity::NonNullable).to_array(),
            ),
        )
        .unwrap()
        .into_array();

        let fill_scalar = Scalar::struct_(
            struct_dtype,
            vec![Scalar::from(Some(-10i32)), Scalar::from(Some(-1i32))],
        );
        let len = 10;
        let sparse_struct = SparseArray::try_new(indices, patch_values, len, fill_scalar).unwrap();

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
        .to_array()
        .into_arrow_preferred()
        .unwrap();

        let actual = sparse_struct
            .to_struct()
            .unwrap()
            .to_array()
            .into_arrow_preferred()
            .unwrap();

        assert_eq!(expected.data_type(), actual.data_type());
        assert_eq!(&expected, &actual);
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
                BoolArray::from_indices(4, vec![0, 1, 2], Validity::NonNullable).to_array(),
            ),
        )
        .unwrap()
        .into_array();

        let fill_scalar = Scalar::null(struct_dtype);
        let len = 10;
        let sparse_struct = SparseArray::try_new(indices, patch_values, len, fill_scalar).unwrap();

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
        .to_array()
        .into_arrow_preferred()
        .unwrap();

        let actual = sparse_struct
            .to_struct()
            .unwrap()
            .to_array()
            .into_arrow_preferred()
            .unwrap();

        assert_eq!(expected.data_type(), actual.data_type());
        assert_eq!(&expected, &actual);
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
        .to_array();
        let len = 10;
        let fill_scalar = Scalar::decimal(DecimalValue::I32(123), decimal_dtype, Nullable);
        let sparse_struct = SparseArray::try_new(indices, patch_values, len, fill_scalar).unwrap();

        let expected = DecimalArray::new(
            buffer![100i128, 200, 123, 123, 123, 123, 123, 300, 4000, 123],
            decimal_dtype,
            // NB: patch indices: [0, 1, 7, 8]; patch validity: [Valid, Valid, Valid, Invalid]; ergo 0, 1, 7 are valid.
            Validity::from_mask(Mask::from_excluded_indices(10, vec![8]), Nullable),
        )
        .to_array()
        .into_arrow_preferred()
        .unwrap();

        let actual = sparse_struct
            .to_decimal()
            .unwrap()
            .to_array()
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

        let array = SparseArray::try_new(
            buffer![0u16, 3, 4, 5, 7, 9, 10].into_array(),
            strings,
            12,
            Scalar::from(Some("123".to_owned())),
        )
        .unwrap();

        let actual = array.to_varbinview().unwrap().into_array();
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

        let actual = actual.into_arrow_preferred().unwrap();
        let expected = expected.into_arrow_preferred().unwrap();

        assert_eq!(actual.data_type(), expected.data_type());
        assert_eq!(&actual, &expected);
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

        let array = SparseArray::try_new(
            buffer![0u16, 3, 4, 5, 7, 9, 10].into_array(),
            strings,
            12,
            Scalar::null(DType::Utf8(Nullable)),
        )
        .unwrap();

        let actual = array.to_varbinview().unwrap().into_array();
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

        let actual = actual.into_arrow_preferred().unwrap();
        let expected = expected.into_arrow_preferred().unwrap();

        assert_eq!(actual.data_type(), expected.data_type());
        assert_eq!(&actual, &expected);
    }

    #[test]
    fn test_sparse_utf8_varbinview_non_nullable() {
        let strings =
            VarBinViewArray::from_iter_str(["hello", "goodbye", "hello", "bonjour", "你好"])
                .into_array();

        let array = SparseArray::try_new(
            buffer![0u16, 3, 4, 5, 8].into_array(),
            strings,
            9,
            Scalar::from("123".to_owned()),
        )
        .unwrap();

        let actual = array.to_varbinview().unwrap().into_array();
        let expected = <VarBinViewArray as FromIterator<_>>::from_iter([
            Some("hello"),
            Some("123"),
            Some("123"),
            Some("goodbye"),
            Some("hello"),
            Some("bonjour"),
            Some("123"),
            Some("123"),
            Some("你好"),
        ])
        .into_array();

        let actual = actual.into_arrow_preferred().unwrap();
        let expected = expected.into_arrow_preferred().unwrap();

        assert_eq!(actual.data_type(), expected.data_type());
        assert_eq!(&actual, &expected);
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

        let array = SparseArray::try_new(
            buffer![0u16, 3, 4, 5, 7, 9, 10].into_array(),
            strings,
            12,
            Scalar::null(DType::Utf8(Nullable)),
        )
        .unwrap();

        let actual = array.to_varbinview().unwrap().into_array();
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

        let actual = actual.into_arrow_preferred().unwrap();
        let expected = expected.into_arrow_preferred().unwrap();

        assert_eq!(actual.data_type(), expected.data_type());
        assert_eq!(&actual, &expected);
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

        let array = SparseArray::try_new(
            buffer![0u16, 3, 4, 5, 7, 9, 10].into_array(),
            binaries,
            12,
            Scalar::from(Some(ByteBuffer::from(b"123".to_vec()))),
        )
        .unwrap();

        let actual = array.to_varbinview().unwrap().into_array();
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

        let actual = actual.into_arrow_preferred().unwrap();
        let expected = expected.into_arrow_preferred().unwrap();

        assert_eq!(actual.data_type(), expected.data_type());
        assert_eq!(&actual, &expected);
    }

    #[test]
    fn test_list_scalar_to_elements_array() {
        let scalar = Scalar::from(Some(vec![1, 2, 3]));
        let array = list_scalar_to_elements_array(scalar.as_list()).unwrap();
        assert_eq!(
            array.unwrap().display_values().to_string(),
            "[1i32, 2i32, 3i32]"
        );

        let scalar = Scalar::null_typed::<Vec<i32>>();
        let array = list_scalar_to_elements_array(scalar.as_list()).unwrap();
        assert!(array.is_none());
    }

    #[test]
    fn test_list_scalar_to_singleton_list_array() {
        let scalar = Scalar::from(Some(vec![1, 2, 3]));
        let array = list_scalar_to_singleton_list_array(scalar.as_list()).unwrap();
        assert!(array.is_some());
        let array = array.unwrap();
        assert_eq!(array.scalar_at(0).unwrap(), scalar);
        assert_eq!(array.len(), 1);

        let scalar = Scalar::null_typed::<Vec<i32>>();
        let array = list_scalar_to_singleton_list_array(scalar.as_list()).unwrap();
        assert!(array.is_none());
    }

    #[test]
    fn test_sparse_list_null_fill() {
        let elements = buffer![1i32, 2, 1, 2].into_array();
        let offsets = buffer![0u32, 1, 2, 3, 4].into_array();
        let lists = ListArray::try_new(elements, offsets, Validity::AllValid)
            .unwrap()
            .into_array();

        let indices = buffer![0u8, 3u8, 4u8, 5u8].into_array();
        let fill_value = Scalar::null(lists.dtype().clone());
        let sparse = SparseArray::try_new(indices, lists, 6, fill_value)
            .unwrap()
            .into_array();

        let actual = sparse.to_canonical().unwrap().into_array();
        let expected = ListArray::try_new(
            buffer![1i32, 2, 1, 2].into_array(),
            buffer![0u32, 1, 1, 1, 2, 3, 4].into_array(),
            Validity::Array(
                BoolArray::from_iter([true, false, false, true, true, true]).into_array(),
            ),
        )
        .unwrap()
        .into_array();

        let actual = actual.into_arrow_preferred().unwrap();
        let expected = expected.into_arrow_preferred().unwrap();

        assert_eq!(actual.data_type(), expected.data_type());
        assert_eq!(&actual, &expected);
    }

    #[test]
    fn test_sparse_list_null_fill_sliced_sparse_values() {
        let elements = buffer![1i32, 2, 1, 2, 1, 2, 1, 2].into_array();
        let offsets = buffer![0u32, 1, 2, 3, 4, 5, 6, 7, 8].into_array();
        let lists = ListArray::try_new(elements, offsets, Validity::AllValid)
            .unwrap()
            .into_array();
        let lists = lists.slice(2, 6).unwrap();

        let indices = buffer![0u8, 3u8, 4u8, 5u8].into_array();
        let fill_value = Scalar::null(lists.dtype().clone());
        let sparse = SparseArray::try_new(indices, lists, 6, fill_value)
            .unwrap()
            .into_array();

        let actual = sparse.to_canonical().unwrap().into_array();
        let expected = ListArray::try_new(
            buffer![1i32, 2, 1, 2].into_array(),
            buffer![0u32, 1, 1, 1, 2, 3, 4].into_array(),
            Validity::Array(
                BoolArray::from_iter([true, false, false, true, true, true]).into_array(),
            ),
        )
        .unwrap()
        .into_array();

        let actual = actual.into_arrow_preferred().unwrap();
        let expected = expected.into_arrow_preferred().unwrap();

        assert_eq!(actual.data_type(), expected.data_type());
        assert_eq!(&actual, &expected);
    }

    #[test]
    fn test_sparse_list_non_null_fill() {
        let elements = buffer![1i32, 2, 1, 2].into_array();
        let offsets = buffer![0u32, 1, 2, 3, 4].into_array();
        let lists = ListArray::try_new(elements, offsets, Validity::AllValid)
            .unwrap()
            .into_array();

        let indices = buffer![0u8, 3u8, 4u8, 5u8].into_array();
        let fill_value = Scalar::from(Some(vec![5i32, 6, 7, 8]));
        let sparse = SparseArray::try_new(indices, lists, 6, fill_value)
            .unwrap()
            .into_array();

        let actual = sparse.to_canonical().unwrap().into_array();
        let expected = ListArray::try_new(
            buffer![1i32, 5, 6, 7, 8, 5, 6, 7, 8, 2, 1, 2].into_array(),
            buffer![0u32, 1, 5, 9, 10, 11, 12].into_array(),
            Validity::AllValid,
        )
        .unwrap()
        .into_array();

        let actual = actual.into_arrow_preferred().unwrap();
        let expected = expected.into_arrow_preferred().unwrap();

        assert_eq!(actual.data_type(), expected.data_type());
        assert_eq!(&actual, &expected);
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

        let array = SparseArray::try_new(
            buffer![0u16, 3, 4, 5, 7, 9, 10].into_array(),
            strings,
            12,
            Scalar::null(DType::Binary(Nullable)),
        )
        .unwrap();

        let actual = array.to_varbinview().unwrap().into_array();
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
        let actual = actual.into_arrow_preferred().unwrap();
        let expected = expected.into_arrow_preferred().unwrap();

        assert_eq!(actual.data_type(), expected.data_type());
        assert_eq!(&actual, &expected);
    }

    #[test]
    fn test_sparse_list_grows_offset_type() {
        let elements = buffer![1i32, 2, 1, 2].into_array();
        let offsets = buffer![0u8, 1, 2, 3, 4].into_array();
        let lists = ListArray::try_new(elements, offsets, Validity::AllValid)
            .unwrap()
            .into_array();

        let indices = buffer![0u8, 1u8, 2u8, 3u8].into_array();
        let fill_value = Scalar::from(Some(vec![42i32; 252])); // 252 + 4 elements = 256 > u8::MAX
        let sparse = SparseArray::try_new(indices, lists, 5, fill_value)
            .unwrap()
            .into_array();

        let actual = sparse.to_canonical().unwrap().into_array();
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
            actual.to_list().unwrap().offsets().dtype(),
            &DType::Primitive(PType::U16, NonNullable)
        );

        let actual = actual.into_arrow_preferred().unwrap();
        let expected = expected.into_arrow_preferred().unwrap();

        assert_eq!(actual.data_type(), expected.data_type());
        assert_eq!(&actual, &expected);
    }
}
