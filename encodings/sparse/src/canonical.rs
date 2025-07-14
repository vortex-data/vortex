// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use num_traits::NumCast;
use vortex_array::arrays::{
    BinaryView, BoolArray, BooleanBuffer, ConstantArray, NullArray, PrimitiveArray, StructArray,
    VarBinViewArray, smallest_storage_type,
};
use vortex_array::builders::{ArrayBuilder as _, DecimalBuilder};
use vortex_array::patches::Patches;
use vortex_array::validity::Validity;
use vortex_array::vtable::CanonicalVTable;
use vortex_array::{Array, Canonical, ToCanonical as _};
use vortex_buffer::{Buffer, buffer, buffer_mut};
use vortex_dtype::{
    DType, DecimalDType, NativePType, Nullability, StructFields, match_each_integer_ptype,
    match_each_native_ptype,
};
use vortex_error::{VortexError, VortexExpect as _, VortexResult, vortex_err};
use vortex_scalar::{
    DecimalScalar, NativeDecimalType, Scalar, StructScalar, Utf8Scalar,
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
            DType::Utf8(nullability) => {
                let patches = array.resolved_patches()?;
                let indices = patches.indices().to_primitive()?;
                let values = patches.values().to_varbinview()?;
                let fill_value = array.fill_scalar().as_utf8();
                let validity = array
                    .validity_mask()
                    .map(|x| Validity::from_mask(x, *nullability))?;
                let len = array.len();

                match_each_integer_ptype!(indices.ptype(), |I| {
                    let indices = indices.buffer::<I>();
                    canonicalize_utf8::<I>(fill_value, indices, values, validity, len)
                })
            }
            DType::Binary(_nullability) => todo!(),
            DType::List(_dtype, _nullability) => todo!(),
            DType::Extension(_ext_dtype) => todo!(),
        }
    }
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

fn canonicalize_utf8<I: NativePType>(
    fill_value: Utf8Scalar,
    indices: Buffer<I>,
    values: VarBinViewArray,
    validity: Validity,
    len: usize,
) -> VortexResult<Canonical> {
    let n_patch_buffers = values.buffers().len();
    let mut buffers = values.buffers().to_vec();

    let fill = if let Some(buffer) = &fill_value.value() {
        buffers.push(buffer.inner().clone());
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

    let array = VarBinViewArray::try_new(
        views.freeze(),
        buffers,
        DType::Utf8(validity.nullability()),
        validity,
    )?;

    Ok(Canonical::VarBinView(array))
}

#[cfg(test)]
mod test {
    use rstest::rstest;
    use vortex_array::arrays::{
        BoolArray, BooleanBufferBuilder, DecimalArray, PrimitiveArray, StructArray, VarBinArray,
        VarBinViewArray,
    };
    use vortex_array::arrow::IntoArrowArray as _;
    use vortex_array::validity::Validity;
    use vortex_array::vtable::ValidityHelper;
    use vortex_array::{IntoArray, ToCanonical};
    use vortex_buffer::buffer;
    use vortex_dtype::Nullability::Nullable;
    use vortex_dtype::{DType, DecimalDType, FieldNames, PType, StructFields};
    use vortex_mask::Mask;
    use vortex_scalar::{DecimalValue, Scalar};

    use crate::SparseArray;

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
    fn test_sparse_varbinview_non_null_fill() {
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
    fn test_sparse_varbinview_null_fill() {
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
    fn test_sparse_varbinview_non_nullable() {
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
    fn test_sparse_varbin_null_fill() {
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
}
