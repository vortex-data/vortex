// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use itertools::Itertools;
use num_traits::NumCast;
use vortex_array::arrays::{
    BinaryView, BoolArray, BooleanBuffer, ConstantArray, FixedSizeListArray, ListArray, NullArray,
    OffsetPType, PrimitiveArray, StructArray, VarBinViewArray, smallest_storage_type,
};
use vortex_array::builders::{ArrayBuilder, DecimalBuilder, ListBuilder, builder_with_capacity};
use vortex_array::patches::Patches;
use vortex_array::validity::Validity;
use vortex_array::vtable::{CanonicalVTable, ValidityHelper};
use vortex_array::{Array, ArrayRef, Canonical, IntoArray as _, ToCanonical as _};
use vortex_buffer::{Buffer, BufferMut, BufferString, ByteBuffer, buffer, buffer_mut};
use vortex_dtype::{
    DType, DecimalDType, NativePType, Nullability, StructFields, match_each_integer_ptype,
    match_each_native_ptype,
};
use vortex_error::{VortexError, VortexExpect as _, vortex_panic};
use vortex_scalar::{
    DecimalScalar, ListScalar, NativeDecimalType, Scalar, StructScalar,
    match_each_decimal_value_type,
};

use crate::{SparseArray, SparseVTable};

impl CanonicalVTable<SparseVTable> for SparseVTable {
    fn canonicalize(array: &SparseArray) -> Canonical {
        if array.patches().num_patches() == 0 {
            return ConstantArray::new(array.fill_scalar().clone(), array.len()).to_canonical();
        }

        match array.dtype() {
            DType::Null => {
                assert!(array.fill_scalar().is_null());
                Canonical::Null(NullArray::new(array.len()))
            }
            DType::Bool(..) => {
                let resolved_patches = array.resolved_patches();
                canonicalize_sparse_bools(&resolved_patches, array.fill_scalar())
            }
            DType::Primitive(ptype, ..) => {
                let resolved_patches = array.resolved_patches();
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
                let resolved_patches = array.resolved_patches();
                canonicalize_sparse_lists(
                    array,
                    resolved_patches,
                    values_dtype.clone(),
                    *nullability,
                )
            }
            DType::FixedSizeList(.., nullability) => {
                canonicalize_sparse_fixed_size_list(array, *nullability)
            }
            DType::Extension(_ext_dtype) => todo!(),
        }
    }
}

/// The elements of this [ListScalar] as an array or `None` if scalar is null.
fn list_scalar_to_elements_array(scalar: ListScalar) -> Option<ArrayRef> {
    let elements = scalar.elements()?;

    let mut builder = builder_with_capacity(scalar.element_dtype(), scalar.len());
    for s in elements {
        builder
            .append_scalar(&s)
            .vortex_expect("Scalar dtype must match");
    }
    Some(builder.finish())
}

/// Create a list-typed array containing one element, scalar, or `None` if scalar is null.
fn list_scalar_to_singleton_list_array(scalar: ListScalar) -> Option<ArrayRef> {
    let nullability = scalar.dtype().nullability();
    let elements = list_scalar_to_elements_array(scalar)?;

    let validity = match nullability {
        Nullability::NonNullable => Validity::NonNullable,
        Nullability::Nullable => Validity::AllValid,
    };

    let n = elements.len();
    Some(
        unsafe {
            ListArray::new_unchecked(elements, buffer![0_u64, n as u64].into_array(), validity)
        }
        .into_array(),
    )
}

#[allow(clippy::cognitive_complexity)]
fn canonicalize_sparse_lists(
    array: &SparseArray,
    resolved_patches: Patches,
    values_dtype: Arc<DType>,
    nullability: Nullability,
) -> Canonical {
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

    let indices = resolved_patches.indices().to_primitive();
    let values = resolved_patches.values().to_list();
    let fill_value = array.fill_scalar().as_list();

    let n_filled = array.len() - resolved_patches.num_patches();
    let total_canonical_values = values.elements().len() + fill_value.len() * n_filled;

    let validity = Validity::from_mask(array.validity_mask(), nullability);

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
) -> Canonical {
    let Some(fill_value_array) = list_scalar_to_singleton_list_array(fill_value) else {
        let sparse_list_elements = values.elements().clone();
        let sparse_list_offsets = values.offsets().to_primitive();
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
            builder.extend_from_array(&fill_value_array);
        }
        builder.extend_from_array(&values.slice(patch_values_index..patch_values_index + 1));
        next_index = next_patched_index + 1;
    }

    for _ in next_index..len {
        builder.extend_from_array(&fill_value_array);
    }

    builder.finish_into_canonical()
}

fn canonicalize_sparse_lists_inner_with_null_fill_value<I: NativePType, O: OffsetPType>(
    indices: &[I],
    elements: ArrayRef,
    offsets: &[O],
    len: usize,
    validity: Validity,
) -> Canonical {
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
    Canonical::List(unsafe {
        ListArray::new_unchecked(elements, dense_offsets.into_array(), validity)
    })
}

/// Canonicalize a sparse [`FixedSizeListArray`] by expanding it into a dense representation.
fn canonicalize_sparse_fixed_size_list(array: &SparseArray, nullability: Nullability) -> Canonical {
    let resolved_patches = array.resolved_patches();
    let indices = resolved_patches.indices().to_primitive();
    let values = resolved_patches.values().to_fixed_size_list();
    let fill_value = array.fill_scalar().as_list();

    let validity = Validity::from_mask(array.validity_mask(), nullability);

    match_each_integer_ptype!(indices.ptype(), |I| {
        canonicalize_sparse_fixed_size_list_inner::<I>(
            indices.as_slice(),
            values,
            fill_value,
            array.len(),
            validity,
        )
    })
}

/// Build a canonical [`FixedSizeListArray`] from sparse patches by interleaving patch values with
/// fill values.
///
/// This algorithm walks through the sparse indices sequentially, filling gaps with the fill value's
/// elements (or defaults if null). Since all lists have the same size, we can directly append
/// elements without tracking offsets.
fn canonicalize_sparse_fixed_size_list_inner<I: NativePType>(
    indices: &[I],
    values: FixedSizeListArray,
    fill_value: ListScalar,
    array_len: usize,
    validity: Validity,
) -> Canonical {
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
        if values.validity().is_valid(patch_idx) {
            let patch_list = values.fixed_size_list_at(patch_idx);
            for i in 0..list_size as usize {
                builder
                    .append_scalar(&patch_list.scalar_at(i))
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
    Canonical::FixedSizeList(unsafe {
        FixedSizeListArray::new_unchecked(elements, list_size, validity, array_len)
    })
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

fn canonicalize_sparse_bools(patches: &Patches, fill_value: &Scalar) -> Canonical {
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

    let bools = BoolArray::from_bool_buffer(
        if fill_bool {
            BooleanBuffer::new_set(patches.array_len())
        } else {
            BooleanBuffer::new_unset(patches.array_len())
        },
        validity,
    );

    Canonical::Bool(bools.patch(patches))
}

fn canonicalize_sparse_primitives<
    T: NativePType + for<'a> TryFrom<&'a Scalar, Error = VortexError>,
>(
    patches: &Patches,
    fill_value: &Scalar,
) -> Canonical {
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

    Canonical::Primitive(parray.patch(patches))
}

fn canonicalize_sparse_struct(
    struct_fields: &StructFields,
    fill_struct: StructScalar,
    dtype: &DType,
    // Resolution is unnecessary b/c we're just pushing the patches into the fields.
    unresolved_patches: &Patches,
    len: usize,
) -> Canonical {
    let (fill_values, top_level_fill_validity) = match fill_struct.fields() {
        Some(fill_values) => (fill_values, Validity::AllValid),
        None => (
            struct_fields.fields().map(Scalar::default_value).collect(),
            Validity::AllInvalid,
        ),
    };
    let patch_values_as_struct = unresolved_patches.values().to_struct();
    let columns_patch_values = patch_values_as_struct.fields();
    let names = patch_values_as_struct.names();
    let validity = if dtype.is_nullable() {
        top_level_fill_validity.patch(
            len,
            unresolved_patches.offset(),
            unresolved_patches.indices(),
            &Validity::from_mask(
                unresolved_patches.values().validity_mask(),
                Nullability::Nullable,
            ),
        )
    } else {
        top_level_fill_validity
            .into_non_nullable()
            .unwrap_or_else(|| vortex_panic!("fill validity should match sparse array nullability"))
    };

    StructArray::try_from_iter_with_validity(
        names.iter().zip_eq(
            columns_patch_values
                .iter()
                .cloned()
                .zip_eq(fill_values)
                .map(|(patch_values, fill_value)| unsafe {
                    SparseArray::new_unchecked(
                        unresolved_patches
                            .clone()
                            .map_values(|_| Ok(patch_values))
                            .vortex_expect("Replacing patch values"),
                        fill_value,
                    )
                }),
        ),
        validity,
    )
    .map(Canonical::Struct)
    .vortex_expect("Creating struct array")
}

fn canonicalize_sparse_decimal<D: NativeDecimalType>(
    decimal_dtype: DecimalDType,
    nullability: Nullability,
    fill_value: DecimalScalar,
    patches: &Patches,
    len: usize,
) -> Canonical {
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
    let array = filled_array.patch(patches);
    Canonical::Decimal(array)
}

fn canonicalize_varbin(
    array: &SparseArray,
    dtype: DType,
    fill_value: Option<ByteBuffer>,
) -> Canonical {
    let patches = array.resolved_patches();
    let indices = patches.indices().to_primitive();
    let values = patches.values().to_varbinview();
    let validity = Validity::from_mask(array.validity_mask(), dtype.nullability());
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
) -> Canonical {
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

    // SAFETY: views are constructed to maintain the invariants
    let array = unsafe {
        VarBinViewArray::new_unchecked(views.freeze(), Arc::from(buffers), dtype, validity)
    };

    Canonical::VarBinView(array)
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use rstest::rstest;
    use vortex_array::arrays::{
        BoolArray, BooleanBufferBuilder, DecimalArray, FixedSizeListArray, ListArray,
        PrimitiveArray, StructArray, VarBinArray, VarBinViewArray,
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

        let flat_bools = sparse_bools.to_bool();
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
        assert!(flat_bools.validity().is_valid(0));
        assert_eq!(
            flat_bools.boolean_buffer().value(1),
            fill_value.unwrap_or_default()
        );
        assert!(!flat_bools.validity().is_valid(1));
        assert_eq!(flat_bools.validity().is_valid(2), fill_value.is_some());
        assert!(!flat_bools.boolean_buffer().value(7));
        assert!(flat_bools.validity().is_valid(7));
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
        BoolArray::from_bool_buffer(buffer.finish(), Validity::from(validity.finish()))
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

        let flat_ints = sparse_ints.to_primitive();
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
        assert!(flat_ints.validity().is_valid(0));
        assert_eq!(flat_ints.as_slice::<i32>()[1], 0);
        assert!(!flat_ints.validity().is_valid(1));
        assert_eq!(
            flat_ints.as_slice::<i32>()[2],
            fill_value.unwrap_or_default()
        );
        assert_eq!(flat_ints.validity().is_valid(2), fill_value.is_some());
        assert_eq!(flat_ints.as_slice::<i32>()[7], 1);
        assert!(flat_ints.validity().is_valid(7));
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

        let actual = array.to_varbinview().into_array();
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

        let actual = array.to_varbinview().into_array();
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

        let actual = array.to_varbinview().into_array();
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

        let actual = array.to_varbinview().into_array();
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

        let actual = array.to_varbinview().into_array();
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
        let array = list_scalar_to_elements_array(scalar.as_list());
        assert_eq!(
            array.unwrap().display_values().to_string(),
            "[1i32, 2i32, 3i32]"
        );

        let scalar = Scalar::null_typed::<Vec<i32>>();
        let array = list_scalar_to_elements_array(scalar.as_list());
        assert!(array.is_none());
    }

    #[test]
    fn test_list_scalar_to_singleton_list_array() {
        let scalar = Scalar::from(Some(vec![1, 2, 3]));
        let array = list_scalar_to_singleton_list_array(scalar.as_list());
        assert!(array.is_some());
        let array = array.unwrap();
        assert_eq!(array.scalar_at(0), scalar);
        assert_eq!(array.len(), 1);

        let scalar = Scalar::null_typed::<Vec<i32>>();
        let array = list_scalar_to_singleton_list_array(scalar.as_list());
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

        let actual = sparse.to_canonical().into_array();
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
        let lists = lists.slice(2..6);

        let indices = buffer![0u8, 3u8, 4u8, 5u8].into_array();
        let fill_value = Scalar::null(lists.dtype().clone());
        let sparse = SparseArray::try_new(indices, lists, 6, fill_value)
            .unwrap()
            .into_array();

        let actual = sparse.to_canonical().into_array();
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

        let actual = sparse.to_canonical().into_array();
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

        let actual = array.to_varbinview().into_array();
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
    fn test_sparse_fixed_size_list_null_fill() {
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
        let sparse = SparseArray::try_new(indices, fsl, 5, fill_value)
            .unwrap()
            .into_array();

        let actual = sparse.to_canonical().into_array();

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

        let actual = actual.into_arrow_preferred().unwrap();
        let expected = expected.into_arrow_preferred().unwrap();

        assert_eq!(actual.data_type(), expected.data_type());
        assert_eq!(&actual, &expected);
    }

    #[test]
    fn test_sparse_fixed_size_list_non_null_fill() {
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
        let sparse = SparseArray::try_new(indices, fsl, 6, fill_value)
            .unwrap()
            .into_array();

        let actual = sparse.to_canonical().into_array();

        // Expected: [1,2], [99,88], [3,4], [99,88], [5,6], [99,88].
        let expected_elements = buffer![1i32, 2, 99, 88, 3, 4, 99, 88, 5, 6, 99, 88].into_array();
        let expected = FixedSizeListArray::try_new(expected_elements, 2, Validity::NonNullable, 6)
            .unwrap()
            .into_array();

        let actual = actual.into_arrow_preferred().unwrap();
        let expected = expected.into_arrow_preferred().unwrap();

        assert_eq!(actual.data_type(), expected.data_type());
        assert_eq!(&actual, &expected);
    }

    #[test]
    fn test_sparse_fixed_size_list_with_validity() {
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
        let sparse = SparseArray::try_new(indices, fsl, 6, fill_value)
            .unwrap()
            .into_array();

        let actual = sparse.to_canonical().into_array();

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

        let actual = actual.into_arrow_preferred().unwrap();
        let expected = expected.into_arrow_preferred().unwrap();

        assert_eq!(actual.data_type(), expected.data_type());
        assert_eq!(&actual, &expected);
    }

    #[test]
    fn test_sparse_fixed_size_list_truly_sparse() {
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

        let sparse = SparseArray::try_new(indices, fsl, 100, fill_value)
            .unwrap()
            .into_array();

        let actual = sparse.to_canonical().into_array();

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

        let actual = actual.into_arrow_preferred().unwrap();
        let expected = expected.into_arrow_preferred().unwrap();

        assert_eq!(actual.data_type(), expected.data_type());
        assert_eq!(&actual, &expected);
    }

    #[test]
    fn test_sparse_fixed_size_list_single_element() {
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
        let sparse = SparseArray::try_new(indices, fsl, 1, fill_value)
            .unwrap()
            .into_array();

        let actual = sparse.to_canonical().into_array();

        // Expected: just [42, 43].
        let expected_elements = buffer![42i32, 43].into_array();
        let expected = FixedSizeListArray::try_new(expected_elements, 2, Validity::NonNullable, 1)
            .unwrap()
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

        let actual = sparse.to_canonical().into_array();
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
            actual.to_list().offsets().dtype(),
            &DType::Primitive(PType::U16, NonNullable)
        );

        let actual = actual.into_arrow_preferred().unwrap();
        let expected = expected.into_arrow_preferred().unwrap();

        assert_eq!(actual.data_type(), expected.data_type());
        assert_eq!(&actual, &expected);
    }
}
