// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_buffer::BitBuffer;
use vortex_buffer::Buffer;
use vortex_buffer::buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::Canonical;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::BoolArray;
use crate::arrays::Constant;
use crate::arrays::ConstantArray;
use crate::arrays::DecimalArray;
use crate::arrays::ExtensionArray;
use crate::arrays::FixedSizeListArray;
use crate::arrays::ListViewArray;
use crate::arrays::NullArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::StructArray;
use crate::arrays::VarBinViewArray;
use crate::arrays::varbinview::BinaryView;
use crate::builders::builder_with_capacity;
use crate::dtype::DType;
use crate::dtype::DecimalType;
use crate::dtype::Nullability;
use crate::match_each_decimal_value;
use crate::match_each_decimal_value_type;
use crate::match_each_native_ptype;
use crate::scalar::DecimalValue;
use crate::scalar::Scalar;
use crate::validity::Validity;

/// Shared implementation for both `canonicalize` and `execute` methods.
pub(crate) fn constant_canonicalize(
    array: ArrayView<'_, Constant>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Canonical> {
    let scalar = array.scalar();

    let validity = match array.dtype().nullability() {
        Nullability::NonNullable => Validity::NonNullable,
        Nullability::Nullable => match scalar.is_null() {
            true => Validity::AllInvalid,
            false => Validity::AllValid,
        },
    };

    Ok(match array.dtype() {
        DType::Null => Canonical::Null(NullArray::new(array.len())),
        DType::Bool(..) => Canonical::Bool(BoolArray::new(
            if scalar.as_bool().value().unwrap_or_default() {
                BitBuffer::new_set(array.len())
            } else {
                BitBuffer::new_unset(array.len())
            },
            validity,
        )),
        DType::Primitive(ptype, ..) => {
            match_each_native_ptype!(ptype, |P| {
                Canonical::Primitive(PrimitiveArray::new(
                    if scalar.is_valid() {
                        Buffer::full(
                            P::try_from(scalar)
                                .vortex_expect("Couldn't unwrap scalar to primitive"),
                            array.len(),
                        )
                    } else {
                        Buffer::zeroed(array.len())
                    },
                    validity,
                ))
            })
        }
        DType::Decimal(decimal_type, ..) => {
            let size = DecimalType::smallest_decimal_value_type(decimal_type);
            let decimal = scalar.as_decimal();
            let Some(value) = decimal.decimal_value() else {
                let all_null = match_each_decimal_value_type!(size, |D| {
                    // SAFETY: All-null decimal arrays with zeroed buffers and matching validity.
                    unsafe {
                        DecimalArray::new_unchecked(
                            Buffer::<D>::zeroed(array.len()),
                            *decimal_type,
                            validity,
                        )
                    }
                });
                return Ok(Canonical::Decimal(all_null));
            };

            let decimal_array = match_each_decimal_value!(value, |value| {
                // SAFETY: Constant decimal values with correct type and validity.
                unsafe {
                    DecimalArray::new_unchecked(
                        Buffer::full(value, array.len()),
                        *decimal_type,
                        validity,
                    )
                }
            });
            Canonical::Decimal(decimal_array)
        }
        DType::Utf8(_) => {
            let value = scalar.as_utf8().value();
            let const_value = value.as_ref().map(|v| v.as_bytes());
            Canonical::VarBinView(constant_canonical_byte_view(
                const_value,
                array.dtype(),
                array.len(),
            ))
        }
        DType::Binary(_) => {
            let value = scalar.as_binary().value().cloned();
            let const_value = value.as_ref().map(|v| v.as_slice());
            Canonical::VarBinView(constant_canonical_byte_view(
                const_value,
                array.dtype(),
                array.len(),
            ))
        }
        DType::Struct(struct_dtype, _) => {
            let value = scalar.as_struct();
            let fields: Vec<_> = match value.fields_iter() {
                Some(fields) => fields
                    .into_iter()
                    .map(|s| ConstantArray::new(s, array.len()).into_array())
                    .collect(),
                None => {
                    assert!(matches!(validity, Validity::AllInvalid));
                    // The struct is entirely null, so fields just need placeholder values with the
                    // correct dtype. We use `default_value` which returns a zero for non-nullable
                    // dtypes and null for nullable dtypes, preserving each field's nullability.
                    struct_dtype
                        .fields()
                        .map(|dt| {
                            let scalar = Scalar::default_value(&dt);
                            ConstantArray::new(scalar, array.len()).into_array()
                        })
                        .collect()
                }
            };
            // SAFETY: Fields are constructed from the same struct scalar, all have same
            // length, dtypes match by construction.
            Canonical::Struct(unsafe {
                StructArray::new_unchecked(fields, struct_dtype.clone(), array.len(), validity)
            })
        }
        DType::Union(..) => todo!("TODO(connor)[Union]: unimplemented"),
        DType::List(..) => Canonical::List(constant_canonical_list_array(scalar, array.len())),
        DType::FixedSizeList(element_dtype, list_size, _) => {
            let value = scalar.as_list();

            Canonical::FixedSizeList(constant_canonical_fixed_size_list_array(
                value.elements(),
                element_dtype,
                *list_size,
                value.dtype().nullability(),
                array.len(),
            ))
        }
        DType::Extension(ext_dtype) => {
            let s = scalar.as_extension();

            let storage_scalar = s.to_storage_scalar();

            // NB: We need to execute the constant array to be canonical because there is a
            // reduction rule that turns `Extension(Constant(..))` into `Constant(Extension(..))`,
            // and if we don't do this we create an infinite cycle.
            // See `ExtensionConstantRule` for more details.
            let storage_self = ConstantArray::new(storage_scalar, array.len())
                .into_array()
                .execute::<Canonical>(ctx)?
                .into_array();

            Canonical::Extension(ExtensionArray::new(ext_dtype.clone(), storage_self))
        }
        DType::Variant(_) => {
            unimplemented!(
                "TODO(variant): canonicalization will use the child-array design in a follow-up"
            )
        }
    })
}

fn constant_canonical_byte_view(
    scalar_bytes: Option<&[u8]>,
    dtype: &DType,
    len: usize,
) -> VarBinViewArray {
    match scalar_bytes {
        None => {
            let views = buffer![BinaryView::empty_view(); len];

            // SAFETY: for all-null the views and buffers are just zeroed, never accessed.
            unsafe {
                VarBinViewArray::new_unchecked(
                    views,
                    Default::default(),
                    dtype.clone(),
                    Validity::AllInvalid,
                )
            }
        }
        Some(scalar_bytes) => {
            // Create a view to hold the scalar bytes.
            // If the scalar cannot be inlined, allocate a single buffer large enough to hold it.
            let view = BinaryView::make_view(scalar_bytes, 0, 0);
            let mut buffers = Vec::new();
            if scalar_bytes.len() >= BinaryView::MAX_INLINED_SIZE {
                buffers.push(Buffer::copy_from(scalar_bytes));
            }

            // Clone our constant view `len` times.
            let views = buffer![view; len];

            // SAFETY: all the views are identical and point to a constant value.
            unsafe {
                VarBinViewArray::new_unchecked(
                    views,
                    Arc::from(buffers),
                    dtype.clone(),
                    Validity::from(dtype.nullability()),
                )
            }
        }
    }
}

/// Creates a [`ListViewArray`] with constant values.
///
/// We basically just project the list scalar value into list view components. If the caller wants
/// a fully decompressed and non-overlapping array, they can rebuild the array.
fn constant_canonical_list_array(scalar: &Scalar, len: usize) -> ListViewArray {
    let list = scalar.as_list();

    // Since "canonicalize" only applies to the top level array, we can simply have 1 scalar in our
    // child `elements` and have all list views point to that scalar.
    let elements = if let Some(elements) = list.elements() {
        // Extract the list elements out of the scalar into a new array.
        let mut builder = builder_with_capacity(
            list.dtype()
                .as_list_element_opt()
                .vortex_expect("list scalar somehow did not have a list DType"),
            list.len(),
        );
        for scalar in &elements {
            builder
                .append_scalar(scalar)
                .vortex_expect("list element scalar was invalid");
        }
        builder.finish()
    } else {
        // Otherwise all values are null, and we don't need to store anything in our `elements`.
        Canonical::empty(list.element_dtype()).into_array()
    };

    let validity = if scalar.dtype().is_nullable() {
        if list.is_null() {
            Validity::AllInvalid
        } else {
            Validity::AllValid
        }
    } else {
        debug_assert!(!list.is_null());
        Validity::NonNullable
    };

    // Somewhat arbitrarily choose `u64` as the type for offsets and sizes.
    let offsets = ConstantArray::new::<u64>(0, len).into_array();
    let sizes = ConstantArray::new::<u64>(list.len() as u64, len).into_array();

    debug_assert!(!offsets.dtype().is_nullable());
    debug_assert!(!sizes.dtype().is_nullable());

    // SAFETY: All views point to the same range [0, list.len()) in the elements array.
    // The elements array contains `len` copies of the same value, offsets are all 0,
    // and sizes are all equal to the list length. The validity matches the scalar's nullability.
    unsafe { ListViewArray::new_unchecked(elements, offsets, sizes, validity) }
}

fn constant_canonical_fixed_size_list_array(
    values: Option<Vec<Scalar>>,
    element_dtype: &DType,
    list_size: u32,
    list_nullability: Nullability,
    len: usize,
) -> FixedSizeListArray {
    match values {
        None => {
            // Even though the scalar is null, we still have to allocate the correct amount of space
            // for the given `DType`.
            let elements_len = list_size as usize * len;
            let mut element_builder = builder_with_capacity(element_dtype, elements_len);
            element_builder.append_defaults(elements_len);
            let elements = element_builder.finish();

            // SAFETY: The elements array has a length that is a multiple of `list_size`, and the
            // validity is `AllInvalid` so we don't care about the length.
            unsafe {
                FixedSizeListArray::new_unchecked(elements, list_size, Validity::AllInvalid, len)
            }
        }
        Some(values) => {
            let mut elements_builder = builder_with_capacity(element_dtype, len * values.len());

            for _ in 0..len {
                for v in &values {
                    elements_builder
                        .append_scalar(v)
                        .vortex_expect("must be a same dtype");
                }
            }

            let elements = elements_builder.finish();
            let validity = Validity::from(list_nullability);

            // SAFETY: The elements array has a length that is a multiple of `list_size`, and the
            // validity is either `NonNullable` or `AllValid` so we don't care about the length.
            unsafe { FixedSizeListArray::new_unchecked(elements, list_size, validity, len) }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use enum_iterator::all;
    use itertools::Itertools;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;

    use crate::IntoArray;
    use crate::LEGACY_SESSION;
    use crate::VortexSessionExecute;
    use crate::arrays::ConstantArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::VarBinArray;
    use crate::arrays::fixed_size_list::FixedSizeListArrayExt;
    use crate::arrays::listview::ListViewArrayExt;
    use crate::arrays::listview::ListViewRebuildMode;
    use crate::arrays::struct_::StructArrayExt;
    use crate::assert_arrays_eq;
    #[expect(deprecated)]
    use crate::canonical::ToCanonical as _;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::dtype::half::f16;
    use crate::expr::stats::Stat;
    use crate::expr::stats::StatsProvider;
    use crate::scalar::Scalar;
    use crate::validity::Validity;

    #[test]
    fn test_canonicalize_null() {
        let const_null = ConstantArray::new(Scalar::null(DType::Null), 42);
        #[expect(deprecated)]
        let actual = const_null.as_array().to_null();
        assert_eq!(actual.len(), 42);
        assert_eq!(
            actual
                .execute_scalar(33, &mut LEGACY_SESSION.create_execution_ctx())
                .unwrap(),
            Scalar::null(DType::Null)
        );
    }

    #[test]
    fn test_canonicalize_const_str() {
        let const_array = ConstantArray::new("four".to_string(), 4);

        let expected = VarBinArray::from(vec!["four", "four", "four", "four"]);
        assert_arrays_eq!(const_array, expected);
    }

    #[test]
    fn test_canonicalize_propagates_stats() -> VortexResult<()> {
        let scalar = Scalar::bool(true, Nullability::NonNullable);
        let const_array = ConstantArray::new(scalar, 4).into_array();
        let stats = const_array.statistics().compute_all(
            &all::<Stat>().collect_vec(),
            &mut LEGACY_SESSION.create_execution_ctx(),
        )?;
        #[expect(deprecated)]
        let canonical = const_array.to_canonical()?.into_array();
        let canonical_stats = canonical.statistics();

        let stats_ref = stats.as_typed_ref(canonical.dtype());

        for stat in all::<Stat>() {
            if stat.dtype(canonical.dtype()).is_none() {
                continue;
            }
            assert_eq!(
                canonical_stats.get(stat),
                stats_ref.get(stat),
                "stat mismatch {stat}"
            );
        }
        Ok(())
    }

    #[test]
    fn test_canonicalize_scalar_values() {
        let f16_value = f16::from_f32(5.722046e-6);
        let f16_scalar = Scalar::primitive(f16_value, Nullability::NonNullable);

        // Create a ConstantArray with the f16 scalar
        let const_array = ConstantArray::new(f16_scalar.clone(), 1).into_array();
        #[expect(deprecated)]
        let canonical_const = const_array.to_primitive();

        // Verify the scalar value is preserved through canonicalization
        assert_eq!(
            canonical_const
                .execute_scalar(0, &mut LEGACY_SESSION.create_execution_ctx())
                .unwrap(),
            f16_scalar
        );
    }

    #[test]
    fn test_canonicalize_lists() -> VortexResult<()> {
        let list_scalar = Scalar::list(
            Arc::new(DType::Primitive(PType::U64, Nullability::NonNullable)),
            vec![1u64.into(), 2u64.into()],
            Nullability::NonNullable,
        );
        let const_array = ConstantArray::new(list_scalar, 2).into_array();
        #[expect(deprecated)]
        let canonical_const = const_array.to_listview();
        let list_array = canonical_const.rebuild(ListViewRebuildMode::MakeZeroCopyToList)?;
        assert_arrays_eq!(
            #[expect(deprecated)]
            list_array.elements().to_primitive(),
            PrimitiveArray::from_iter([1u64, 2, 1, 2])
        );
        assert_arrays_eq!(
            #[expect(deprecated)]
            list_array.offsets().to_primitive(),
            PrimitiveArray::from_iter([0u64, 2])
        );
        assert_arrays_eq!(
            #[expect(deprecated)]
            list_array.sizes().to_primitive(),
            PrimitiveArray::from_iter([2u64, 2])
        );
        Ok(())
    }

    #[test]
    fn test_canonicalize_empty_list() {
        let list_scalar = Scalar::list(
            Arc::new(DType::Primitive(PType::U64, Nullability::NonNullable)),
            vec![],
            Nullability::NonNullable,
        );
        let const_array = ConstantArray::new(list_scalar, 2).into_array();
        #[expect(deprecated)]
        let canonical_const = const_array.to_listview();
        #[expect(deprecated)]
        let elements_prim = canonical_const.elements().to_primitive();
        assert!(elements_prim.is_empty());
        assert_arrays_eq!(
            #[expect(deprecated)]
            canonical_const.offsets().to_primitive(),
            PrimitiveArray::from_iter([0u64, 0])
        );
        assert_arrays_eq!(
            #[expect(deprecated)]
            canonical_const.sizes().to_primitive(),
            PrimitiveArray::from_iter([0u64, 0])
        );
    }

    #[test]
    fn test_canonicalize_null_list() {
        let list_scalar = Scalar::null(DType::List(
            Arc::new(DType::Primitive(PType::U64, Nullability::NonNullable)),
            Nullability::Nullable,
        ));
        let const_array = ConstantArray::new(list_scalar, 2).into_array();
        #[expect(deprecated)]
        let canonical_const = const_array.to_listview();
        #[expect(deprecated)]
        let elements_prim = canonical_const.elements().to_primitive();
        assert!(elements_prim.is_empty());
        assert_arrays_eq!(
            #[expect(deprecated)]
            canonical_const.offsets().to_primitive(),
            PrimitiveArray::from_iter([0u64, 0])
        );
        assert_arrays_eq!(
            #[expect(deprecated)]
            canonical_const.sizes().to_primitive(),
            PrimitiveArray::from_iter([0u64, 0])
        );
    }

    #[test]
    fn test_canonicalize_nullable_struct() {
        let array = ConstantArray::new(
            Scalar::null(DType::struct_(
                [(
                    "non_null_field",
                    DType::Primitive(PType::I8, Nullability::NonNullable),
                )],
                Nullability::Nullable,
            )),
            3,
        );

        #[expect(deprecated)]
        let struct_array = array.as_array().to_struct();
        assert_eq!(struct_array.len(), 3);
        assert_eq!(
            struct_array
                .valid_count(&mut LEGACY_SESSION.create_execution_ctx())
                .unwrap(),
            0
        );

        let field = struct_array
            .unmasked_field_by_name("non_null_field")
            .unwrap();

        assert_eq!(
            field.dtype(),
            &DType::Primitive(PType::I8, Nullability::NonNullable)
        );
    }

    #[test]
    fn test_canonicalize_fixed_size_list_non_null() {
        // Test with a non-null fixed-size list constant.
        let fsl_scalar = Scalar::fixed_size_list(
            Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
            vec![
                Scalar::primitive(10i32, Nullability::NonNullable),
                Scalar::primitive(20i32, Nullability::NonNullable),
                Scalar::primitive(30i32, Nullability::NonNullable),
            ],
            Nullability::NonNullable,
        );

        let const_array = ConstantArray::new(fsl_scalar, 4).into_array();
        #[expect(deprecated)]
        let canonical = const_array.to_fixed_size_list();

        assert_eq!(canonical.len(), 4);
        assert_eq!(canonical.list_size(), 3);
        assert!(matches!(canonical.validity(), Ok(Validity::NonNullable)));

        // Check that each list is [10, 20, 30].
        for i in 0..4 {
            let list = canonical.fixed_size_list_elements_at(i).unwrap();
            #[expect(deprecated)]
            let list_primitive = list.to_primitive();
            assert_arrays_eq!(list_primitive, PrimitiveArray::from_iter([10i32, 20, 30]));
        }
    }

    #[test]
    fn test_canonicalize_fixed_size_list_nullable() {
        // Test with a nullable but non-null fixed-size list constant.
        let fsl_scalar = Scalar::fixed_size_list(
            Arc::new(DType::Primitive(PType::F64, Nullability::NonNullable)),
            vec![
                Scalar::primitive(1.5f64, Nullability::NonNullable),
                Scalar::primitive(2.5f64, Nullability::NonNullable),
            ],
            Nullability::Nullable,
        );

        let const_array = ConstantArray::new(fsl_scalar, 3).into_array();
        #[expect(deprecated)]
        let canonical = const_array.to_fixed_size_list();

        assert_eq!(canonical.len(), 3);
        assert_eq!(canonical.list_size(), 2);
        assert!(matches!(canonical.validity(), Ok(Validity::AllValid)));

        // Check elements.
        #[expect(deprecated)]
        let elements = canonical.elements().to_primitive();
        assert_arrays_eq!(
            elements,
            PrimitiveArray::from_iter([1.5f64, 2.5, 1.5, 2.5, 1.5, 2.5])
        );
    }

    #[test]
    fn test_canonicalize_fixed_size_list_null() {
        // Test with a null fixed-size list constant.
        let fsl_scalar = Scalar::null(DType::FixedSizeList(
            Arc::new(DType::Primitive(PType::U64, Nullability::NonNullable)),
            4,
            Nullability::Nullable,
        ));

        let const_array = ConstantArray::new(fsl_scalar, 5).into_array();
        #[expect(deprecated)]
        let canonical = const_array.to_fixed_size_list();

        assert_eq!(canonical.len(), 5);
        assert_eq!(canonical.list_size(), 4);
        assert!(matches!(canonical.validity(), Ok(Validity::AllInvalid)));

        // Elements should be defaults (zeros).
        #[expect(deprecated)]
        let elements = canonical.elements().to_primitive();
        assert_eq!(elements.len(), 20); // 5 lists * 4 elements each
        assert!(elements.as_slice::<u64>().iter().all(|&x| x == 0));
    }

    #[test]
    fn test_canonicalize_fixed_size_list_empty() {
        // Test with size-0 lists (edge case).
        let fsl_scalar = Scalar::fixed_size_list(
            Arc::new(DType::Primitive(PType::I8, Nullability::NonNullable)),
            vec![],
            Nullability::NonNullable,
        );

        let const_array = ConstantArray::new(fsl_scalar, 10).into_array();
        #[expect(deprecated)]
        let canonical = const_array.to_fixed_size_list();

        assert_eq!(canonical.len(), 10);
        assert_eq!(canonical.list_size(), 0);
        assert!(matches!(canonical.validity(), Ok(Validity::NonNullable)));

        // Elements array should be empty.
        assert!(canonical.elements().is_empty());
    }

    #[test]
    fn test_canonicalize_fixed_size_list_nested() {
        // Test with nested data types (list of strings).
        let fsl_scalar = Scalar::fixed_size_list(
            Arc::new(DType::Utf8(Nullability::NonNullable)),
            vec![Scalar::from("hello"), Scalar::from("world")],
            Nullability::NonNullable,
        );

        let const_array = ConstantArray::new(fsl_scalar, 2).into_array();
        #[expect(deprecated)]
        let canonical = const_array.to_fixed_size_list();

        assert_eq!(canonical.len(), 2);
        assert_eq!(canonical.list_size(), 2);

        // Check elements are repeated correctly.
        #[expect(deprecated)]
        let elements = canonical.elements().to_varbinview();
        assert_eq!(
            elements
                .execute_scalar(0, &mut LEGACY_SESSION.create_execution_ctx())
                .unwrap(),
            "hello".into()
        );
        assert_eq!(
            elements
                .execute_scalar(1, &mut LEGACY_SESSION.create_execution_ctx())
                .unwrap(),
            "world".into()
        );
        assert_eq!(
            elements
                .execute_scalar(2, &mut LEGACY_SESSION.create_execution_ctx())
                .unwrap(),
            "hello".into()
        );
        assert_eq!(
            elements
                .execute_scalar(3, &mut LEGACY_SESSION.create_execution_ctx())
                .unwrap(),
            "world".into()
        );
    }

    #[test]
    fn test_canonicalize_fixed_size_list_single_element() {
        // Test with a single-element list.
        let fsl_scalar = Scalar::fixed_size_list(
            Arc::new(DType::Primitive(PType::I16, Nullability::NonNullable)),
            vec![Scalar::primitive(42i16, Nullability::NonNullable)],
            Nullability::NonNullable,
        );

        let const_array = ConstantArray::new(fsl_scalar, 1).into_array();
        #[expect(deprecated)]
        let canonical = const_array.to_fixed_size_list();

        assert_eq!(canonical.len(), 1);
        assert_eq!(canonical.list_size(), 1);

        #[expect(deprecated)]
        let elements = canonical.elements().to_primitive();
        assert_arrays_eq!(elements, PrimitiveArray::from_iter([42i16]));
    }

    #[test]
    fn test_canonicalize_fixed_size_list_with_null_elements() {
        // Test FSL with nullable element type where some elements are null.
        let fsl_scalar = Scalar::fixed_size_list(
            Arc::new(DType::Primitive(PType::I32, Nullability::Nullable)),
            vec![
                Scalar::primitive(100i32, Nullability::Nullable),
                Scalar::null(DType::Primitive(PType::I32, Nullability::Nullable)),
                Scalar::primitive(200i32, Nullability::Nullable),
            ],
            Nullability::NonNullable,
        );

        let const_array = ConstantArray::new(fsl_scalar, 3).into_array();
        #[expect(deprecated)]
        let canonical = const_array.to_fixed_size_list();

        assert_eq!(canonical.len(), 3);
        assert_eq!(canonical.list_size(), 3);
        assert!(matches!(canonical.validity(), Ok(Validity::NonNullable)));

        // Check elements including nulls.
        #[expect(deprecated)]
        let elements = canonical.elements().to_primitive();
        assert_eq!(
            elements
                .execute_scalar(0, &mut LEGACY_SESSION.create_execution_ctx())
                .unwrap(),
            Scalar::from(100i32)
        );
        assert_eq!(
            elements
                .execute_scalar(1, &mut LEGACY_SESSION.create_execution_ctx())
                .unwrap(),
            Scalar::null(DType::Primitive(PType::I32, Nullability::Nullable))
        );
        assert_eq!(
            elements
                .execute_scalar(2, &mut LEGACY_SESSION.create_execution_ctx())
                .unwrap(),
            Scalar::from(200i32)
        );

        // Check element validity.
        let element_validity = elements
            .validity()
            .vortex_expect("constant canonical element validity should be derivable");
        assert!(element_validity.is_valid(0).unwrap());
        assert!(!element_validity.is_valid(1).unwrap());
        assert!(element_validity.is_valid(2).unwrap());

        // Pattern should repeat.
        assert!(element_validity.is_valid(3).unwrap());
        assert!(!element_validity.is_valid(4).unwrap());
        assert!(element_validity.is_valid(5).unwrap());
    }

    #[test]
    fn test_canonicalize_fixed_size_list_large() {
        // Test with a large constant array.
        let fsl_scalar = Scalar::fixed_size_list(
            Arc::new(DType::Primitive(PType::U8, Nullability::NonNullable)),
            vec![
                Scalar::primitive(1u8, Nullability::NonNullable),
                Scalar::primitive(2u8, Nullability::NonNullable),
                Scalar::primitive(3u8, Nullability::NonNullable),
                Scalar::primitive(4u8, Nullability::NonNullable),
                Scalar::primitive(5u8, Nullability::NonNullable),
            ],
            Nullability::NonNullable,
        );

        let const_array = ConstantArray::new(fsl_scalar, 1000).into_array();
        #[expect(deprecated)]
        let canonical = const_array.to_fixed_size_list();

        assert_eq!(canonical.len(), 1000);
        assert_eq!(canonical.list_size(), 5);

        #[expect(deprecated)]
        let elements = canonical.elements().to_primitive();
        assert_eq!(elements.len(), 5000);

        // Check pattern repeats correctly.
        for i in 0..1000 {
            let base = i * 5;
            assert_eq!(
                elements
                    .execute_scalar(base, &mut LEGACY_SESSION.create_execution_ctx())
                    .unwrap(),
                Scalar::from(1u8)
            );
            assert_eq!(
                elements
                    .execute_scalar(base + 1, &mut LEGACY_SESSION.create_execution_ctx())
                    .unwrap(),
                Scalar::from(2u8)
            );
            assert_eq!(
                elements
                    .execute_scalar(base + 2, &mut LEGACY_SESSION.create_execution_ctx())
                    .unwrap(),
                Scalar::from(3u8)
            );
            assert_eq!(
                elements
                    .execute_scalar(base + 3, &mut LEGACY_SESSION.create_execution_ctx())
                    .unwrap(),
                Scalar::from(4u8)
            );
            assert_eq!(
                elements
                    .execute_scalar(base + 4, &mut LEGACY_SESSION.create_execution_ctx())
                    .unwrap(),
                Scalar::from(5u8)
            );
        }
    }
}
