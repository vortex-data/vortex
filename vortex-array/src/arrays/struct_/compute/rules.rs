// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::ConstantArray;
use crate::arrays::Struct;
use crate::arrays::StructArray;
use crate::arrays::dict::TakeReduceAdaptor;
use crate::arrays::scalar_fn::ExactScalarFn;
use crate::arrays::scalar_fn::ScalarFnArrayView;
use crate::arrays::scalar_fn::ScalarFnFactoryExt;
use crate::arrays::slice::SliceReduceAdaptor;
use crate::arrays::struct_::StructArrayExt;
use crate::builtins::ArrayBuiltins;
use crate::optimizer::rules::ArrayParentReduceRule;
use crate::optimizer::rules::ParentRuleSet;
use crate::scalar_fn::EmptyOptions;
use crate::scalar_fn::fns::cast::Cast;
use crate::scalar_fn::fns::cast::CastMode;
use crate::scalar_fn::fns::get_item::GetItem;
use crate::scalar_fn::fns::mask::Mask;
use crate::scalar_fn::fns::mask::MaskReduceAdaptor;
use crate::validity::Validity;

pub(crate) const PARENT_RULES: ParentRuleSet<Struct> = ParentRuleSet::new(&[
    ParentRuleSet::lift(&StructCastPushDownRule),
    ParentRuleSet::lift(&StructGetItemRule),
    ParentRuleSet::lift(&MaskReduceAdaptor(Struct)),
    ParentRuleSet::lift(&SliceReduceAdaptor(Struct)),
    ParentRuleSet::lift(&TakeReduceAdaptor(Struct)),
]);

/// Rule to push down cast into struct fields.
///
/// In [`CastMode::ByName`] this also supports schema evolution by allowing new nullable
/// fields to appear in the target type (filled with nulls) and source-only fields to be
/// dropped. In [`CastMode::ByPosition`] the source and target struct must have the same
/// number of fields and they are matched positionally regardless of name.
#[derive(Debug)]
struct StructCastPushDownRule;
impl ArrayParentReduceRule<Struct> for StructCastPushDownRule {
    type Parent = ExactScalarFn<Cast>;

    fn reduce_parent(
        &self,
        array: ArrayView<'_, Struct>,
        parent: ScalarFnArrayView<Cast>,
        _child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        let dtype = parent.options.target();
        let options = parent.options.options();
        let Some(target_fields) = dtype.as_struct_fields_opt() else {
            return Ok(None);
        };

        let source_fields = array.struct_fields();

        let new_fields = match options.mode() {
            CastMode::ByPosition => {
                vortex_ensure!(
                    source_fields.nfields() == target_fields.nfields(),
                    "CAST by position requires source ({}) and target ({}) struct to have the same number of fields",
                    source_fields.nfields(),
                    target_fields.nfields()
                );
                (0..source_fields.nfields())
                    .zip_eq(target_fields.fields())
                    .map(|(src_idx, target_dtype)| {
                        array
                            .unmasked_field(src_idx)
                            .cast_opts(target_dtype, *options)
                    })
                    .try_collect()?
            }
            CastMode::ByName => {
                vortex_ensure!(
                    source_fields.names().iter().all_unique(),
                    "CAST by name requires unique field names in the source struct; \
                     use by-position mode for structs with duplicate field names"
                );
                vortex_ensure!(
                    target_fields.names().iter().all_unique(),
                    "CAST by name requires unique field names in the target struct; \
                     use by-position mode for structs with duplicate field names"
                );
                let mut new_fields = Vec::with_capacity(target_fields.nfields());
                for (target_name, target_dtype) in
                    target_fields.names().iter().zip(target_fields.fields())
                {
                    match array.unmasked_field_by_name(target_name).ok() {
                        Some(field) => {
                            new_fields.push(field.cast_opts(target_dtype, *options)?);
                        }
                        None => {
                            vortex_ensure!(
                                target_dtype.is_nullable(),
                                "Cannot add non-nullable field '{}' during struct cast",
                                target_name
                            );
                            new_fields.push(
                                ConstantArray::new(
                                    crate::scalar::Scalar::null(target_dtype),
                                    array.len(),
                                )
                                .into_array(),
                            );
                        }
                    }
                }
                new_fields
            }
        };

        let validity = array
            .validity()?
            .cast_nullability(dtype.nullability(), array.len())?;

        let new_struct = unsafe {
            StructArray::new_unchecked(new_fields, target_fields.clone(), array.len(), validity)
        };

        Ok(Some(new_struct.into_array()))
    }
}

/// Rule to flatten get_item from struct by field name
#[derive(Debug)]
pub(crate) struct StructGetItemRule;
impl ArrayParentReduceRule<Struct> for StructGetItemRule {
    type Parent = ExactScalarFn<GetItem>;

    fn reduce_parent(
        &self,
        child: ArrayView<'_, Struct>,
        parent: ScalarFnArrayView<'_, GetItem>,
        _child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        let field_name = parent.options;
        let field = child
            .unmasked_field_by_name_opt(field_name)
            .ok_or_else(|| {
                vortex_err!(
                    "Field '{}' missing from struct array {}",
                    field_name,
                    child.struct_fields().names()
                )
            })?;

        match child.validity()? {
            Validity::NonNullable | Validity::AllValid => {
                // If the struct is non-nullable or all valid, the field's validity is unchanged
                Ok(Some(field.clone()))
            }
            Validity::AllInvalid => {
                // If everything is invalid, the field is also all invalid
                Ok(Some(
                    ConstantArray::new(
                        crate::scalar::Scalar::null(field.dtype().clone()),
                        field.len(),
                    )
                    .into_array(),
                ))
            }
            Validity::Array(mask) => {
                // If the validity is an array, we need to combine it with the field's validity
                Mask.try_new_array(field.len(), EmptyOptions, [field.clone(), mask])
                    .map(Some)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;

    use crate::IntoArray;
    use crate::arrays::StructArray;
    use crate::arrays::VarBinViewArray;
    use crate::arrays::struct_::StructArrayExt;
    use crate::arrays::struct_::compute::rules::ConstantArray;
    use crate::assert_arrays_eq;
    use crate::builtins::ArrayBuiltins;
    #[expect(deprecated)]
    use crate::canonical::ToCanonical as _;
    use crate::dtype::DType;
    use crate::dtype::FieldNames;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::dtype::StructFields;
    use crate::scalar::Scalar;
    use crate::validity::Validity;

    #[test]
    fn test_struct_cast_field_reorder() {
        // Source: {a, b}, Target: {c, b, a} - reordered + new null field
        let source = StructArray::try_new(
            FieldNames::from(["a", "b"]),
            vec![
                VarBinViewArray::from_iter_str(["A"]).into_array(),
                VarBinViewArray::from_iter_str(["B"]).into_array(),
            ],
            1,
            Validity::NonNullable,
        )
        .unwrap();

        let utf8_null = DType::Utf8(Nullability::Nullable);
        let target = DType::Struct(
            StructFields::new(
                FieldNames::from(["c", "b", "a"]),
                vec![utf8_null.clone(); 3],
            ),
            Nullability::NonNullable,
        );

        // Use `ArrayBuiltins::cast` which goes through the optimizer and applies
        // `StructCastPushDownRule`.
        #[expect(deprecated)]
        let result = source.into_array().cast(target).unwrap().to_struct();
        assert_arrays_eq!(
            result.unmasked_field_by_name("a").unwrap(),
            VarBinViewArray::from_iter_nullable_str([Some("A")])
        );
        assert_arrays_eq!(
            result.unmasked_field_by_name("b").unwrap(),
            VarBinViewArray::from_iter_nullable_str([Some("B")])
        );
        assert_arrays_eq!(
            result.unmasked_field_by_name("c").unwrap(),
            ConstantArray::new(Scalar::null(utf8_null), 1)
        );
    }

    /// Regression test: casting a struct to a non-struct DType must not panic. Previously,
    /// `StructCastPushDownRule` called `as_struct_fields()` which panics on non-struct types.
    #[test]
    fn cast_struct_to_non_struct_does_not_panic() {
        let source = StructArray::try_new(
            FieldNames::from(["x"]),
            vec![buffer![1i32, 2, 3].into_array()],
            3,
            Validity::NonNullable,
        )
        .unwrap();

        // Casting a struct to a primitive type should not panic. Before the fix,
        // `StructCastPushDownRule` would panic via `as_struct_fields()` on the non-struct target.
        let result = source
            .into_array()
            .cast(DType::Primitive(PType::I32, Nullability::NonNullable));
        // Whether this errors or succeeds depends on execution, but the key invariant is that the
        // optimizer rule does not panic.
        if let Ok(arr) = &result {
            assert_eq!(
                arr.dtype(),
                &DType::Primitive(PType::I32, Nullability::NonNullable)
            );
        }
    }

    #[test]
    fn cast_struct_drop_field() {
        // Casting to a struct with a subset of fields should succeed.
        let source = StructArray::try_new(
            FieldNames::from(["a", "b", "c"]),
            vec![
                buffer![1i32, 2, 3].into_array(),
                buffer![10i64, 20, 30].into_array(),
                buffer![100u8, 200, 255].into_array(),
            ],
            3,
            Validity::NonNullable,
        )
        .unwrap();

        let target = DType::Struct(
            StructFields::new(
                FieldNames::from(["a", "c"]),
                vec![
                    DType::Primitive(PType::I32, Nullability::NonNullable),
                    DType::Primitive(PType::U8, Nullability::NonNullable),
                ],
            ),
            Nullability::NonNullable,
        );

        #[expect(deprecated)]
        let result = source.into_array().cast(target).unwrap().to_struct();
        assert_eq!(result.unmasked_fields().len(), 2);
        assert_arrays_eq!(
            result.unmasked_field_by_name("a").unwrap(),
            buffer![1i32, 2, 3].into_array()
        );
        assert_arrays_eq!(
            result.unmasked_field_by_name("c").unwrap(),
            buffer![100u8, 200, 255].into_array()
        );
    }

    #[test]
    fn cast_struct_field_type_widening() {
        // Casting struct fields to wider types (i32 -> i64).
        let source = StructArray::try_new(
            FieldNames::from(["val"]),
            vec![buffer![1i32, 2, 3].into_array()],
            3,
            Validity::NonNullable,
        )
        .unwrap();

        let target = DType::Struct(
            StructFields::new(
                FieldNames::from(["val"]),
                vec![DType::Primitive(PType::I64, Nullability::NonNullable)],
            ),
            Nullability::NonNullable,
        );

        #[expect(deprecated)]
        let result = source.into_array().cast(target).unwrap().to_struct();
        assert_eq!(
            result.unmasked_field_by_name("val").unwrap().dtype(),
            &DType::Primitive(PType::I64, Nullability::NonNullable)
        );
        assert_arrays_eq!(
            result.unmasked_field_by_name("val").unwrap(),
            buffer![1i64, 2, 3].into_array()
        );
    }

    #[test]
    fn cast_struct_add_non_nullable_field_fails() {
        // Adding a non-nullable field via cast should fail.
        let source = StructArray::try_new(
            FieldNames::from(["a"]),
            vec![buffer![1i32].into_array()],
            1,
            Validity::NonNullable,
        )
        .unwrap();

        let target = DType::Struct(
            StructFields::new(
                FieldNames::from(["a", "b"]),
                vec![
                    DType::Primitive(PType::I32, Nullability::NonNullable),
                    DType::Primitive(PType::I32, Nullability::NonNullable),
                ],
            ),
            Nullability::NonNullable,
        );

        assert!(source.into_array().cast(target).is_err());
    }

    #[test]
    fn cast_struct_by_position_ignores_names() {
        use crate::scalar_fn::fns::cast::CastOptions;

        // Source: {a, b}, Target: {x, y} - by_position should match by index, not name.
        let source = StructArray::try_new(
            FieldNames::from(["a", "b"]),
            vec![
                buffer![1i32, 2, 3].into_array(),
                buffer![10i64, 20, 30].into_array(),
            ],
            3,
            Validity::NonNullable,
        )
        .unwrap();

        let target = DType::Struct(
            StructFields::new(
                FieldNames::from(["x", "y"]),
                vec![
                    DType::Primitive(PType::I32, Nullability::NonNullable),
                    DType::Primitive(PType::I64, Nullability::NonNullable),
                ],
            ),
            Nullability::NonNullable,
        );

        let result = source
            .into_array()
            .cast_opts(target.clone(), CastOptions::by_position())
            .unwrap()
            .to_struct();
        assert_eq!(result.dtype(), &target);
        assert_arrays_eq!(
            result.unmasked_field_by_name("x").unwrap(),
            buffer![1i32, 2, 3].into_array()
        );
        assert_arrays_eq!(
            result.unmasked_field_by_name("y").unwrap(),
            buffer![10i64, 20, 30].into_array()
        );
    }

    #[test]
    fn cast_struct_by_position_field_count_mismatch_fails() {
        use crate::scalar_fn::fns::cast::CastOptions;

        let source = StructArray::try_new(
            FieldNames::from(["a", "b"]),
            vec![buffer![1i32].into_array(), buffer![10i64].into_array()],
            1,
            Validity::NonNullable,
        )
        .unwrap();

        let target = DType::Struct(
            StructFields::new(
                FieldNames::from(["x"]),
                vec![DType::Primitive(PType::I32, Nullability::NonNullable)],
            ),
            Nullability::NonNullable,
        );

        assert!(
            source
                .into_array()
                .cast_opts(target, CastOptions::by_position())
                .is_err()
        );
    }

    #[test]
    fn cast_struct_by_position_handles_duplicate_field_names() {
        use crate::scalar_fn::fns::cast::CastOptions;

        // Source has duplicate field names; by_position matches positionally, so both fields
        // retain their original data (no collision).
        let source = StructArray::try_new(
            FieldNames::from(["a", "a"]),
            vec![
                buffer![1i32, 2, 3].into_array(),
                buffer![10i64, 20, 30].into_array(),
            ],
            3,
            Validity::NonNullable,
        )
        .unwrap();

        let target = source.dtype().as_nullable();

        let result = source
            .into_array()
            .cast_opts(target.clone(), CastOptions::by_position())
            .unwrap()
            .to_struct();
        assert_eq!(result.dtype(), &target);
        assert_arrays_eq!(result.unmasked_field(0), buffer![1i32, 2, 3].into_array());
        assert_arrays_eq!(
            result.unmasked_field(1),
            buffer![10i64, 20, 30].into_array()
        );
    }

    #[test]
    fn cast_struct_by_name_duplicate_source_names_fails() {
        use crate::scalar_fn::fns::cast::CastOptions;

        // `find` cannot disambiguate duplicate source names, so by_name must reject them.
        let source = StructArray::try_new(
            FieldNames::from(["a", "a"]),
            vec![buffer![1i32].into_array(), buffer![10i64].into_array()],
            1,
            Validity::NonNullable,
        )
        .unwrap();

        let target = source.dtype().as_nullable();

        let err = source
            .into_array()
            .cast_opts(target, CastOptions::by_name())
            .unwrap_err();
        assert!(
            err.to_string().contains("unique"),
            "expected uniqueness error, got: {err}"
        );
    }

    #[test]
    fn cast_struct_by_name_duplicate_target_names_fails() {
        use crate::scalar_fn::fns::cast::CastOptions;

        // Duplicate target names would collapse silently under by_name, so we reject them.
        let source = StructArray::try_new(
            FieldNames::from(["a", "b"]),
            vec![buffer![1i32].into_array(), buffer![10i64].into_array()],
            1,
            Validity::NonNullable,
        )
        .unwrap();

        let target = DType::Struct(
            StructFields::new(
                FieldNames::from(["a", "a"]),
                vec![
                    DType::Primitive(PType::I32, Nullability::NonNullable),
                    DType::Primitive(PType::I64, Nullability::NonNullable),
                ],
            ),
            Nullability::NonNullable,
        );

        let err = source
            .into_array()
            .cast_opts(target, CastOptions::by_name())
            .unwrap_err();
        assert!(
            err.to_string().contains("unique"),
            "expected uniqueness error, got: {err}"
        );
    }
}
