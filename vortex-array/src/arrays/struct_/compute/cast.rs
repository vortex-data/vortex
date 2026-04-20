// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::ConstantArray;
use crate::arrays::Struct;
use crate::arrays::StructArray;
use crate::arrays::struct_::StructArrayExt;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::scalar::Scalar;
use crate::scalar_fn::fns::cast::CastKernel;
use crate::scalar_fn::fns::cast::CastMode;
use crate::scalar_fn::fns::cast::CastOptions;

impl CastKernel for Struct {
    fn cast(
        array: ArrayView<'_, Struct>,
        dtype: &DType,
        options: &CastOptions,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let Some(target_sdtype) = dtype.as_struct_fields_opt() else {
            return Ok(None);
        };

        let source_sdtype = array.struct_fields();

        let cast_fields = match options.mode() {
            CastMode::ByPosition => {
                vortex_ensure!(
                    target_sdtype.nfields() == source_sdtype.nfields(),
                    "CAST by position requires source ({}) and target ({}) struct to have the same number of fields",
                    source_sdtype.nfields(),
                    target_sdtype.nfields()
                );
                array
                    .iter_unmasked_fields()
                    .zip_eq(target_sdtype.fields())
                    .map(|(field, target_type)| field.cast_opts(target_type, *options))
                    .try_collect()?
            }
            CastMode::ByName => {
                vortex_ensure!(
                    source_sdtype.names().iter().all_unique(),
                    "CAST by name requires unique field names in the source struct; \
                     use by-position mode for structs with duplicate field names"
                );
                vortex_ensure!(
                    target_sdtype.names().iter().all_unique(),
                    "CAST by name requires unique field names in the target struct; \
                     use by-position mode for structs with duplicate field names"
                );
                let mut cast_fields = Vec::with_capacity(target_sdtype.nfields());
                for (target_name, target_type) in
                    target_sdtype.names().iter().zip_eq(target_sdtype.fields())
                {
                    match source_sdtype.find(target_name) {
                        None => {
                            vortex_ensure!(
                                target_type.is_nullable(),
                                "Cannot add non-nullable field '{}' during struct cast",
                                target_name
                            );
                            cast_fields.push(
                                ConstantArray::new(Scalar::null(target_type), array.len())
                                    .into_array(),
                            );
                        }
                        Some(src_field_idx) => {
                            cast_fields.push(
                                array
                                    .unmasked_field(src_field_idx)
                                    .cast_opts(target_type, *options)?,
                            );
                        }
                    }
                }
                cast_fields
            }
        };

        let validity = array
            .validity()?
            .cast_nullability(dtype.nullability(), array.len())?;

        StructArray::try_new(
            target_sdtype.names().clone(),
            cast_fields,
            array.len(),
            validity,
        )
        .map(|a| Some(a.into_array()))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_buffer::buffer;
    use vortex_session::VortexSession;

    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::StructArray;
    use crate::arrays::VarBinArray;
    use crate::arrays::struct_::StructArrayExt;
    use crate::builtins::ArrayBuiltins;
    use crate::compute::conformance::cast::test_cast_conformance;
    use crate::dtype::DType;
    use crate::dtype::DecimalDType;
    use crate::dtype::FieldNames;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::session::ArraySession;
    use crate::validity::Validity;

    static SESSION: LazyLock<VortexSession> =
        LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

    #[rstest]
    #[case(create_test_struct(false))]
    #[case(create_test_struct(true))]
    #[case(create_nested_struct())]
    #[case(create_simple_struct())]
    fn test_cast_struct_conformance(#[case] array: StructArray) {
        test_cast_conformance(&array.into_array());
    }

    fn create_test_struct(nullable: bool) -> StructArray {
        let names = FieldNames::from(["a", "b"]);

        let a = buffer![1i32, 2, 3].into_array();
        let b = VarBinArray::from_iter(
            vec![Some("x"), None, Some("z")],
            DType::Utf8(Nullability::Nullable),
        )
        .into_array();

        StructArray::try_new(
            names,
            vec![a, b],
            3,
            if nullable {
                Validity::AllValid
            } else {
                Validity::NonNullable
            },
        )
        .unwrap()
    }

    fn create_nested_struct() -> StructArray {
        // Create inner struct
        let inner_names = FieldNames::from(["x", "y"]);

        let x = buffer![1.0f32, 2.0, 3.0].into_array();
        let y = buffer![4.0f32, 5.0, 6.0].into_array();
        let inner_struct = StructArray::try_new(inner_names, vec![x, y], 3, Validity::NonNullable)
            .unwrap()
            .into_array();

        // Create outer struct with inner struct as a field
        let outer_names: FieldNames = ["id", "point"].into();
        // Outer struct would have fields: id (I64) and point (inner struct)

        let ids = buffer![100i64, 200, 300].into_array();

        StructArray::try_new(
            outer_names,
            vec![ids, inner_struct],
            3,
            Validity::NonNullable,
        )
        .unwrap()
    }

    fn create_simple_struct() -> StructArray {
        let names = FieldNames::from(["value"]);
        // Simple struct with a single U8 field

        let values = buffer![42u8].into_array();

        StructArray::try_new(names, vec![values], 1, Validity::NonNullable).unwrap()
    }

    #[test]
    fn cast_nullable_all_invalid() {
        let empty_struct = StructArray::try_new(
            FieldNames::from(["a"]),
            vec![PrimitiveArray::new::<i32>(buffer![], Validity::AllInvalid).into_array()],
            0,
            Validity::AllInvalid,
        )
        .unwrap()
        .into_array();

        let target_dtype = DType::struct_(
            [("a", DType::Primitive(PType::I32, Nullability::NonNullable))],
            Nullability::NonNullable,
        );

        let result = empty_struct.cast(target_dtype.clone()).unwrap();
        assert_eq!(result.dtype(), &target_dtype);
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn cast_by_position_handles_duplicate_field_names() {
        use crate::assert_arrays_eq;
        use crate::scalar_fn::fns::cast::CastOptions;

        let names = FieldNames::from(["a", "a"]);
        let field1 = buffer![1i32, 2, 3].into_array();
        let field2 = buffer![10i64, 20, 30].into_array();

        let struct_array =
            StructArray::try_new(names, vec![field1, field2], 3, Validity::NonNullable).unwrap();

        let target_dtype = struct_array.dtype().as_nullable();

        let result = struct_array
            .into_array()
            .cast_opts(target_dtype.clone(), CastOptions::by_position())
            .unwrap()
            .execute::<StructArray>(&mut SESSION.create_execution_ctx())
            .unwrap();
        assert_eq!(result.dtype(), &target_dtype);
        assert_eq!(result.len(), 3);
        assert_eq!(result.struct_fields().nfields(), 2);
        assert_arrays_eq!(result.unmasked_field(0), buffer![1i32, 2, 3].into_array());
        assert_arrays_eq!(
            result.unmasked_field(1),
            buffer![10i64, 20, 30].into_array()
        );
    }

    #[test]
    fn cast_by_name_duplicate_source_names_fails() {
        use crate::scalar_fn::fns::cast::CastOptions;

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
    fn cast_by_name_duplicate_target_names_fails() {
        use crate::scalar_fn::fns::cast::CastOptions;

        let source = StructArray::try_new(
            FieldNames::from(["a", "b"]),
            vec![buffer![1i32].into_array(), buffer![10i64].into_array()],
            1,
            Validity::NonNullable,
        )
        .unwrap();

        let target = DType::struct_(
            [
                ("a", DType::Primitive(PType::I32, Nullability::NonNullable)),
                ("a", DType::Primitive(PType::I64, Nullability::NonNullable)),
            ],
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

    #[test]
    fn cast_add_fields() {
        let names = FieldNames::from(["a", "b"]);
        let field1 = buffer![1i32, 2, 3].into_array();
        let field2 = buffer![10i64, 20, 30].into_array();
        let target_dtype = DType::struct_(
            [
                ("a", field1.dtype().clone()),
                ("b", field2.dtype().clone()),
                (
                    "c",
                    DType::Decimal(DecimalDType::new(38, 10), Nullability::Nullable),
                ),
            ],
            Nullability::NonNullable,
        );

        let struct_array =
            StructArray::try_new(names, vec![field1, field2], 3, Validity::NonNullable).unwrap();

        let result = struct_array
            .into_array()
            .cast(target_dtype.clone())
            .unwrap();
        assert_eq!(result.dtype(), &target_dtype);
        assert_eq!(result.len(), 3);
        let nfields = result
            .execute::<StructArray>(&mut SESSION.create_execution_ctx())
            .unwrap()
            .struct_fields()
            .nfields();
        assert_eq!(nfields, 3);
    }

    #[test]
    fn cast_by_position_renames_fields() {
        use crate::assert_arrays_eq;
        use crate::scalar_fn::fns::cast::CastOptions;

        // Source: {a, b}, Target: {x, y} - same number of fields, different names.
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

        let target = DType::struct_(
            [
                ("x", DType::Primitive(PType::I32, Nullability::NonNullable)),
                ("y", DType::Primitive(PType::I64, Nullability::NonNullable)),
            ],
            Nullability::NonNullable,
        );

        let result = source
            .into_array()
            .cast_opts(target.clone(), CastOptions::by_position())
            .unwrap()
            .execute::<StructArray>(&mut SESSION.create_execution_ctx())
            .unwrap();

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
    fn cast_by_position_field_count_mismatch_fails() {
        use crate::scalar_fn::fns::cast::CastOptions;

        let source = StructArray::try_new(
            FieldNames::from(["a", "b"]),
            vec![buffer![1i32].into_array(), buffer![10i64].into_array()],
            1,
            Validity::NonNullable,
        )
        .unwrap();

        let target = DType::struct_(
            [("x", DType::Primitive(PType::I32, Nullability::NonNullable))],
            Nullability::NonNullable,
        );

        assert!(
            source
                .into_array()
                .cast_opts(target, CastOptions::by_position())
                .is_err()
        );
    }
}
