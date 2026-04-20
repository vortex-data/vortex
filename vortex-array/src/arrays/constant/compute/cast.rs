// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Constant;
use crate::arrays::ConstantArray;
use crate::dtype::DType;
use crate::scalar_fn::fns::cast::CastOptions;
use crate::scalar_fn::fns::cast::CastReduce;

impl CastReduce for Constant {
    fn cast(
        array: ArrayView<'_, Constant>,
        dtype: &DType,
        options: &CastOptions,
    ) -> VortexResult<Option<ArrayRef>> {
        let scalar = array.scalar().cast_opts(dtype, *options)?;
        Ok(Some(ConstantArray::new(scalar, array.len()).into_array()))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_session::VortexSession;

    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::arrays::ConstantArray;
    use crate::arrays::StructArray;
    use crate::arrays::struct_::StructArrayExt;
    use crate::builtins::ArrayBuiltins;
    use crate::compute::conformance::cast::test_cast_conformance;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::scalar::Scalar;
    use crate::scalar_fn::fns::cast::CastOptions;
    use crate::session::ArraySession;

    static SESSION: LazyLock<VortexSession> =
        LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

    #[rstest]
    #[case(ConstantArray::new(Scalar::from(42u32), 5).into_array())]
    #[case(ConstantArray::new(Scalar::from(-100i32), 10).into_array())]
    #[case(ConstantArray::new(Scalar::from(3.5f32), 3).into_array())]
    #[case(ConstantArray::new(Scalar::from(true), 7).into_array())]
    #[case(ConstantArray::new(Scalar::null_native::<i32>(), 4).into_array())]
    #[case(ConstantArray::new(Scalar::from(255u8), 1).into_array())]
    fn test_cast_constant_conformance(#[case] array: crate::ArrayRef) {
        test_cast_conformance(&array);
    }

    fn source_struct_dtype() -> DType {
        DType::struct_(
            [
                ("a", DType::Primitive(PType::I32, Nullability::NonNullable)),
                ("b", DType::Primitive(PType::I64, Nullability::NonNullable)),
            ],
            Nullability::NonNullable,
        )
    }

    fn source_struct_scalar() -> Scalar {
        Scalar::struct_(
            source_struct_dtype(),
            vec![Scalar::from(1i32), Scalar::from(10i64)],
        )
    }

    #[test]
    fn cast_constant_struct_by_position_renames_fields() {
        let source = ConstantArray::new(source_struct_scalar(), 3).into_array();
        let target = DType::struct_(
            [
                ("x", DType::Primitive(PType::I32, Nullability::NonNullable)),
                ("y", DType::Primitive(PType::I64, Nullability::NonNullable)),
            ],
            Nullability::NonNullable,
        );

        let result = source
            .cast_opts(target.clone(), CastOptions::by_position())
            .unwrap();
        assert_eq!(result.dtype(), &target);
        assert_eq!(result.len(), 3);
        let s = result
            .execute::<StructArray>(&mut SESSION.create_execution_ctx())
            .unwrap();
        assert_eq!(
            s.unmasked_field_by_name("x")
                .unwrap()
                .execute_scalar(0, &mut SESSION.create_execution_ctx())
                .unwrap(),
            Scalar::from(1i32)
        );
        assert_eq!(
            s.unmasked_field_by_name("y")
                .unwrap()
                .execute_scalar(0, &mut SESSION.create_execution_ctx())
                .unwrap(),
            Scalar::from(10i64)
        );
    }

    #[test]
    fn cast_constant_struct_by_name_schema_evolution() {
        let source = ConstantArray::new(source_struct_scalar(), 2).into_array();
        let target = DType::struct_(
            [
                ("a", DType::Primitive(PType::I32, Nullability::NonNullable)),
                ("b", DType::Primitive(PType::I64, Nullability::NonNullable)),
                ("c", DType::Primitive(PType::I32, Nullability::Nullable)),
            ],
            Nullability::NonNullable,
        );

        let result = source
            .cast_opts(target.clone(), CastOptions::by_name())
            .unwrap();
        assert_eq!(result.dtype(), &target);
        let s = result
            .execute::<StructArray>(&mut SESSION.create_execution_ctx())
            .unwrap();
        assert_eq!(
            s.unmasked_field_by_name("a")
                .unwrap()
                .execute_scalar(0, &mut SESSION.create_execution_ctx())
                .unwrap(),
            Scalar::from(1i32)
        );
        assert_eq!(
            s.unmasked_field_by_name("b")
                .unwrap()
                .execute_scalar(0, &mut SESSION.create_execution_ctx())
                .unwrap(),
            Scalar::from(10i64)
        );
        assert!(
            s.unmasked_field_by_name("c")
                .unwrap()
                .execute_scalar(0, &mut SESSION.create_execution_ctx())
                .unwrap()
                .is_null()
        );
    }

    #[test]
    fn cast_constant_struct_by_name_reorders_fields() {
        let source = ConstantArray::new(source_struct_scalar(), 1).into_array();
        let target = DType::struct_(
            [
                ("b", DType::Primitive(PType::I64, Nullability::NonNullable)),
                ("a", DType::Primitive(PType::I32, Nullability::NonNullable)),
            ],
            Nullability::NonNullable,
        );

        let result = source
            .cast_opts(target.clone(), CastOptions::by_name())
            .unwrap();
        assert_eq!(result.dtype(), &target);
        let s = result
            .execute::<StructArray>(&mut SESSION.create_execution_ctx())
            .unwrap();
        assert_eq!(
            s.unmasked_field_by_name("a")
                .unwrap()
                .execute_scalar(0, &mut SESSION.create_execution_ctx())
                .unwrap(),
            Scalar::from(1i32)
        );
        assert_eq!(
            s.unmasked_field_by_name("b")
                .unwrap()
                .execute_scalar(0, &mut SESSION.create_execution_ctx())
                .unwrap(),
            Scalar::from(10i64)
        );
    }

    #[test]
    fn cast_constant_struct_by_name_requires_nullable_added_field() {
        let source = ConstantArray::new(source_struct_scalar(), 1).into_array();
        let target = DType::struct_(
            [
                ("a", DType::Primitive(PType::I32, Nullability::NonNullable)),
                ("b", DType::Primitive(PType::I64, Nullability::NonNullable)),
                ("c", DType::Primitive(PType::I32, Nullability::NonNullable)),
            ],
            Nullability::NonNullable,
        );

        assert!(
            source.cast_opts(target, CastOptions::by_name()).is_err(),
            "adding non-nullable field should fail"
        );
    }

    #[test]
    fn cast_constant_struct_by_position_field_count_mismatch_fails() {
        let source = ConstantArray::new(source_struct_scalar(), 1).into_array();
        let target = DType::struct_(
            [("x", DType::Primitive(PType::I32, Nullability::NonNullable))],
            Nullability::NonNullable,
        );

        assert!(
            source
                .cast_opts(target, CastOptions::by_position())
                .is_err()
        );
    }

    #[test]
    fn cast_constant_struct_by_name_duplicate_source_names_fails() {
        // Shape differs from the target so the cast cannot short-circuit via
        // `eq_ignore_nullability`; forces ByName resolution which must reject duplicate names.
        let dup_src_dtype = DType::struct_(
            [
                ("a", DType::Primitive(PType::I32, Nullability::NonNullable)),
                ("a", DType::Primitive(PType::I64, Nullability::NonNullable)),
            ],
            Nullability::NonNullable,
        );
        let source = ConstantArray::new(
            Scalar::struct_(dup_src_dtype, vec![Scalar::from(1i32), Scalar::from(10i64)]),
            1,
        )
        .into_array();

        let target = DType::struct_(
            [
                ("a", DType::Primitive(PType::I32, Nullability::NonNullable)),
                ("b", DType::Primitive(PType::I64, Nullability::NonNullable)),
            ],
            Nullability::NonNullable,
        );
        let err = source
            .cast_opts(target, CastOptions::by_name())
            .unwrap_err();
        assert!(
            err.to_string().contains("unique"),
            "expected uniqueness error, got: {err}"
        );
    }
}
