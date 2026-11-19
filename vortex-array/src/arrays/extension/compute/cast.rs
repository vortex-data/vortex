// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Extension;
use crate::arrays::ExtensionArray;
use crate::arrays::extension::ExtensionArrayExt;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::scalar_fn::fns::cast::CastReduce;

impl CastReduce for Extension {
    fn cast(array: ArrayView<'_, Extension>, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
        if !array.dtype().eq_ignore_nullability(dtype) {
            return Ok(None);
        }

        let DType::Extension(ext_dtype) = dtype else {
            unreachable!("Already verified we have an extension dtype");
        };

        let new_storage = array
            .storage_array()
            .cast(ext_dtype.storage_dtype().clone())?;

        Ok(Some(
            ExtensionArray::new(ext_dtype.clone(), new_storage).into_array(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_buffer::Buffer;
    use vortex_buffer::buffer;
    use vortex_error::VortexExpect;
    use vortex_session::VortexSession;

    use super::*;
    use crate::IntoArray;
    use crate::RecursiveCanonical;
    use crate::arrays::ConstantArray;
    use crate::arrays::ListArray;
    use crate::arrays::NullArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::StructArray;
    use crate::builtins::ArrayBuiltins;
    use crate::compute::conformance::cast::test_cast_conformance;
    use crate::dtype::DType;
    use crate::dtype::FieldNames;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::dtype::StructFields;
    use crate::executor::VortexSessionExecute;
    use crate::expr::cast;
    use crate::expr::root;
    use crate::extension::datetime::TimeUnit;
    use crate::extension::datetime::Timestamp;
    use crate::scalar::Scalar;
    use crate::test_harness::ExpectedOutcome;
    use crate::test_harness::assert_execution_paths;
    use crate::validity::Validity;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    #[test]
    fn cast_same_ext_dtype() {
        let ext_dtype = Timestamp::new(TimeUnit::Milliseconds, Nullability::NonNullable).erased();
        let storage = Buffer::<i64>::empty().into_array();

        let arr = ExtensionArray::new(ext_dtype.clone(), storage);

        let output = arr
            .clone()
            .into_array()
            .cast(DType::Extension(ext_dtype.clone()))
            .unwrap();
        assert_eq!(arr.len(), output.len());
        assert_eq!(arr.dtype(), output.dtype());
        assert_eq!(output.dtype(), &DType::Extension(ext_dtype));
    }

    #[test]
    fn cast_same_ext_dtype_differet_nullability() {
        let ext_dtype = Timestamp::new(TimeUnit::Milliseconds, Nullability::NonNullable).erased();
        let storage = Buffer::<i64>::empty().into_array();

        let arr = ExtensionArray::new(ext_dtype.clone(), storage);
        assert!(!arr.dtype().is_nullable());

        let new_dtype = DType::Extension(ext_dtype).with_nullability(Nullability::Nullable);

        let output = arr.clone().into_array().cast(new_dtype.clone()).unwrap();
        assert_eq!(arr.len(), output.len());
        assert!(arr.dtype().eq_ignore_nullability(output.dtype()));
        assert_eq!(output.dtype(), &new_dtype);
    }

    #[rstest]
    #[case::to_storage(
        create_timestamp_array(TimeUnit::Milliseconds, false).into_array(),
        DType::Primitive(PType::I64, Nullability::NonNullable),
    )]
    #[case::from_storage(
        buffer![1_000i64, 2_000].into_array(),
        timestamp_dtype(TimeUnit::Milliseconds, Nullability::NonNullable),
    )]
    #[case::different_unit(
        create_timestamp_array(TimeUnit::Milliseconds, false).into_array(),
        timestamp_dtype(TimeUnit::Nanoseconds, Nullability::NonNullable),
    )]
    #[case::nullable_to_storage(
        create_timestamp_array(TimeUnit::Milliseconds, true).into_array(),
        DType::Primitive(PType::I64, Nullability::Nullable),
    )]
    #[case::null_to_non_nullable_timestamp(
        NullArray::new(2).into_array(),
        timestamp_dtype(TimeUnit::Milliseconds, Nullability::NonNullable),
    )]
    #[case::typed_null_to_timestamp(
        PrimitiveArray::from_option_iter([None::<i64>, None]).into_array(),
        timestamp_dtype(TimeUnit::Milliseconds, Nullability::Nullable),
    )]
    #[case::nested_in_struct(
        in_struct(create_timestamp_array(TimeUnit::Milliseconds, false).into_array()),
        struct_dtype(DType::Primitive(PType::I64, Nullability::NonNullable)),
    )]
    #[case::nested_in_list(
        in_list(create_timestamp_array(TimeUnit::Milliseconds, false).into_array()),
        list_dtype(DType::Primitive(PType::I64, Nullability::NonNullable), Nullability::NonNullable),
    )]
    #[case::null_list_constant(
        ConstantArray::new(
            Scalar::null(list_dtype(
                timestamp_dtype(TimeUnit::Milliseconds, Nullability::NonNullable),
                Nullability::Nullable,
            )),
            2,
        )
        .into_array(),
        list_dtype(DType::Primitive(PType::I64, Nullability::NonNullable), Nullability::Nullable),
    )]
    #[case::empty_list_constant(
        ConstantArray::new(
            Scalar::list_empty(
                Arc::new(timestamp_dtype(TimeUnit::Milliseconds, Nullability::NonNullable)),
                Nullability::NonNullable,
            ),
            2,
        )
        .into_array(),
        list_dtype(DType::Primitive(PType::I64, Nullability::NonNullable), Nullability::NonNullable),
    )]
    fn cast_never_reinterprets_storage(#[case] input: ArrayRef, #[case] target: DType) {
        assert!(cast(root(), target.clone()).bind(input.dtype()).is_err());
        assert!(input.cast(target.clone()).is_err());

        assert_execution_paths(
            &[input],
            |inputs| inputs[0].cast(target.clone()),
            ExpectedOutcome::Error,
            &mut SESSION.create_execution_ctx(),
        );
    }

    #[test]
    fn untyped_null_casts_to_nullable_timestamp_on_every_path() {
        let target = timestamp_dtype(TimeUnit::Milliseconds, Nullability::Nullable);
        let expected = ExtensionArray::new(
            target.as_extension().clone(),
            PrimitiveArray::from_option_iter([None::<i64>, None]).into_array(),
        );

        assert_execution_paths(
            &[NullArray::new(2).into_array()],
            |inputs| inputs[0].cast(target.clone()),
            ExpectedOutcome::Array(expected.into_array()),
            &mut SESSION.create_execution_ctx(),
        );
    }

    #[test]
    fn cast_timestamp_nullability_on_every_path() {
        let input = create_timestamp_array(TimeUnit::Milliseconds, false);
        let target = input.dtype().as_nullable();
        let expected = ExtensionArray::new(
            target.as_extension().clone(),
            PrimitiveArray::from_option_iter([
                Some(1_000_000i64),
                Some(2_000_000),
                Some(3_000_000),
                Some(4_000_000),
                Some(5_000_000),
            ])
            .into_array(),
        );

        assert_execution_paths(
            &[input.into_array()],
            |inputs| inputs[0].cast(target.clone()),
            ExpectedOutcome::Array(expected.into_array()),
            &mut SESSION.create_execution_ctx(),
        );
    }

    #[test]
    fn same_type_cast_returns_the_storage_error() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let input = create_timestamp_array(TimeUnit::Milliseconds, true);
        let storage = input.storage_array();
        let storage_error = storage
            .cast(storage.dtype().as_nonnullable())
            .and_then(|cast| cast.execute::<RecursiveCanonical>(&mut ctx))
            .err()
            .expect("storage with nulls cannot cast to non-nullable");

        let target = input.dtype().as_nonnullable();
        let error = input
            .into_array()
            .cast(target)
            .and_then(|cast| cast.execute::<RecursiveCanonical>(&mut ctx))
            .err()
            .expect("timestamp with nulls cannot cast to non-nullable");

        assert_eq!(
            error.to_string().lines().next(),
            storage_error.to_string().lines().next(),
        );
        Ok(())
    }

    #[rstest]
    #[case(create_timestamp_array(TimeUnit::Milliseconds, false))]
    #[case(create_timestamp_array(TimeUnit::Microseconds, true))]
    #[case(create_timestamp_array(TimeUnit::Nanoseconds, false))]
    #[case(create_timestamp_array(TimeUnit::Seconds, true))]
    fn test_cast_extension_conformance(#[case] array: ExtensionArray) {
        test_cast_conformance(&array.into_array(), &mut SESSION.create_execution_ctx());
    }

    fn create_timestamp_array(time_unit: TimeUnit, nullable: bool) -> ExtensionArray {
        let dtype = timestamp_dtype(time_unit, nullable.into());

        let storage = if nullable {
            PrimitiveArray::from_option_iter([
                Some(1_000_000i64), // 1 second in microseconds
                None,
                Some(2_000_000),
                Some(3_000_000),
                None,
            ])
            .into_array()
        } else {
            buffer![1_000_000i64, 2_000_000, 3_000_000, 4_000_000, 5_000_000].into_array()
        };

        ExtensionArray::new(dtype.as_extension().clone(), storage)
    }

    fn in_struct(field: ArrayRef) -> ArrayRef {
        StructArray::from_fields(&[("a", field)])
            .vortex_expect("a struct of one field is valid")
            .into_array()
    }

    fn struct_dtype(field: DType) -> DType {
        DType::Struct(
            StructFields::new(FieldNames::from(["a"]), vec![field]),
            Nullability::NonNullable,
        )
    }

    /// The lists `[0, 2)` and `[2, 5)` of `elements`, which must hold five values.
    fn in_list(elements: ArrayRef) -> ArrayRef {
        ListArray::try_new(
            elements,
            buffer![0u32, 2, 5].into_array(),
            Validity::NonNullable,
            &mut SESSION.create_execution_ctx(),
        )
        .vortex_expect("the offsets fit five elements")
        .into_array()
    }

    fn list_dtype(element: DType, nullability: Nullability) -> DType {
        DType::List(Arc::new(element), nullability)
    }

    fn timestamp_dtype(time_unit: TimeUnit, nullability: Nullability) -> DType {
        DType::Extension(
            Timestamp::new_with_tz(time_unit, Some("UTC".into()), nullability).erased(),
        )
    }
}
