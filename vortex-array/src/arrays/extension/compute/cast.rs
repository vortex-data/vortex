// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

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
    fn cast(
        array: ArrayView<'_, Extension>,
        dtype: &DType,
    ) -> vortex_error::VortexResult<Option<ArrayRef>> {
        if !array.dtype().eq_ignore_nullability(dtype) {
            // Target is not the same extension type.
            // Delegate to the storage array's cast.
            return Ok(Some(array.storage_array().cast(dtype.clone())?));
        }

        let DType::Extension(ext_dtype) = dtype else {
            unreachable!("Already verified we have an extension dtype");
        };

        let new_storage = match array
            .storage_array()
            .cast(ext_dtype.storage_dtype().clone())
        {
            Ok(arr) => arr,
            Err(e) => {
                tracing::warn!("Failed to cast storage array: {e}");
                return Ok(None);
            }
        };

        Ok(Some(
            ExtensionArray::new(ext_dtype.clone(), new_storage).into_array(),
        ))
    }
}

#[cfg(test)]
mod tests {

    use rstest::rstest;
    use vortex_buffer::Buffer;
    use vortex_buffer::buffer;

    use super::*;
    use crate::IntoArray;
    use crate::arrays::PrimitiveArray;
    use crate::assert_arrays_eq;
    use crate::builtins::ArrayBuiltins;
    use crate::compute::conformance::cast::test_cast_conformance;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::extension::datetime::TimeUnit;
    use crate::extension::datetime::Timestamp;

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

    #[test]
    fn cast_different_ext_dtype() {
        let original_dtype =
            Timestamp::new(TimeUnit::Milliseconds, Nullability::NonNullable).erased();
        // Note NS here instead of MS
        let target_dtype = Timestamp::new(TimeUnit::Nanoseconds, Nullability::NonNullable).erased();

        let storage = buffer![1i64].into_array();
        let arr = ExtensionArray::new(original_dtype, storage);

        #[expect(deprecated)]
        let result = arr
            .into_array()
            .cast(DType::Extension(target_dtype))
            .and_then(|a| a.to_canonical().map(|c| c.into_array()));
        assert!(result.is_err());
    }

    #[test]
    fn cast_timestamp_to_i64() -> vortex_error::VortexResult<()> {
        let ext_dtype = Timestamp::new_with_tz(
            TimeUnit::Nanoseconds,
            Some("UTC".into()),
            Nullability::NonNullable,
        )
        .erased();
        let storage = buffer![1i64, 2, 3].into_array();
        let arr = ExtensionArray::new(ext_dtype, storage).into_array();

        let result = arr.cast(DType::Primitive(PType::I64, Nullability::NonNullable))?;
        assert_eq!(
            result.dtype(),
            &DType::Primitive(PType::I64, Nullability::NonNullable)
        );
        assert_arrays_eq!(result, buffer![1i64, 2, 3].into_array());
        Ok(())
    }

    #[rstest]
    #[case(create_timestamp_array(TimeUnit::Milliseconds, false))]
    #[case(create_timestamp_array(TimeUnit::Microseconds, true))]
    #[case(create_timestamp_array(TimeUnit::Nanoseconds, false))]
    #[case(create_timestamp_array(TimeUnit::Seconds, true))]
    fn test_cast_extension_conformance(#[case] array: ExtensionArray) {
        test_cast_conformance(&array.into_array());
    }

    fn create_timestamp_array(time_unit: TimeUnit, nullable: bool) -> ExtensionArray {
        let ext_dtype =
            Timestamp::new_with_tz(time_unit, Some("UTC".into()), nullable.into()).erased();

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

        ExtensionArray::new(ext_dtype, storage)
    }
}
