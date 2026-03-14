// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::Extension;
use crate::arrays::ExtensionArray;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::scalar_fn::fns::cast::CastReduce;

impl CastReduce for Extension {
    fn cast(array: &ExtensionArray, dtype: &DType) -> vortex_error::VortexResult<Option<ArrayRef>> {
        // Fast path: same extension type (ignoring nullability), just cast the storage.
        if array.dtype().eq_ignore_nullability(dtype) {
            let DType::Extension(ext_dtype) = dtype else {
                unreachable!("Already verified we have an extension dtype");
            };

            let new_storage = array
                .storage_array()
                .cast(ext_dtype.storage_dtype().clone())?;

            return Ok(Some(
                ExtensionArray::new(ext_dtype.clone(), new_storage).into_array(),
            ));
        }

        // Otherwise we defer to the extension vtable.
        array.ext_dtype().cast_from_ext(array, dtype)
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
    use crate::builtins::ArrayBuiltins;
    use crate::compute::conformance::cast::test_cast_conformance;
    use crate::dtype::Nullability;
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
        assert!(!arr.dtype.is_nullable());

        let new_dtype = DType::Extension(ext_dtype).with_nullability(Nullability::Nullable);

        let output = arr.clone().into_array().cast(new_dtype.clone()).unwrap();
        assert_eq!(arr.len(), output.len());
        assert!(arr.dtype().eq_ignore_nullability(output.dtype()));
        assert_eq!(output.dtype(), &new_dtype);
    }

    #[test]
    fn cast_timestamp_ms_to_ns() {
        let source_dtype =
            Timestamp::new(TimeUnit::Milliseconds, Nullability::NonNullable).erased();
        let target_dtype = Timestamp::new(TimeUnit::Nanoseconds, Nullability::NonNullable).erased();

        let storage = buffer![1i64, 2, 3].into_array();
        let arr = ExtensionArray::new(source_dtype, storage).into_array();

        let result = arr.cast(DType::Extension(target_dtype.clone())).unwrap();
        assert_eq!(result.dtype(), &DType::Extension(target_dtype));

        // Verify values were scaled: ms → ns is ×1_000_000
        let ext = result.to_canonical().unwrap().as_extension().clone();
        let prim = ext
            .storage_array()
            .to_canonical()
            .unwrap()
            .as_primitive()
            .clone();
        assert_eq!(prim.as_slice::<i64>(), &[1_000_000, 2_000_000, 3_000_000]);
    }

    #[test]
    fn cast_timestamp_s_to_us() {
        let source_dtype = Timestamp::new(TimeUnit::Seconds, Nullability::NonNullable).erased();
        let target_dtype =
            Timestamp::new(TimeUnit::Microseconds, Nullability::NonNullable).erased();

        let storage = buffer![10i64, 20].into_array();
        let arr = ExtensionArray::new(source_dtype, storage).into_array();

        let result = arr.cast(DType::Extension(target_dtype)).unwrap();
        let ext = result.to_canonical().unwrap().as_extension().clone();
        let prim = ext
            .storage_array()
            .to_canonical()
            .unwrap()
            .as_primitive()
            .clone();
        assert_eq!(prim.as_slice::<i64>(), &[10_000_000, 20_000_000]);
    }

    #[test]
    fn cast_timestamp_tz_mismatch_fails() {
        use std::sync::Arc;

        let utc_dtype = Timestamp::new_with_tz(
            TimeUnit::Seconds,
            Some(Arc::from("UTC")),
            Nullability::NonNullable,
        )
        .erased();
        let no_tz_dtype = Timestamp::new(TimeUnit::Nanoseconds, Nullability::NonNullable).erased();

        let storage = buffer![1i64].into_array();
        let arr = ExtensionArray::new(utc_dtype, storage).into_array();

        // Timezone mismatch: cast creates a lazy expression, error surfaces on evaluation.
        let result = arr
            .cast(DType::Extension(no_tz_dtype))
            .and_then(|a| a.to_canonical().map(|c| c.into_array()));
        assert!(result.is_err());
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
