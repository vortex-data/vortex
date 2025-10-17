// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::DType;

use crate::arrays::{ExtensionArray, ExtensionVTable};
use crate::compute::{self, CastKernel, CastKernelAdapter};
use crate::{ArrayRef, IntoArray, register_kernel};

impl CastKernel for ExtensionVTable {
    fn cast(
        &self,
        array: &ExtensionArray,
        dtype: &DType,
    ) -> vortex_error::VortexResult<Option<ArrayRef>> {
        if !array.dtype().eq_ignore_nullability(dtype) {
            return Ok(None);
        }

        let DType::Extension(ext_dtype) = dtype else {
            unreachable!("Already verified we have an extension dtype");
        };

        let new_storage = match compute::cast(array.storage(), ext_dtype.storage_dtype()) {
            Ok(arr) => arr,
            Err(e) => {
                log::warn!("Failed to cast storage array: {e}");
                return Ok(None);
            }
        };

        Ok(Some(
            ExtensionArray::new(ext_dtype.clone(), new_storage).into_array(),
        ))
    }
}

register_kernel!(CastKernelAdapter(ExtensionVTable).lift());

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use rstest::rstest;
    use vortex_buffer::{Buffer, buffer};
    use vortex_dtype::datetime::{TIMESTAMP_ID, TemporalMetadata, TimeUnit};
    use vortex_dtype::{ExtDType, Nullability, PType};

    use super::*;
    use crate::IntoArray;
    use crate::arrays::PrimitiveArray;
    use crate::compute::cast;
    use crate::compute::conformance::cast::test_cast_conformance;

    #[test]
    fn cast_same_ext_dtype() {
        let ext_dtype = Arc::new(ExtDType::new(
            TIMESTAMP_ID.clone(),
            Arc::new(PType::I64.into()),
            Some(TemporalMetadata::Timestamp(TimeUnit::Milliseconds, None).into()),
        ));
        let storage = Buffer::<i64>::empty().into_array();

        let arr = ExtensionArray::new(ext_dtype.clone(), storage);

        let output = cast(arr.as_ref(), &DType::Extension(ext_dtype.clone())).unwrap();
        assert_eq!(arr.len(), output.len());
        assert_eq!(arr.dtype(), output.dtype());
        assert_eq!(output.dtype(), &DType::Extension(ext_dtype));
    }

    #[test]
    fn cast_same_ext_dtype_differet_nullability() {
        let ext_dtype = Arc::new(ExtDType::new(
            TIMESTAMP_ID.clone(),
            Arc::new(PType::I64.into()),
            Some(TemporalMetadata::Timestamp(TimeUnit::Milliseconds, None).into()),
        ));
        let storage = Buffer::<i64>::empty().into_array();

        let arr = ExtensionArray::new(ext_dtype.clone(), storage);
        assert!(!arr.dtype.is_nullable());

        let new_dtype = DType::Extension(ext_dtype).with_nullability(Nullability::Nullable);

        let output = cast(arr.as_ref(), &new_dtype).unwrap();
        assert_eq!(arr.len(), output.len());
        assert!(arr.dtype().eq_ignore_nullability(output.dtype()));
        assert_eq!(output.dtype(), &new_dtype);
    }

    #[test]
    fn cast_different_ext_dtype() {
        let original_dtype = Arc::new(ExtDType::new(
            TIMESTAMP_ID.clone(),
            Arc::new(PType::I64.into()),
            Some(TemporalMetadata::Timestamp(TimeUnit::Milliseconds, None).into()),
        ));
        let target_dtype = Arc::new(ExtDType::new(
            TIMESTAMP_ID.clone(),
            Arc::new(PType::I64.into()),
            // Note NS here instead of MS
            Some(TemporalMetadata::Timestamp(TimeUnit::Nanoseconds, None).into()),
        ));

        let storage = buffer![1i64].into_array();
        let arr = ExtensionArray::new(original_dtype, storage);

        assert!(cast(arr.as_ref(), &DType::Extension(target_dtype)).is_err());
    }

    #[rstest]
    #[case(create_timestamp_array(TimeUnit::Milliseconds, false))]
    #[case(create_timestamp_array(TimeUnit::Microseconds, true))]
    #[case(create_timestamp_array(TimeUnit::Nanoseconds, false))]
    #[case(create_timestamp_array(TimeUnit::Seconds, true))]
    fn test_cast_extension_conformance(#[case] array: ExtensionArray) {
        test_cast_conformance(array.as_ref());
    }

    fn create_timestamp_array(time_unit: TimeUnit, nullable: bool) -> ExtensionArray {
        let ext_dtype = Arc::new(ExtDType::new(
            TIMESTAMP_ID.clone(),
            Arc::new(if nullable {
                DType::Primitive(PType::I64, Nullability::Nullable)
            } else {
                DType::Primitive(PType::I64, Nullability::NonNullable)
            }),
            Some(TemporalMetadata::Timestamp(time_unit, Some("UTC".to_string())).into()),
        ));

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
