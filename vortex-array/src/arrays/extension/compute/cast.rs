// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::DType;

use crate::arrays::{ExtensionArray, ExtensionVTable};
use crate::compute::{CastKernel, CastKernelAdapter, cast};
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

        let new_storage = match cast(array.storage(), ext_dtype.storage_dtype()) {
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

    use vortex_dtype::datetime::{TIMESTAMP_ID, TemporalMetadata, TimeUnit};
    use vortex_dtype::{ExtDType, Nullability, PType};

    use super::*;
    use crate::arrays::PrimitiveArray;

    #[test]
    fn cast_same_ext_dtype() {
        let ext_dtype = Arc::new(ExtDType::new(
            TIMESTAMP_ID.clone(),
            Arc::new(PType::I64.into()),
            Some(TemporalMetadata::Timestamp(TimeUnit::Ms, None).into()),
        ));
        let storage = PrimitiveArray::from_iter(Vec::<i64>::new()).into_array();

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
            Some(TemporalMetadata::Timestamp(TimeUnit::Ms, None).into()),
        ));
        let storage = PrimitiveArray::from_iter(Vec::<i64>::new()).into_array();

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
            Some(TemporalMetadata::Timestamp(TimeUnit::Ms, None).into()),
        ));
        let target_dtype = Arc::new(ExtDType::new(
            TIMESTAMP_ID.clone(),
            Arc::new(PType::I64.into()),
            // Note NS here instead of MS
            Some(TemporalMetadata::Timestamp(TimeUnit::Ns, None).into()),
        ));

        let storage = PrimitiveArray::from_iter(Vec::<i64>::new()).into_array();
        let arr = ExtensionArray::new(original_dtype, storage);

        assert!(cast(arr.as_ref(), &DType::Extension(target_dtype)).is_err());
    }
}
