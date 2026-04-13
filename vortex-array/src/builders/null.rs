// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;

use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::NullArray;
use crate::builders::ArrayBuilder;
use crate::canonical::Canonical;
use crate::dtype::DType;
use crate::scalar::Scalar;

/// The builder for building a [`NullArray`].
pub struct NullBuilder {
    length: usize,
}

impl Default for NullBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl NullBuilder {
    pub fn new() -> Self {
        Self { length: 0 }
    }
}

impl ArrayBuilder for NullBuilder {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn dtype(&self) -> &DType {
        &DType::Null
    }

    fn len(&self) -> usize {
        self.length
    }

    fn append_zeros(&mut self, n: usize) {
        self.length += n;
    }

    unsafe fn append_nulls_unchecked(&mut self, n: usize) {
        self.length += n;
    }

    fn append_scalar(&mut self, scalar: &Scalar) -> VortexResult<()> {
        vortex_ensure!(
            scalar.dtype() == self.dtype(),
            "NullBuilder expected scalar with dtype {}, got {}",
            self.dtype(),
            scalar.dtype()
        );

        self.append_null();
        Ok(())
    }

    unsafe fn extend_from_array_unchecked(&mut self, array: &ArrayRef) {
        self.append_nulls(array.len());
    }

    fn reserve_exact(&mut self, _additional: usize) {}

    unsafe fn set_validity_unchecked(&mut self, _validity: Mask) {}

    fn finish(&mut self) -> ArrayRef {
        NullArray::new(self.length).into_array()
    }

    fn finish_into_canonical(&mut self) -> Canonical {
        Canonical::Null(NullArray::new(self.length))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builders::ArrayBuilder;
    use crate::dtype::DType;
    use crate::scalar::Scalar;

    #[test]
    fn test_append_scalar() {
        let mut builder = NullBuilder::new();

        // Test appending null scalar.
        let null_scalar = Scalar::null(DType::Null);
        builder.append_scalar(&null_scalar).unwrap();
        builder.append_scalar(&null_scalar).unwrap();
        builder.append_scalar(&null_scalar).unwrap();

        let array = builder.finish();
        assert_eq!(array.len(), 3);

        // For null arrays, all values are null - nothing to check for actual values.
        // Just verify the array is indeed a null array with the right length.

        // Test wrong dtype error.
        let mut builder = NullBuilder::new();
        let wrong_scalar = Scalar::from(42i32);
        assert!(builder.append_scalar(&wrong_scalar).is_err());
    }
}
