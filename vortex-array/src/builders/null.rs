// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;

use vortex_error::VortexResult;
use vortex_error::vortex_ensure_eq;

use crate::ArrayRef;
use crate::ExecutionCtx;
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
        vortex_ensure_eq!(
            scalar.dtype(),
            self.dtype(),
            "NullBuilder received a scalar with the wrong dtype"
        );

        self.append_null();
        Ok(())
    }

    fn reserve_exact(&mut self, _additional: usize) {}

    fn finish(&mut self) -> ArrayRef {
        NullArray::new(std::mem::take(&mut self.length)).into_array()
    }

    fn finish_into_canonical(&mut self, _ctx: &mut ExecutionCtx) -> Canonical {
        Canonical::Null(NullArray::new(std::mem::take(&mut self.length)))
    }
}
