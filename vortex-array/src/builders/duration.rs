// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;

use vortex_buffer::BufferAllocatorRef;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure_eq;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Duration;
use crate::arrays::DurationArray;
use crate::arrays::duration::DurationArraySlotsExt;
use crate::builders::ArrayBuilder;
use crate::builders::ChildBuilder;
use crate::canonical::Canonical;
use crate::dtype::DType;
use crate::dtype::DurationUnit;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::scalar::Scalar;

/// Builds an array of durations in a single unit.
pub struct DurationBuilder {
    dtype: DType,
    values: ChildBuilder,
}

impl DurationBuilder {
    pub(super) fn with_capacity_in(
        unit: DurationUnit,
        nullability: Nullability,
        capacity: usize,
        allocator: &BufferAllocatorRef,
    ) -> Self {
        Self {
            dtype: DType::Duration(unit, nullability),
            values: ChildBuilder::with_capacity(
                &DType::Primitive(PType::I64, nullability),
                capacity,
                allocator,
            ),
        }
    }

    pub(crate) fn append_duration(
        &mut self,
        array: ArrayView<'_, Duration>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        self.values.append_array(array.values(), ctx)
    }

    fn finish_duration(&mut self) -> DurationArray {
        let DType::Duration(unit, _) = self.dtype else {
            unreachable!("DurationBuilder has a duration dtype");
        };
        DurationArray::try_new(unit, self.values.finish())
            .vortex_expect("DurationBuilder stores i64 values")
    }
}

impl ArrayBuilder for DurationBuilder {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn len(&self) -> usize {
        self.values.len()
    }

    fn append_zeros(&mut self, n: usize) {
        self.values.append_zeros(n)
    }

    unsafe fn append_nulls_unchecked(&mut self, n: usize) {
        self.values.append_nulls(n)
    }

    fn append_scalar(&mut self, scalar: &Scalar) -> VortexResult<()> {
        vortex_ensure_eq!(scalar.dtype(), self.dtype());
        let scalar = Scalar::try_new(
            DType::Primitive(PType::I64, self.dtype().nullability()),
            scalar.value().cloned(),
        )?;
        self.values.append_scalar(&scalar)
    }

    fn reserve_chunks(&mut self, additional: usize) {
        self.values.reserve_chunks(additional)
    }

    fn reserve_exact(&mut self, capacity: usize) {
        self.values.reserve_exact(capacity)
    }

    fn finish(&mut self) -> ArrayRef {
        self.finish_duration().into_array()
    }

    fn finish_into_canonical(&mut self, _ctx: &mut ExecutionCtx) -> Canonical {
        Canonical::Duration(self.finish_duration())
    }
}
