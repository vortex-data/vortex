// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;

use vortex_buffer::BitBufferMut;
use vortex_buffer::BufferAllocatorRef;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure_eq;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Bool;
use crate::arrays::BoolArray;
use crate::arrays::bool::BoolArrayExt;
use crate::builders::ArrayBuilder;
use crate::builders::DEFAULT_BUILDER_CAPACITY;
use crate::builders::LazyBitBufferBuilder;
use crate::canonical::Canonical;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::scalar::Scalar;

pub struct BoolBuilder {
    dtype: DType,
    inner: BitBufferMut,
    nulls: LazyBitBufferBuilder,
}

impl BoolBuilder {
    /// Creates a builder with the default capacity.
    #[deprecated(note = "use `new_in` with an explicit allocator")]
    pub fn new(nullability: Nullability) -> Self {
        Self::new_in(nullability, BufferAllocatorRef::static_ref())
    }

    /// Creates a builder with the default capacity using `allocator`.
    pub fn new_in(nullability: Nullability, allocator: &BufferAllocatorRef) -> Self {
        Self::with_capacity_in(nullability, DEFAULT_BUILDER_CAPACITY, allocator)
    }

    /// Creates a builder with the given capacity.
    #[deprecated(note = "use `with_capacity_in` with an explicit allocator")]
    pub fn with_capacity(nullability: Nullability, capacity: usize) -> Self {
        Self::with_capacity_in(nullability, capacity, BufferAllocatorRef::static_ref())
    }

    /// Creates a builder with the given capacity and allocator.
    pub fn with_capacity_in(
        nullability: Nullability,
        capacity: usize,
        allocator: &BufferAllocatorRef,
    ) -> Self {
        Self {
            inner: BitBufferMut::with_capacity_in(capacity, allocator.clone()),
            nulls: LazyBitBufferBuilder::new(capacity, allocator.clone()),
            dtype: DType::Bool(nullability),
        }
    }

    /// Appends a boolean value to the builder.
    pub fn append_value(&mut self, value: bool) {
        self.append_values(value, 1)
    }

    /// Appends the same boolean value multiple times to the builder.
    ///
    /// This method appends the given boolean value `n` times.
    pub fn append_values(&mut self, value: bool, n: usize) {
        self.inner.append_n(value, n);
        self.nulls.append_n_non_nulls(n)
    }

    /// Finishes the builder directly into a [`BoolArray`].
    pub fn finish_into_bool(&mut self) -> BoolArray {
        assert_eq!(
            self.nulls.len(),
            self.inner.len(),
            "Null count and value count should match when calling BoolBuilder::finish."
        );

        // SAFETY: the assert above establishes the only invariant, that the validity covers
        // exactly the bits appended.
        unsafe {
            BoolArray::new_unchecked(
                self.inner.take().freeze(),
                self.nulls.finish_with_nullability(self.dtype.nullability()),
            )
        }
    }

    pub(crate) fn append_bool_array(
        &mut self,
        array: ArrayView<'_, Bool>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        let mask = BoolArrayExt::validity(&array).execute_mask(array.len(), ctx)?;
        self.inner.append_buffer(&array.to_bit_buffer());
        self.nulls.append_validity_mask(&mask);
        Ok(())
    }
}

impl ArrayBuilder for BoolBuilder {
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
        self.inner.len()
    }

    fn append_zeros(&mut self, n: usize) {
        self.append_values(false, n)
    }

    unsafe fn append_nulls_unchecked(&mut self, n: usize) {
        self.inner.append_n(false, n);
        self.nulls.append_n_nulls(n)
    }

    fn append_scalar(&mut self, scalar: &Scalar) -> VortexResult<()> {
        vortex_ensure_eq!(
            scalar.dtype(),
            self.dtype(),
            "BoolBuilder received a scalar with the wrong dtype"
        );

        match scalar.as_bool().value() {
            Some(value) => self.append_value(value),
            None => self.append_null(),
        }

        Ok(())
    }

    fn reserve_exact(&mut self, additional: usize) {
        self.inner.reserve(additional);
        self.nulls.reserve_exact(additional);
    }

    fn finish(&mut self) -> ArrayRef {
        self.finish_into_bool().into_array()
    }

    fn finish_into_canonical(&mut self, _ctx: &mut ExecutionCtx) -> Canonical {
        Canonical::Bool(self.finish_into_bool())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use vortex_buffer::BitBuffer;
    use vortex_buffer::BufferAllocatorRef;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::assert_arrays_eq;
    use crate::builders::ArrayBuilder;
    use crate::builders::BoolBuilder;
    use crate::builders::bool::BoolArray;
    use crate::builtins::ArrayBuiltins;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::validity::Validity;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    #[test]
    fn test_append_bool_array_error_does_not_mutate_builder() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let mut builder = BoolBuilder::with_capacity_in(
            Nullability::Nullable,
            4,
            BufferAllocatorRef::static_ref(),
        );
        builder.append_value(true);

        let failing_validity = BoolArray::from_iter([Some(true), None])
            .into_array()
            .cast(DType::Bool(Nullability::NonNullable))?;
        let failing = BoolArray::new(
            BitBuffer::from_iter([false, true]),
            Validity::Array(failing_validity),
        );

        assert!(
            failing
                .into_array()
                .append_to_builder(&mut builder, &mut ctx)
                .is_err()
        );
        assert_eq!(builder.len(), 1);
        let result = builder.finish_into_bool();
        assert_arrays_eq!(result, BoolArray::from_iter([Some(true)]), &mut ctx);
        Ok(())
    }
}
