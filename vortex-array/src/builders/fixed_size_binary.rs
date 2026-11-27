// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;

use vortex_buffer::BufferAllocatorRef;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure_eq;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::FixedSizeBinary;
use crate::arrays::FixedSizeBinaryArray;
use crate::arrays::fixed_size_binary::FixedSizeBinaryArrayExt;
use crate::buffer::BufferHandle;
use crate::builders::ArrayBuilder;
use crate::builders::DEFAULT_BUILDER_CAPACITY;
use crate::builders::LazyBitBufferBuilder;
use crate::canonical::Canonical;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::scalar::Scalar;

/// Builds a [`FixedSizeBinaryArray`] one value at a time.
pub struct FixedSizeBinaryBuilder {
    dtype: DType,
    byte_width: usize,
    len: usize,
    values: ByteBufferMut,
    nulls: LazyBitBufferBuilder,
}

impl FixedSizeBinaryBuilder {
    /// Creates a builder for values of `byte_width` bytes.
    pub fn new_in(
        byte_width: u32,
        nullability: Nullability,
        allocator: &BufferAllocatorRef,
    ) -> Self {
        Self::with_capacity_in(DEFAULT_BUILDER_CAPACITY, byte_width, nullability, allocator)
    }

    /// Creates a builder for values of `byte_width` bytes with room for `capacity` rows.
    pub fn with_capacity_in(
        capacity: usize,
        byte_width: u32,
        nullability: Nullability,
        allocator: &BufferAllocatorRef,
    ) -> Self {
        let width = byte_width as usize;
        Self {
            dtype: DType::FixedSizeBinary(byte_width, nullability),
            byte_width: width,
            len: 0,
            values: ByteBufferMut::with_capacity_in(
                capacity.saturating_mul(width),
                allocator.clone(),
            ),
            nulls: LazyBitBufferBuilder::new(capacity, allocator.clone()),
        }
    }

    /// Returns the number of bytes in each value.
    pub fn byte_width(&self) -> u32 {
        match self.dtype {
            DType::FixedSizeBinary(byte_width, _) => byte_width,
            _ => unreachable!("FixedSizeBinaryBuilder requires a fixed-size binary dtype"),
        }
    }

    /// Appends one non-null value.
    ///
    /// # Panics
    ///
    /// Panics if `value` does not have exactly the builder's byte width.
    pub fn append_value(&mut self, value: impl AsRef<[u8]>) {
        let value = value.as_ref();
        assert_eq!(
            value.len(),
            self.byte_width,
            "fixed-size binary value has {} bytes, expected {}",
            value.len(),
            self.byte_width
        );
        self.values.extend_from_slice(value);
        self.len += 1;
        self.nulls.append_non_null();
    }

    /// Appends an optional value, as a null when `value` is `None`.
    ///
    /// # Panics
    ///
    /// Panics if a present value does not have exactly the builder's byte width, or if `value` is
    /// `None` and the builder is non-nullable.
    pub fn append_option(&mut self, value: Option<impl AsRef<[u8]>>) {
        match value {
            Some(value) => self.append_value(value),
            None => self.append_null(),
        }
    }

    pub(crate) fn append_fixed_size_binary_array(
        &mut self,
        array: ArrayView<'_, FixedSizeBinary>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        vortex_ensure_eq!(
            array.byte_width(),
            self.byte_width(),
            "FixedSizeBinaryBuilder received an array of another byte width"
        );

        let mask = array.validity()?.execute_mask(array.len(), ctx)?;
        self.values.extend_from_slice(array.values().as_slice());
        self.len += array.len();
        self.nulls.append_validity_mask(&mask);
        Ok(())
    }

    /// Finishes the builder into a [`FixedSizeBinaryArray`], leaving it empty.
    pub fn finish_into_fixed_size_binary(&mut self) -> FixedSizeBinaryArray {
        let validity = self.nulls.finish_with_nullability(self.dtype.nullability());
        let values = self.values.take().freeze();
        let len = std::mem::take(&mut self.len);
        assert!(
            validity
                .maybe_len()
                .is_none_or(|validity_len| validity_len == len),
            "validity of length {:?} does not cover the {len} values appended",
            validity.maybe_len(),
        );

        // SAFETY: Every appended row added `byte_width` bytes and one validity bit.
        unsafe {
            FixedSizeBinaryArray::new_unchecked_handle(
                BufferHandle::new_host(values),
                self.byte_width(),
                len,
                validity,
            )
        }
    }
}

impl ArrayBuilder for FixedSizeBinaryBuilder {
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
        self.len
    }

    fn append_zeros(&mut self, n: usize) {
        self.values.push_n(0, n * self.byte_width);
        self.len += n;
        self.nulls.append_n_non_nulls(n);
    }

    unsafe fn append_nulls_unchecked(&mut self, n: usize) {
        self.values.push_n(0, n * self.byte_width);
        self.len += n;
        self.nulls.append_n_nulls(n);
    }

    fn append_scalar(&mut self, scalar: &Scalar) -> VortexResult<()> {
        vortex_ensure_eq!(
            scalar.dtype(),
            self.dtype(),
            "FixedSizeBinaryBuilder received a scalar with the wrong dtype"
        );

        match scalar.as_binary().value() {
            None => self.append_null(),
            Some(value) => self.append_value(value),
        }
        Ok(())
    }

    fn reserve_exact(&mut self, additional: usize) {
        self.values
            .reserve(additional.saturating_mul(self.byte_width));
        self.nulls.reserve_exact(additional);
    }

    fn finish(&mut self) -> ArrayRef {
        self.finish_into_fixed_size_binary().into_array()
    }

    fn finish_into_canonical(&mut self, _ctx: &mut ExecutionCtx) -> Canonical {
        Canonical::FixedSizeBinary(self.finish_into_fixed_size_binary())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_buffer::BufferAllocatorRef;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::arrays::FixedSizeBinaryArray;
    use crate::assert_arrays_eq;
    use crate::builders::ArrayBuilder;
    use crate::builders::FixedSizeBinaryBuilder;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::scalar::Scalar;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    fn builder(byte_width: u32, nullability: Nullability) -> FixedSizeBinaryBuilder {
        FixedSizeBinaryBuilder::new_in(byte_width, nullability, BufferAllocatorRef::static_ref())
    }

    #[test]
    fn appends_values_nulls_and_scalars() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let mut builder = builder(2, Nullability::Nullable);
        builder.append_value(b"ab");
        builder.append_null();
        builder.append_scalar(&Scalar::fixed_size_binary(
            b"cd".to_vec(),
            Nullability::Nullable,
        ))?;
        builder.append_scalar(&Scalar::null(DType::FixedSizeBinary(
            2,
            Nullability::Nullable,
        )))?;
        builder.append_zero();
        assert_eq!(builder.len(), 5);

        let expected = FixedSizeBinaryArray::from_option_iter(
            [Some(b"ab"), None, Some(b"cd"), None, Some(&[0u8, 0])],
            2,
        );
        assert_arrays_eq!(builder.finish(), expected, &mut ctx);
        assert!(builder.is_empty());
        Ok(())
    }

    #[test]
    fn extends_from_array() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let source = FixedSizeBinaryArray::from_option_iter([Some(b"xyz"), None, Some(b"123")], 3);
        let mut builder = builder(3, Nullability::Nullable);
        builder.append_value(b"abc");
        source
            .into_array()
            .append_to_builder(&mut builder, &mut ctx)?;

        let expected = FixedSizeBinaryArray::from_option_iter(
            [Some(b"abc"), Some(b"xyz"), None, Some(b"123")],
            3,
        );
        assert_arrays_eq!(builder.finish(), expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn rejects_scalar_of_other_byte_width() {
        let mut builder = builder(3, Nullability::NonNullable);
        assert!(
            builder
                .append_scalar(&Scalar::fixed_size_binary(
                    b"ab".to_vec(),
                    Nullability::NonNullable
                ))
                .is_err()
        );
    }

    #[rstest]
    #[case::nullable(Nullability::Nullable)]
    #[case::non_nullable(Nullability::NonNullable)]
    fn zero_width_counts_rows(#[case] nullability: Nullability) {
        let mut builder = builder(0, nullability);
        builder.append_value(b"");
        builder.append_defaults(2);
        let array = builder.finish();
        assert_eq!(array.len(), 3);
        assert_eq!(array.dtype(), &DType::FixedSizeBinary(0, nullability));
    }

    #[test]
    #[should_panic(expected = "fixed-size binary value has 1 bytes, expected 2")]
    fn append_value_checks_width() {
        builder(2, Nullability::NonNullable).append_value(b"a");
    }
}
