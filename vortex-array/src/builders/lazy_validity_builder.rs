use arrow_buffer::{BooleanBuffer, BooleanBufferBuilder, NullBuffer};
use vortex_dtype::Nullability;
use vortex_dtype::Nullability::{NonNullable, Nullable};
use vortex_error::{vortex_panic, VortexExpect, VortexResult};
use vortex_mask::Mask;

use crate::validity::Validity;

/// This is borrowed from arrow's null buffer builder, however we expose a `append_buffer`
/// method to append a boolean buffer directly.
pub struct LazyNullBufferBuilder {
    inner: Option<BooleanBufferBuilder>,
    len: usize,
    capacity: usize,
}

impl LazyNullBufferBuilder {
    /// Creates a new empty builder.
    /// `capacity` is the number of bits in the null buffer.
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: None,
            len: 0,
            capacity,
        }
    }

    #[inline]
    pub fn append_n_non_nulls(&mut self, n: usize) {
        if let Some(buf) = self.inner.as_mut() {
            buf.append_n(n, true)
        } else {
            self.len += n;
        }
    }

    #[inline]
    pub fn append_non_null(&mut self) {
        if let Some(buf) = self.inner.as_mut() {
            buf.append(true)
        } else {
            self.len += 1;
        }
    }

    #[inline]
    pub fn append_n_nulls(&mut self, n: usize) {
        self.materialize_if_needed();
        self.inner
            .as_mut()
            .vortex_expect("cannot append null to non-nullable builder")
            .append_n(n, false);
    }

    #[allow(dead_code)]
    #[inline]
    pub fn append_null(&mut self) {
        self.materialize_if_needed();
        self.inner
            .as_mut()
            .vortex_expect("cannot append null to non-nullable builder")
            .append(false);
    }

    #[allow(dead_code)]
    #[inline]
    pub fn append(&mut self, not_null: bool) {
        if not_null {
            self.append_non_null()
        } else {
            self.append_null()
        }
    }

    #[inline]
    pub fn append_buffer(&mut self, bool_buffer: &BooleanBuffer) {
        self.materialize_if_needed();
        self.inner
            .as_mut()
            .vortex_expect("buffer just materialized")
            .append_buffer(bool_buffer);
    }

    pub fn append_validity_mask(&mut self, validity_mask: Mask) {
        match validity_mask {
            Mask::AllTrue(len) => self.append_n_non_nulls(len),
            Mask::AllFalse(len) => self.append_n_nulls(len),
            Mask::Values(is_valid) => self.append_buffer(is_valid.boolean_buffer()),
        }
    }

    pub fn append_validity(&mut self, validity: &Validity, length: usize) -> VortexResult<()> {
        self.append_validity_mask(validity.to_logical(length)?);
        Ok(())
    }

    pub fn set_bit(&mut self, index: usize, v: bool) {
        self.materialize_if_needed();
        self.inner
            .as_mut()
            .vortex_expect("buffer just materialized")
            .set_bit(index, v);
    }

    pub fn len(&self) -> usize {
        // self.len is the length of the builder if the inner buffer is not materialized
        self.inner.as_ref().map(|i| i.len()).unwrap_or(self.len)
    }

    pub fn truncate(&mut self, len: usize) {
        if let Some(b) = self.inner.as_mut() {
            b.truncate(len)
        }
        self.len = len;
    }

    pub fn reserve(&mut self, n: usize) {
        self.materialize_if_needed();
        self.inner
            .as_mut()
            .vortex_expect("buffer just materialized")
            .reserve(n);
    }

    pub fn finish(&mut self) -> Option<NullBuffer> {
        self.len = 0;
        Some(NullBuffer::new(self.inner.take()?.finish()))
    }

    pub fn finish_with_nullability(&mut self, nullability: Nullability) -> Validity {
        let nulls = self.finish();
        match (nullability, nulls) {
            (NonNullable, None) => Validity::NonNullable,
            (Nullable, None) => Validity::AllValid,
            (Nullable, Some(arr)) => Validity::from(arr),
            _ => vortex_panic!("Invalid nullability/nulls combination"),
        }
    }

    #[inline]
    fn materialize_if_needed(&mut self) {
        if self.inner.is_none() {
            self.materialize()
        }
    }

    // This only happens once per builder
    #[cold]
    #[inline(never)]
    fn materialize(&mut self) {
        if self.inner.is_none() {
            let mut b = BooleanBufferBuilder::new(self.len.max(self.capacity));
            b.append_n(self.len, true);
            self.inner = Some(b);
        }
    }
}
