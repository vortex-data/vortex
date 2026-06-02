// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::BitBuffer;
use vortex_buffer::BitBufferMut;
use vortex_error::VortexExpect;
use vortex_error::vortex_panic;
use vortex_mask::Mask;

use crate::dtype::Nullability;
use crate::dtype::Nullability::NonNullable;
use crate::dtype::Nullability::Nullable;
use crate::validity::Validity;

/// This is borrowed from arrow's null buffer builder, however we expose a `append_buffer`
/// method to append a boolean buffer directly.
pub struct LazyBitBufferBuilder {
    inner: Option<BitBufferMut>,
    len: usize,
    capacity: usize,
}

impl LazyBitBufferBuilder {
    /// Creates a new empty builder.
    /// `capacity` is the number of bits in the null buffer.
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: None,
            len: 0,
            capacity,
        }
    }

    /// Creates a builder pre-populated from a validity mask, taking ownership of the mask's buffer
    /// instead of copying it where possible.
    ///
    /// This is the counterpart to [`append_validity_mask`](Self::append_validity_mask) for callers
    /// that want to *replace* the builder's contents with the mask rather than extend them: because
    /// we own the mask, we can move its buffer in instead of copying it.
    pub fn from_validity_mask(validity_mask: Mask) -> Self {
        match validity_mask {
            // An unmaterialized builder already represents `len` non-null values, so an all-valid
            // mask stays lazy.
            Mask::AllTrue(len) => Self {
                inner: None,
                len,
                capacity: len,
            },
            Mask::AllFalse(len) => Self::from_buffer(BitBufferMut::new_unset(len)),
            // Take ownership of the underlying buffer; `into_bit_buffer` and `try_into_mut` only
            // copy when the buffer is shared, otherwise this is a move.
            values @ Mask::Values(_) => Self::from_buffer(
                values
                    .into_bit_buffer()
                    .try_into_mut()
                    .unwrap_or_else(|buffer| BitBufferMut::copy_from(&buffer)),
            ),
        }
    }

    /// Creates a builder backed by an already-materialized buffer.
    fn from_buffer(inner: BitBufferMut) -> Self {
        Self {
            inner: Some(inner),
            len: 0,
            capacity: 0,
        }
    }

    /// Appends `n` non-null values to the builder.
    #[inline]
    pub fn append_n_non_nulls(&mut self, n: usize) {
        if let Some(buf) = self.inner.as_mut() {
            buf.append_n(true, n)
        } else {
            self.len += n;
        }
    }

    /// Appends a single non-null value to the builder.
    #[inline]
    pub fn append_non_null(&mut self) {
        if let Some(buf) = self.inner.as_mut() {
            buf.append(true)
        } else {
            self.len += 1;
        }
    }

    /// Appends `n` null values to the builder.
    #[inline]
    pub fn append_n_nulls(&mut self, n: usize) {
        self.materialize_if_needed();
        self.inner
            .as_mut()
            .vortex_expect("cannot append null to non-nullable builder")
            .append_n(false, n);
    }

    /// Appends a single null value to the builder.
    #[inline]
    pub fn append_null(&mut self) {
        self.materialize_if_needed();
        self.inner
            .as_mut()
            .vortex_expect("cannot append null to non-nullable builder")
            .append(false);
    }

    /// Appends values from a boolean buffer where `true` indicates non-null.
    #[inline]
    pub fn append_buffer(&mut self, bool_buffer: &BitBuffer) {
        self.materialize_if_needed();
        self.inner
            .as_mut()
            .vortex_expect("buffer just materialized")
            .append_buffer(bool_buffer);
    }

    /// Appends values from a validity mask.
    ///
    /// Takes the mask by reference: the `Mask::Values` case copies the underlying buffer into the
    /// running null buffer, so there is nothing to gain from owning the mask.
    pub fn append_validity_mask(&mut self, validity_mask: &Mask) {
        match validity_mask {
            Mask::AllTrue(len) => self.append_n_non_nulls(*len),
            Mask::AllFalse(len) => self.append_n_nulls(*len),
            Mask::Values(is_valid) => self.append_buffer(is_valid.bit_buffer()),
        }
    }

    /// Sets the validity bit at the given index.
    pub fn set_bit(&mut self, index: usize, v: bool) {
        self.materialize_if_needed();
        self.inner
            .as_mut()
            .vortex_expect("buffer just materialized")
            .set_to(index, v);
    }

    /// Returns the current length of the builder.
    pub fn len(&self) -> usize {
        // self.len is the length of the builder if the inner buffer is not materialized
        self.inner.as_ref().map(|i| i.len()).unwrap_or(self.len)
    }

    fn finish(&mut self) -> Option<BitBuffer> {
        self.len = 0;
        self.inner.take().map(|b| b.freeze())
    }

    /// Finishes the builder and returns a `Validity` based on the given nullability.
    pub fn finish_with_nullability(&mut self, nullability: Nullability) -> Validity {
        let nulls = self.finish();

        match (nullability, nulls) {
            (NonNullable, None) => Validity::NonNullable,
            (Nullable, None) => Validity::AllValid,
            (Nullable, Some(arr)) => Validity::from(arr),
            _ => vortex_panic!("Invalid nullability/nulls combination"),
        }
    }

    /// Ensures the builder can hold `additional` extra values.
    pub fn reserve_exact(&mut self, additional: usize) {
        if self.inner.is_none() {
            self.capacity += additional;
        } else {
            self.inner
                .as_mut()
                .vortex_expect("buffer just materialized")
                .reserve(additional);
        }
    }

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
            let mut bit_mut = BitBufferMut::with_capacity(self.len.max(self.capacity));
            bit_mut.append_n(true, self.len);
            self.inner = Some(bit_mut);
        }
    }
}
