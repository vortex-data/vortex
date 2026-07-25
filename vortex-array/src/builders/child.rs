// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::ChunkedArray;
use crate::arrays::MaskedArray;
use crate::builders::ArrayBuilder;
use crate::builders::builder_with_capacity;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::scalar::Scalar;
use crate::validity::Validity;

/// The shortest array that a [`ChildBuilder`] keeps as a chunk of its own.
///
/// Nested builders routinely append very short arrays (the elements of a single list, for
/// example), and giving each one its own chunk would produce a [`ChunkedArray`] with more chunks
/// than values. Below this length, copying the values costs less than the indirection that every
/// later access to the chunk would pay.
pub(super) const MIN_CHUNK_LEN: usize = 64;

/// Accumulates the child of a nested [`ArrayBuilder`] without canonicalizing appended arrays.
///
/// Nested builders receive values from two sources: individual [`Scalar`]s, which have to be
/// materialized into a canonical builder, and whole arrays, whose encoding the builder has no
/// reason to decode. A `ChildBuilder` keeps the latter as chunks and materializes only the former,
/// stitching everything back together into a [`ChunkedArray`] on [`finish`](Self::finish) when
/// more than one chunk accumulated.
///
/// This keeps canonical arrays canonical only at the top level, which is all that [`Canonical`]
/// promises: the fields of a `StructArray`, the elements of a list, and the storage of an
/// extension array may all stay compressed.
///
/// [`Canonical`]: crate::Canonical
pub struct ChildBuilder {
    /// The [`DType`] shared by every chunk and by the scalar builder.
    dtype: DType,

    /// Completed chunks, in logical order. Never contains an empty chunk.
    chunks: Vec<ArrayRef>,

    /// The summed length of `chunks`.
    chunks_len: usize,

    /// Builder holding the scalars appended after the last chunk.
    pending: Box<dyn ArrayBuilder>,
}

impl ChildBuilder {
    /// Creates a new `ChildBuilder` whose scalar builder is pre-allocated for `capacity` values.
    pub fn with_capacity(dtype: &DType, capacity: usize) -> Self {
        Self {
            dtype: dtype.clone(),
            chunks: Vec::new(),
            chunks_len: 0,
            pending: builder_with_capacity(dtype, capacity),
        }
    }

    /// The number of values appended so far.
    pub fn len(&self) -> usize {
        self.chunks_len + self.pending.len()
    }

    /// Appends every value of `array` to the child.
    ///
    /// Arrays long enough to earn a chunk are kept in their original encoding; shorter ones are
    /// materialized into the scalar builder.
    pub fn append_array(&mut self, array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<()> {
        vortex_ensure!(
            array.dtype() == &self.dtype,
            "Cannot append an array of dtype {} to a child builder of dtype {}",
            array.dtype(),
            self.dtype,
        );

        if array.is_empty() {
            return Ok(());
        }

        if array.len() < MIN_CHUNK_LEN {
            return array.append_to_builder(self.pending.as_mut(), ctx);
        }

        self.flush_pending();
        self.chunks_len += array.len();
        self.chunks.push(array.clone());

        Ok(())
    }

    /// Appends a single [`Scalar`] to the child.
    pub fn append_scalar(&mut self, scalar: &Scalar) -> VortexResult<()> {
        self.pending.append_scalar(scalar)
    }

    /// Appends `n` "zero" values to the child.
    ///
    /// See [`ArrayBuilder::append_zeros`].
    pub fn append_zeros(&mut self, n: usize) {
        self.pending.append_zeros(n)
    }

    /// Appends `n` null values to the child.
    ///
    /// See [`ArrayBuilder::append_nulls`].
    pub fn append_nulls(&mut self, n: usize) {
        self.pending.append_nulls(n)
    }

    /// Appends `n` default values to the child.
    ///
    /// See [`ArrayBuilder::append_defaults`].
    pub fn append_defaults(&mut self, n: usize) {
        self.pending.append_defaults(n)
    }

    /// Allocates space for `additional` more values in the scalar builder.
    pub fn reserve_exact(&mut self, additional: usize) {
        self.pending.reserve_exact(additional)
    }

    /// Overrides the validity of every value appended so far.
    ///
    /// # Safety
    ///
    /// `validity` must have the same length as [`self.len()`](Self::len).
    ///
    /// # Panics
    ///
    /// Panics if a chunk that was kept in its original encoding contains nulls, since replacing
    /// the validity of such a chunk would require decoding it.
    pub unsafe fn set_validity_unchecked(&mut self, validity: Mask) {
        if !self.dtype.is_nullable() {
            return;
        }

        if self.chunks.is_empty() {
            // Fast path: every value lives in the scalar builder, which owns its null buffer.
            unsafe { self.pending.set_validity_unchecked(validity) };
            return;
        }

        // The chunks carry their own validity, so the override has to be pushed into each of them.
        self.flush_pending();
        let mut offset = 0;
        for chunk in &mut self.chunks {
            let end = offset + chunk.len();
            *chunk = MaskedArray::try_new(
                chunk.clone(),
                Validity::from_mask(validity.slice(offset..end), Nullability::Nullable),
            )
            .vortex_expect("cannot override the validity of a child chunk that contains nulls")
            .into_array();
            offset = end;
        }
    }

    /// Finishes the child, combining the accumulated chunks into a [`ChunkedArray`] when there is
    /// more than one of them.
    pub fn finish(&mut self) -> ArrayRef {
        if self.chunks.is_empty() {
            return self.pending.finish();
        }

        self.flush_pending();
        self.chunks_len = 0;

        let mut chunks = std::mem::take(&mut self.chunks);
        if chunks.len() == 1 {
            return chunks.remove(0);
        }

        ChunkedArray::try_new(chunks, self.dtype.clone())
            .vortex_expect("every child chunk has the child dtype")
            .into_array()
    }

    /// Moves whatever the scalar builder holds into `chunks`, keeping the chunks in logical order.
    fn flush_pending(&mut self) {
        if self.pending.is_empty() {
            return;
        }
        self.chunks_len += self.pending.len();
        let pending = self.pending.finish();
        self.chunks.push(pending);
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use super::ChildBuilder;
    use super::MIN_CHUNK_LEN;
    use crate::ArrayRef;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::arrays::Chunked;
    use crate::arrays::ChunkedArray;
    use crate::arrays::Constant;
    use crate::arrays::ConstantArray;
    use crate::arrays::Primitive;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::chunked::ChunkedArrayExt;
    use crate::assert_arrays_eq;
    use crate::dtype::DType;
    use crate::dtype::Nullability::NonNullable;
    use crate::dtype::PType::I32;

    /// A non-canonical array of `len` values, all equal to `value`.
    fn constant(value: i32, len: usize) -> ArrayRef {
        ConstantArray::new(value, len).into_array()
    }

    #[test]
    fn test_long_arrays_are_kept_as_chunks() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let mut builder = ChildBuilder::with_capacity(&DType::from(I32), 0);

        builder.append_array(&constant(1, MIN_CHUNK_LEN), &mut ctx)?;
        builder.append_array(&constant(2, MIN_CHUNK_LEN), &mut ctx)?;
        assert_eq!(builder.len(), 2 * MIN_CHUNK_LEN);

        let child = builder.finish();
        let chunked = child.as_::<Chunked>();
        assert_eq!(chunked.nchunks(), 2);
        // The chunks were never decoded.
        assert!(chunked.iter_chunks().all(|c| c.is::<Constant>()));

        Ok(())
    }

    #[test]
    fn test_short_arrays_are_materialized() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let mut builder = ChildBuilder::with_capacity(&DType::from(I32), 0);

        builder.append_array(&constant(1, MIN_CHUNK_LEN - 1), &mut ctx)?;
        assert_eq!(builder.len(), MIN_CHUNK_LEN - 1);

        let child = builder.finish();
        assert!(child.is::<Primitive>());

        Ok(())
    }

    #[test]
    fn test_scalars_interleaved_with_chunks_keep_their_order() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let mut builder = ChildBuilder::with_capacity(&DType::from(I32), 0);

        builder.append_scalar(&1i32.into())?;
        builder.append_array(&constant(2, MIN_CHUNK_LEN), &mut ctx)?;
        builder.append_scalar(&3i32.into())?;

        let child = builder.finish();
        assert_eq!(child.len(), MIN_CHUNK_LEN + 2);

        let expected = ChunkedArray::try_new(
            vec![
                buffer![1i32].into_array(),
                constant(2, MIN_CHUNK_LEN),
                buffer![3i32].into_array(),
            ],
            DType::from(I32),
        )?
        .into_array();
        assert_arrays_eq!(&child, &expected, &mut ctx);

        Ok(())
    }

    #[test]
    fn test_single_chunk_is_not_wrapped() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let mut builder = ChildBuilder::with_capacity(&DType::from(I32), 0);

        builder.append_array(&constant(7, MIN_CHUNK_LEN), &mut ctx)?;

        let child = builder.finish();
        assert!(child.is::<Constant>());

        Ok(())
    }

    #[test]
    fn test_finish_resets_the_builder() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let mut builder = ChildBuilder::with_capacity(&DType::from(I32), 0);

        builder.append_array(&constant(1, MIN_CHUNK_LEN), &mut ctx)?;
        builder.append_scalar(&2i32.into())?;
        assert_eq!(builder.finish().len(), MIN_CHUNK_LEN + 1);

        assert_eq!(builder.len(), 0);
        builder.append_scalar(&3i32.into())?;

        let expected = PrimitiveArray::new(buffer![3i32], NonNullable.into()).into_array();
        assert_arrays_eq!(&builder.finish(), &expected, &mut ctx);

        Ok(())
    }
}
