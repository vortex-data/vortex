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

/// The default shortest array that a [`ChildBuilder`] keeps as a chunk of its own.
///
/// Nested builders routinely append very short arrays (the elements of a single list, for
/// example), and giving each one its own chunk would produce a [`ChunkedArray`] with more chunks
/// than values. Below this length, copying the values costs less than the indirection that every
/// later access to the chunk would pay.
///
/// Callers that would rather keep every chunk boundary, however short, can lower the threshold
/// with [`ArrayBuilder::set_min_chunk_len`].
pub(super) const DEFAULT_MIN_CHUNK_LEN: usize = 64;

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

    /// The shortest appended array that is kept as a chunk rather than copied.
    min_chunk_len: usize,
}

impl ChildBuilder {
    /// Creates a new `ChildBuilder` whose scalar builder is pre-allocated for `capacity` values.
    pub fn with_capacity(dtype: &DType, capacity: usize) -> Self {
        Self {
            dtype: dtype.clone(),
            chunks: Vec::new(),
            chunks_len: 0,
            pending: builder_with_capacity(dtype, capacity),
            min_chunk_len: DEFAULT_MIN_CHUNK_LEN,
        }
    }

    /// Sets the shortest appended array that is kept as a chunk rather than copied.
    ///
    /// See [`ArrayBuilder::set_min_chunk_len`].
    pub fn set_min_chunk_len(&mut self, min_chunk_len: usize) {
        self.min_chunk_len = min_chunk_len;
        self.pending.set_min_chunk_len(min_chunk_len);
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

        if array.len() < self.min_chunk_len {
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
    use std::sync::Arc;

    use rstest::rstest;
    use vortex_buffer::buffer;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;
    use vortex_mask::Mask;

    use super::ChildBuilder;
    use super::DEFAULT_MIN_CHUNK_LEN as MIN_CHUNK_LEN;
    use crate::ArrayRef;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::arrays::Chunked;
    use crate::arrays::ChunkedArray;
    use crate::arrays::Constant;
    use crate::arrays::ConstantArray;
    use crate::arrays::ListView;
    use crate::arrays::ListViewArray;
    use crate::arrays::Masked;
    use crate::arrays::Primitive;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::chunked::ChunkedArrayExt;
    use crate::arrays::listview::ListViewArraySlotsExt;
    use crate::assert_arrays_eq;
    use crate::dtype::DType;
    use crate::dtype::Nullability::NonNullable;
    use crate::dtype::Nullability::Nullable;
    use crate::dtype::PType::I32;
    use crate::scalar::Scalar;
    use crate::validity::Validity;

    /// A non-canonical array of `len` values, all equal to `value`.
    fn constant(value: i32, len: usize) -> ArrayRef {
        ConstantArray::new(value, len).into_array()
    }

    /// A non-canonical *nullable* array of `len` non-null values, all equal to `value`.
    fn nullable_constant(value: i32, len: usize) -> ArrayRef {
        ConstantArray::new(Scalar::primitive(value, Nullable), len).into_array()
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
    fn test_empty_arrays_never_become_chunks() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let mut builder = ChildBuilder::with_capacity(&DType::from(I32), 0);
        let empty = constant(1, MIN_CHUNK_LEN).slice(0..0)?;

        builder.append_array(&empty, &mut ctx)?;
        builder.append_array(&constant(1, MIN_CHUNK_LEN), &mut ctx)?;
        builder.append_array(&empty, &mut ctx)?;
        builder.append_array(&constant(2, MIN_CHUNK_LEN), &mut ctx)?;
        builder.append_array(&empty, &mut ctx)?;

        assert_eq!(builder.len(), 2 * MIN_CHUNK_LEN);
        assert_eq!(builder.finish().as_::<Chunked>().nchunks(), 2);

        Ok(())
    }

    /// An empty child must still finish as an empty array rather than as an empty [`ChunkedArray`].
    #[test]
    fn test_empty_child_finishes_without_chunks() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let mut builder = ChildBuilder::with_capacity(&DType::from(I32), 0);

        builder.append_array(&constant(1, MIN_CHUNK_LEN).slice(0..0)?, &mut ctx)?;

        let child = builder.finish();
        assert!(child.is_empty());
        assert!(child.is::<Primitive>());

        Ok(())
    }

    /// The dtype check has to run before the length check, so that a mismatched array is rejected
    /// whether it would have been chunked or materialized.
    #[rstest]
    #[case::would_be_materialized(MIN_CHUNK_LEN - 1)]
    #[case::would_be_chunked(MIN_CHUNK_LEN)]
    fn test_appending_a_mismatched_dtype_is_rejected(#[case] len: usize) {
        let mut ctx = array_session().create_execution_ctx();
        let mut builder = ChildBuilder::with_capacity(&DType::from(I32), 0);

        let wrong_dtype = ConstantArray::new(1i64, len).into_array();
        assert!(builder.append_array(&wrong_dtype, &mut ctx).is_err());
    }

    /// Everything the scalar builder can produce has to be flushed ahead of the next chunk.
    #[test]
    fn test_zeros_and_nulls_around_chunks_keep_their_order() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let dtype = DType::Primitive(I32, Nullable);
        let mut builder = ChildBuilder::with_capacity(&dtype, 0);

        builder.append_array(&nullable_constant(1, MIN_CHUNK_LEN), &mut ctx)?;
        builder.append_nulls(2);
        builder.append_array(&nullable_constant(2, MIN_CHUNK_LEN), &mut ctx)?;
        builder.append_zeros(1);

        let child = builder.finish();
        assert_eq!(child.len(), 2 * MIN_CHUNK_LEN + 3);
        assert_eq!(child.as_::<Chunked>().nchunks(), 4);

        let expected = PrimitiveArray::from_option_iter(
            std::iter::repeat_n(Some(1i32), MIN_CHUNK_LEN)
                .chain([None, None])
                .chain(std::iter::repeat_n(Some(2i32), MIN_CHUNK_LEN))
                .chain([Some(0)]),
        )
        .into_array();
        assert_arrays_eq!(&child, &expected, &mut ctx);

        Ok(())
    }

    /// Overriding the validity once chunks exist has to push the override into each chunk, sliced
    /// to that chunk's own range.
    #[test]
    fn test_set_validity_pushes_the_override_into_every_chunk() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let dtype = DType::Primitive(I32, Nullable);
        let mut builder = ChildBuilder::with_capacity(&dtype, 0);

        builder.append_array(&nullable_constant(1, MIN_CHUNK_LEN), &mut ctx)?;
        builder.append_array(&nullable_constant(2, MIN_CHUNK_LEN), &mut ctx)?;

        // Straddle the chunk boundary, so an override sliced wrongly cannot pass.
        let invalid = [MIN_CHUNK_LEN - 1, MIN_CHUNK_LEN];
        let validity = Mask::from_iter((0..2 * MIN_CHUNK_LEN).map(|i| !invalid.contains(&i)));
        unsafe { builder.set_validity_unchecked(validity) };

        let child = builder.finish();
        let chunked = child.as_::<Chunked>();
        assert_eq!(chunked.nchunks(), 2);
        // The override was layered over the chunks rather than decoding them.
        assert!(chunked.iter_chunks().all(|chunk| chunk.is::<Masked>()));

        let expected =
            PrimitiveArray::from_option_iter((0..2 * MIN_CHUNK_LEN).map(|i| {
                (!invalid.contains(&i)).then_some(if i < MIN_CHUNK_LEN { 1i32 } else { 2 })
            }))
            .into_array();
        assert_arrays_eq!(&child, &expected, &mut ctx);

        Ok(())
    }

    /// Values still sitting in the scalar builder are part of the override too.
    #[test]
    fn test_set_validity_covers_pending_scalars() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let dtype = DType::Primitive(I32, Nullable);
        let mut builder = ChildBuilder::with_capacity(&dtype, 0);

        builder.append_array(&nullable_constant(1, MIN_CHUNK_LEN), &mut ctx)?;
        builder.append_scalar(&Scalar::primitive(2i32, Nullable))?;

        let mut validity = vec![true; MIN_CHUNK_LEN + 1];
        validity[MIN_CHUNK_LEN] = false;
        unsafe { builder.set_validity_unchecked(Mask::from_iter(validity)) };

        let child = builder.finish();
        let expected = PrimitiveArray::from_option_iter(
            std::iter::repeat_n(Some(1i32), MIN_CHUNK_LEN).chain([None]),
        )
        .into_array();
        assert_arrays_eq!(&child, &expected, &mut ctx);

        Ok(())
    }

    /// A non-nullable child cannot carry nulls, so the override is dropped and the chunks are left
    /// exactly as they were appended.
    #[test]
    fn test_set_validity_is_a_noop_for_a_non_nullable_child() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let mut builder = ChildBuilder::with_capacity(&DType::from(I32), 0);

        builder.append_array(&constant(1, MIN_CHUNK_LEN), &mut ctx)?;
        builder.append_array(&constant(2, MIN_CHUNK_LEN), &mut ctx)?;
        unsafe { builder.set_validity_unchecked(Mask::new_false(2 * MIN_CHUNK_LEN)) };

        let child = builder.finish();
        let chunked = child.as_::<Chunked>();
        assert!(chunked.iter_chunks().all(|chunk| chunk.is::<Constant>()));

        let expected = ChunkedArray::try_new(
            vec![constant(1, MIN_CHUNK_LEN), constant(2, MIN_CHUNK_LEN)],
            DType::from(I32),
        )?
        .into_array();
        assert_arrays_eq!(&child, &expected, &mut ctx);

        Ok(())
    }

    /// Replacing the validity of a chunk that already contains nulls would mean decoding it, which
    /// is exactly what the chunk exists to avoid.
    #[test]
    #[should_panic(expected = "cannot override the validity of a child chunk that contains nulls")]
    fn test_set_validity_rejects_a_chunk_that_contains_nulls() {
        let mut ctx = array_session().create_execution_ctx();
        let dtype = DType::Primitive(I32, Nullable);
        let mut builder = ChildBuilder::with_capacity(&dtype, 0);

        let with_nulls = ConstantArray::new(Scalar::null(dtype), MIN_CHUNK_LEN).into_array();
        builder
            .append_array(&with_nulls, &mut ctx)
            .vortex_expect("append");

        unsafe { builder.set_validity_unchecked(Mask::new_true(MIN_CHUNK_LEN)) };
    }

    /// A threshold of zero keeps every chunk boundary, however short.
    #[test]
    fn test_min_chunk_len_zero_keeps_every_chunk() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let mut builder = ChildBuilder::with_capacity(&DType::from(I32), 0);
        builder.set_min_chunk_len(0);

        builder.append_array(&constant(1, 1), &mut ctx)?;
        builder.append_array(&constant(2, 1), &mut ctx)?;

        let child = builder.finish();
        let chunked = child.as_::<Chunked>();
        assert_eq!(chunked.nchunks(), 2);
        assert!(chunked.iter_chunks().all(|chunk| chunk.is::<Constant>()));

        Ok(())
    }

    /// The threshold reaches the children of a child: an array short enough to be materialized
    /// lands in the scalar builder, which is itself a nested builder with children of its own.
    #[test]
    fn test_min_chunk_len_applies_transitively() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let dtype = DType::List(Arc::new(DType::from(I32)), NonNullable);
        let mut builder = ChildBuilder::with_capacity(&dtype, 0);
        builder.set_min_chunk_len(4 * MIN_CHUNK_LEN);

        // Two lists — far too few to be a chunk — holding enough elements between them that those
        // elements would become a chunk if the threshold stopped at the outer builder.
        let lists = ListViewArray::new(
            constant(1, 2 * MIN_CHUNK_LEN),
            buffer![0u64, MIN_CHUNK_LEN as u64].into_array(),
            buffer![MIN_CHUNK_LEN as u64; 2].into_array(),
            Validity::NonNullable,
        )
        .into_array();
        builder.append_array(&lists, &mut ctx)?;

        let child = builder.finish();
        assert!(child.as_::<ListView>().elements().is::<Primitive>());

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
