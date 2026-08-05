// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::ChunkedArray;
use crate::builders::ArrayBuilder;
use crate::builders::builder_with_capacity;
use crate::dtype::DType;
use crate::scalar::Scalar;

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

    /// Appends every value of `array` to the child as a chunk of its own, keeping its encoding.
    ///
    /// However short the array, it becomes a chunk: the caller had a whole array to hand, and
    /// deciding on its behalf that its values are cheaper copied than referenced would be guessing
    /// at a boundary only the caller can see. Callers that would rather have the values copied
    /// should append them as scalars.
    ///
    /// Nothing is decoded here, so `_ctx` goes unused; it stays in the signature so that the
    /// nested builders forwarding their [`ExecutionCtx`] here do not have to explain why they
    /// don't.
    pub fn append_array(&mut self, array: &ArrayRef, _ctx: &mut ExecutionCtx) -> VortexResult<()> {
        vortex_ensure!(
            array.dtype() == &self.dtype,
            "Cannot append an array of dtype {} to a child builder of dtype {}",
            array.dtype(),
            self.dtype,
        );

        if array.is_empty() {
            return Ok(());
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

        unsafe { ChunkedArray::new_unchecked(chunks, self.dtype.clone()) }.into_array()
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
    use rstest::rstest;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use super::ChildBuilder;
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
    use crate::dtype::Nullability::Nullable;
    use crate::dtype::PType::I32;
    use crate::scalar::Scalar;

    /// An arbitrary array length. `ChildBuilder` treats no length specially, so the tests only
    /// need a length long enough to tell chunks apart.
    const CHUNK_LEN: usize = 64;

    /// A non-canonical array of `len` values, all equal to `value`.
    fn constant(value: i32, len: usize) -> ArrayRef {
        ConstantArray::new(value, len).into_array()
    }

    /// A non-canonical *nullable* array of `len` non-null values, all equal to `value`.
    fn nullable_constant(value: i32, len: usize) -> ArrayRef {
        ConstantArray::new(Scalar::primitive(value, Nullable), len).into_array()
    }

    #[test]
    fn test_appended_arrays_are_kept_as_chunks() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let mut builder = ChildBuilder::with_capacity(&DType::from(I32), 0);

        builder.append_array(&constant(1, CHUNK_LEN), &mut ctx)?;
        builder.append_array(&constant(2, CHUNK_LEN), &mut ctx)?;
        assert_eq!(builder.len(), 2 * CHUNK_LEN);

        let child = builder.finish();
        let chunked = child.as_::<Chunked>();
        assert_eq!(chunked.nchunks(), 2);
        // The chunks were never decoded.
        assert!(chunked.iter_chunks().all(|c| c.is::<Constant>()));

        Ok(())
    }

    /// However short the appended array, its encoding survives: a single-value array is a chunk
    /// too. A caller that wants the values copied appends them as scalars instead.
    #[test]
    fn test_short_arrays_are_kept_as_chunks_too() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let mut builder = ChildBuilder::with_capacity(&DType::from(I32), 0);

        builder.append_array(&constant(1, 1), &mut ctx)?;
        builder.append_array(&constant(2, 1), &mut ctx)?;
        assert_eq!(builder.len(), 2);

        let child = builder.finish();
        let chunked = child.as_::<Chunked>();
        assert_eq!(chunked.nchunks(), 2);
        assert!(chunked.iter_chunks().all(|c| c.is::<Constant>()));

        Ok(())
    }

    #[test]
    fn test_scalars_interleaved_with_chunks_keep_their_order() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let mut builder = ChildBuilder::with_capacity(&DType::from(I32), 0);

        builder.append_scalar(&1i32.into())?;
        builder.append_array(&constant(2, CHUNK_LEN), &mut ctx)?;
        builder.append_scalar(&3i32.into())?;

        let child = builder.finish();
        assert_eq!(child.len(), CHUNK_LEN + 2);

        let expected = ChunkedArray::try_new(
            vec![
                buffer![1i32].into_array(),
                constant(2, CHUNK_LEN),
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

        builder.append_array(&constant(7, CHUNK_LEN), &mut ctx)?;

        let child = builder.finish();
        assert!(child.is::<Constant>());

        Ok(())
    }

    #[test]
    fn test_empty_arrays_never_become_chunks() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let mut builder = ChildBuilder::with_capacity(&DType::from(I32), 0);
        let empty = constant(1, CHUNK_LEN).slice(0..0)?;

        builder.append_array(&empty, &mut ctx)?;
        builder.append_array(&constant(1, CHUNK_LEN), &mut ctx)?;
        builder.append_array(&empty, &mut ctx)?;
        builder.append_array(&constant(2, CHUNK_LEN), &mut ctx)?;
        builder.append_array(&empty, &mut ctx)?;

        assert_eq!(builder.len(), 2 * CHUNK_LEN);
        assert_eq!(builder.finish().as_::<Chunked>().nchunks(), 2);

        Ok(())
    }

    /// An empty child must still finish as an empty array rather than as an empty [`ChunkedArray`].
    #[test]
    fn test_empty_child_finishes_without_chunks() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let mut builder = ChildBuilder::with_capacity(&DType::from(I32), 0);

        builder.append_array(&constant(1, CHUNK_LEN).slice(0..0)?, &mut ctx)?;

        let child = builder.finish();
        assert!(child.is_empty());
        assert!(child.is::<Primitive>());

        Ok(())
    }

    /// The dtype check has to run before the empty check, so that a mismatched array is rejected
    /// whether or not it would have become a chunk.
    #[rstest]
    #[case::empty(0)]
    #[case::non_empty(CHUNK_LEN)]
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

        builder.append_array(&nullable_constant(1, CHUNK_LEN), &mut ctx)?;
        builder.append_nulls(2);
        builder.append_array(&nullable_constant(2, CHUNK_LEN), &mut ctx)?;
        builder.append_zeros(1);

        let child = builder.finish();
        assert_eq!(child.len(), 2 * CHUNK_LEN + 3);
        assert_eq!(child.as_::<Chunked>().nchunks(), 4);

        let expected = PrimitiveArray::from_option_iter(
            std::iter::repeat_n(Some(1i32), CHUNK_LEN)
                .chain([None, None])
                .chain(std::iter::repeat_n(Some(2i32), CHUNK_LEN))
                .chain([Some(0)]),
        )
        .into_array();
        assert_arrays_eq!(&child, &expected, &mut ctx);

        Ok(())
    }

    #[test]
    fn test_finish_resets_the_builder() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let mut builder = ChildBuilder::with_capacity(&DType::from(I32), 0);

        builder.append_array(&constant(1, CHUNK_LEN), &mut ctx)?;
        builder.append_scalar(&2i32.into())?;
        assert_eq!(builder.finish().len(), CHUNK_LEN + 1);

        assert_eq!(builder.len(), 0);
        builder.append_scalar(&3i32.into())?;

        let expected = PrimitiveArray::new(buffer![3i32], NonNullable.into()).into_array();
        assert_arrays_eq!(&builder.finish(), &expected, &mut ctx);

        Ok(())
    }
}
