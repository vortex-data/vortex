// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Chunked decode for [`ListView`] arrays with primitive elements.
//!
//! The canonical-by-canonical path for `ListView<Primitive>` decompresses the offsets,
//! sizes, and elements buffers up-front. When offsets and sizes are themselves
//! bit-packed (the common case), that's three full materializations before a consumer
//! sees a single row.
//!
//! The chunked path emits **row windows**: each window is a `(offsets, sizes,
//! elements_slice)` triple covering up to [`CHUNK_LEN`] rows. The producer keeps a
//! reusable [`Scratch`] for the offsets and another for the sizes (so the steady-state
//! decompress footprint is two scratches, not three full buffers), and the elements
//! slice is just a borrowed view onto the shared, already-materialized elements buffer.
//!
//! For v1 we materialize the elements buffer once up-front (since `ListView` allows
//! arbitrary, possibly-overlapping offsets we don't know in advance which slice each
//! chunk needs without a pre-pass). A future iteration can replace this with a
//! per-chunk min-offset / max-offset+size pre-pass and decompress only the referenced
//! slice — the producer API already supports that shape.

use std::mem::MaybeUninit;

use vortex_buffer::Buffer;
use vortex_error::VortexResult;

use super::CHUNK_LEN;
use super::scratch::Scratch;
use crate::ArrayRef;
use crate::arrays::ListView;
use crate::arrays::PrimitiveArray;
use crate::arrays::listview::ListViewArrayExt;
use crate::arrays::primitive::PrimitiveArrayExt;
use crate::dtype::NativePType;
use crate::dtype::Nullability;
use crate::executor::ExecutionCtx;
use crate::match_each_integer_ptype;

/// A row window emitted by [`ListChunkProducer`].
///
/// The lifetime is tied to the producer-owned `Scratch` buffers — the consumer must
/// finish reading the chunk before requesting the next.
pub struct ListChunk<'a, O, S, E> {
    /// Offsets for this window of rows.
    pub offsets: &'a [O],
    /// Sizes for this window of rows.
    pub sizes: &'a [S],
    /// Shared reference to the (fully materialized) elements buffer.
    pub elements: &'a [E],
}

/// Streams chunks of `(offsets, sizes)` for a list-of-primitive array.
///
/// `O` is the offset native type (`u32` or `u64`), `S` the size native type, and `E` the
/// element native type.
pub struct ListChunkProducer<O: NativePType, S: NativePType, E: NativePType> {
    offsets: Buffer<O>,
    sizes: Buffer<S>,
    elements: Buffer<E>,
    cursor: usize,
}

impl<O, S, E> ListChunkProducer<O, S, E>
where
    O: NativePType,
    S: NativePType,
    E: NativePType,
{
    /// Construct directly from canonicalized buffers.
    pub fn new(offsets: Buffer<O>, sizes: Buffer<S>, elements: Buffer<E>) -> Self {
        Self {
            offsets,
            sizes,
            elements,
            cursor: 0,
        }
    }

    /// Number of rows that remain to be emitted.
    pub fn remaining(&self) -> usize {
        self.offsets.as_slice().len().saturating_sub(self.cursor)
    }

    /// Total number of rows.
    pub fn len(&self) -> usize {
        self.offsets.as_slice().len()
    }

    /// Whether there are any rows at all.
    pub fn is_empty(&self) -> bool {
        self.offsets.as_slice().is_empty()
    }

    /// Drive the producer to completion, calling `f` with each row chunk as typed
    /// slices. This is the fast-path API for callers who know the offset/size types
    /// at compile time — it has no dyn-dispatch in the hot loop.
    pub fn for_each_chunk_typed<F>(&mut self, mut f: F)
    where
        F: FnMut(&[O], &[S], &[E]),
    {
        let mut o_scratch = Scratch::<O>::new();
        let mut s_scratch = Scratch::<S>::new();
        while let Some(chunk) = self.next_chunk(&mut o_scratch, &mut s_scratch) {
            f(chunk.offsets, chunk.sizes, chunk.elements);
        }
    }

    /// Pull the next row window. Returns slices directly out of the already-canonical
    /// `offsets`/`sizes`/`elements` buffers — no memcpy, no scratch hop.
    ///
    /// The scratch arguments are kept for API symmetry with future chunked-bit-unpack
    /// variants where the producer materialises its own offsets/sizes per chunk; for
    /// the current canonical-then-chunk producer they are unused.
    pub fn next_chunk<'a>(
        &'a mut self,
        _offset_scratch: &'a mut Scratch<O>,
        _size_scratch: &'a mut Scratch<S>,
    ) -> Option<ListChunk<'a, O, S, E>> {
        let total = self.offsets.as_slice().len();
        if self.cursor >= total {
            return None;
        }
        let take = CHUNK_LEN.min(total - self.cursor);
        let offsets = &self.offsets.as_slice()[self.cursor..self.cursor + take];
        let sizes = &self.sizes.as_slice()[self.cursor..self.cursor + take];
        self.cursor += take;
        Some(ListChunk {
            offsets,
            sizes,
            elements: self.elements.as_slice(),
        })
    }
}

/// Build a typed [`ListChunkProducer`] directly. The caller commits to the offset and
/// size native types up front; the producer is callable without dyn dispatch in the hot
/// loop.
///
/// Errors if the array's offset/size/element ptypes don't match `<O, S, E>`.
pub fn build_listview_producer_typed<O, S, E>(
    array: ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ListChunkProducer<O, S, E>>
where
    O: NativePType,
    S: NativePType,
    E: NativePType,
{
    let Some(lv) = array.as_opt::<ListView>() else {
        vortex_error::vortex_bail!(
            "build_listview_producer_typed: expected ListView, got {}",
            array.encoding_id()
        );
    };
    if !matches!(array.dtype().nullability(), Nullability::NonNullable) {
        vortex_error::vortex_bail!("build_listview_producer_typed: only non-nullable for v1");
    }
    let offsets = lv.offsets().clone().execute::<PrimitiveArray>(ctx)?;
    let sizes = lv.sizes().clone().execute::<PrimitiveArray>(ctx)?;
    let elements = lv.elements().clone().execute::<PrimitiveArray>(ctx)?;
    if O::PTYPE != offsets.ptype()
        || S::PTYPE != sizes.ptype()
        || E::PTYPE != elements.ptype()
    {
        vortex_error::vortex_bail!(
            "build_listview_producer_typed: ptype mismatch ({}, {}, {}) vs requested ({}, {}, {})",
            offsets.ptype(),
            sizes.ptype(),
            elements.ptype(),
            O::PTYPE,
            S::PTYPE,
            E::PTYPE,
        );
    }
    Ok(ListChunkProducer::<O, S, E>::new(
        offsets.into_buffer::<O>(),
        sizes.into_buffer::<S>(),
        elements.into_buffer::<E>(),
    ))
}

/// Build a [`ListChunkProducer`] for a `ListView<Primitive>` array.
///
/// Canonicalizes offsets, sizes and elements once via the existing executor, then returns
/// a producer that streams row windows. The producer can be used to drive a downstream
/// consumer without ever materializing a full `ListViewArray`.
pub fn build_listview_primitive_producer<E: NativePType>(
    array: ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<BoxedListChunkProducer<E>> {
    let Some(lv) = array.as_opt::<ListView>() else {
        vortex_error::vortex_bail!(
            "build_listview_primitive_producer: expected ListView, got {}",
            array.encoding_id()
        );
    };
    if !matches!(array.dtype().nullability(), Nullability::NonNullable) {
        vortex_error::vortex_bail!(
            "build_listview_primitive_producer: only non-nullable for v1"
        );
    }
    let offsets = lv.offsets().clone().execute::<PrimitiveArray>(ctx)?;
    let sizes = lv.sizes().clone().execute::<PrimitiveArray>(ctx)?;
    let elements = lv.elements().clone().execute::<PrimitiveArray>(ctx)?;
    if E::PTYPE != elements.ptype() {
        vortex_error::vortex_bail!(
            "build_listview_primitive_producer: element type {} does not match {}",
            E::PTYPE,
            elements.ptype()
        );
    }
    let elements_buf = elements.into_buffer::<E>();
    Ok(match_each_integer_ptype!(offsets.ptype(), |O| {
        match_each_integer_ptype!(sizes.ptype(), |S| {
            BoxedListChunkProducer::new::<O, S>(
                offsets.into_buffer::<O>(),
                sizes.into_buffer::<S>(),
                elements_buf,
            )
        })
    }))
}

/// Type-erased entry point handle that lets callers dispatch to a `ListChunkProducer`
/// without monomorphizing on every `(O, S, E)` triple.
pub struct BoxedListChunkProducer<E: NativePType> {
    inner: Box<dyn ListChunkProducerErased<E>>,
}

impl<E: NativePType> BoxedListChunkProducer<E> {
    fn new<O, S>(offsets: Buffer<O>, sizes: Buffer<S>, elements: Buffer<E>) -> Self
    where
        O: NativePType + num_traits::AsPrimitive<usize> + Send + 'static,
        S: NativePType + num_traits::AsPrimitive<usize> + Send + 'static,
        E: NativePType + Send + 'static,
    {
        Self {
            inner: Box::new(ErasedProducer::<O, S, E>::new(ListChunkProducer::new(
                offsets, sizes, elements,
            ))),
        }
    }

    /// Total rows.
    pub fn len(&self) -> usize {
        self.inner.len_erased()
    }

    /// Whether the producer has any rows.
    pub fn is_empty(&self) -> bool {
        self.inner.len_erased() == 0
    }

    /// Drive one chunk and call `f` with the (offsets, sizes, elements) triple. The
    /// closure may borrow the chunk for the duration of the call. Returns `Some(())` if
    /// a chunk was produced; `None` when exhausted.
    ///
    /// We hide the offset/size native types behind a `dyn` adaptor so callers don't
    /// have to monomorphize over every integer combination.
    pub fn for_each_chunk<F>(&mut self, mut f: F)
    where
        F: FnMut(ListChunkErased<'_, E>),
    {
        while let Some(()) = self.inner.next_chunk_erased(&mut f) {}
    }
}

/// Type-erased chunk view passed to [`BoxedListChunkProducer::for_each_chunk`].
pub struct ListChunkErased<'a, E: NativePType> {
    /// Number of rows in this chunk.
    pub n: usize,
    /// Function that returns row `i`'s offset as `usize`.
    pub offset_of: &'a dyn Fn(usize) -> usize,
    /// Function that returns row `i`'s size as `usize`.
    pub size_of: &'a dyn Fn(usize) -> usize,
    /// The shared elements buffer.
    pub elements: &'a [E],
}

trait ListChunkProducerErased<E: NativePType>: Send {
    fn len_erased(&self) -> usize;
    fn next_chunk_erased(
        &mut self,
        f: &mut dyn FnMut(ListChunkErased<'_, E>),
    ) -> Option<()>;
}

struct ErasedProducer<O: NativePType, S: NativePType, E: NativePType> {
    inner: ListChunkProducer<O, S, E>,
    /// Persisted scratches so we don't pay a heap allocation per chunk.
    offset_scratch: Scratch<O>,
    size_scratch: Scratch<S>,
}

impl<O: NativePType, S: NativePType, E: NativePType> ErasedProducer<O, S, E> {
    fn new(inner: ListChunkProducer<O, S, E>) -> Self {
        Self {
            inner,
            offset_scratch: Scratch::<O>::new(),
            size_scratch: Scratch::<S>::new(),
        }
    }
}

impl<O, S, E> ListChunkProducerErased<E> for ErasedProducer<O, S, E>
where
    O: NativePType + num_traits::AsPrimitive<usize> + Send,
    S: NativePType + num_traits::AsPrimitive<usize> + Send,
    E: NativePType + Send,
{
    fn len_erased(&self) -> usize {
        self.inner.len()
    }

    fn next_chunk_erased(
        &mut self,
        f: &mut dyn FnMut(ListChunkErased<'_, E>),
    ) -> Option<()> {
        let chunk = self
            .inner
            .next_chunk(&mut self.offset_scratch, &mut self.size_scratch)?;
        let offsets = chunk.offsets;
        let sizes = chunk.sizes;
        let elements = chunk.elements;
        let off_fn = |i: usize| -> usize { offsets[i].as_() };
        let sz_fn = |i: usize| -> usize { sizes[i].as_() };
        f(ListChunkErased {
            n: offsets.len(),
            offset_of: &off_fn,
            size_of: &sz_fn,
            elements,
        });
        Some(())
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use super::*;
    use crate::IntoArray;
    use crate::LEGACY_SESSION;
    use crate::VortexSessionExecute;
    use crate::arrays::ListViewArray;
    use crate::validity::Validity;

    fn ctx() -> ExecutionCtx {
        LEGACY_SESSION.create_execution_ctx()
    }

    #[test]
    fn list_chunked_round_trip() -> VortexResult<()> {
        // 3 lists: [1,2], [3], [4,5,6]
        let elements = buffer![1i32, 2, 3, 4, 5, 6];
        let offsets = buffer![0u32, 2, 3];
        let sizes = buffer![2u32, 1, 3];
        let lv = ListViewArray::new(
            elements.into_array(),
            offsets.into_array(),
            sizes.into_array(),
            Validity::NonNullable,
        );
        let mut producer = build_listview_primitive_producer::<i32>(lv.into_array(), &mut ctx())?;

        let mut row_data: Vec<Vec<i32>> = Vec::new();
        producer.for_each_chunk(|chunk| {
            for i in 0..chunk.n {
                let o = (chunk.offset_of)(i);
                let s = (chunk.size_of)(i);
                row_data.push(chunk.elements[o..o + s].to_vec());
            }
        });
        assert_eq!(row_data, vec![vec![1, 2], vec![3], vec![4, 5, 6]]);
        Ok(())
    }
}
