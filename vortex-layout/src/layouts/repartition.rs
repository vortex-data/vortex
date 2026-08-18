// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::VecDeque;
use std::sync::Arc;

use async_trait::async_trait;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ChunkedArray;
use vortex_array::dtype::DType;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::BufferedBytesReservation;
use crate::BufferedBytesTracker;
use crate::LayoutRef;
use crate::LayoutStrategy;
use crate::LayoutWriter;
use crate::LayoutWriterContext;
use crate::segments::SegmentSinkRef;
use crate::sequence::SequenceId;

#[derive(Clone)]
pub struct RepartitionWriterOptions {
    /// The minimum uncompressed size in bytes for a block.
    pub block_size_minimum: u64,
    /// The multiple of the number of rows in each block.
    pub block_len_multiple: usize,
    /// Optional target uncompressed size in bytes for a block.
    ///
    /// The repartition writer attempts to produce partitions with this uncompressed size. This is
    /// only a best effort attempt: the partitions may be arbitrarily larger or smaller. Reasons for
    /// this include:
    ///
    /// 1. The size of one element may not perfectly divide the target size, resulting in blocks
    ///    that are either too large or too small.
    ///
    /// 2. Variable length types are expensive to pack due to the need to read each element length.
    ///
    /// 3. View types are expensive to pack due to each view sharing an arbitrary slice of data.
    pub block_size_target: Option<u64>,
    pub canonicalize: bool,
}

impl RepartitionWriterOptions {
    /// Compute the effective block length for a given [`DType`].
    ///
    /// For fixed-width types where [`DType::element_size`] is known and large enough that
    /// `element_size * block_len_multiple` would exceed `block_size_target`, this reduces the
    /// block length so each block stays close to the target byte size.
    fn effective_block_len(&self, dtype: &DType) -> usize {
        let Some(block_size_target) = self.block_size_target else {
            return self.block_len_multiple;
        };
        match dtype.element_size() {
            Some(elem_size) if elem_size > 0 => {
                // `div_ceil` ensures we overshoot the block_size_target; therefore preventing
                // Prevent adjacent 0.9 MiB inputs from being combined into one 1.8 MiB block.
                let max_rows = usize::try_from(block_size_target.div_ceil(elem_size as u64))
                    .unwrap_or(usize::MAX);
                self.block_len_multiple.min(max_rows).max(1)
            }
            _ => self.block_len_multiple,
        }
    }
}

/// Repartition a stream of arrays into blocks.
///
/// Each emitted block (except the last) is at least `block_size_minimum` bytes and contains a
/// multiple of `block_len_multiple` rows.
#[derive(Clone)]
pub struct RepartitionStrategy {
    child: Arc<dyn LayoutStrategy>,
    options: RepartitionWriterOptions,
}

impl RepartitionStrategy {
    pub fn new<S: LayoutStrategy>(child: S, options: RepartitionWriterOptions) -> Self {
        Self {
            child: Arc::new(child),
            options,
        }
    }
}

impl LayoutStrategy for RepartitionStrategy {
    fn new_writer(
        &self,
        ctx: LayoutWriterContext,
        segment_sink: SegmentSinkRef,
        dtype: DType,
        session: &VortexSession,
    ) -> VortexResult<Box<dyn LayoutWriter>> {
        let block_len = self.options.effective_block_len(&dtype);
        let buffered_bytes = ctx.buffered_bytes_tracker().clone();
        Ok(Box::new(RepartitionLayoutWriter {
            child: self
                .child
                .new_writer(ctx, segment_sink, dtype.clone(), session)?,
            chunks: ChunksBuffer::new(self.options.block_size_minimum, block_len, buffered_bytes),
            dtype,
            block_len,
            canonicalize: self.options.canonicalize,
            exec_ctx: session.create_execution_ctx(),
        }))
    }
}

struct RepartitionLayoutWriter {
    child: Box<dyn LayoutWriter>,
    chunks: ChunksBuffer,
    dtype: DType,
    block_len: usize,
    canonicalize: bool,
    exec_ctx: vortex_array::ExecutionCtx,
}

impl RepartitionLayoutWriter {
    fn canonicalize(&mut self, chunk: ArrayRef) -> VortexResult<ArrayRef> {
        if self.canonicalize {
            Ok(chunk.execute::<Canonical>(&mut self.exec_ctx)?.into_array())
        } else {
            Ok(chunk)
        }
    }

    fn build_output(
        &mut self,
        mut chunks: Vec<(SequenceId, ArrayRef)>,
    ) -> VortexResult<(SequenceId, ArrayRef)> {
        let (sequence_id, last) = chunks.pop().vortex_expect("output chunks are non-empty");
        let chunked = ChunkedArray::try_new(
            chunks
                .into_iter()
                .map(|(_, chunk)| chunk)
                .chain(std::iter::once(last)),
            self.dtype.clone(),
        )?;
        Ok((
            sequence_id,
            chunked
                .into_array()
                .execute::<Canonical>(&mut self.exec_ctx)?
                .into_array(),
        ))
    }
}

#[async_trait]
impl LayoutWriter for RepartitionLayoutWriter {
    async fn write(&mut self, sequence_id: SequenceId, chunk: ArrayRef) -> VortexResult<()> {
        let chunk = self.canonicalize(chunk)?;
        let mut sequence = sequence_id.descend();
        let mut offset = 0;
        while offset < chunk.len() {
            let end = (offset + self.block_len).min(chunk.len());
            self.chunks
                .push_back(sequence.advance(), chunk.slice(offset..end)?);
            offset = end;

            if self.chunks.have_enough() {
                let chunks = self.chunks.collect_exact_blocks()?;
                assert!(!chunks.is_empty());
                let (sequence_id, output) = self.build_output(chunks)?;
                if !output.is_empty() {
                    self.child.write(sequence_id, output).await?;
                }
            }
        }
        Ok(())
    }

    async fn finish(&mut self, sequence_id: SequenceId) -> VortexResult<()> {
        if !self.chunks.data.is_empty() {
            let chunks = self
                .chunks
                .data
                .drain(..)
                .map(|(sequence_id, array, _, _reservation)| (sequence_id, array))
                .collect();
            let (sequence_id, output) = self.build_output(chunks)?;
            if !output.is_empty() {
                self.child.write(sequence_id, output).await?;
            }
        }
        self.child.finish(sequence_id).await
    }

    async fn close(self: Box<Self>) -> VortexResult<LayoutRef> {
        self.child.close().await
    }
}

struct ChunksBuffer {
    /// Each entry stores the chunk and the `nbytes()` snapshot taken at push time.
    /// This avoids accounting mismatches when interior-mutable arrays (e.g. `SharedArray`)
    /// change their reported size after being pushed.
    data: VecDeque<(SequenceId, ArrayRef, u64, BufferedBytesReservation)>,
    row_count: usize,
    nbytes: u64,
    block_size_minimum: u64,
    block_len_multiple: usize,
    buffered_bytes: BufferedBytesTracker,
}

impl ChunksBuffer {
    fn new(
        block_size_minimum: u64,
        block_len_multiple: usize,
        buffered_bytes: BufferedBytesTracker,
    ) -> Self {
        Self {
            data: Default::default(),
            row_count: 0,
            nbytes: 0,
            block_size_minimum,
            block_len_multiple,
            buffered_bytes,
        }
    }

    fn have_enough(&self) -> bool {
        self.nbytes >= self.block_size_minimum && self.row_count >= self.block_len_multiple
    }

    fn collect_exact_blocks(&mut self) -> VortexResult<Vec<(SequenceId, ArrayRef)>> {
        let nblocks = self.row_count / self.block_len_multiple;
        let mut res = Vec::with_capacity(self.data.len());
        let mut remaining = nblocks * self.block_len_multiple;
        while remaining > 0 {
            let (sequence_id, chunk, _, _reservation) = self
                .pop_front()
                .vortex_expect("must have at least one chunk");
            let len = chunk.len();

            if len > remaining {
                let left = chunk.slice(0..remaining)?;
                let right = chunk.slice(remaining..len)?;
                let mut sequence = sequence_id.descend();
                res.push((sequence.advance(), left));
                self.push_front(sequence.advance(), right);
                remaining = 0;
            } else {
                res.push((sequence_id, chunk));
                remaining -= len;
            }
        }
        Ok(res)
    }

    fn push_back(&mut self, sequence_id: SequenceId, chunk: ArrayRef) {
        let nb = chunk.nbytes();
        self.row_count += chunk.len();
        self.nbytes += nb;
        self.data
            .push_back((sequence_id, chunk, nb, self.buffered_bytes.reserve(nb)));
    }

    fn push_front(&mut self, sequence_id: SequenceId, chunk: ArrayRef) {
        let nb = chunk.nbytes();
        self.row_count += chunk.len();
        self.nbytes += nb;
        self.data
            .push_front((sequence_id, chunk, nb, self.buffered_bytes.reserve(nb)));
    }

    fn pop_front(&mut self) -> Option<(SequenceId, ArrayRef, u64, BufferedBytesReservation)> {
        let res = self.data.pop_front();
        if let Some((_, chunk, nb, _)) = res.as_ref() {
            self.row_count -= chunk.len();
            self.nbytes -= nb;
        }
        res
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_array::ArrayContext;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::array_session;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::arrays::FixedSizeListArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::SharedArray;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability::NonNullable;
    use vortex_array::dtype::PType;
    use vortex_array::validity::Validity;
    use vortex_error::VortexResult;
    use vortex_io::runtime::single::block_on;
    use vortex_io::session::RuntimeSessionExt;

    use super::*;
    use crate::LayoutStrategy;
    use crate::layouts::chunked::writer::ChunkedLayoutStrategy;
    use crate::layouts::flat::writer::FlatLayoutStrategy;
    use crate::segments::TestSegments;
    use crate::sequence::SequenceId;
    use crate::sequence::SequentialArrayStreamExt;
    use crate::test::new_session;

    const ONE_MEG: u64 = 1 << 20;

    #[test]
    fn effective_block_len_small_elements() {
        // f64 = 8 bytes/element. 8192 * 8 = 64 KiB << 1 MiB, so no reduction.
        let dtype = DType::Primitive(PType::F64, NonNullable);
        let options = RepartitionWriterOptions {
            block_size_minimum: 0,
            block_len_multiple: 8192,
            block_size_target: Some(ONE_MEG),
            canonicalize: false,
        };
        assert_eq!(options.effective_block_len(&dtype), 8192);
    }

    #[test]
    fn effective_block_len_large_elements() {
        // FixedSizeList(f64, 1000) = 8000 bytes/element.
        // div_ceil(1 MiB, 8000) = 132, so effective block len = min(8192, 132) = 132.
        let dtype = DType::FixedSizeList(
            Arc::new(DType::Primitive(PType::F64, NonNullable)),
            1000,
            NonNullable,
        );
        let options = RepartitionWriterOptions {
            block_size_minimum: 0,
            block_len_multiple: 8192,
            block_size_target: Some(ONE_MEG),
            canonicalize: false,
        };
        assert_eq!(options.effective_block_len(&dtype), 132);
    }

    #[test]
    fn effective_block_len_variable_width() {
        // Utf8 has no known element_size, so block_len_multiple is unchanged.
        let dtype = DType::Utf8(NonNullable);
        let options = RepartitionWriterOptions {
            block_size_minimum: 0,
            block_len_multiple: 8192,
            block_size_target: Some(ONE_MEG),
            canonicalize: false,
        };
        assert_eq!(options.effective_block_len(&dtype), 8192);
    }

    #[test]
    fn effective_block_len_very_large_elements() {
        // FixedSizeList(f64, 1_000_000) = 8_000_000 bytes/element.
        // 1 MiB / 8_000_000 = 0, clamped to max(1) = 1.
        let dtype = DType::FixedSizeList(
            Arc::new(DType::Primitive(PType::F64, NonNullable)),
            1_000_000,
            NonNullable,
        );
        let options = RepartitionWriterOptions {
            block_size_minimum: 0,
            block_len_multiple: 8192,
            block_size_target: Some(ONE_MEG),
            canonicalize: false,
        };
        assert_eq!(options.effective_block_len(&dtype), 1);
    }

    #[test]
    fn repartition_large_element_type_produces_small_blocks() -> VortexResult<()> {
        // Create a FixedSizeList(f64, 1000) array with 1000 lists.
        // Each list is 8000 bytes, so 1000 lists = 8 MiB total.
        // With block_size_target = 1 MiB, effective block_len = 133.
        // We expect the repartition to produce blocks of 132 rows each.
        let list_size: u32 = 1000;
        let num_lists: usize = 1000;
        let total_elements = list_size as usize * num_lists;

        let elements = PrimitiveArray::from_iter((0..total_elements).map(|i| i as f64));
        let fsl = FixedSizeListArray::new(
            elements.into_array(),
            list_size,
            Validity::NonNullable,
            num_lists,
        );

        let ctx = ArrayContext::empty();
        let segments = Arc::new(TestSegments::default());
        let (ptr, eof) = SequenceId::root().split();

        let child = ChunkedLayoutStrategy::new(FlatLayoutStrategy::default());
        let strategy = RepartitionStrategy::new(
            child,
            RepartitionWriterOptions {
                block_size_minimum: 0,
                block_len_multiple: 8192,
                block_size_target: Some(ONE_MEG),
                canonicalize: false,
            },
        );

        let stream = fsl.into_array().to_array_stream().sequenced(ptr);
        let layout = block_on(|handle| async move {
            let session = new_session().with_handle(handle);
            strategy
                .write_stream(
                    ctx.into(),
                    Arc::<TestSegments>::clone(&segments),
                    stream,
                    eof,
                    &session,
                )
                .await
        })?;

        // The layout should be a ChunkedLayout with multiple children.
        // With 1000 rows and effective block_len = 132:
        //   - 7 full blocks of 132 rows = 924 rows
        //   - 1 remainder block of 76 rows
        //   - Total: 8 blocks, 1000 rows
        assert_eq!(layout.row_count(), num_lists as u64);

        // All non-last children should have 131 rows.
        let nchildren = layout.nchildren();
        assert!(nchildren > 1, "expected multiple chunks, got {nchildren}");

        for i in 0..nchildren - 1 {
            let child = layout.slot(i)?.vortex_expect("chunk slot present");
            assert_eq!(
                child.row_count(),
                132,
                "chunk {i} has {} rows, expected 131",
                child.row_count()
            );
        }

        // Last child gets the remainder.
        let last = layout
            .slot(nchildren - 1)?
            .vortex_expect("chunk slot present");
        assert_eq!(last.row_count(), 1000 - 132 * (nchildren as u64 - 1));

        Ok(())
    }

    #[test]
    fn repartition_small_element_type_unchanged() -> VortexResult<()> {
        // For f64 (8 bytes/element), effective block_len stays at 8192.
        // With 10000 elements and block_size_minimum=0, we get one block of 8192
        // and one remainder of 1808.
        let num_elements: usize = 10000;
        let elements = PrimitiveArray::from_iter((0..num_elements).map(|i| i as f64));

        let ctx = ArrayContext::empty();
        let segments = Arc::new(TestSegments::default());
        let (ptr, eof) = SequenceId::root().split();

        let child = ChunkedLayoutStrategy::new(FlatLayoutStrategy::default());
        let strategy = RepartitionStrategy::new(
            child,
            RepartitionWriterOptions {
                block_size_minimum: 0,
                block_len_multiple: 8192,
                block_size_target: Some(ONE_MEG),
                canonicalize: false,
            },
        );

        let stream = elements.into_array().to_array_stream().sequenced(ptr);
        let layout = block_on(|handle| async move {
            let session = new_session().with_handle(handle);
            strategy
                .write_stream(
                    ctx.into(),
                    Arc::<TestSegments>::clone(&segments),
                    stream,
                    eof,
                    &session,
                )
                .await
        })?;

        assert_eq!(layout.row_count(), num_elements as u64);
        assert_eq!(layout.nchildren(), 2);
        assert_eq!(
            layout
                .slot(0)?
                .vortex_expect("chunk slot present")
                .row_count(),
            8192
        );
        assert_eq!(
            layout
                .slot(1)?
                .vortex_expect("chunk slot present")
                .row_count(),
            1808
        );

        Ok(())
    }

    /// Regression test: `SharedArray` slices sharing an `Arc<Mutex<SharedState>>` can
    /// transition from Source to Cached when any one of them is canonicalized. This caused
    /// `pop_front` to panic with `attempt to subtract with overflow` because the buffer's
    /// running `nbytes` total was accumulated with the smaller Source-era values while
    /// `pop_front` subtracted the larger Cached-era values.
    #[test]
    fn chunks_buffer_pop_front_no_panic_after_shared_execution() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let n = 20_000usize;
        let block_len = 10_000usize;

        let constant = ConstantArray::new(42i64, n);
        let shared = SharedArray::new(constant.into_array());
        let shared_handle = shared.clone();
        let arr = shared.into_array();

        let s1 = arr.slice(0..block_len)?;
        let s2 = arr.slice(block_len..n)?;

        let tracker = BufferedBytesTracker::new();
        let mut buf = ChunksBuffer::new(0, block_len, tracker.clone());
        let mut sequence = SequenceId::root();
        buf.push_back(sequence.advance(), s1);
        buf.push_back(sequence.advance(), s2);

        assert!(tracker.buffered_bytes() > 0);

        let output = buf.pop_front().unwrap();
        drop(output);

        // Transition SharedState from Source to Cached for ALL slices sharing this Arc.
        use vortex_array::arrays::shared::SharedArrayExt;
        let _canonical =
            shared_handle.get_or_compute(|source| source.clone().execute::<Canonical>(&mut ctx))?;

        // Before the fix this panicked with "attempt to subtract with overflow".
        let s2 = buf.pop_front().unwrap();
        drop(s2);
        assert_eq!(buf.nbytes, 0);
        assert_eq!(buf.row_count, 0);
        assert_eq!(tracker.buffered_bytes(), 0);

        Ok(())
    }
}
