// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Content-defined chunking (CDC) for the Vortex write path.
//!
//! **Unstable / experimental**: this module is a prototype and its API, defaults, and produced
//! chunk boundaries may change between releases.
//!
//! # Why content-defined chunk boundaries?
//!
//! Storage systems such as the Hugging Face Hub's Xet backend deduplicate files at the level of
//! ~64 KiB *byte* chunks whose boundaries are chosen by a rolling hash over the file's bytes
//! (GearHash, see [Xet chunking spec]). Byte-level CDC is robust to data being shifted around a
//! file, but it cannot see through re-encoded data: it only deduplicates ranges that are
//! byte-identical between two files.
//!
//! Vortex compresses each chunk of a column independently and deterministically, so two files
//! that contain an identical *logical* chunk of a column produce identical bytes for that chunk.
//! The default write strategy, however, cuts columns into fixed row-count blocks
//! (multiples of 8192 rows). Inserting or deleting a single row therefore shifts the contents of
//! every downstream block by one row, every re-encoded block differs, and byte-level
//! deduplication recovers almost nothing.
//!
//! [`CdcRepartitionStrategy`] replaces fixed row-count repartitioning with boundaries chosen by
//! a rolling hash over the *logical values* of the column, the same trick Parquet's
//! `use_content_defined_chunking` writer option applies to data pages. After an insert, delete,
//! or localized edit, the boundary positions re-synchronize with the surrounding content within
//! roughly one chunk, so all other chunks re-encode to byte-identical segments that Xet's
//! byte-level chunker can deduplicate.
//!
//! # How boundaries are chosen
//!
//! Each incoming chunk is canonicalized and reduced to one digest per row, along with how many
//! bytes that row's content would occupy serialized. The eight bytes of each row's digest update
//! a 64-bit GEAR rolling hash `h = (h << 1) + table[byte]`, whose value depends only on the last
//! eight rows fed. The table is [`gearhash::DEFAULT_TABLE`], which the [Xet chunking spec]
//! references normatively; a test pins its contents, since cut positions (and therefore the
//! written bytes) are a function of it. A chunk ends after a row whose update leaves the top
//! [`boundary_mask_bits`](ContentDefinedChunkingOptions::boundary_mask_bits) bits of `h` zero,
//! once the chunk spans at least
//! [`min_chunk_bytes`](ContentDefinedChunkingOptions::min_chunk_bytes) of serialized values, or
//! after the first row that takes it to
//! [`max_chunk_bytes`](ContentDefinedChunkingOptions::max_chunk_bytes).
//!
//! Chunk size budgets are measured in *serialized value bytes* (a deterministic function of the
//! logical content), not encoded on-disk bytes.
//!
//! [Xet chunking spec]: https://huggingface.co/docs/xet/chunking

mod digest;

use std::sync::Arc;

use async_stream::try_stream;
use async_trait::async_trait;
use futures::StreamExt as _;
use futures::pin_mut;
use gearhash::Hasher;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ChunkedArray;
use vortex_array::dtype::DType;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;

use self::digest::RowDigest;
use self::digest::row_digests;
use crate::LayoutRef;
use crate::LayoutStrategy;
use crate::LayoutWriterContext;
use crate::segments::SegmentSinkRef;
use crate::sequence::SendableSequentialStream;
use crate::sequence::SequencePointer;
use crate::sequence::SequentialStreamAdapter;
use crate::sequence::SequentialStreamExt;

/// Options controlling content-defined chunk boundary selection.
#[derive(Clone, Debug)]
pub struct ContentDefinedChunkingOptions {
    /// The minimum serialized size of a chunk in bytes. Boundary candidates closer than this to
    /// the previous boundary are ignored.
    pub min_chunk_bytes: u64,
    /// The maximum serialized size of a chunk in bytes. A boundary is forced at the first row
    /// end at or beyond this size (so a chunk may overshoot by at most one row).
    pub max_chunk_bytes: u64,
    /// The number of leading bits of the rolling hash that must all be zero after a row for a
    /// chunk to end there, so candidates appear roughly every `2^boundary_mask_bits` rows; the
    /// byte budgets above then clamp chunks into `[min_chunk_bytes, max_chunk_bytes]`.
    pub boundary_mask_bits: u32,
}

impl Default for ContentDefinedChunkingOptions {
    fn default() -> Self {
        // In the same range as the defaults of Parquet's content-defined chunking writer option
        // (min 256 KiB / max 1 MiB, applied to uncompressed values). The smaller minimum lets
        // boundaries re-synchronize faster after an edit shifts rows.
        Self {
            min_chunk_bytes: 128 * 1024,
            max_chunk_bytes: 1024 * 1024,
            boundary_mask_bits: 15,
        }
    }
}

impl ContentDefinedChunkingOptions {
    fn boundary_mask(&self) -> u64 {
        if self.boundary_mask_bits == 0 || self.boundary_mask_bits > 63 {
            vortex_panic!(
                "boundary_mask_bits must be in 1..=63, got {}",
                self.boundary_mask_bits
            );
        }
        u64::MAX << (64 - self.boundary_mask_bits)
    }
}

/// Repartition a stream of arrays into blocks whose boundaries are content-defined.
///
/// **Unstable / experimental**: see the [module docs](self) for the motivation and algorithm.
///
/// Identical runs of rows produce identical boundary decisions no matter where they appear in
/// the stream, so files written from edited versions of the same data share most of their
/// chunks, and therefore most of their bytes.
#[derive(Clone)]
pub struct CdcRepartitionStrategy {
    child: Arc<dyn LayoutStrategy>,
    options: ContentDefinedChunkingOptions,
}

impl CdcRepartitionStrategy {
    /// Create a new CDC repartitioning strategy wrapping `child`.
    ///
    /// # Panics
    ///
    /// If the options are inconsistent (`min_chunk_bytes >= max_chunk_bytes`, or
    /// `boundary_mask_bits` outside `1..=63`).
    pub fn new<S: LayoutStrategy>(child: S, options: ContentDefinedChunkingOptions) -> Self {
        if options.min_chunk_bytes >= options.max_chunk_bytes {
            vortex_panic!(
                "min_chunk_bytes ({}) must be smaller than max_chunk_bytes ({})",
                options.min_chunk_bytes,
                options.max_chunk_bytes
            );
        }
        // Force mask validation eagerly so misconfiguration fails at strategy construction.
        let _ = options.boundary_mask();
        Self {
            child: Arc::new(child),
            options,
        }
    }
}

#[async_trait]
impl LayoutStrategy for CdcRepartitionStrategy {
    async fn write_stream(
        &self,
        ctx: LayoutWriterContext,
        segment_sink: SegmentSinkRef,
        stream: SendableSequentialStream,
        eof: SequencePointer,
        session: &VortexSession,
    ) -> VortexResult<LayoutRef> {
        let dtype = stream.dtype().clone();
        let dtype_clone = dtype.clone();
        let options = self.options.clone();
        let cdc_session = session.clone();

        let repartitioned_stream = try_stream! {
            let stream = stream.peekable();
            pin_mut!(stream);

            let mut exec_ctx = cdc_session.create_execution_ctx();
            let mut cutter = RollingCutter::new(&options);
            // Canonical slices accumulated since the previous emitted boundary.
            let mut pending: Vec<ArrayRef> = Vec::new();

            while let Some(chunk) = stream.as_mut().next().await {
                let (sequence_id, chunk) = chunk?;
                let mut sequence_pointer = sequence_id.descend();

                let canonical = chunk.execute::<Canonical>(&mut exec_ctx)?;
                let digests = row_digests(&canonical, &mut exec_ctx)?;
                let canonical = canonical.into_array();

                let mut start = 0;
                for cut in cutter.process_rows(&digests) {
                    pending.push(canonical.slice(start..cut)?);
                    start = cut;
                    let block = take_block(&mut pending, &dtype_clone, &mut exec_ctx)?;
                    yield (sequence_pointer.advance(), block);
                }
                if start < canonical.len() {
                    pending.push(canonical.slice(start..canonical.len())?);
                }

                if stream.as_mut().peek().await.is_none() && !pending.is_empty() {
                    let block = take_block(&mut pending, &dtype_clone, &mut exec_ctx)?;
                    yield (sequence_pointer.advance(), block);
                }
            }
        };

        self.child
            .write_stream(
                ctx,
                segment_sink,
                SequentialStreamAdapter::new(dtype, repartitioned_stream).sendable(),
                eof,
                session,
            )
            .await
    }
}

/// Concatenate the slices accumulated since the previous boundary into one canonical block.
fn take_block(
    pending: &mut Vec<ArrayRef>,
    dtype: &DType,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    Ok(ChunkedArray::try_new(pending.drain(..), dtype.clone())?
        .into_array()
        .execute::<Canonical>(ctx)?
        .into_array())
}

/// Rolling GEAR hash state that survives across incoming chunks of a column stream.
struct RollingCutter {
    hasher: Hasher<'static>,
    serialized_bytes: u64,
    min_chunk_bytes: u64,
    max_chunk_bytes: u64,
    boundary_mask: u64,
}

impl RollingCutter {
    fn new(options: &ContentDefinedChunkingOptions) -> Self {
        Self {
            hasher: Hasher::default(),
            serialized_bytes: 0,
            min_chunk_bytes: options.min_chunk_bytes,
            max_chunk_bytes: options.max_chunk_bytes,
            boundary_mask: options.boundary_mask(),
        }
    }

    /// Feed a chunk's rows and return the ascending row ends (exclusive) after which a chunk
    /// boundary is placed.
    fn process_rows(&mut self, rows: &[RowDigest]) -> Vec<usize> {
        let mut cuts = Vec::new();
        for (row, digest) in rows.iter().enumerate() {
            self.hasher.update(&digest.hash.to_le_bytes());
            self.serialized_bytes += digest.width;
            if self.serialized_bytes >= self.max_chunk_bytes
                || (self.serialized_bytes >= self.min_chunk_bytes
                    && self.hasher.is_match(self.boundary_mask))
            {
                self.hasher.set_hash(0);
                self.serialized_bytes = 0;
                cuts.push(row + 1);
            }
        }
        cuts
    }
}

#[cfg(test)]
mod tests;
