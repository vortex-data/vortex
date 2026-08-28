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
//! Every row is reduced to a 64-bit *whitened digest* per leaf value (a SplitMix64-style mix of
//! the validity marker and the value's bytes). The digest's eight bytes update a 64-bit GEAR
//! rolling hash `h = (h << 1) + table[byte]`, whose value depends only on the last few rows
//! fed. The table is [`gearhash::DEFAULT_TABLE`], which the [Xet chunking spec] references
//! normatively; a test pins its contents, since cut positions (and therefore the written bytes)
//! are a function of it. A boundary becomes *eligible* at any digest byte where the top
//! [`boundary_mask_bits`](ContentDefinedChunkingOptions::boundary_mask_bits) bits of `h` are all
//! zero, and the pending chunk already spans at least
//! [`min_chunk_bytes`](ContentDefinedChunkingOptions::min_chunk_bytes) of serialized values. The
//! cut is taken at the end of the row that produced the eligible byte, so chunks always split on
//! row boundaries. If no eligible byte appears before
//! [`max_chunk_bytes`](ContentDefinedChunkingOptions::max_chunk_bytes) of serialized values, a
//! cut is forced at the next row end.
//!
//! Hashing whitened digests instead of raw value bytes matters for columnar data: typical
//! columns (sequential ids, near-constant timestamps, low-cardinality categories) have very low
//! per-byte entropy, which starves a raw GEAR hash of boundary candidates and makes cut
//! positions degrade into fixed-size strides that never re-align after a row shift. Mixing each
//! value through SplitMix64 restores a uniform candidate distribution for any content while
//! remaining a pure function of the row's logical bytes.
//!
//! Chunk size budgets are measured in *serialized value bytes* (a deterministic function of the
//! logical content), not encoded on-disk bytes.
//!
//! [Xet chunking spec]: https://huggingface.co/docs/xet/chunking

pub mod xet;

use std::sync::Arc;

use async_stream::try_stream;
use async_trait::async_trait;
use futures::StreamExt as _;
use futures::pin_mut;
use gearhash::DEFAULT_TABLE;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::extension::ExtensionArraySlotsExt;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_array::arrays::varbinview::VarBinViewArrayExt;
use vortex_array::dtype::Nullability;
use vortex_array::match_each_native_ptype;
use vortex_array::validity::Validity;
use vortex_buffer::BitBuffer;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;
use vortex_mask::Mask;
use vortex_session::VortexSession;

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
    /// The number of leading hash bits that must all be zero for a digest byte to be a boundary
    /// candidate. Each row feeds eight digest bytes per leaf value, so candidates appear
    /// roughly every `2^boundary_mask_bits / 8` rows; the byte budgets above then clamp chunks
    /// into `[min_chunk_bytes, max_chunk_bytes]`.
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
            boundary_mask_bits: 18,
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
                let feeds = row_feeds(&canonical, &mut exec_ctx)?;
                let canonical = canonical.into_array();

                let cuts = cutter.process_rows(&feeds, canonical.len());
                let mut start = 0usize;
                for cut in cuts {
                    let part = canonical.slice(start..cut)?;
                    start = cut;
                    pending.push(part);
                    let block = ChunkedArray::try_new(pending.drain(..), dtype_clone.clone())?
                        .into_array()
                        .execute::<Canonical>(&mut exec_ctx)?
                        .into_array();
                    if !block.is_empty() {
                        yield (sequence_pointer.advance(), block);
                    }
                }
                if start < canonical.len() {
                    let len = canonical.len();
                    pending.push(canonical.slice(start..len)?);
                }

                if stream.as_mut().peek().await.is_none() && !pending.is_empty() {
                    let block = ChunkedArray::try_new(pending.drain(..), dtype_clone.clone())?
                        .into_array()
                        .execute::<Canonical>(&mut exec_ctx)?
                        .into_array();
                    if !block.is_empty() {
                        yield (sequence_pointer.advance(), block);
                    }
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

/// A SplitMix64-style finalizer used to whiten row content before it reaches the GEAR hash.
#[inline]
fn mix64(value: u64) -> u64 {
    let mut z = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Fold a byte string into a digest, eight little-endian bytes at a time.
#[inline]
fn fold_bytes(mut digest: u64, bytes: &[u8]) -> u64 {
    let mut chunks = bytes.chunks_exact(8);
    for word in &mut chunks {
        let mut buf = [0u8; 8];
        buf.copy_from_slice(word);
        digest = mix64(digest ^ u64::from_le_bytes(buf));
    }
    let tail = chunks.remainder();
    if !tail.is_empty() {
        let mut word = [0u8; 8];
        word[..tail.len()].copy_from_slice(tail);
        digest = mix64(digest ^ u64::from_le_bytes(word));
    }
    digest
}

/// Rolling GEAR hash state that survives across incoming chunks of a column stream.
struct RollingCutter {
    hash: u64,
    serialized_bytes: u64,
    boundary_eligible: bool,
    min_chunk_bytes: u64,
    max_chunk_bytes: u64,
    boundary_mask: u64,
}

impl RollingCutter {
    fn new(options: &ContentDefinedChunkingOptions) -> Self {
        Self {
            hash: 0,
            serialized_bytes: 0,
            boundary_eligible: false,
            min_chunk_bytes: options.min_chunk_bytes,
            max_chunk_bytes: options.max_chunk_bytes,
            boundary_mask: options.boundary_mask(),
        }
    }

    /// Roll the whitened digest of one row value into the hash, after accounting for the
    /// value's serialized width.
    #[inline]
    fn feed_digest(&mut self, digest: u64, serialized_width: u64) {
        self.serialized_bytes += serialized_width;
        for byte in digest.to_le_bytes() {
            self.hash = (self.hash << 1).wrapping_add(DEFAULT_TABLE[byte as usize]);
            if self.serialized_bytes >= self.min_chunk_bytes && self.hash & self.boundary_mask == 0
            {
                self.boundary_eligible = true;
            }
        }
    }

    /// Close out the current row, returning whether a chunk boundary falls after it.
    #[inline]
    fn end_row(&mut self) -> bool {
        if self.boundary_eligible || self.serialized_bytes >= self.max_chunk_bytes {
            self.hash = 0;
            self.serialized_bytes = 0;
            self.boundary_eligible = false;
            return true;
        }
        false
    }

    /// Feed `row_count` rows described by `feeds` and return the ascending row ends (exclusive)
    /// after which a chunk boundary is placed.
    fn process_rows(&mut self, feeds: &[RowFeed], row_count: usize) -> Vec<usize> {
        let mut cuts = Vec::new();
        for row in 0..row_count {
            for feed in feeds {
                let (marker, marker_width) = match &feed.marker {
                    MarkerFeed::NonNullable => (0u64, 0u64),
                    MarkerFeed::Constant(byte) => (1 + u64::from(*byte), 1),
                    MarkerFeed::Bytes(bytes) => (1 + u64::from(bytes[row]), 1),
                };
                let (digest, width) = match &feed.values {
                    ValueFeed::Fixed { bytes, width } => (
                        fold_bytes(marker, &bytes.as_slice()[row * width..(row + 1) * width]),
                        *width as u64,
                    ),
                    ValueFeed::Views { array } => {
                        let view = &array.views()[row];
                        let digest = mix64(marker ^ u64::from(view.len()));
                        let digest = if view.is_inlined() {
                            fold_bytes(digest, view.as_inlined().value())
                        } else {
                            let r = view.as_view();
                            fold_bytes(
                                digest,
                                &array.buffer(r.buffer_index as usize).as_slice()[r.as_range()],
                            )
                        };
                        (digest, 4 + u64::from(view.len()))
                    }
                    // No content is visible, so the digest is constant: no boundary candidates
                    // arise and cuts degrade to `max_chunk_bytes` strides.
                    ValueFeed::Opaque => (mix64(marker), 8),
                };
                self.feed_digest(digest, marker_width + width);
            }
            if self.end_row() {
                cuts.push(row + 1);
            }
        }
        cuts
    }
}

/// Per-row validity marker bytes fed into the rolling hash ahead of the value bytes.
enum MarkerFeed {
    /// The dtype is non-nullable: no marker byte is fed.
    NonNullable,
    /// Every row feeds the same marker byte (all-valid or all-null).
    Constant(u8),
    /// Row `i` feeds `bytes[i]` (1 = valid, 0 = null).
    Bytes(Vec<u8>),
}

/// The serialized value bytes of one (possibly nested) leaf of a chunk.
enum ValueFeed {
    /// Fixed-width values: row `i` feeds `bytes[i * width..(i + 1) * width]`.
    Fixed { bytes: ByteBuffer, width: usize },
    /// Variable-width values behind binary views: each row feeds its length (4 LE bytes)
    /// followed by its content bytes.
    Views { array: VarBinViewArray },
    /// Values this prototype cannot inspect (lists, maps, unions, ...): rows feed no value
    /// bytes, degrading boundary selection to `max_chunk_bytes`-sized cuts.
    Opaque,
}

struct RowFeed {
    marker: MarkerFeed,
    values: ValueFeed,
}

/// Flatten a canonical chunk into the ordered list of feeds that serialize each row.
fn row_feeds(canonical: &Canonical, ctx: &mut ExecutionCtx) -> VortexResult<Vec<RowFeed>> {
    let mut feeds = Vec::new();
    collect_row_feeds(canonical, ctx, &mut feeds)?;
    Ok(feeds)
}

fn collect_row_feeds(
    canonical: &Canonical,
    ctx: &mut ExecutionCtx,
    feeds: &mut Vec<RowFeed>,
) -> VortexResult<()> {
    match canonical {
        Canonical::Primitive(array) => {
            let marker = marker_feed(&array.validity()?, array.len(), ctx)?;
            let width = array.ptype().byte_width();
            let bytes = match_each_native_ptype!(array.ptype(), |P| {
                array.to_buffer::<P>().into_byte_buffer()
            });
            feeds.push(RowFeed {
                marker,
                values: ValueFeed::Fixed { bytes, width },
            });
        }
        Canonical::Bool(array) => {
            let marker = marker_feed(&array.validity()?, array.len(), ctx)?;
            let bytes = ByteBuffer::from(bits_to_bytes(&array.clone().into_bit_buffer()));
            feeds.push(RowFeed {
                marker,
                values: ValueFeed::Fixed { bytes, width: 1 },
            });
        }
        Canonical::VarBinView(array) => {
            let marker = marker_feed(&array.varbinview_validity(), array.len(), ctx)?;
            feeds.push(RowFeed {
                marker,
                values: ValueFeed::Views {
                    array: array.clone(),
                },
            });
        }
        Canonical::Struct(array) => {
            let marker = marker_feed(&array.struct_validity(), array.len(), ctx)?;
            feeds.push(RowFeed {
                marker,
                values: ValueFeed::Opaque,
            });
            for field in array.iter_unmasked_fields() {
                let child = field.clone().execute::<Canonical>(ctx)?;
                collect_row_feeds(&child, ctx, feeds)?;
            }
        }
        Canonical::Extension(array) => {
            let storage = array.storage().clone().execute::<Canonical>(ctx)?;
            collect_row_feeds(&storage, ctx, feeds)?;
        }
        // Decimal, list, map, fixed-size list, union, variant, and null values are not yet
        // serialized by this prototype: their rows contribute no value bytes, so boundary
        // selection for them degrades to fixed `max_chunk_bytes`-sized cuts.
        _ => {
            feeds.push(RowFeed {
                marker: MarkerFeed::NonNullable,
                values: ValueFeed::Opaque,
            });
        }
    }
    Ok(())
}

fn marker_feed(
    validity: &Validity,
    row_count: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<MarkerFeed> {
    if validity.nullability() == Nullability::NonNullable {
        return Ok(MarkerFeed::NonNullable);
    }
    Ok(match validity.execute_mask(row_count, ctx)? {
        Mask::AllTrue(_) => MarkerFeed::Constant(1),
        Mask::AllFalse(_) => MarkerFeed::Constant(0),
        Mask::Values(values) => MarkerFeed::Bytes(bits_to_bytes(values.bit_buffer())),
    })
}

/// Materialize a bit buffer into one byte per bit, so row loops avoid per-bit accessor calls.
fn bits_to_bytes(bits: &BitBuffer) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(bits.len());
    bytes.extend((0..bits.len()).map(|i| bits.value(i) as u8));
    bytes
}

#[cfg(test)]
mod tests;
