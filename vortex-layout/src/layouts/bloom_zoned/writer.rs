// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use async_trait::async_trait;
use fastbloom::BloomFilter;
use futures::StreamExt;
use std::mem::take;
use std::str::from_utf8_unchecked;
use std::sync::Arc;

use parking_lot::Mutex;
use std::thread::available_parallelism;
use vortex_array::arrays::VarBinArray;
use vortex_array::{ArrayContext, ArrayRef, Canonical, IntoArray};
use vortex_buffer::ByteBuffer;
use vortex_dtype::{DType, Nullability};
use vortex_error::{VortexResult, vortex_bail};
use vortex_io::runtime::Handle;

use crate::layouts::bloom_zoned::{BloomZonedLayout, serialize_bloom};
use crate::segments::SegmentSinkRef;
use crate::sequence::{
    SendableSequentialStream, SequencePointer, SequentialArrayStreamExt, SequentialStreamAdapter,
    SequentialStreamExt,
};
use crate::{IntoLayout, LayoutRef, LayoutStrategy};
use vortex_mask::AllOr;

/// Configuration options for BloomZonedStrategy
pub struct BloomZonedOptions {
    /// Target false positive rate for bloom filters (eg, 0.01 = 1%)
    pub false_positive_rate: f64,
    /// Number of rows per zone (bloom filter granularity)
    pub zone_len: usize,
    /// Seed for bloom filter hash functions
    pub seed: u128,
    /// Number of chunks to process in parallel
    pub concurrency: usize,
}

impl Default for BloomZonedOptions {
    fn default() -> Self {
        Self {
            false_positive_rate: 0.01,
            zone_len: 8192,
            seed: 0x0123_4567_89ab_cdef_0123_4567_89ab_cdef,
            concurrency: available_parallelism().map(|v| v.get()).unwrap_or(1),
        }
    }
}

/// Layout strategy that adds bloom filters to zones of data for efficient pruning
pub struct BloomZonedStrategy {
    child: Arc<dyn LayoutStrategy>,
    bloom_strategy: Arc<dyn LayoutStrategy>,
    options: BloomZonedOptions,
}

impl BloomZonedStrategy {
    pub fn new<Child: LayoutStrategy, Bloom: LayoutStrategy>(
        child: Child,
        bloom_strategy: Bloom,
        options: BloomZonedOptions,
    ) -> Self {
        Self {
            child: Arc::new(child),
            bloom_strategy: Arc::new(bloom_strategy),
            options,
        }
    }
}

/// Per-chunk bloom filter computed in parallel
struct ChunkBloom {
    /// Serialized bloom filter bytes for this chunk
    bloom_bytes: ByteBuffer,
}

impl ChunkBloom {
    /// Build a bloom filter for a chunk
    fn from_chunk(
        array: &ArrayRef,
        false_positive_rate: f64,
        seed: u128,
    ) -> VortexResult<Option<Self>> {
        let dtype = array.dtype();
        if !dtype.is_utf8() {
            // Skip non-utf8 columns silently, currently only support string data
            return Ok(None);
        }

        let canonical = array.to_canonical();
        let Canonical::VarBinView(view_array) = canonical else {
            vortex_bail!("Unable to build bloom filter from non-varbin array");
        };
        if !view_array.dtype().is_utf8() {
            vortex_bail!("Bloom filter expects utf8 canonical array");
        }

        let row_count = view_array.len();
        let mut bloom = BloomFilter::with_false_pos(false_positive_rate)
            .seed(&seed)
            .expected_items(row_count.max(1));

        // Insert all valid UTF-8 values into bloom filter
        match view_array.validity_mask().slices() {
            AllOr::All => {
                for idx in 0..view_array.len() {
                    let bytes = view_array.bytes_at(idx);
                    let string = unsafe { from_utf8_unchecked(bytes.as_slice()) };
                    bloom.insert(string);
                }
            }
            AllOr::None => {
                // No valid values, return empty bloom
            }
            AllOr::Some(ranges) => {
                for &(start, stop) in ranges {
                    for index in start..stop {
                        let bytes = view_array.bytes_at(index);
                        let string = unsafe { from_utf8_unchecked(bytes.as_slice()) };
                        bloom.insert(string);
                    }
                }
            }
        }

        // Serialize bloom filter using the shared serialization function
        let bloom_bytes = serialize_bloom(&bloom);

        Ok(Some(ChunkBloom { bloom_bytes }))
    }
}

/// Accumulates per-chunk blooms (one bloom per chunk = one zone)
struct BloomZoneAccumulator {
    zones: Vec<ByteBuffer>,
}

impl BloomZoneAccumulator {
    fn new() -> Self {
        Self { zones: Vec::new() }
    }

    /// Push a pre-computed chunk bloom (each chunk = one zone)
    fn push_chunk(&mut self, chunk_bloom: &ChunkBloom) {
        self.zones.push(chunk_bloom.bloom_bytes.clone());
    }

    /// Return bloom zones table (one row per chunk)
    fn as_bloom_table(&mut self) -> Option<ArrayRef> {
        if self.zones.is_empty() {
            return None;
        }

        // Create VarBinArray of bloom filter bytes (one per chunk/zone)
        let zones = take(&mut self.zones);
        Some(
            VarBinArray::from_iter(
                zones.into_iter().map(Some).collect::<Vec<_>>(),
                DType::Binary(Nullability::NonNullable),
            )
            .into_array(),
        )
    }
}

#[async_trait]
impl LayoutStrategy for BloomZonedStrategy {
    async fn write_stream(
        &self,
        ctx: ArrayContext,
        segment_sink: SegmentSinkRef,
        stream: SendableSequentialStream,
        mut eof: SequencePointer,
        handle: Handle,
    ) -> VortexResult<LayoutRef> {
        let handle2 = handle.clone();
        let zone_len = self.options.zone_len;
        let false_positive_rate = self.options.false_positive_rate;
        let seed = self.options.seed;

        let bloom_accumulator = Arc::new(Mutex::new(BloomZoneAccumulator::new()));

        // Capture dtype before transforming stream
        let dtype = stream.dtype().clone();

        // We can compute per-chunk bloom filters in parallel, so we spawn tasks for each chunk
        let stream = stream
            .map(move |chunk| {
                handle2.spawn_cpu(move || {
                    let (sequence_id, chunk) = chunk?;
                    // Build bloom filter for this chunk (CPU-intensive: canonicalization, hashing)
                    let chunk_bloom = ChunkBloom::from_chunk(&chunk, false_positive_rate, seed)?;
                    VortexResult::Ok((sequence_id, chunk, chunk_bloom))
                })
            })
            .buffered(self.options.concurrency);

        // Now we accumulate the blooms we computed above, this time we cannot spawn because we
        // need to feed the accumulator an ordered stream.
        let bloom_accumulator2 = bloom_accumulator.clone();
        let stream = SequentialStreamAdapter::new(
            dtype,
            stream.map(move |item| {
                let (sequence_id, chunk, chunk_bloom) = item?;
                // Accumulate the already-computed bloom filter
                if let Some(bloom) = chunk_bloom {
                    bloom_accumulator2.lock().push_chunk(&bloom);
                }
                Ok((sequence_id, chunk))
            }),
        )
        .sendable();
        // The eof used for the data child should appear _before_ our own bloom tables.
        let data_eof = eof.split_off();
        let data_layout = self
            .child
            .write_stream(
                ctx.clone(),
                segment_sink.clone(),
                stream,
                data_eof,
                handle.clone(),
            )
            .await?;

        let Some(bloom_table) = bloom_accumulator.lock().as_bloom_table() else {
            // If we have no blooms (e.g. non-utf8 data), then we just return the child layout.
            return Ok(data_layout);
        };

        // Write bloom zones using the bloom strategy
        let bloom_stream = bloom_table.to_array_stream().sequenced(eof.split_off());
        let bloom_layout = self
            .bloom_strategy
            .write_stream(ctx, segment_sink, bloom_stream, eof, handle)
            .await?;

        Ok(BloomZonedLayout::new(data_layout, bloom_layout, zone_len, seed).into_layout())
    }

    fn buffered_bytes(&self) -> u64 {
        self.child.buffered_bytes() + self.bloom_strategy.buffered_bytes()
    }
}
