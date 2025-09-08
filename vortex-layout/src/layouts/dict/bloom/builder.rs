// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::{Array, ToCanonical};
use vortex_error::VortexResult;

use crate::layouts::dict::bloom::BloomFilter;
use crate::layouts::zoned::accumulator::Accumulator;

/// Build a bloom filter by accumulating tokens derived from the non-null values of
/// a [UTF-8][vortex_dtype::DType::Utf8] array.
#[derive(Default)]
pub struct BloomFilterAccumulator {
    zone_filters: Vec<BloomFilter>,
}

/// Target false positivity rate for bloom filters
const FP_TARGET: f64 = 0.01;

impl Accumulator for BloomFilterAccumulator {
    type Value = Vec<BloomFilter>;

    fn push_chunk(&mut self, chunk: &dyn Array) -> VortexResult<()> {
        // Only UTF-8 columns can populate a Bloom filter.
        if !chunk.dtype().is_utf8() {
            return Ok(());
        }

        let chunk = chunk.to_varbinview();

        // NOTE(aduffy): the chunk length is a weak estimate. When we tokenize, that can yield
        //  either substantially more OR substantially fewer unique tokens than the number of
        //  strings in the chunk. Not really sure how to do much better though without paying a high
        //  memory/compute cost.
        let mut filter = BloomFilter::new_sbbf_ndv_fp(chunk.len(), FP_TARGET);

        for index in 0..chunk.len() {
            if chunk.is_valid(index) {
                let bytes = chunk.bytes_at(index);
                // SAFETY: DType is checked to be UTF-8 above
                let string = unsafe { std::str::from_utf8_unchecked(bytes.as_ref()) };
                filter.insert(string);
            }
        }

        self.zone_filters.push(filter);
        Ok(())
    }

    fn finish(self) -> VortexResult<Option<Self::Value>> {
        if self.zone_filters.is_empty() {
            Ok(None)
        } else {
            Ok(Some(self.zone_filters))
        }
    }
}
