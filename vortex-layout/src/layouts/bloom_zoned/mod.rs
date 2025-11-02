// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod reader;
mod writer;

pub use writer::{BloomZonedOptions, BloomZonedStrategy};

use std::sync::Arc;

use crate::segments::{SegmentId, SegmentSource};
use crate::{
    LayoutChildType, LayoutChildren, LayoutEncodingRef, LayoutId, LayoutReaderRef, LayoutRef,
    VTable, vtable,
};
use fastbloom::BloomFilter;
use vortex_array::{ArrayContext, DeserializeMetadata, SerializeMetadata};
use vortex_buffer::ByteBuffer;
use vortex_dtype::{DType, Nullability};
use vortex_error::{VortexExpect, VortexResult, vortex_bail, vortex_panic};

use reader::BloomZonedReader;

vtable!(BloomZoned);

/// Serialize a bloom filter to bytes: [num_hashes: u32][words: &[u64]]
///
/// The format is: 4 bytes for num_hashes in little-endian, followed by
/// the bloom filter words as u64s in little-endian.
pub(crate) fn serialize_bloom(bloom: &BloomFilter) -> ByteBuffer {
    let words = bloom.as_slice();
    let num_hashes = bloom.num_hashes();

    // 4 bytes for num_hashes, followed by the bloom filter words as u64s (8 bytes each)
    let mut bytes = Vec::with_capacity(4 + words.len() * 8);
    bytes.extend_from_slice(&num_hashes.to_le_bytes());
    for &word in words {
        bytes.extend_from_slice(&word.to_le_bytes());
    }

    ByteBuffer::from(bytes)
}

/// Deserialize a bloom filter from bytes: [num_hashes: u32][words: &[u64]]
///
/// The seed must match the seed used when the bloom filter was created.
pub(crate) fn deserialize_bloom(bytes: &[u8], seed: u128) -> VortexResult<BloomFilter> {
    if bytes.len() < 4 {
        vortex_bail!("Bloom filter bytes too short: {}", bytes.len());
    }

    let num_hashes = u32::from_le_bytes(bytes[0..4].try_into()?);
    let payload = &bytes[4..];

    if payload.len() % 8 != 0 {
        vortex_bail!(
            "Bloom filter byte length mismatch: bloom words must be multiples of 8, got {}",
            payload.len()
        );
    }

    let mut words = Vec::with_capacity(payload.len() / 8);
    for chunk in payload.chunks_exact(8) {
        words.push(u64::from_le_bytes(chunk.try_into()?));
    }

    Ok(BloomFilter::from_vec(words).seed(&seed).hashes(num_hashes))
}

impl VTable for BloomZonedVTable {
    type Layout = BloomZonedLayout;
    type Encoding = BloomZonedLayoutEncoding;
    type Metadata = BloomZonedMetadata;

    fn id(_encoding: &Self::Encoding) -> LayoutId {
        LayoutId::new_ref("smithdb.bloom_zoned")
    }

    fn encoding(_layout: &Self::Layout) -> LayoutEncodingRef {
        LayoutEncodingRef::new_ref(BloomZonedLayoutEncoding.as_ref())
    }

    fn row_count(layout: &Self::Layout) -> u64 {
        layout.data.row_count()
    }

    fn dtype(layout: &Self::Layout) -> &DType {
        layout.data.dtype()
    }

    fn metadata(layout: &Self::Layout) -> Self::Metadata {
        BloomZonedMetadata {
            zone_len: u32::try_from(layout.zone_len).vortex_expect("Invalid zone length"),
            seed: layout.seed(),
        }
    }

    fn segment_ids(_layout: &Self::Layout) -> Vec<SegmentId> {
        vec![]
    }

    fn nchildren(_layout: &Self::Layout) -> usize {
        2
    }

    fn child(layout: &Self::Layout, idx: usize) -> VortexResult<LayoutRef> {
        match idx {
            0 => Ok(layout.data.clone()),
            1 => Ok(layout.bloom_zones.clone()),
            _ => vortex_bail!("Invalid child index: {}", idx),
        }
    }

    fn child_type(_layout: &Self::Layout, idx: usize) -> LayoutChildType {
        match idx {
            0 => LayoutChildType::Transparent("data".into()),
            1 => LayoutChildType::Auxiliary("bloom_zones".into()),
            _ => vortex_panic!("Invalid child index: {}", idx),
        }
    }

    fn new_reader(
        layout: &Self::Layout,
        name: Arc<str>,
        segment_source: Arc<dyn SegmentSource>,
    ) -> VortexResult<LayoutReaderRef> {
        Ok(Arc::new(BloomZonedReader::try_new(
            layout.clone(),
            name,
            segment_source,
        )?))
    }

    fn build(
        _encoding: &Self::Encoding,
        dtype: &DType,
        _row_count: u64,
        metadata: &<Self::Metadata as DeserializeMetadata>::Output,
        _segment_ids: Vec<SegmentId>,
        children: &dyn LayoutChildren,
        _ctx: ArrayContext,
    ) -> VortexResult<Self::Layout> {
        let data = children.child(0, dtype)?;

        // Bloom zones should be binary data
        let bloom_zones_dtype = DType::Binary(Nullability::NonNullable);
        let bloom_zones = children.child(1, &bloom_zones_dtype)?;

        Ok(BloomZonedLayout::new(
            data,
            bloom_zones,
            metadata.zone_len as usize,
            metadata.seed,
        ))
    }
}

#[derive(Debug)]
pub struct BloomZonedLayoutEncoding;

#[derive(Clone, Debug)]
pub struct BloomZonedLayout {
    data: LayoutRef,
    bloom_zones: LayoutRef,
    zone_len: usize,
    seed: u128,
}

impl BloomZonedLayout {
    pub fn new(data: LayoutRef, bloom_zones: LayoutRef, zone_len: usize, seed: u128) -> Self {
        if zone_len == 0 {
            vortex_panic!("Zone length must be greater than 0");
        }
        Self {
            data,
            bloom_zones,
            zone_len,
            seed,
        }
    }

    pub fn data(&self) -> &LayoutRef {
        &self.data
    }

    pub fn bloom_zones(&self) -> &LayoutRef {
        &self.bloom_zones
    }

    pub fn zone_len(&self) -> usize {
        self.zone_len
    }

    pub fn nzones(&self) -> usize {
        usize::try_from(self.bloom_zones.row_count()).vortex_expect("Invalid number of zones")
    }

    pub fn seed(&self) -> u128 {
        self.seed
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct BloomZonedMetadata {
    /// Number of rows per zone
    pub(super) zone_len: u32,
    pub(super) seed: u128,
}

impl SerializeMetadata for BloomZonedMetadata {
    fn serialize(self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(20);
        bytes.extend_from_slice(&self.zone_len.to_le_bytes());
        bytes.extend_from_slice(&self.seed.to_le_bytes());
        bytes
    }
}

impl DeserializeMetadata for BloomZonedMetadata {
    type Output = Self;

    fn deserialize(metadata: &[u8]) -> VortexResult<Self::Output> {
        if metadata.len() != 20 {
            vortex_bail!(
                "Invalid BloomZonedMetadata size: expected 20 bytes, got {}",
                metadata.len()
            );
        }

        let zone_len = u32::from_le_bytes(metadata[0..4].try_into()?);
        let seed = u128::from_le_bytes(metadata[4..20].try_into()?);

        Ok(Self { zone_len, seed })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::{fixture, rstest};
    use std::sync::Arc;

    use crate::LayoutRef;
    use crate::LayoutStrategy;
    use crate::layouts::chunked::writer::ChunkedLayoutStrategy;
    use crate::layouts::flat::writer::FlatLayoutStrategy;
    use crate::segments::{SegmentSource, TestSegments};
    use crate::sequence::{SequenceId, SequentialArrayStreamExt};
    use vortex_array::ArrayContext;
    use vortex_array::IntoArray;
    use vortex_array::arrays::{ChunkedArray, PrimitiveArray, VarBinArray};
    use vortex_dtype::{DType, Nullability};
    use vortex_expr::{eq, list_contains, lit, root};
    use vortex_io::runtime::single::block_on;
    use vortex_mask::Mask;
    use vortex_scalar::Scalar;

    #[rstest]
    #[case::min_value(0, 0)]
    #[case::max_value(u32::MAX, u128::MAX)]
    #[case::typical_value(8192, 0x0123_4567_89ab_cdef_0000_0000_0000_0001)]
    #[case::small_value(1024, 42)]
    #[case::medium_value(4096, 0xfedc_ba98_7654_3210_0011_2233_4455_6677)]
    #[case::large_value(16384, 0xabcdef0123456789abcdef0123456789)]
    #[case::one(1, 7)]
    fn test_metadata_serialization(#[case] zone_len: u32, #[case] seed: u128) {
        let metadata = BloomZonedMetadata { zone_len, seed };
        let serialized = metadata.clone().serialize();
        let deserialized = BloomZonedMetadata::deserialize(&serialized).unwrap();
        assert_eq!(deserialized, metadata);
    }

    #[rstest]
    #[case::too_short(3)]
    #[case::too_long(5)]
    #[case::way_too_short(1)]
    #[case::almost_correct(19)]
    #[case::way_too_long(21)]
    fn test_deserialize_invalid_size(#[case] size: usize) {
        let invalid_metadata = vec![0u8; size];
        let result = BloomZonedMetadata::deserialize(&invalid_metadata);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Invalid BloomZonedMetadata size")
        );
    }

    #[test]
    fn test_serialize_size() {
        let metadata = BloomZonedMetadata {
            zone_len: 1024,
            seed: 123,
        };
        let serialized = metadata.serialize();
        assert_eq!(
            serialized.len(),
            20,
            "Serialized metadata should be 20 bytes"
        );
    }

    #[test]
    fn test_metadata_byte_order() {
        // Test that values are correctly encoded in little-endian
        let metadata = BloomZonedMetadata {
            zone_len: 0x12345678,
            seed: 0x0123_4567_89ab_cdef_fedc_ba98_7654_3210,
        };
        let serialized = metadata.serialize();

        // Verify little-endian encoding
        assert_eq!(&serialized[0..4], &[0x78, 0x56, 0x34, 0x12]); // zone_len
        assert_eq!(
            &serialized[4..20],
            &0x0123_4567_89ab_cdef_fedc_ba98_7654_3210u128.to_le_bytes()
        );
    }

    #[rstest]
    #[case::small(vec!["a", "b"], 0.01, 123)]
    #[case::medium(vec!["alpha", "bravo", "charlie", "delta"], 0.001, 456)]
    #[case::large(vec!["one", "two", "three", "four", "five", "six", "seven"], 0.0001, 789)]
    #[case::single(vec!["single"], 0.01, 1)]
    #[case::high_fpr(vec!["x", "y", "z"], 0.1, 999)]
    fn test_bloom_serialize_deserialize_round_trip(
        #[case] values: Vec<&str>,
        #[case] false_positive_rate: f64,
        #[case] seed: u128,
    ) {
        // Create and populate bloom filter
        let mut bloom = BloomFilter::with_false_pos(false_positive_rate)
            .seed(&seed)
            .expected_items(values.len().max(1));

        for value in &values {
            bloom.insert(value);
        }

        let serialized = serialize_bloom(&bloom);
        let deserialized =
            deserialize_bloom(serialized.as_slice(), seed).expect("deserialize should succeed");

        // Verify all original values are present
        for value in &values {
            assert!(
                deserialized.contains(value),
                "Value '{}' should be present after round-trip",
                value
            );
        }
    }

    #[test]
    fn test_bloom_serialize_empty() {
        let seed = 42u128;
        let bloom = BloomFilter::with_false_pos(0.01)
            .seed(&seed)
            .expected_items(1);

        let serialized = serialize_bloom(&bloom);
        let deserialized =
            deserialize_bloom(serialized.as_slice(), seed).expect("deserialize should succeed");

        // Should not contain anything
        assert!(!deserialized.contains("anything"));
    }

    #[test]
    fn test_bloom_deserialize_invalid_length() {
        // Too short (less than 4 bytes for num_hashes)
        let err = deserialize_bloom(&[0, 0, 0], 0).unwrap_err();
        assert!(err.to_string().contains("too short"));

        // Invalid word alignment (not multiple of 8)
        let err = deserialize_bloom(&[1, 0, 0, 0, 1, 2, 3], 0).unwrap_err();
        assert!(err.to_string().contains("bloom words"));
    }

    #[fixture]
    fn bloom_fixture() -> (Arc<dyn SegmentSource>, LayoutRef, usize, usize, u128) {
        let zone_len = 3;
        let seed = 0x0123_4567_89ab_cdef_0000_0000_0000_0001;
        let options = BloomZonedOptions {
            false_positive_rate: 1e-9,
            zone_len,
            seed,
            concurrency: 1,
        };

        let chunks = [
            vec!["alpha", "bravo", "charlie"],
            vec!["delta", "echo", "foxtrot"],
            vec!["golf", "hotel", "india"],
        ];

        let chunk_arrays: Vec<_> = chunks
            .iter()
            .map(|chunk| {
                VarBinArray::from_iter(
                    chunk.iter().copied().map(Some),
                    DType::Utf8(Nullability::NonNullable),
                )
                .into_array()
            })
            .collect();

        let row_count: usize = chunk_arrays.iter().map(|array| array.len()).sum();

        let chunked_array = ChunkedArray::from_iter(chunk_arrays).into_array();

        let segments = Arc::new(TestSegments::default());
        let (ptr, eof) = SequenceId::root().split();

        let strategy = BloomZonedStrategy::new(
            ChunkedLayoutStrategy::new(FlatLayoutStrategy::default()),
            FlatLayoutStrategy::default(),
            options,
        );

        let array_stream = chunked_array.to_array_stream().sequenced(ptr);

        let layout = block_on(|handle| {
            strategy.write_stream(
                ArrayContext::empty(),
                segments.clone(),
                array_stream,
                eof,
                handle,
            )
        })
        .unwrap();

        let segment_source: Arc<dyn SegmentSource> = segments;
        (segment_source, layout, row_count, zone_len, seed)
    }

    #[rstest]
    fn test_bloom_layout_writer_produces_layout(
        #[from(bloom_fixture)] (segments, layout, row_count, zone_len, seed): (
            Arc<dyn SegmentSource>,
            LayoutRef,
            usize,
            usize,
            u128,
        ),
    ) {
        assert_eq!(usize::try_from(layout.row_count()).unwrap(), row_count);
        let bloom_layout = layout.as_ref().as_::<BloomZonedVTable>();
        assert_eq!(bloom_layout.zone_len(), zone_len);
        assert_eq!(bloom_layout.seed(), seed);
        assert_eq!(bloom_layout.nzones(), 3);

        // Ensure we can instantiate a reader
        assert!(
            layout
                .new_reader("col".into(), segments)
                .unwrap()
                .dtype()
                .is_utf8()
        );
    }

    #[rstest]
    fn test_bloom_reader_prunes_eq(
        #[from(bloom_fixture)] (segments, layout, row_count, _zone_len, _seed): (
            Arc<dyn SegmentSource>,
            LayoutRef,
            usize,
            usize,
            u128,
        ),
    ) {
        let reader = layout
            .new_reader("col".into(), Arc::clone(&segments))
            .unwrap();

        let eq_expr = eq(root(), lit("delta"));
        let mask = block_on(|_handle| async {
            let future = reader.clone().pruning_evaluation(
                &(0..row_count as u64),
                &eq_expr,
                Mask::new_true(row_count),
            )?;
            future.await
        })
        .unwrap();

        let values: Vec<bool> = (0..row_count).map(|idx| mask.value(idx)).collect();
        assert_eq!(
            values,
            vec![false, false, false, true, true, true, false, false, false]
        );

        let missing_expr = eq(root(), lit("missing"));
        let missing_mask = block_on(|_handle| async {
            let future = reader.pruning_evaluation(
                &(0..row_count as u64),
                &missing_expr,
                Mask::new_true(row_count),
            )?;
            future.await
        })
        .unwrap();
        assert!(missing_mask.all_false());
    }

    #[rstest]
    fn test_bloom_reader_prunes_in(
        #[from(bloom_fixture)] (segments, layout, row_count, _zone_len, _seed): (
            Arc<dyn SegmentSource>,
            LayoutRef,
            usize,
            usize,
            u128,
        ),
    ) {
        let reader = layout.new_reader("col".into(), segments).unwrap();

        let list_scalar = Scalar::list(
            Arc::new(DType::Utf8(Nullability::NonNullable)),
            vec![Scalar::from("alpha"), Scalar::from("delta")],
            Nullability::NonNullable,
        );
        let in_expr = list_contains(lit(list_scalar), root());

        let mask = block_on(|_handle| async {
            let future = reader.pruning_evaluation(
                &(0..row_count as u64),
                &in_expr,
                Mask::new_true(row_count),
            )?;
            future.await
        })
        .unwrap();

        let values: Vec<bool> = (0..row_count).map(|idx| mask.value(idx)).collect();
        assert_eq!(
            values,
            vec![true, true, true, true, true, true, false, false, false]
        );
    }

    #[test]
    fn test_bloom_writer_skips_non_utf8() {
        let options = BloomZonedOptions {
            false_positive_rate: 1e-6,
            zone_len: 2,
            seed: 1234,
            concurrency: 1,
        };

        let segments = Arc::new(TestSegments::default());
        let (ptr, eof) = SequenceId::root().split();

        let strategy = BloomZonedStrategy::new(
            ChunkedLayoutStrategy::new(FlatLayoutStrategy::default()),
            FlatLayoutStrategy::default(),
            options,
        );

        let chunk_arrays = vec![
            PrimitiveArray::from_iter([1i32, 2, 3]).into_array(),
            PrimitiveArray::from_iter([4i32, 5, 6]).into_array(),
        ];

        let chunked_array = ChunkedArray::from_iter(chunk_arrays).into_array();
        let array_stream = chunked_array.to_array_stream().sequenced(ptr);

        let layout = block_on(|handle| {
            strategy.write_stream(ArrayContext::empty(), segments, array_stream, eof, handle)
        })
        .unwrap();

        assert!(!layout.as_ref().is::<BloomZonedVTable>());
    }
}
