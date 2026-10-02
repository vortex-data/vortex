// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use rstest::rstest;

use super::*;

const BLOCK: Alignment = Alignment::new(4096);

#[rstest]
// Contiguous packing pads only up to the segment's own alignment.
#[case(SegmentPadding::None, 10, 1 << 20, Alignment::new(8), 6)]
#[case(SegmentPadding::None, 16, 1 << 20, Alignment::new(8), 0)]
// Always pads to the block, however small the segment.
#[case(SegmentPadding::block_aligned(), 10, 4, Alignment::new(8), 4086)]
#[case(SegmentPadding::block_aligned(), 4096, 4, Alignment::new(8), 0)]
// A segment demanding more than a block is aligned to its own requirement.
#[case(SegmentPadding::block_aligned(), 10, 4, Alignment::new(1 << 16), 65526)]
// Grouping leaves a segment that fits in the rest of the current block where it is.
#[case(SegmentPadding::grouped(), 10, 4, Alignment::new(8), 6)]
#[case(SegmentPadding::grouped(), 4000, 96, Alignment::new(8), 0)]
// One byte more than the block has room for, and it moves to the next block.
#[case(SegmentPadding::grouped(), 4000, 97, Alignment::new(8), 96)]
// Anything at least a block long is left packed: a straddle costs it far less than padding.
#[case(SegmentPadding::grouped(), 10, 1 << 20, Alignment::new(8), 6)]
#[case(SegmentPadding::grouped(), 4000, 4096, Alignment::new(8), 0)]
// A segment demanding more than a block still only pays its own alignment.
#[case(SegmentPadding::grouped(), 10, 4, Alignment::new(1 << 16), 65526)]
// Proportional pads a large segment, but leaves a small one packed.
#[case(SegmentPadding::proportional(), 10, 1 << 20, Alignment::new(8), 4086)]
#[case(SegmentPadding::proportional(), 10, 4, Alignment::new(8), 6)]
// Exactly on budget: 4086 * 64 <= 261_504.
#[case(SegmentPadding::proportional(), 10, 261_504, Alignment::new(8), 4086)]
#[case(SegmentPadding::proportional(), 10, 261_503, Alignment::new(8), 6)]
fn pads_segments(
    #[case] padding: SegmentPadding,
    #[case] byte_offset: u64,
    #[case] length: u64,
    #[case] alignment: Alignment,
    #[case] expected: u64,
) {
    assert_eq!(padding.padding(byte_offset, length, alignment), expected);
}

#[rstest]
#[case(SegmentPadding::block_aligned())]
#[case(SegmentPadding::grouped())]
#[case(SegmentPadding::proportional())]
#[case(SegmentPadding::None)]
fn padding_always_satisfies_the_segments_own_alignment(#[case] padding: SegmentPadding) {
    for byte_offset in 0..300u64 {
        for alignment in [1, 2, 8, 64, 256] {
            let alignment = Alignment::new(alignment);
            let offset = byte_offset + padding.padding(byte_offset, 1 << 20, alignment);
            assert!(alignment.is_offset_aligned(offset as usize));
        }
    }
}

/// The proportional budget must hold across a whole file, not just per segment.
#[test]
fn proportional_padding_is_bounded_by_the_overhead_budget() {
    let padding = SegmentPadding::proportional();
    let mut offset = 0u64;
    let mut padded = 0u64;
    let mut data = 0u64;
    for i in 0..10_000u64 {
        // A spread of segment sizes from a few bytes to a few MiB.
        let length = 3 + (i * 7919) % (4 << 20);
        let pad = padding.padding(offset, length, Alignment::new(8));
        offset += pad + length;
        padded += pad;
        data += length;
    }
    assert!(
        padded * u64::from(DEFAULT_MAX_OVERHEAD_RATIO) <= data,
        "padding {padded} exceeded the 1/{DEFAULT_MAX_OVERHEAD_RATIO} budget of {data} bytes"
    );
}

/// Block alignment must never cost more than a block per segment.
/// The whole point of grouping: sub-block segments touch one block each, as they would under
/// block alignment, for a fraction of the padding.
#[test]
fn grouping_reads_like_block_alignment_for_less_padding() {
    let (mut grouped_offset, mut always_offset) = (0u64, 0u64);
    let (mut grouped_pad, mut always_pad) = (0u64, 0u64);
    for i in 0..10_000u64 {
        // Sub-block segments, which is where the two policies are supposed to agree on reads.
        let length = 1 + (i * 7919) % ((1 << 12) - 1);
        let alignment = Alignment::new(1 << (i % 5));

        let pad = SegmentPadding::grouped().padding(grouped_offset, length, alignment);
        grouped_offset += pad;
        grouped_pad += pad;
        let pad = SegmentPadding::block_aligned().padding(always_offset, length, alignment);
        always_offset += pad;
        always_pad += pad;

        assert_eq!(
            blocks_spanned(grouped_offset, length),
            blocks_spanned(always_offset, length),
            "segment {i} of {length} bytes spans a different number of blocks"
        );

        grouped_offset += length;
        always_offset += length;
    }
    // Grouping pads a fraction of what block-aligning does on this distribution.
    assert!(
        grouped_pad * 2 < always_pad,
        "grouping padded {grouped_pad} against {always_pad} block-aligned"
    );
}

/// A grouped segment never straddles a block it could have fit inside.
#[test]
fn grouping_never_splits_a_segment_that_fits_in_a_block() {
    let mut offset = 0u64;
    for i in 0..10_000u64 {
        let length = 1 + (i * 4099) % (1 << 13);
        offset += SegmentPadding::grouped().padding(offset, length, Alignment::new(8));
        if length < BLOCK.as_usize() as u64 {
            assert_eq!(blocks_spanned(offset, length), 1, "segment {i} was split");
        }
        offset += length;
    }
}

fn blocks_spanned(offset: u64, length: u64) -> u64 {
    let block = BLOCK.as_usize() as u64;
    (offset % block + length).div_ceil(block)
}

#[rstest]
#[case("none", SegmentPadding::None)]
#[case("always", SegmentPadding::block_aligned())]
#[case("grouped", SegmentPadding::grouped())]
#[case("proportional", SegmentPadding::proportional())]
#[case(" grouped ", SegmentPadding::grouped())]
#[case(
    "proportional:8",
    SegmentPadding::Proportional { block: BLOCK, max_overhead_ratio: 8 }
)]
fn parses_a_padding_policy(#[case] value: &str, #[case] expected: SegmentPadding) {
    assert_eq!(SegmentPadding::parse(value), Some(expected));
}

#[rstest]
#[case("")]
#[case("grouped:4096")]
#[case("aligned")]
#[case("proportional:0")]
#[case("proportional:-1")]
#[case("proportional:")]
fn rejects_an_unknown_padding_policy(#[case] value: &str) {
    assert_eq!(SegmentPadding::parse(value), None);
}

#[test]
fn block_alignment_costs_at_most_one_block_per_segment() {
    let padding = SegmentPadding::block_aligned();
    for byte_offset in [0u64, 1, 7, 4095, 4096, 1 << 20, (1 << 20) + 3] {
        let pad = padding.padding(byte_offset, 1024, Alignment::new(8));
        assert!(pad < BLOCK.as_usize() as u64);
        assert_eq!((byte_offset + pad) % BLOCK.as_usize() as u64, 0);
    }
}

mod end_to_end {
    use std::sync::Arc;
    use std::sync::LazyLock;

    use tempfile::TempDir;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::array_session;
    use vortex_array::arrays::ChunkedArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::StructArray;
    use vortex_array::arrays::VarBinArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::memory::BufferAllocatorRef;
    use vortex_array::stream::ArrayStreamExt;
    use vortex_buffer::Alignment;
    use vortex_error::VortexResult;
    use vortex_io::runtime::Handle;
    use vortex_io::session::RuntimeSession;
    use vortex_io::session::RuntimeSessionExt;
    use vortex_io::std_file::FileReadAt;
    use vortex_io::std_file::FileReadAtOptions;
    use vortex_io::std_file::FileWrite;
    use vortex_layout::session::LayoutSession;
    use vortex_session::VortexSession;

    use super::*;
    use crate::OpenOptionsSessionExt;
    use crate::WriteOptionsSessionExt;
    use crate::footer::SegmentSpec;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = array_session()
            .with::<LayoutSession>()
            .with::<RuntimeSession>();
        crate::register_default_encodings(&session);
        crate::enable_all_registered_array_encodings(&session);
        session
    });

    /// Enough chunks and columns to produce a spread of small and large segments.
    fn sample() -> VortexResult<vortex_array::ArrayRef> {
        let numbers =
            ChunkedArray::from_iter((0..8).map(|c| {
                PrimitiveArray::from_iter((0..2000i64).map(|i| i * (c + 1))).into_array()
            }))
            .into_array();
        let strings = ChunkedArray::from_iter((0..8).map(|c| {
            VarBinArray::from_iter(
                (0..2000).map(|i| Some(format!("chunk-{c}-row-{i}"))),
                vortex_array::dtype::DType::Utf8(vortex_array::dtype::Nullability::Nullable),
            )
            .into_array()
        }))
        .into_array();
        Ok(StructArray::from_fields(&[("numbers", numbers), ("strings", strings)])?.into_array())
    }

    async fn write_to(path: &std::path::Path, padding: SegmentPadding) -> VortexResult<()> {
        let write = FileWrite::create(path, SESSION.handle()).await?;
        SESSION
            .write_options()
            .with_segment_padding(padding)
            .write(write, sample()?.to_array_stream())
            .await?;
        Ok(())
    }

    async fn segment_specs(path: &std::path::Path) -> VortexResult<Arc<[SegmentSpec]>> {
        let file = SESSION.open_options().open_path(path).await?;
        Ok(Arc::clone(file.footer().segment_map()))
    }

    fn block_of(spec: &SegmentSpec) -> u64 {
        DEFAULT_BLOCK_SIZE.max(spec.alignment).as_usize() as u64
    }

    #[tokio::test]
    async fn block_alignment_aligns_every_segment() -> VortexResult<()> {
        let dir = TempDir::new()?;
        let path = dir.path().join("aligned.vortex");
        write_to(&path, SegmentPadding::block_aligned()).await?;

        let specs = segment_specs(&path).await?;
        assert!(!specs.is_empty());
        for spec in specs.iter() {
            assert_eq!(
                spec.offset % block_of(spec),
                0,
                "segment at {} is not block aligned",
                spec.offset
            );
        }
        Ok(())
    }

    #[tokio::test]
    async fn contiguous_packing_leaves_segments_unaligned() -> VortexResult<()> {
        let dir = TempDir::new()?;
        let path = dir.path().join("packed.vortex");
        write_to(&path, SegmentPadding::None).await?;

        let specs = segment_specs(&path).await?;
        assert!(
            specs.iter().any(|spec| spec.offset % block_of(spec) != 0),
            "expected contiguously packed segments to straddle block boundaries"
        );
        Ok(())
    }

    /// Proportional padding must align the segments that dominate the bytes read while leaving the
    /// long tail of small segments packed.
    #[tokio::test]
    async fn proportional_padding_aligns_only_the_large_segments() -> VortexResult<()> {
        let dir = TempDir::new()?;
        let path = dir.path().join("proportional.vortex");
        write_to(&path, SegmentPadding::proportional()).await?;

        let specs = segment_specs(&path).await?;
        let budget = u64::from(DEFAULT_MAX_OVERHEAD_RATIO);
        for spec in specs.iter() {
            let block = block_of(spec);
            if spec.offset % block != 0 {
                // Only a segment too small to afford a block boundary may be left unaligned.
                let required = (block - spec.offset % block) % block;
                assert!(
                    required * budget > u64::from(spec.length),
                    "segment of {} bytes at {} could have afforded {required} bytes of padding",
                    spec.length,
                    spec.offset
                );
            }
        }
        Ok(())
    }

    async fn read_back(
        path: &std::path::Path,
        handle: Handle,
        options: FileReadAtOptions,
    ) -> VortexResult<vortex_array::ArrayRef> {
        let reader = FileReadAt::open_with_options(
            path,
            handle,
            BufferAllocatorRef::statically_allocated(),
            options,
        )?;
        SESSION
            .open_options()
            .open(Arc::new(reader))
            .await?
            .scan()?
            .into_array_stream()?
            .read_all()
            .await
    }

    /// Every padding policy must round-trip identically under both buffered and direct reads,
    /// including the contiguously packed layout that files written before this option used.
    #[rstest]
    #[case(SegmentPadding::None)]
    #[case(SegmentPadding::block_aligned())]
    #[case(SegmentPadding::grouped())]
    #[case(SegmentPadding::proportional())]
    #[tokio::test]
    async fn direct_reads_round_trip_every_padding_policy(
        #[case] padding: SegmentPadding,
    ) -> VortexResult<()> {
        let dir = TempDir::new()?;
        let path = dir.path().join("data.vortex");
        write_to(&path, padding).await?;

        let handle = SESSION.handle();
        let direct = {
            #[cfg(target_os = "linux")]
            {
                FileReadAtOptions::default().with_direct_io()
            }
            #[cfg(not(target_os = "linux"))]
            {
                FileReadAtOptions::default()
            }
        };

        let mut ctx = SESSION.create_execution_ctx();
        let expected = read_back(&path, handle.clone(), FileReadAtOptions::default()).await?;
        let actual = read_back(&path, handle, direct).await?;
        assert_arrays_eq!(expected, actual, &mut ctx);
        Ok(())
    }

    /// Block alignment must be a pure layout change: the same bytes, at padded offsets.
    #[tokio::test]
    async fn block_alignment_only_moves_segments() -> VortexResult<()> {
        let dir = TempDir::new()?;
        let packed = dir.path().join("packed.vortex");
        let aligned = dir.path().join("aligned.vortex");
        write_to(&packed, SegmentPadding::None).await?;
        write_to(&aligned, SegmentPadding::block_aligned()).await?;

        let packed_specs = segment_specs(&packed).await?;
        let aligned_specs = segment_specs(&aligned).await?;
        assert_eq!(packed_specs.len(), aligned_specs.len());
        for (packed, aligned) in packed_specs.iter().zip(aligned_specs.iter()) {
            assert_eq!(packed.length, aligned.length);
            assert_eq!(packed.alignment, aligned.alignment);
        }

        assert!(
            std::fs::metadata(&aligned)?.len() > std::fs::metadata(&packed)?.len(),
            "block alignment should cost padding bytes"
        );
        Ok(())
    }

    #[test]
    fn write_options_default_to_contiguous_packing() {
        assert_eq!(SegmentPadding::default(), SegmentPadding::None);
    }

    const _: () = assert!(DEFAULT_BLOCK_SIZE.as_usize() == 4096);
    const _: Alignment = DEFAULT_BLOCK_SIZE;
}
