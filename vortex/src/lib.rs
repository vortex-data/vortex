// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// https://github.com/rust-lang/cargo/pull/11645#issuecomment-1536905941
#![doc = include_str!(concat!("../", env!("CARGO_PKG_README")))]

pub mod session;

pub use vortex_array::*;
#[cfg(feature = "files")]
pub use vortex_file as file;
pub use {
    vortex_buffer as buffer, vortex_dtype as dtype, vortex_error as error, vortex_expr as expr,
    vortex_flatbuffers as flatbuffers, vortex_io as io, vortex_ipc as ipc, vortex_layout as layout,
    vortex_mask as mask, vortex_metrics as metrics, vortex_proto as proto, vortex_scalar as scalar,
    vortex_scan as scan, vortex_utils as utils,
};

pub mod compressor {
    pub use vortex_btrblocks::BtrBlocksCompressor;
    #[cfg(feature = "zstd")]
    pub use vortex_layout::layouts::compact::CompactCompressor;
}

pub mod encodings {
    #[cfg(feature = "zstd")]
    pub use vortex_zstd as zstd;
    pub use {
        vortex_alp as alp, vortex_bytebool as bytebool, vortex_datetime_parts as datetime_parts,
        vortex_decimal_byte_parts as decimal_byte_parts, vortex_dict as dict,
        vortex_fastlanes as fastlanes, vortex_fsst as fsst, vortex_pco as pco,
        vortex_runend as runend, vortex_sequence as sequence, vortex_sparse as sparse,
        vortex_zigzag as zigzag,
    };
}

/// These tests are included in the getting started documentation, so be mindful of which imports
/// to keep inside the test functions, and which to just use from the outer scope. The examples
/// get too verbose if we include _everything_.
#[cfg(test)]
mod test {
    use itertools::Itertools;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::iter::ArrayIteratorExt;
    use vortex_array::validity::Validity;
    use vortex_array::vtable::ValidityHelper;
    use vortex_array::{ArrayRef, IntoArray, ToCanonical};
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_expr::{gt, lit, root};
    use vortex_file::{VortexOpenOptions, VortexWriteOptions, WriteStrategyBuilder};
    use vortex_layout::layouts::compact::CompactCompressor;

    use crate as vortex;

    #[test]
    fn convert() -> anyhow::Result<()> {
        // [convert]
        use std::fs::File;

        use arrow_array::RecordBatchReader;
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
        use vortex::Array;
        use vortex::arrays::ChunkedArray;
        use vortex::dtype::DType;
        use vortex::dtype::arrow::FromArrowType;
        use vortex_array::arrow::FromArrowArray;

        let reader = ParquetRecordBatchReaderBuilder::try_new(File::open(
            "../docs/_static/example.parquet",
        )?)?
        .build()?;

        let dtype = DType::from_arrow(reader.schema());
        let chunks = reader
            .map_ok(|record_batch| ArrayRef::from_arrow(record_batch, false))
            .try_collect()?;
        let vortex_array = ChunkedArray::try_new(chunks, dtype)?.into_array();
        // [convert]

        assert_eq!(vortex_array.len(), 1000);

        Ok(())
    }

    #[test]
    fn compress() -> VortexResult<()> {
        // [compress]
        use vortex::compressor::{BtrBlocksCompressor, CompactCompressor};

        let array = PrimitiveArray::new(buffer![42u64; 100_000], Validity::NonNullable);

        // You can compress an array in-memory with the BtrBlocks compressor
        let compressed = BtrBlocksCompressor::default().compress(array.as_ref())?;
        println!(
            "BtrBlocks size: {} / {}",
            compressed.nbytes(),
            array.nbytes()
        );

        // Or apply generally stronger compression with the compact compressor
        let compressed = CompactCompressor::default()
            .with_values_per_page(8192)
            .compress(array.as_ref())?;
        println!("Compact size: {} / {}", compressed.nbytes(), array.nbytes());
        // [compress]

        Ok(())
    }

    #[tokio::test]
    async fn read_write() -> VortexResult<()> {
        // [write]
        let array = PrimitiveArray::new(buffer![0u64, 1, 2, 3, 4], Validity::NonNullable);

        // Write a Vortex file with the default compression and layout strategy.
        VortexWriteOptions::default()
            .write(
                &mut tokio::fs::File::create("example.vortex").await?,
                array.to_array_stream(),
            )
            .await?;

        // [write]

        // [read]
        let array = VortexOpenOptions::new()
            .open("example.vortex")
            .await?
            .scan()?
            .with_filter(gt(root(), lit(2u64)))
            .into_array_iter()?
            .read_all()?;

        assert_eq!(array.len(), 2);

        // [read]

        std::fs::remove_file("example.vortex")?;

        Ok(())
    }

    #[tokio::test]
    async fn compact_read_write() -> VortexResult<()> {
        // [compact write]
        let array = PrimitiveArray::new(buffer![0u64, 1, 2, 3, 4], Validity::NonNullable);

        VortexWriteOptions::default()
            .with_strategy(
                WriteStrategyBuilder::new()
                    .with_compressor(CompactCompressor::default())
                    .build(),
            )
            .write(
                &mut tokio::fs::File::create("example_compact.vortex").await?,
                array.to_array_stream(),
            )
            .await?;

        // [compact read]
        let recovered_array = VortexOpenOptions::new()
            .open("example_compact.vortex")
            .await?
            .scan()?
            .into_array_iter()?
            .read_all()?;

        assert_eq!(recovered_array.len(), array.len());
        let recovered_primitive = recovered_array.to_primitive();
        assert_eq!(recovered_primitive.validity(), array.validity());
        assert_eq!(recovered_primitive.buffer::<u64>(), array.buffer::<u64>());

        std::fs::remove_file("example_compact.vortex")?;

        Ok(())
    }
}
