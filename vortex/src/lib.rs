// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// https://github.com/rust-lang/cargo/pull/11645#issuecomment-1536905941
#![doc = include_str!(concat!("../", env!("CARGO_PKG_README")))]

pub use vortex_array as array;
use vortex_array::ArraySession;
// vortex::compute is deprecated and will be ported over to expressions.
pub use vortex_array::compute;
// vortex::expr is in the process of having its dependencies inverted, and will eventually be
// pulled back out into a vortex_expr crate.
pub use vortex_array::expr;
use vortex_array::expr::session::ExprSession;
pub use vortex_buffer as buffer;
pub use vortex_dtype as dtype;
pub use vortex_error as error;
#[cfg(feature = "files")]
pub use vortex_file as file;
pub use vortex_flatbuffers as flatbuffers;
pub use vortex_io as io;
use vortex_io::session::RuntimeSession;
pub use vortex_ipc as ipc;
pub use vortex_layout as layout;
use vortex_layout::session::LayoutSession;
pub use vortex_mask as mask;
pub use vortex_metrics as metrics;
use vortex_metrics::VortexMetrics;
pub use vortex_proto as proto;
pub use vortex_scalar as scalar;
pub use vortex_scan as scan;
pub use vortex_session as session;
use vortex_session::VortexSession;
pub use vortex_utils as utils;

pub mod compressor {
    pub use vortex_btrblocks::BtrBlocksCompressor;
    #[cfg(feature = "zstd")]
    pub use vortex_layout::layouts::compact::CompactCompressor;
}

pub mod encodings {
    pub use vortex_alp as alp;
    pub use vortex_bytebool as bytebool;
    pub use vortex_datetime_parts as datetime_parts;
    pub use vortex_decimal_byte_parts as decimal_byte_parts;
    pub use vortex_fastlanes as fastlanes;
    pub use vortex_fsst as fsst;
    pub use vortex_pco as pco;
    pub use vortex_runend as runend;
    pub use vortex_sequence as sequence;
    pub use vortex_sparse as sparse;
    pub use vortex_zigzag as zigzag;
    #[cfg(feature = "zstd")]
    pub use vortex_zstd as zstd;
}

/// Extension trait to create a default Vortex session.
pub trait VortexSessionDefault {
    /// Creates a default Vortex session with the standard arrays, layouts, and expressions.
    fn default() -> VortexSession;
}

impl VortexSessionDefault for VortexSession {
    fn default() -> VortexSession {
        let session = VortexSession::empty()
            .with::<VortexMetrics>()
            .with::<ArraySession>()
            .with::<LayoutSession>()
            .with::<ExprSession>()
            .with::<RuntimeSession>();

        #[cfg(feature = "files")]
        file::register_default_encodings(&session);

        session
    }
}

/// These tests are included in the getting started documentation, so be mindful of which imports
/// to keep inside the test functions, and which to just use from the outer scope. The examples
/// get too verbose if we include _everything_.
#[cfg(test)]
mod test {
    use itertools::Itertools;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::ToCanonical;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::expr::gt;
    use vortex_array::expr::lit;
    use vortex_array::expr::root;
    use vortex_array::stream::ArrayStreamExt;
    use vortex_array::validity::Validity;
    use vortex_array::vtable::ValidityHelper;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_file::OpenOptionsSessionExt;
    use vortex_file::WriteOptionsSessionExt;
    use vortex_file::WriteStrategyBuilder;
    use vortex_layout::layouts::compact::CompactCompressor;
    use vortex_session::VortexSession;

    use crate as vortex;
    use crate::VortexSessionDefault;

    #[test]
    fn convert() -> anyhow::Result<()> {
        // [convert]
        use std::fs::File;

        use arrow_array::RecordBatchReader;
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
        use vortex::array::Array;
        use vortex::array::arrays::ChunkedArray;
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
        use vortex::compressor::BtrBlocksCompressor;
        use vortex::compressor::CompactCompressor;

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
        let session = VortexSession::default();

        // [write]
        let array = PrimitiveArray::new(buffer![0u64, 1, 2, 3, 4], Validity::NonNullable);

        // Write a Vortex file with the default compression and layout strategy.
        session
            .write_options()
            .write(
                &mut tokio::fs::File::create("example.vortex").await?,
                array.to_array_stream(),
            )
            .await?;

        // [write]

        // [read]
        let array = session
            .open_options()
            .open("example.vortex")
            .await?
            .scan()?
            .with_filter(gt(root(), lit(2u64)))
            .into_array_stream()?
            .read_all()
            .await?;

        assert_eq!(array.len(), 2);

        // [read]

        std::fs::remove_file("example.vortex")?;

        Ok(())
    }

    #[tokio::test]
    async fn compact_read_write() -> VortexResult<()> {
        let session = VortexSession::default();

        // [compact write]
        let array = PrimitiveArray::new(buffer![0u64, 1, 2, 3, 4], Validity::NonNullable);

        session
            .write_options()
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
        let recovered_array = session
            .open_options()
            .open("example_compact.vortex")
            .await?
            .scan()?
            .into_array_stream()?
            .read_all()
            .await?;

        assert_eq!(recovered_array.len(), array.len());
        let recovered_primitive = recovered_array.to_primitive();
        assert_eq!(recovered_primitive.validity(), array.validity());
        assert_eq!(recovered_primitive.buffer::<u64>(), array.buffer::<u64>());

        std::fs::remove_file("example_compact.vortex")?;

        Ok(())
    }
}
