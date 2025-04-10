// https://github.com/rust-lang/cargo/pull/11645#issuecomment-1536905941
#![doc = include_str!(concat!("../", env!("CARGO_PKG_README")))]

pub use vortex_array::*;
#[cfg(feature = "files")]
pub use vortex_file as file;
#[cfg(feature = "files")]
pub use vortex_io as io;
pub use {
    vortex_btrblocks as compressor, vortex_buffer as buffer, vortex_dtype as dtype,
    vortex_error as error, vortex_expr as expr, vortex_flatbuffers as flatbuffers,
    vortex_ipc as ipc, vortex_layout as layout, vortex_mask as mask, vortex_proto as proto,
    vortex_scalar as scalar,
};

pub mod encodings {
    pub use {
        vortex_alp as alp, vortex_bytebool as bytebool, vortex_datetime_parts as datetime_parts,
        vortex_dict as dict, vortex_fastlanes as fastlanes, vortex_fsst as fsst,
        vortex_runend as runend, vortex_sparse as sparse, vortex_zigzag as zigzag,
    };
}

/// These tests are included in the getting started documentation, so be mindful of which imports
/// to keep inside the test functions, and which to just use from the outer scope. The examples
/// get too verbose if we include _everything_.
#[cfg(test)]
mod test {
    use itertools::Itertools;
    use vortex_array::TryIntoArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::stream::{ArrayStreamArrayExt, ArrayStreamExt};
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_expr::{gt, ident, lit};
    use vortex_file::{VortexOpenOptions, VortexWriteOptions};

    use crate as vortex;

    #[test]
    fn convert() -> VortexResult<()> {
        // [convert]
        use std::fs::File;

        use arrow::array::RecordBatchReader;
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
        use vortex::Array;
        use vortex::arrays::ChunkedArray;
        use vortex::dtype::DType;
        use vortex::dtype::arrow::FromArrowType;

        let reader = ParquetRecordBatchReaderBuilder::try_new(File::open(
            "../docs/_static/example.parquet",
        )?)?
        .build()?;

        let dtype = DType::from_arrow(reader.schema());
        let chunks = reader
            .map(|record_batch| record_batch?.try_into_array())
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
        use vortex::nbytes::NBytes;

        let array = PrimitiveArray::new(buffer![42u64; 100_000], Validity::NonNullable);

        let compressed = BtrBlocksCompressor.compress(&array)?;
        println!("{} / {}", compressed.nbytes(), array.nbytes());
        // [compress]

        Ok(())
    }

    #[tokio::test]
    async fn read_write() -> VortexResult<()> {
        // [write]
        use vortex::io::TokioFile;

        let array = PrimitiveArray::new(buffer![0u64, 1, 2, 3, 4], Validity::NonNullable);

        // Write a Vortex file with the default compression and layout strategy.
        VortexWriteOptions::default()
            .write(
                tokio::fs::File::create("example.vortex").await?,
                array.to_array_stream(),
            )
            .await?;

        // [write]

        // [read]
        let array = VortexOpenOptions::file()
            .open(TokioFile::open("example.vortex")?)
            .await?
            .scan()?
            .with_filter(gt(ident(), lit(2u64)))
            .into_array_stream()?
            .read_all()
            .await?;

        assert_eq!(array.len(), 2);

        // [read]

        std::fs::remove_file("example.vortex")?;

        Ok(())
    }
}
