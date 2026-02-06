// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// https://github.com/rust-lang/cargo/pull/11645#issuecomment-1536905941
#![doc = include_str!(concat!("../", env!("CARGO_PKG_README")))]

// vortex::compute is deprecated and will be ported over to expressions.
pub use vortex_array::compute;
// vortex::expr is in the process of having its dependencies inverted, and will eventually be
// pulled back out into a vortex_expr crate.
pub use vortex_array::expr;
use vortex_array::expr::session::ExprSession;
use vortex_array::session::ArraySession;
use vortex_dtype::session::DTypeSession;
use vortex_io::session::RuntimeSession;
use vortex_layout::session::LayoutSession;
use vortex_session::VortexSession;

// We re-export like so in order to allow users to search inside subcrates when using the Rust docs.

pub mod array {
    pub use vortex_array::*;
}

pub mod buffer {
    pub use vortex_buffer::*;
}

pub mod compute2 {
    pub use vortex_compute::*;
}

pub mod compressor {
    pub use vortex_btrblocks::BtrBlocksCompressor;
    #[cfg(feature = "zstd")]
    pub use vortex_layout::layouts::compact::CompactCompressor;
}

pub mod dtype {
    pub use vortex_dtype::*;
}

pub mod error {
    pub use vortex_error::*;
}

#[cfg(feature = "files")]
pub mod file {
    pub use vortex_file::*;
}

pub mod flatbuffers {
    pub use vortex_flatbuffers::*;
}

pub mod io {
    pub use vortex_io::*;
}

pub mod ipc {
    pub use vortex_ipc::*;
}

pub mod layout {
    pub use vortex_layout::*;
}

pub mod mask {
    pub use vortex_mask::*;
}

pub mod metrics {
    pub use vortex_metrics::*;
}

pub mod proto {
    pub use vortex_proto::*;
}

pub mod scalar {
    pub use vortex_scalar::*;
}

pub mod scan {
    pub use vortex_scan::*;
}

pub mod session {
    pub use vortex_session::*;
}

pub mod utils {
    pub use vortex_utils::*;
}

pub mod encodings {
    pub mod alp {
        pub use vortex_alp::*;
    }

    pub mod bytebool {
        pub use vortex_bytebool::*;
    }

    pub mod datetime_parts {
        pub use vortex_datetime_parts::*;
    }

    pub mod decimal_byte_parts {
        pub use vortex_decimal_byte_parts::*;
    }

    pub mod fastlanes {
        pub use vortex_fastlanes::*;
    }

    pub mod fsst {
        pub use vortex_fsst::*;
    }

    pub mod pco {
        pub use vortex_pco::*;
    }

    pub mod runend {
        pub use vortex_runend::*;
    }

    pub mod sequence {
        pub use vortex_sequence::*;
    }

    pub mod sparse {
        pub use vortex_sparse::*;
    }

    pub mod zigzag {
        pub use vortex_zigzag::*;
    }

    #[cfg(feature = "zstd")]
    pub mod zstd {
        pub use vortex_zstd::*;
    }
}

/// Extension trait to create a default Vortex session.
pub trait VortexSessionDefault {
    /// Creates a default Vortex session with the standard arrays, layouts, and expressions.
    fn default() -> VortexSession;
}

impl VortexSessionDefault for VortexSession {
    #[allow(unused_mut)]
    fn default() -> VortexSession {
        let mut session = VortexSession::empty()
            .with::<DTypeSession>()
            .with::<ArraySession>()
            .with::<LayoutSession>()
            .with::<ExprSession>()
            .with::<RuntimeSession>();

        #[cfg(all(feature = "cuda", target_os = "linux"))]
        // Even if the CUDA feature is enabled we need to check at
        // runtime whether CUDA is available in the current environment.
        if vortex_cuda::cuda_available() {
            use vortex_cuda::CudaSessionExt;
            session = session.with::<vortex_cuda::CudaSession>();
            vortex_cuda::initialize_cuda(&session.cuda_session());
        }

        #[cfg(feature = "files")]
        file::register_default_encodings(&mut session);

        session
    }
}

/// These tests are included in the getting started documentation, so be mindful of which imports
/// to keep inside the test functions, and which to just use from the outer scope. The examples
/// get too verbose if we include _everything_.
#[cfg(test)]
mod test {
    use std::path::PathBuf;

    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::ToCanonical;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::StructArray;
    use vortex_array::expr::gt;
    use vortex_array::expr::lit;
    use vortex_array::expr::root;
    use vortex_array::expr::select;
    use vortex_array::stream::ArrayStreamExt;
    use vortex_array::validity::Validity;
    use vortex_array::vtable::ValidityHelper;
    use vortex_buffer::buffer;
    use vortex_dtype::FieldNames;
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
        let chunks: Vec<_> = reader
            .map(|record_batch| {
                let batch = record_batch?;
                ArrayRef::from_arrow(batch, false)
            })
            .collect::<VortexResult<_>>()?;
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
            .with_zstd_values_per_page(8192)
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
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("example.vortex");

        session
            .write_options()
            .write(
                &mut tokio::fs::File::create(&path).await?,
                array.to_array_stream(),
            )
            .await?;

        // [write]

        // [read]
        let array = session
            .open_options()
            .open_path(path.clone())
            .await?
            .scan()?
            .with_filter(gt(root(), lit(2u64)))
            .into_array_stream()?
            .read_all()
            .await?;

        assert_eq!(array.len(), 2);

        // [read]

        std::fs::remove_file(&path)?;

        Ok(())
    }

    #[tokio::test]
    async fn compact_read_write() -> VortexResult<()> {
        let session = VortexSession::default();

        // [compact write]
        let array = PrimitiveArray::new(buffer![0u64, 1, 2, 3, 4], Validity::NonNullable);

        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("example_compact.vortex");

        session
            .write_options()
            .with_strategy(
                WriteStrategyBuilder::default()
                    .with_compressor(CompactCompressor::default())
                    .build(),
            )
            .write(
                &mut tokio::fs::File::create(&path).await?,
                array.to_array_stream(),
            )
            .await?;

        // [compact read]
        let recovered_array = session
            .open_options()
            .open_path(path.clone())
            .await?
            .scan()?
            .into_array_stream()?
            .read_all()
            .await?;

        assert_eq!(recovered_array.len(), array.len());
        let recovered_primitive = recovered_array.to_primitive();
        assert_eq!(recovered_primitive.validity(), array.validity());
        assert_eq!(
            recovered_primitive.to_buffer::<u64>(),
            array.to_buffer::<u64>()
        );

        std::fs::remove_file(&path)?;

        Ok(())
    }

    #[tokio::test]
    async fn projection_read_write() -> VortexResult<()> {
        let session = VortexSession::default();

        // Build a simple two-column struct array: { id: u64, value: u64 }
        let ids = PrimitiveArray::new(buffer![1u64, 2, 3, 4, 5], Validity::NonNullable);
        let values = PrimitiveArray::new(buffer![10u64, 20, 30, 40, 50], Validity::NonNullable);

        let array = StructArray::try_new(
            FieldNames::from(["id", "value"]),
            vec![ids.into_array(), values.into_array()],
            5,
            Validity::NonNullable,
        )?
        .into_array();

        // Write a Vortex file containing both columns.
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("example_projection.vortex");

        session
            .write_options()
            .write(
                &mut tokio::fs::File::create(&path).await?,
                array.to_array_stream(),
            )
            .await?;

        // Read the file back, but project down to just the "value" column.
        let projected = session
            .open_options()
            .open_path(path.clone())
            .await?
            .scan()?
            .with_projection(select(["value"], root()))
            .into_array_stream()?
            .read_all()
            .await?;

        // Projection keeps the same number of rows but only one column.
        assert_eq!(projected.len(), 5);

        std::fs::remove_file(&path)?;

        Ok(())
    }
}
