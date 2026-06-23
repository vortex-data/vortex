// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// https://github.com/rust-lang/cargo/pull/11645#issuecomment-1536905941
#![doc = include_str!(concat!("../", env!("CARGO_PKG_README")))]
//! # Rust API Map
//!
//! The `vortex` crate is the batteries-included entry point. It re-exports the core crates under
//! stable module names so Rust users can start here and drill down into the crate that owns a
//! concept:
//!
//! - [`array`](mod@array) contains [`ArrayRef`](array::ArrayRef), canonical arrays, expressions, scalar
//!   functions, statistics, and Arrow conversion.
//! - [`dtype`] and [`scalar`] contain the logical type system and single-value representation.
//! - [`file`](mod@file) contains Vortex file readers and writers when the `files` feature is enabled.
//! - [`layout`] contains serialized layout trees, layout readers, scan builders, and segment
//!   sources/sinks.
//! - [`compressor`] contains the default BtrBlocks-style adaptive compressor.
//! - [`encodings`] exposes maintained encoding crates such as FastLanes, Pco, and Zstd.
//! - [`session`] contains [`VortexSession`](session::VortexSession), the registry container used to
//!   make array, layout, scalar-function, and runtime plugins available.
//!
//! Most applications should create a default session with [`VortexSessionDefault`]. Lower-level
//! crates expose narrower session builders when you are embedding only part of Vortex.
//!
//! ```rust
//! use vortex::VortexSessionDefault;
//! use vortex::session::VortexSession;
//!
//! let session = VortexSession::default();
//! ```
//!
//! ## Arrays and Compression
//!
//! Arrays are logical values plus a physical encoding. The default compressor chooses an encoding
//! tree that preserves the array's [`DType`](dtype::DType) while often reducing bytes in memory or
//! on disk.
//!
//! ```rust
//! use vortex::VortexSessionDefault;
//! use vortex::array::{IntoArray, VortexSessionExecute};
//! use vortex::array::arrays::PrimitiveArray;
//! use vortex::buffer::buffer;
//! use vortex::compressor::BtrBlocksCompressor;
//! use vortex::session::VortexSession;
//! use vortex::array::validity::Validity;
//!
//! # fn example() -> vortex::error::VortexResult<()> {
//! let session = VortexSession::default();
//! let array = PrimitiveArray::new(buffer![42u64; 1024], Validity::NonNullable).into_array();
//! let compressed = BtrBlocksCompressor::default()
//!     .compress(&array, &mut session.create_execution_ctx())?;
//!
//! assert_eq!(compressed.dtype(), array.dtype());
//! assert_eq!(compressed.len(), array.len());
//! # Ok(())
//! # }
//! ```
//!
//! ## Files and Scans
//!
//! Vortex files store a layout tree plus segment bytes. Read and write APIs hang off the session
//! via extension traits, and scans use expressions for projection and filtering.
//!
//! ```rust,no_run
//! use vortex::VortexSessionDefault;
//! use vortex::array::{IntoArray, stream::ArrayStreamExt};
//! use vortex::array::arrays::PrimitiveArray;
//! use vortex::array::expr::{gt, lit, root};
//! use vortex::array::validity::Validity;
//! use vortex::buffer::{ByteBufferMut, buffer};
//! use vortex::file::{OpenOptionsSessionExt, WriteOptionsSessionExt};
//! use vortex::session::VortexSession;
//!
//! # async fn example() -> vortex::error::VortexResult<()> {
//! let session = VortexSession::default();
//! let array = PrimitiveArray::new(buffer![0u64, 1, 2, 3, 4], Validity::NonNullable);
//!
//! let mut bytes = ByteBufferMut::empty();
//! session
//!     .write_options()
//!     .write(&mut bytes, array.into_array().to_array_stream())
//!     .await?;
//!
//! let filtered = session
//!     .open_options()
//!     .open_buffer(bytes)?
//!     .scan()?
//!     .with_filter(gt(root(), lit(2u64)))
//!     .into_array_stream()?
//!     .read_all()
//!     .await?;
//!
//! assert_eq!(filtered.len(), 2);
//! # Ok(())
//! # }
//! ```

// vortex::compute is deprecated and will be ported over to expressions.
pub use vortex_array::aggregate_fn;
use vortex_array::aggregate_fn::session::AggregateFnSession;
use vortex_array::arrow::ArrowSession;
pub use vortex_array::compute;
use vortex_array::dtype::session::DTypeSession;
// vortex::expr is in the process of having its dependencies inverted, and will eventually be
// pulled back out into a vortex_expr crate.
pub use vortex_array::expr;
use vortex_array::memory::MemorySession;
use vortex_array::optimizer::kernels::KernelSession;
pub use vortex_array::scalar_fn;
use vortex_array::scalar_fn::session::ScalarFnSession;
use vortex_array::session::ArraySession;
use vortex_array::stats::session::StatsSession;
use vortex_io::session::RuntimeSession;
use vortex_layout::session::LayoutSession;
use vortex_session::VortexSession;

// We re-export like so in order to allow users to search inside subcrates when using the Rust docs.

/// Core array APIs, canonical arrays, expressions, scalar functions, statistics, and Arrow
/// conversion.
pub mod array {
    pub use vortex_array::*;

    // TODO(connor): We should probably manually pull up everything we need besides these 3 modules.
    // Note that there `vortex::dtype`, `vortex::extension`, and `vortex::scalar` are all exported
    // twice.
}

/// Aligned buffers and byte buffers used by arrays, layouts, IPC, and file IO.
pub mod buffer {
    pub use vortex_buffer::*;
}

/// Default adaptive compression APIs based on the maintained BtrBlocks-style compressor.
pub mod compressor {
    pub use vortex_btrblocks::BtrBlocksCompressor;
    pub use vortex_btrblocks::BtrBlocksCompressorBuilder;
    pub use vortex_btrblocks::Scheme;
    pub use vortex_btrblocks::SchemeId;
}

/// Logical Vortex data types.
pub mod dtype {
    pub use vortex_array::dtype::*;
}

/// Error and result types shared across Vortex crates.
pub mod error {
    pub use vortex_error::*;
}

/// Built-in extension dtypes such as UUID and temporal types.
pub mod extension {
    pub use vortex_array::extension::*;
}

#[cfg(feature = "files")]
/// Vortex file readers, writers, open options, write options, and file-format footer APIs.
pub mod file {
    pub use vortex_file::*;
}

/// Generated flatbuffer bindings used by Vortex serialization.
pub mod flatbuffers {
    pub use vortex_flatbuffers::*;
}

/// Async and blocking IO abstractions used by file readers and writers.
pub mod io {
    pub use vortex_io::*;
}

/// IPC serialization helpers for Vortex arrays.
pub mod ipc {
    pub use vortex_ipc::*;
}

/// Serialized layout trees, layout readers, scan builders, and segment sources/sinks.
pub mod layout {
    pub use vortex_layout::*;
}

/// Selection masks used by array and scan operations.
pub mod mask {
    pub use vortex_mask::*;
}

/// Metrics traits and default metrics registry implementations.
pub mod metrics {
    pub use vortex_metrics::*;
}

/// Generated protocol buffer bindings used by Vortex metadata.
pub mod proto {
    pub use vortex_proto::*;
}

/// Scalar values and typed scalar views.
pub mod scalar {
    pub use vortex_array::scalar::*;
}

/// Data-source abstractions for scan integrations.
pub mod scan {
    pub use vortex_scan::*;
}

/// Session registries and session extension traits.
pub mod session {
    pub use vortex_session::*;
}

/// Small utility types used across Vortex crates.
pub mod utils {
    pub use vortex_utils::*;
}

/// Maintained array encoding crates.
pub mod encodings {
    /// Adaptive Lossless floating-point encodings.
    pub mod alp {
        pub use vortex_alp::*;
    }

    /// Byte-per-value boolean encoding.
    pub mod bytebool {
        pub use vortex_bytebool::*;
    }

    /// Date/time decomposition encodings.
    pub mod datetime_parts {
        pub use vortex_datetime_parts::*;
    }

    /// Decimal byte-part decomposition encodings.
    pub mod decimal_byte_parts {
        pub use vortex_decimal_byte_parts::*;
    }

    /// FastLanes integer encodings: bit-packing, delta, frame-of-reference, and RLE.
    pub mod fastlanes {
        pub use vortex_fastlanes::*;
    }

    /// Fast Static Symbol Table string encoding.
    pub mod fsst {
        pub use vortex_fsst::*;
    }

    /// Pco numeric compression encoding.
    pub mod pco {
        pub use vortex_pco::*;
    }

    /// Arrow-compatible run-end encoding.
    pub mod runend {
        pub use vortex_runend::*;
    }

    /// Fixed-step sequence encoding.
    pub mod sequence {
        pub use vortex_sequence::*;
    }

    /// Sparse fill-value-plus-patches encoding.
    pub mod sparse {
        pub use vortex_sparse::*;
    }

    /// Zig-zag integer transform encoding.
    pub mod zigzag {
        pub use vortex_zigzag::*;
    }

    #[cfg(feature = "zstd")]
    /// Zstd-backed binary/string compression encodings.
    pub mod zstd {
        pub use vortex_zstd::*;
    }
}

/// Extension trait to create a default Vortex session.
pub trait VortexSessionDefault {
    /// Creates a default Vortex session with standard arrays, layouts, scalar functions,
    /// optimizer kernels, expressions, aggregate functions, and runtime support.
    fn default() -> VortexSession;
}

impl VortexSessionDefault for VortexSession {
    fn default() -> VortexSession {
        let session = VortexSession::empty()
            .with::<DTypeSession>()
            .with::<ArraySession>()
            .with::<KernelSession>()
            .with::<LayoutSession>()
            .with::<ScalarFnSession>()
            .with::<StatsSession>()
            .with::<AggregateFnSession>()
            .with::<ArrowSession>()
            .with::<MemorySession>()
            .with::<RuntimeSession>();

        #[cfg(feature = "files")]
        let session = {
            let session = session.with::<file::multi::MultiFileSession>();
            file::register_default_encodings(&session);
            session
        };

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
    use vortex_array::VortexSessionExecute;
    use vortex_array::array_session;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::StructArray;
    use vortex_array::dtype::FieldNames;
    use vortex_array::expr::gt;
    use vortex_array::expr::lit;
    use vortex_array::expr::root;
    use vortex_array::expr::select;
    use vortex_array::stream::ArrayStreamExt;
    use vortex_array::validity::Validity;
    use vortex_btrblocks::BtrBlocksCompressorBuilder;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_file::OpenOptionsSessionExt;
    use vortex_file::WriteOptionsSessionExt;
    use vortex_file::WriteStrategyBuilder;
    use vortex_session::VortexSession;

    use crate as vortex;
    use crate::VortexSessionDefault;

    #[test]
    fn convert() -> anyhow::Result<()> {
        // [convert]
        use std::fs::File;

        use arrow_array::RecordBatchReader;
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
        use vortex::array::arrays::ChunkedArray;
        use vortex::array::arrow::FromArrowArray;
        use vortex::dtype::DType;
        use vortex::dtype::arrow::FromArrowType;

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

        let array = PrimitiveArray::new(buffer![42u64; 100_000], Validity::NonNullable);

        // You can compress an array in-memory with the BtrBlocks compressor
        let session = VortexSession::default();
        let compressed = BtrBlocksCompressor::default().compress(
            &array.clone().into_array(),
            &mut session.create_execution_ctx(),
        )?;
        println!(
            "BtrBlocks size: {} / {}",
            compressed.nbytes(),
            array.into_array().nbytes()
        );
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
                array.into_array().to_array_stream(),
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
                    .with_btrblocks_builder(BtrBlocksCompressorBuilder::default().with_compact())
                    .build(),
            )
            .write(
                &mut tokio::fs::File::create(&path).await?,
                array.clone().into_array().to_array_stream(),
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

        let mut ctx = array_session().create_execution_ctx();

        let recovered_primitive = recovered_array.execute::<PrimitiveArray>(&mut ctx)?;
        assert!(recovered_primitive.validity()?.mask_eq(
            &array.validity()?,
            array.len(),
            &mut ctx
        )?);
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
                array.into_array().to_array_stream(),
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
