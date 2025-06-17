use std::sync::Arc;

use arcref::ArcRef;
use futures::{FutureExt as _, StreamExt as _};
use vortex_array::stats::Stat;
use vortex_array::{Array, ArrayContext, ArrayRef, Canonical, IntoArray};
use vortex_dtype::PType;
use vortex_error::VortexResult;
use vortex_pco::PcoArray;
use vortex_zstd::ZstdArray;

use crate::scan::{TaskExecutor, TaskExecutorExt as _};
use crate::segments::SequenceWriter;
use crate::{
    LayoutStrategy, SendableLayoutWriter, SendableSequentialStream, SequentialStreamAdapter,
    SequentialStreamExt as _,
};

fn is_pco_number_type(ptype: PType) -> bool {
    matches!(
        ptype,
        PType::F16
            | PType::F32
            | PType::F64
            | PType::I16
            | PType::I32
            | PType::I64
            | PType::U16
            | PType::U32
            | PType::U64
    )
}

/// A simple compressor that uses the "compact" strategy:
/// - Pco for supported numeric types (16, 32, and 64-bit floats and ints)
/// - Zstd for everything else (primitive arrays only)
#[derive(Debug, Clone)]
pub struct CompactCompressor {
    pco_level: usize,
    zstd_level: i32,
    values_per_page: usize,
}

impl CompactCompressor {
    pub fn with_pco_level(mut self, level: usize) -> Self {
        self.pco_level = level;
        self
    }

    pub fn with_zstd_level(mut self, level: i32) -> Self {
        self.zstd_level = level;
        self
    }

    /// Sets the number of non-null primitive values to store per
    /// separately-decompressible page/frame.
    ///
    /// Fewer values per page can reduce the time to query a small slice of rows, but too
    /// few can increase compressed size and (de)compression time. The default is 0, which
    /// is used for maximally-large pages.
    pub fn with_values_per_page(mut self, values_per_page: usize) -> Self {
        self.values_per_page = values_per_page;
        self
    }

    pub fn compress(&self, array: &dyn Array) -> VortexResult<ArrayRef> {
        self.compress_canonical(array.to_canonical()?)
    }

    /// Compress a single array using the compact strategy
    pub fn compress_canonical(&self, canonical: Canonical) -> VortexResult<ArrayRef> {
        match canonical {
            Canonical::Primitive(primitive) => {
                let ptype = primitive.ptype();

                if is_pco_number_type(ptype) {
                    let pco_array =
                        PcoArray::from_primitive(&primitive, self.pco_level, self.values_per_page)?;
                    Ok(pco_array.into_array())
                } else {
                    let zstd_array = ZstdArray::from_primitive(
                        &primitive,
                        self.zstd_level,
                        self.values_per_page,
                    )?;
                    Ok(zstd_array.into_array())
                }
            }
            // For non-primitive arrays, return as-is for now.
            _ => Ok(array.to_array()),
        }
    }
}

impl Default for CompactCompressor {
    fn default() -> Self {
        Self {
            pco_level: pco::DEFAULT_COMPRESSION_LEVEL,
            zstd_level: 3,
            values_per_page: 0, // don't break up pages for faster access
        }
    }
}

/// A layout writer that compresses chunks using the "compact" strategy:
/// - pco for supported numeric types (16, 32, and 64-bit floats and ints)
/// - zstd for everything else (primitive arrays only)
pub struct CompactCompressedStrategy {
    child: ArcRef<dyn LayoutStrategy>,
    executor: Arc<dyn TaskExecutor>,
    parallelism: usize,
    compressor: CompactCompressor,
}

impl CompactCompressedStrategy {
    pub fn new(
        child: ArcRef<dyn LayoutStrategy>,
        executor: Arc<dyn TaskExecutor>,
        parallelism: usize,
    ) -> Self {
        Self {
            child,
            executor,
            parallelism,
            compressor: CompactCompressor::default(),
        }
    }

    pub fn with_compressor(
        child: ArcRef<dyn LayoutStrategy>,
        executor: Arc<dyn TaskExecutor>,
        parallelism: usize,
        compressor: CompactCompressor,
    ) -> Self {
        Self {
            child,
            executor,
            parallelism,
            compressor,
        }
    }
}

impl LayoutStrategy for CompactCompressedStrategy {
    fn write_stream(
        &self,
        ctx: &ArrayContext,
        sequence_writer: SequenceWriter,
        stream: SendableSequentialStream,
    ) -> SendableLayoutWriter {
        let executor = self.executor.clone();

        let dtype = stream.dtype().clone();
        let stream = stream
            .map(move |chunk| {
                let compressor = self.compressor.clone();
                async move {
                    let (sequence_id, chunk) = chunk?;
                    // Compute the stats for the chunk prior to compression
                    chunk
                        .statistics()
                        .compute_all(&Stat::all().collect::<Vec<_>>())?;
                    Ok((sequence_id, compressor.compress(&chunk)?))
                }
                .boxed()
            })
            .map(move |compress_future| executor.spawn(compress_future))
            .buffered(self.parallelism);

        self.child.write_stream(
            ctx,
            sequence_writer,
            SequentialStreamAdapter::new(dtype, stream).sendable(),
        )
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;

    use super::*;

    #[test]
    fn test_compact_compressor_pco_types() {
        let compressor = CompactCompressor::default();

        // Test pco-supported types
        let f64_array =
            PrimitiveArray::new(buffer![1.0f64, 2.0, 3.0, 4.0, 5.0], Validity::NonNullable);
        let compressed = compressor.compress(f64_array.as_ref()).unwrap();

        // Verify we can decompress back to original
        let decompressed = compressed.to_canonical().unwrap();
        assert_eq!(decompressed.len(), 5);

        // Test i32 (pco-supported)
        let i32_array = PrimitiveArray::new(buffer![1i32, 2, 3, 4, 5], Validity::NonNullable);
        let compressed = compressor.compress(i32_array.as_ref()).unwrap();
        let decompressed = compressed.to_canonical().unwrap();
        assert_eq!(decompressed.len(), 5);

        // Test u64 (pco-supported)
        let u64_array = PrimitiveArray::new(buffer![1u64, 2, 3, 4, 5], Validity::NonNullable);
        let compressed = compressor.compress(u64_array.as_ref()).unwrap();
        let decompressed = compressed.to_canonical().unwrap();
        assert_eq!(decompressed.len(), 5);
    }

    #[test]
    fn test_compact_compressor_zstd_types() {
        let compressor = CompactCompressor::default();

        // Test zstd-supported types (non-pco)
        let i8_array = PrimitiveArray::new(buffer![1i8, 2, 3, 4, 5], Validity::NonNullable);
        let compressed = compressor.compress(i8_array.as_ref()).unwrap();

        // Verify we can decompress back to original
        let decompressed = compressed.to_canonical().unwrap();
        assert_eq!(decompressed.len(), 5);

        // Test u8 (zstd)
        let u8_array = PrimitiveArray::new(buffer![1u8, 2, 3, 4, 5], Validity::NonNullable);
        let compressed = compressor.compress(u8_array.as_ref()).unwrap();
        let decompressed = compressed.to_canonical().unwrap();
        assert_eq!(decompressed.len(), 5);
    }

    #[test]
    fn test_compact_compressor_custom_options() {
        let compressor = CompactCompressor::new()
            .with_pco_options(12, 1024)
            .with_zstd_options(6, 100);

        let f64_array = PrimitiveArray::new(buffer![1.0f64; 1000], Validity::NonNullable);
        let compressed = compressor.compress(f64_array.as_ref()).unwrap();
        let decompressed = compressed.to_canonical().unwrap();
        assert_eq!(decompressed.len(), 1000);
    }
}
