// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Parquet writer settings for the GPU benchmark.
//!
//! The GPU backend rewrites each dataset before reading it back with cuDF. The CPU suite's
//! settings are tuned for host reads, and reusing them would measure a device decoder on a file
//! laid out for a different one; these settings size pages and row groups for a GPU read
//! instead. Each choice and its reason is tabulated in the README.

use clap::ValueEnum;
use parquet::basic::Compression;
use parquet::basic::ZstdLevel;
use parquet::file::properties::DEFAULT_MAX_ROW_GROUP_ROW_COUNT;
use parquet::file::properties::EnabledStatistics;
use parquet::file::properties::WriterProperties;
use parquet::file::properties::WriterVersion;

/// Target size of a data page written for GPU reads.
///
/// Pages are the unit a GPU reader decompresses in parallel, so they need to be large enough
/// to amortize per-page setup and numerous enough to fill the device. ~1 MiB is the page size
/// cuDF's Parquet reader is tuned around.
pub const GPU_DATA_PAGE_SIZE: usize = 1024 * 1024;

/// Row cap per data page.
///
/// `parquet`'s default caps pages at 20k rows, which produces pages far below
/// [`GPU_DATA_PAGE_SIZE`] for narrow columns and leaves the device underfed.
const GPU_DATA_PAGE_ROW_COUNT_LIMIT: usize = 1_000_000;

/// Rows per physical partition in both GPU benchmark formats.
///
/// A Parquet row group and a Vortex chunk are the same thing for this comparison: the unit the
/// reader plans and dispatches over. Pinning both to one value is what makes the two numbers
/// comparable — otherwise Parquet reads ~1M-row row groups while Vortex inherits the Arrow
/// reader's ~8K-row batches, turning one launch into hundreds.
pub const GPU_ROW_GROUP_SIZE: usize = DEFAULT_MAX_ROW_GROUP_ROW_COUNT;

/// Parquet page codecs the GPU benchmark can write.
///
/// A deliberate two-codec subset, not a limit of Parquet or of cuDF — both handle more. These
/// two are the ones worth comparing: Snappy is the Parquet default and the fastest to decode on
/// the device, and Zstd is what the CPU Parquet benchmark writes, so picking it answers "what
/// does the GPU do with the file the CPU suite already measures?". Adding a codec here is a
/// two-line change if another becomes interesting.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, ValueEnum)]
pub enum GpuCodec {
    /// The Parquet default, and the codec with the highest device-side throughput.
    #[default]
    Snappy,
    /// Matches the codec used by the CPU Parquet benchmark, at lower device throughput.
    Zstd,
}

impl GpuCodec {
    /// The Parquet compression setting for this codec.
    pub fn to_parquet(self) -> Compression {
        match self {
            GpuCodec::Snappy => Compression::SNAPPY,
            GpuCodec::Zstd => Compression::ZSTD(ZstdLevel::default()),
        }
    }

    /// Short lowercase name, used in measurement labels.
    pub fn name(self) -> &'static str {
        match self {
            GpuCodec::Snappy => "snappy",
            GpuCodec::Zstd => "zstd",
        }
    }
}

/// Writer properties tuned for a GPU read.
pub fn gpu_writer_properties(codec: GpuCodec) -> WriterProperties {
    WriterProperties::builder()
        // V1 data pages compress the whole page body. V2 pages place uncompressed
        // repetition/definition levels ahead of the compressed values in the same body, which
        // not every GPU reader path handles.
        .set_writer_version(WriterVersion::PARQUET_1_0)
        .set_compression(codec.to_parquet())
        // Dictionary encoding keeps the decompressed payload small and is the encoding GPU
        // Parquet readers decode fastest.
        .set_dictionary_enabled(true)
        .set_data_page_size_limit(GPU_DATA_PAGE_SIZE)
        .set_data_page_row_count_limit(GPU_DATA_PAGE_ROW_COUNT_LIMIT)
        // Stated explicitly rather than left to the default, because the Vortex side is
        // rebatched to the same constant and the two have to move together.
        .set_max_row_group_row_count(Some(GPU_ROW_GROUP_SIZE))
        // Per-page statistics only inflate the page headers a reader has to walk.
        .set_statistics_enabled(EnabledStatistics::Chunk)
        .build()
}

#[cfg(test)]
mod tests {
    use parquet::file::properties::WriterVersion;

    use super::*;

    #[test]
    fn gpu_properties_use_v1_pages_and_the_requested_codec() {
        let properties = gpu_writer_properties(GpuCodec::Snappy);
        assert_eq!(properties.writer_version(), WriterVersion::PARQUET_1_0);
        assert_eq!(properties.compression(&"x".into()), Compression::SNAPPY);
        assert_eq!(properties.data_page_size_limit(), GPU_DATA_PAGE_SIZE);
    }
}
