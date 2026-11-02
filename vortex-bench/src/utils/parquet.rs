// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Shared Parquet writer settings for generated benchmark data.

use parquet::basic::Compression;
use parquet::basic::ZstdLevel;
use parquet::file::properties::WriterProperties;
use parquet::file::properties::WriterPropertiesBuilder;

/// Writer settings for generated benchmark Parquet: zstd level 3, with the parquet-rs default
/// page and dictionary limits.
///
/// The default 20,000-row page limit keeps many pages per column chunk, which DataFusion needs for
/// page-level pruning. Larger pages shrink the files but slow down selective queries.
///
/// Row group size is left to the caller.
pub fn parquet_writer_properties() -> WriterPropertiesBuilder {
    WriterProperties::builder().set_compression(Compression::ZSTD(
        ZstdLevel::try_new(3).unwrap_or_else(|_| unreachable!("3 is a valid zstd level")),
    ))
}
