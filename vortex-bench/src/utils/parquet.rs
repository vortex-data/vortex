// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Shared Parquet writer settings for generated benchmark data.

use parquet::basic::Compression;
use parquet::basic::ZstdLevel;
use parquet::file::properties::WriterProperties;
use parquet::file::properties::WriterPropertiesBuilder;

/// Columns whose dictionary outgrows this fall back to plain encoding. The parquet-rs default
/// (1 MiB) keeps dictionaries for unique keys, which compress far worse than plain values. A small
/// limit approximates the per-column choice that DuckDB's writer makes.
const DICTIONARY_PAGE_SIZE_LIMIT: usize = 64 * 1024;

/// Writer settings for generated benchmark Parquet: zstd level 3, pages bounded by bytes only,
/// and small dictionaries.
///
/// Row group size is left to the caller.
pub fn parquet_writer_properties() -> WriterPropertiesBuilder {
    WriterProperties::builder()
        .set_compression(Compression::ZSTD(
            ZstdLevel::try_new(3).unwrap_or_else(|_| unreachable!("3 is a valid zstd level")),
        ))
        .set_dictionary_page_size_limit(DICTIONARY_PAGE_SIZE_LIMIT)
        // The default 20,000-row page limit splits row groups into many small pages, which gives
        // zstd too little context to compress repeating columns.
        .set_data_page_row_count_limit(usize::MAX)
}
