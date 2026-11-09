// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Splitting generated benchmark tables into files of similar size.
//!
//! A split table is written as `{table}_{part:04}.{ext}` with 1-based parts. The split is decided
//! once, on the Parquet files, and every other format is converted from them file by file, so all
//! formats cover identical row ranges per file.

use glob::Pattern;

/// Target compressed size of each generated benchmark file.
pub const TARGET_FILE_COMPRESSED_BYTES: u64 = 200 * 1024 * 1024;

/// Number of files to split a table of `compressed_bytes` into, so each is close to
/// [`TARGET_FILE_COMPRESSED_BYTES`].
pub fn num_files(compressed_bytes: f64) -> usize {
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        clippy::cast_sign_loss,
        reason = "estimated table sizes are far below the f64 and usize limits"
    )]
    let files = (compressed_bytes / TARGET_FILE_COMPRESSED_BYTES as f64).round() as usize;
    files.max(1)
}

/// File name of the 1-based `part` of a split table.
pub fn part_file_name(table: &str, part: usize, ext: &str) -> String {
    format!("{table}_{part:04}.{ext}")
}

/// Glob matching every part of a split table, and no other table sharing its prefix (such as
/// `customer_address` for `customer`).
#[expect(clippy::expect_used)]
pub fn part_pattern(table: &str, ext: &str) -> Pattern {
    format!("{table}_[0-9][0-9][0-9][0-9].{ext}")
        .parse()
        .expect("valid glob pattern")
}
