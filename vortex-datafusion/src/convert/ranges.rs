// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use datafusion_datasource::FileRange;
use vortex::ArrayRef;
use vortex::scan::ScanBuilder;

/// If the file has a [`FileRange`](datafusion::datasource::listing::FileRange), we translate it into a row range in the file for the scan.
pub(crate) fn apply_byte_range(
    file_range: FileRange,
    total_size: u64,
    row_count: u64,
    scan_builder: ScanBuilder<ArrayRef>,
) -> ScanBuilder<ArrayRef> {
    let row_range = byte_range_to_row_range(
        file_range.start as u64..file_range.end as u64,
        row_count,
        total_size,
    );

    scan_builder.with_row_range(row_range)
}

fn byte_range_to_row_range(byte_range: Range<u64>, row_count: u64, total_size: u64) -> Range<u64> {
    let average_row = total_size / row_count;
    assert!(average_row > 0, "A row must always have at least one byte");

    let start_row = byte_range.start / average_row;
    let end_row = byte_range.end / average_row;

    // We take the min here as `end_row` might overshoot
    start_row..u64::min(row_count, end_row)
}

#[cfg(test)]
mod tests {
    use std::ops::Range;

    use itertools::Itertools;
    use rstest::rstest;

    use crate::convert::ranges::byte_range_to_row_range;

    #[rstest]
    #[case(0..100, 100, 100, 0..100)]
    #[case(0..105, 100, 105, 0..100)]
    #[case(0..50, 100, 105, 0..50)]
    #[case(50..105, 100, 105, 50..100)]
    #[case(0..1, 4, 8, 0..0)]
    #[case(1..8, 4, 8, 0..4)]
    fn test_range_translation(
        #[case] byte_range: Range<u64>,
        #[case] row_count: u64,
        #[case] total_size: u64,
        #[case] expected: Range<u64>,
    ) {
        assert_eq!(
            byte_range_to_row_range(byte_range, row_count, total_size),
            expected
        );
    }

    #[test]
    fn test_consecutive_ranges() {
        let row_count = 100;
        let total_size = 429;
        let bytes_a = 0..143;
        let bytes_b = 143..286;
        let bytes_c = 286..429;

        let rows_a = byte_range_to_row_range(bytes_a, row_count, total_size);
        let rows_b = byte_range_to_row_range(bytes_b, row_count, total_size);
        let rows_c = byte_range_to_row_range(bytes_c, row_count, total_size);

        assert_eq!(rows_a.end - rows_a.start, 35);
        assert_eq!(rows_b.end - rows_b.start, 36);
        assert_eq!(rows_c.end - rows_c.start, 29);

        assert_eq!(rows_a.start, 0);
        assert_eq!(rows_c.end, 100);
        for (left, right) in [rows_a, rows_b, rows_c].iter().tuple_windows() {
            assert_eq!(left.end, right.start);
        }
    }
}
