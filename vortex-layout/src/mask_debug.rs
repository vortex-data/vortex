// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Write as _;
use std::ops::Range;
use std::time::Duration;

use vortex_mask::AllOr;
use vortex_mask::Mask;

const MAX_SAMPLE_RANGES: usize = 8;
const FNV_OFFSET: u64 = 0xcbf29ce484222325;
const FNV_PRIME: u64 = 0x100000001b3;

#[derive(Debug)]
pub(crate) struct MaskCoordinateSummary {
    pub rows: usize,
    pub true_rows: usize,
    pub density: f64,
    pub first_row: Option<u64>,
    pub last_row: Option<u64>,
    pub coord_hash: u64,
    pub coord_sum: u64,
    pub coord_xor: u64,
    pub sample_ranges: String,
}

pub(crate) fn mask_coordinate_summary(
    mask: &Mask,
    row_range: &Range<u64>,
) -> MaskCoordinateSummary {
    debug_assert_eq!(
        mask.len() as u64,
        row_range.end - row_range.start,
        "mask coordinate summary requires a mask in the same coordinate space as row_range",
    );
    let first_row = mask.first().map(|idx| row_range.start + idx as u64);
    let last_row = mask.last().map(|idx| row_range.start + idx as u64);
    let (coord_hash, coord_sum, coord_xor, sample_ranges) = summarize_ranges(mask, row_range);

    MaskCoordinateSummary {
        rows: mask.len(),
        true_rows: mask.true_count(),
        density: mask.density(),
        first_row,
        last_row,
        coord_hash,
        coord_sum,
        coord_xor,
        sample_ranges,
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn log_mask_batch(
    message: &'static str,
    scan_label: Option<&str>,
    local_range: &Range<u64>,
    coord_range: &Range<u64>,
    mask: &Mask,
    elapsed: Option<Duration>,
    extra_rows: Option<usize>,
) {
    if !tracing::enabled!(tracing::Level::DEBUG) {
        return;
    }

    let coords = mask_coordinate_summary(mask, coord_range);
    tracing::debug!(
        scan_label = scan_label.unwrap_or(""),
        row_start = local_range.start,
        row_end = local_range.end,
        coord_start = coord_range.start,
        coord_end = coord_range.end,
        batch_input_rows = mask.len(),
        batch_output_rows = mask.true_count(),
        batch_extra_rows = extra_rows,
        elapsed_ms = elapsed.map(|duration| duration.as_secs_f64() * 1000.0),
        coord_rows = coords.rows,
        coord_true_rows = coords.true_rows,
        coord_density = coords.density,
        coord_first_row = ?coords.first_row,
        coord_last_row = ?coords.last_row,
        coord_hash = coords.coord_hash,
        coord_sum = coords.coord_sum,
        coord_xor = coords.coord_xor,
        coord_sample = coords.sample_ranges.as_str(),
        message
    );
}

fn summarize_ranges(mask: &Mask, row_range: &Range<u64>) -> (u64, u64, u64, String) {
    match mask.slices() {
        AllOr::All => {
            let mut hash = hash_start(1);
            hash = hash_range(hash, row_range.start, row_range.end);
            (
                hash,
                sum_range(row_range.start, row_range.end),
                xor_range(row_range.start, row_range.end),
                format!("[{}..{})", row_range.start, row_range.end),
            )
        }
        AllOr::None => (hash_start(2), 0, 0, "[]".to_string()),
        AllOr::Some(slices) => {
            let mut hash = hash_start(3);
            let mut sum = 0_u64;
            let mut xor = 0_u64;
            let mut sample = String::from("[");
            for (idx, &(start, end)) in slices.iter().enumerate() {
                let start = row_range.start + start as u64;
                let end = row_range.start + end as u64;
                hash = hash_range(hash, start, end);
                sum = sum.wrapping_add(sum_range(start, end));
                xor ^= xor_range(start, end);
                if idx < MAX_SAMPLE_RANGES {
                    if idx > 0 {
                        sample.push_str(", ");
                    }
                    let _ = write!(sample, "{start}..{end}");
                }
            }
            if slices.len() > MAX_SAMPLE_RANGES {
                let _ = write!(sample, ", +{} more", slices.len() - MAX_SAMPLE_RANGES);
            }
            sample.push(']');
            (hash, sum, xor, sample)
        }
    }
}

fn hash_start(tag: u64) -> u64 {
    hash_u64(FNV_OFFSET, tag)
}

fn hash_range(hash: u64, start: u64, end: u64) -> u64 {
    hash_u64(hash_u64(hash, start), end)
}

fn hash_u64(mut hash: u64, value: u64) -> u64 {
    for byte in value.to_le_bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

fn sum_range(start: u64, end: u64) -> u64 {
    if start >= end {
        return 0;
    }

    let n = u128::from(end - start);
    let first = u128::from(start);
    let last = u128::from(end - 1);
    let sum = if n % 2 == 0 {
        (n / 2).wrapping_mul(first + last)
    } else {
        n.wrapping_mul((first + last) / 2)
    };
    let mut low = [0_u8; 8];
    low.copy_from_slice(&sum.to_le_bytes()[..8]);
    u64::from_le_bytes(low)
}

fn xor_range(start: u64, end: u64) -> u64 {
    if start >= end {
        return 0;
    }
    xor_upto(end - 1) ^ if start == 0 { 0 } else { xor_upto(start - 1) }
}

fn xor_upto(value: u64) -> u64 {
    match value & 3 {
        0 => value,
        1 => 1,
        2 => value + 1,
        _ => 0,
    }
}
