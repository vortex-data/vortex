// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Cheap, scale-free features of an integer chunk, computed independently of the compressor.

use std::collections::BTreeMap;

use anyhow::bail;
use arrow_array::Array;
use arrow_array::ArrayRef;
use arrow_array::cast::AsArray;
use arrow_array::types::Int8Type;
use arrow_array::types::Int16Type;
use arrow_array::types::Int32Type;
use arrow_array::types::Int64Type;
use arrow_array::types::UInt8Type;
use arrow_array::types::UInt16Type;
use arrow_array::types::UInt32Type;
use arrow_array::types::UInt64Type;
use arrow_schema::DataType;

/// The CSV header for [`compute`].
pub const HEADER: &str = "ptype_bits,signed,len,null_frac,bits_bp,bits_for,bits_zz,for_gain,\
avg_run,distinct,distinct_frac,top1_frac,top10_cover,sorted_frac,bits_delta,step_mode_frac,\
bits_p50,bits_p90,bits_p99,trailing_zeros";

fn bits(value: u128) -> u32 {
    u128::BITS - value.leading_zeros()
}

/// Computes the features of an integer column chunk as one CSV fragment.
pub fn compute(array: &ArrayRef) -> anyhow::Result<String> {
    let len = array.len();
    let (ptype_bits, signed, values): (u32, bool, Vec<i128>) = match array.data_type() {
        DataType::Int8 => (8, true, collect(array.as_primitive::<Int8Type>().iter())),
        DataType::Int16 => (16, true, collect(array.as_primitive::<Int16Type>().iter())),
        DataType::Int32 => (32, true, collect(array.as_primitive::<Int32Type>().iter())),
        DataType::Int64 => (64, true, collect(array.as_primitive::<Int64Type>().iter())),
        DataType::UInt8 => (8, false, collect(array.as_primitive::<UInt8Type>().iter())),
        DataType::UInt16 => (16, false, collect(array.as_primitive::<UInt16Type>().iter())),
        DataType::UInt32 => (32, false, collect(array.as_primitive::<UInt32Type>().iter())),
        DataType::UInt64 => (64, false, collect(array.as_primitive::<UInt64Type>().iter())),
        other => bail!("not an integer type: {other}"),
    };

    let n = values.len();
    let null_frac = if len == 0 {
        0.0
    } else {
        1.0 - n as f64 / len as f64
    };
    if n == 0 {
        return Ok(format!(
            "{ptype_bits},{},{len},{null_frac},,,,,,0,,,,,,,,,,",
            u8::from(signed)
        ));
    }

    let min = values.iter().copied().min().unwrap_or_default();
    let max = values.iter().copied().max().unwrap_or_default();
    let bits_bp = if min >= 0 {
        f64::from(bits(max.unsigned_abs()))
    } else {
        f64::from(ptype_bits)
    };
    let bits_for = f64::from(bits((max - min).unsigned_abs()));
    let bits_zz = f64::from(bits(min.unsigned_abs().max(max.unsigned_abs())) + 1);

    let mut runs = 1usize;
    let mut sorted = 0usize;
    for pair in values.windows(2) {
        if pair[0] != pair[1] {
            runs += 1;
        }
        if pair[0] <= pair[1] {
            sorted += 1;
        }
    }
    let avg_run = n as f64 / runs as f64;
    let sorted_frac = if n > 1 {
        sorted as f64 / (n - 1) as f64
    } else {
        1.0
    };

    let mut counts: BTreeMap<i128, usize> = BTreeMap::new();
    for v in &values {
        *counts.entry(*v).or_default() += 1;
    }
    let distinct = counts.len();
    let mut freqs: Vec<usize> = counts.values().copied().collect();
    freqs.sort_unstable_by(|a, b| b.cmp(a));
    let top1_frac = freqs[0] as f64 / n as f64;
    let top10_cover = freqs.iter().take(10).sum::<usize>() as f64 / n as f64;

    let (bits_delta, step_mode_frac) = if n > 1 {
        let deltas: Vec<i128> = values.windows(2).map(|p| p[1] - p[0]).collect();
        let dmin = deltas.iter().copied().min().unwrap_or_default();
        let dmax = deltas.iter().copied().max().unwrap_or_default();
        let mut delta_counts: BTreeMap<i128, usize> = BTreeMap::new();
        for d in &deltas {
            *delta_counts.entry(*d).or_default() += 1;
        }
        let mode = delta_counts.values().copied().max().unwrap_or_default();
        (
            f64::from(bits((dmax - dmin).unsigned_abs())),
            mode as f64 / deltas.len() as f64,
        )
    } else {
        (0.0, 1.0)
    };

    let mut widths: Vec<u32> = values.iter().map(|v| bits((v - min).unsigned_abs())).collect();
    widths.sort_unstable();
    let pct = |p: f64| widths[((widths.len() - 1) as f64 * p) as usize];

    let trailing_zeros = values
        .iter()
        .filter(|v| **v != 0)
        .map(|v| v.trailing_zeros())
        .min()
        .unwrap_or(0);

    Ok(format!(
        "{ptype_bits},{},{len},{null_frac:.6},{bits_bp},{bits_for},{bits_zz},{},{avg_run:.4},{distinct},{:.6},{top1_frac:.6},{top10_cover:.6},{sorted_frac:.6},{bits_delta},{step_mode_frac:.6},{},{},{},{trailing_zeros}",
        u8::from(signed),
        bits_bp - bits_for,
        distinct as f64 / n as f64,
        pct(0.5),
        pct(0.9),
        pct(0.99),
    ))
}

fn collect<T: Into<i128>>(iter: impl Iterator<Item = Option<T>>) -> Vec<i128> {
    iter.flatten().map(Into::into).collect()
}
