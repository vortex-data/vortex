// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Cheap, scale-free features of an integer chunk.
//!
//! The same code computes features for the training CSV (from Arrow) and for the model-driven
//! compressor (from Vortex arrays), so training and inference see the same numbers.

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
use vortex::utils::aliases::hash_map::HashMap;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::match_each_integer_ptype;

/// The base feature names, in CSV order.
pub const NAMES: [&str; 20] = [
    "ptype_bits",
    "signed",
    "len",
    "null_frac",
    "bits_bp",
    "bits_for",
    "bits_zz",
    "for_gain",
    "avg_run",
    "distinct",
    "distinct_frac",
    "top1_frac",
    "top10_cover",
    "sorted_frac",
    "bits_delta",
    "step_mode_frac",
    "bits_p50",
    "bits_p90",
    "bits_p99",
    "trailing_zeros",
];

/// The CSV header for [`compute`].
pub fn header() -> String {
    NAMES.join(",")
}

fn bits(value: u128) -> u32 {
    u128::BITS - value.leading_zeros()
}

/// Values per sampling window and the number of windows features are computed over.
///
/// Contiguous windows keep runs, deltas and sortedness intact while bounding the cost.
const WINDOW: usize = 512;
const WINDOWS: usize = 8;

/// The position ranges features are computed over: evenly spaced contiguous windows.
fn windows(len: usize) -> Vec<std::ops::Range<usize>> {
    if len <= WINDOW * WINDOWS {
        return vec![0..len];
    }
    let stride = len / WINDOWS;
    (0..WINDOWS)
        .map(|i| {
            let start = i * stride + (stride - WINDOW) / 2;
            start..start + WINDOW
        })
        .collect()
}

/// Features of a chunk, in [`NAMES`] order. `NaN` marks features undefined for all-null chunks.
///
/// `values` are the valid values inside the sampling windows; `null_frac` is over the whole chunk.
pub fn from_values(
    ptype_bits: u32,
    signed: bool,
    len: usize,
    null_frac: f64,
    values: &[i128],
) -> [f64; 20] {
    let n = values.len();
    let mut out = [f64::NAN; 20];
    out[0] = f64::from(ptype_bits);
    out[1] = f64::from(u8::from(signed));
    out[2] = len as f64;
    out[3] = null_frac;
    if n == 0 {
        out[9] = 0.0;
        return out;
    }

    let mut min = values[0];
    let mut max = values[0];
    let mut runs = 1usize;
    let mut sorted = 0usize;
    let mut dmin = i128::MAX;
    let mut dmax = i128::MIN;
    // Values come from at most 64-bit integers, so their low 64 bits identify them for counting.
    let mut counts: HashMap<u64, u32> = HashMap::with_capacity(n);
    let mut delta_counts: HashMap<u64, u32> = HashMap::with_capacity(n);
    let mut trailing = u32::MAX;
    for (i, &v) in values.iter().enumerate() {
        min = min.min(v);
        max = max.max(v);
        *counts.entry(v as u64).or_default() += 1;
        if v != 0 {
            trailing = trailing.min(v.trailing_zeros());
        }
        if i > 0 {
            let prev = values[i - 1];
            if prev != v {
                runs += 1;
            }
            if prev <= v {
                sorted += 1;
            }
            let d = v - prev;
            dmin = dmin.min(d);
            dmax = dmax.max(d);
            *delta_counts.entry(d as u64).or_default() += 1;
        }
    }

    let bits_bp = if min >= 0 {
        f64::from(bits(max.unsigned_abs()))
    } else {
        f64::from(ptype_bits)
    };
    let bits_for = f64::from(bits((max - min).unsigned_abs()));
    let bits_zz = f64::from(bits(min.unsigned_abs().max(max.unsigned_abs())) + 1);

    let distinct = counts.len();
    let mut freqs: Vec<u32> = counts.into_values().collect();
    let top_k = freqs.len().min(10);
    if top_k < freqs.len() {
        freqs.select_nth_unstable_by(top_k - 1, |a, b| b.cmp(a));
    }
    let top10: u64 = freqs[..top_k].iter().map(|&c| u64::from(c)).sum();
    let top1 = freqs[..top_k].iter().copied().max().unwrap_or(0);

    let mut width_hist = [0usize; 129];
    for &v in values {
        width_hist[bits((v - min).unsigned_abs()) as usize] += 1;
    }
    let pct = |p: f64| {
        let target = ((n - 1) as f64 * p) as usize;
        let mut seen = 0;
        for (width, &count) in width_hist.iter().enumerate() {
            seen += count;
            if seen > target {
                return width as f64;
            }
        }
        128.0
    };

    let (bits_delta, step_mode_frac) = if n > 1 {
        let mode = delta_counts.values().copied().max().unwrap_or(0);
        (
            f64::from(bits((dmax - dmin).unsigned_abs())),
            f64::from(mode) / (n - 1) as f64,
        )
    } else {
        (0.0, 1.0)
    };

    out[4] = bits_bp;
    out[5] = bits_for;
    out[6] = bits_zz;
    out[7] = bits_bp - bits_for;
    out[8] = n as f64 / runs as f64;
    out[9] = distinct as f64;
    out[10] = distinct as f64 / n as f64;
    out[11] = f64::from(top1) / n as f64;
    out[12] = top10 as f64 / n as f64;
    out[13] = if n > 1 {
        sorted as f64 / (n - 1) as f64
    } else {
        1.0
    };
    out[14] = bits_delta;
    out[15] = step_mode_frac;
    out[16] = pct(0.5);
    out[17] = pct(0.9);
    out[18] = pct(0.99);
    out[19] = if trailing == u32::MAX {
        0.0
    } else {
        f64::from(trailing)
    };
    out
}

/// Features of an Arrow integer column chunk, as one CSV fragment.
pub fn compute(array: &ArrayRef) -> anyhow::Result<String> {
    let len = array.len();
    let null_frac = if len == 0 {
        0.0
    } else {
        array.null_count() as f64 / len as f64
    };
    let ranges = windows(len);
    let (ptype_bits, signed, values): (u32, bool, Vec<Option<i128>>) = match array.data_type() {
        DataType::Int8 => (8, true, collect(array.as_primitive::<Int8Type>().iter(), &ranges)),
        DataType::Int16 => (16, true, collect(array.as_primitive::<Int16Type>().iter(), &ranges)),
        DataType::Int32 => (32, true, collect(array.as_primitive::<Int32Type>().iter(), &ranges)),
        DataType::Int64 => (64, true, collect(array.as_primitive::<Int64Type>().iter(), &ranges)),
        DataType::UInt8 => (8, false, collect(array.as_primitive::<UInt8Type>().iter(), &ranges)),
        DataType::UInt16 => (16, false, collect(array.as_primitive::<UInt16Type>().iter(), &ranges)),
        DataType::UInt32 => (32, false, collect(array.as_primitive::<UInt32Type>().iter(), &ranges)),
        DataType::UInt64 => (64, false, collect(array.as_primitive::<UInt64Type>().iter(), &ranges)),
        other => bail!("not an integer type: {other}"),
    };
    let values: Vec<i128> = values.into_iter().flatten().collect();
    let f = from_values(ptype_bits, signed, len, null_frac, &values);
    Ok(f.iter()
        .map(|v| if v.is_nan() { String::new() } else { format!("{v}") })
        .collect::<Vec<_>>()
        .join(","))
}

fn collect<T: Into<i128> + Copy>(
    iter: impl Iterator<Item = Option<T>>,
    ranges: &[std::ops::Range<usize>],
) -> Vec<Option<i128>> {
    let all: Vec<Option<T>> = iter.collect();
    ranges
        .iter()
        .flat_map(|r| all[r.clone()].iter().map(|v| v.map(Into::into)))
        .collect()
}

/// Features of a canonical Vortex integer array.
pub fn from_primitive(array: &PrimitiveArray, ctx: &mut ExecutionCtx) -> anyhow::Result<[f64; 20]> {
    let len = array.len();
    let mask = array.validity()?.execute_mask(len, ctx)?;
    let null_frac = if len == 0 {
        0.0
    } else {
        1.0 - mask.true_count() as f64 / len as f64
    };
    let ranges = windows(len);
    let ptype = array.ptype();
    let all_valid = mask.all_true();
    let values: Vec<i128> = match_each_integer_ptype!(ptype, |T| {
        let slice = array.as_slice::<T>();
        ranges
            .iter()
            .flat_map(|r| r.clone())
            .filter(|&i| all_valid || mask.value(i))
            .map(|i| i128::from(slice[i]))
            .collect()
    });
    Ok(from_values(
        ptype.bit_width() as u32,
        ptype.is_signed_int(),
        len,
        null_frac,
        &values,
    ))
}

/// Adds the derived features the model also uses: `log_len` and closed-form size estimates.
///
/// Must match `add_estimates` in `train.py`.
pub fn with_derived(base: &[f64; 20]) -> Vec<(&'static str, f64)> {
    let get = |name: &str| base[NAMES.iter().position(|n| *n == name).unwrap_or(0)];
    let mut out: Vec<(&'static str, f64)> = NAMES.iter().copied().zip(base.iter().copied()).collect();
    let len = get("len");
    let w = get("ptype_bits");
    let n = len * (1.0 - get("null_frac"));
    let pos = len.log2().max(1.0);
    let runs = (n / get("avg_run").max(1.0)).max(1.0);
    let patches = (n * (1.0 - get("top1_frac"))).max(1.0);
    let bits_bp = if get("bits_bp") >= 0.0 { get("bits_bp") } else { w };
    let distinct = get("distinct");
    let estimates = [
        ("est_bp", n * bits_bp),
        ("est_for", n * get("bits_for")),
        ("est_zz", n * get("bits_zz")),
        ("est_delta", n * get("bits_delta")),
        ("est_dict", distinct * w + n * distinct.max(2.0).log2()),
        ("est_runend", runs * (get("bits_for") + pos)),
        ("est_sparse", patches * (get("bits_for") + pos)),
        ("est_p90_patched", n * get("bits_p90") + n * 0.1 * (w + pos)),
    ];
    out.push(("log_len", len.log2()));
    for (name, bits) in estimates {
        out.push((name, (bits.max(1.0) / (len * w)).log2()));
    }
    out
}
