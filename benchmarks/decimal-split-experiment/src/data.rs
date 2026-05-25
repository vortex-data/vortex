// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Synthetic and real (TPC-H) decimal column generators.

use arrow_array::Array;
use arrow_array::Decimal128Array;
use arrow_array::RecordBatch;
use arrow_buffer::i256;
use arrow_schema::DataType;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use tpchgen::generators::LineItemGenerator;
use tpchgen_arrow::LineItemArrow;
use tpchgen_arrow::RecordBatchIterator;

/// How large the magnitudes of a synthetic decimal column are. This is the
/// dimension that decides whether the high limbs are dead weight.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Magnitude {
    /// Values fit comfortably in the low 64-bit limb (typical money columns).
    /// High limb is all-zero -> the "small decimal" case.
    Small,
    /// Values span ~96 bits: the high limb carries real entropy.
    Medium,
    /// Values span the full i128 range.
    Large,
}

impl Magnitude {
    pub fn label(self) -> &'static str {
        match self {
            Magnitude::Small => "small (fits i64)",
            Magnitude::Medium => "medium (~96-bit)",
            Magnitude::Large => "large (full i128)",
        }
    }
}

/// Generate a synthetic i128 decimal column with the requested magnitude.
pub fn gen_i128(n: usize, mag: Magnitude, seed: u64) -> Vec<i128> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| match mag {
            // Cents up to ~$10M: a realistic price/amount column.
            Magnitude::Small => i128::from(rng.random_range(0i64..1_000_000_000i64)),
            Magnitude::Medium => {
                let hi = rng.random_range(0u64..(1u64 << 32));
                let lo = rng.random::<u64>();
                ((u128::from(hi) << 64) | u128::from(lo)) as i128
            }
            Magnitude::Large => {
                // Full-entropy hi limb, but clear the top 3 bits so two values
                // sum below the precision-38 ceiling (10^38 < 2^126) and Arrow's
                // checked kernel does not reject the result.
                let hi = rng.random::<u64>() & (u64::MAX >> 3);
                let lo = rng.random::<u64>();
                ((u128::from(hi) << 64) | u128::from(lo)) as i128
            }
        })
        .collect()
}

/// Generate an i128 column with block-wise structure: each `block`-sized chunk
/// is, with probability `const_frac`, "small" (high limb constant 0) and
/// otherwise full-range. Returns the values plus per-block metadata
/// (`Some(0)` for constant-high blocks, `None` for varying ones) - the kind of
/// per-chunk stat a real encoding records.
pub fn gen_i128_blocked(
    n: usize,
    block: usize,
    const_frac: f64,
    seed: u64,
) -> (Vec<i128>, Vec<Option<u64>>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut values = Vec::with_capacity(n);
    let num_blocks = n.div_ceil(block);
    let mut meta = Vec::with_capacity(num_blocks);
    for _ in 0..num_blocks {
        let is_const = rng.random::<f64>() < const_frac;
        meta.push(if is_const { Some(0u64) } else { None });
        let remaining = n - values.len();
        for _ in 0..block.min(remaining) {
            if is_const {
                values.push(i128::from(rng.random_range(0i64..1_000_000_000i64)));
            } else {
                let hi = rng.random::<u64>() & (u64::MAX >> 3);
                let lo = rng.random::<u64>();
                values.push(((u128::from(hi) << 64) | u128::from(lo)) as i128);
            }
        }
    }
    (values, meta)
}

/// Generate a synthetic i256 decimal column.
pub fn gen_i256(n: usize, mag: Magnitude, seed: u64) -> Vec<i256> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            let mut limbs = [0u64; 4];
            match mag {
                Magnitude::Small => {
                    limbs[0] = rng.random_range(0u64..1_000_000_000u64);
                }
                Magnitude::Medium => {
                    limbs[0] = rng.random::<u64>();
                    limbs[1] = rng.random_range(0u64..(1u64 << 32));
                }
                Magnitude::Large => {
                    for l in &mut limbs {
                        *l = rng.random::<u64>();
                    }
                    // Clear the top 5 bits of the value so two values sum below
                    // the precision-76 ceiling (10^76 < 2^253), keeping Arrow's
                    // checked kernel happy.
                    limbs[3] &= u64::MAX >> 5;
                }
            }
            let mut bytes = [0u8; 32];
            for (k, l) in limbs.iter().enumerate() {
                bytes[k * 8..k * 8 + 8].copy_from_slice(&l.to_le_bytes());
            }
            i256::from_le_bytes(bytes)
        })
        .collect()
}

/// A named real decimal column pulled from a TPC-H table.
pub struct RealColumn {
    pub name: String,
    pub precision: u8,
    pub scale: i8,
    pub values: Vec<i128>,
}

/// Generate TPC-H `lineitem` locally and extract every Decimal128 column.
///
/// `lineitem` carries `l_quantity`, `l_extendedprice`, `l_discount` and
/// `l_tax`, all small-magnitude money/quantity decimals - exactly the real
/// case the split layout should help. Returns the columns concatenated across
/// the streamed batches.
pub fn tpch_lineitem_decimals(scale_factor: f64) -> Vec<RealColumn> {
    let generator = LineItemGenerator::new(scale_factor, 1, 1);
    let iter = LineItemArrow::new(generator).with_batch_size(64 * 1024);
    let schema = iter.schema().clone();

    let decimal_fields: Vec<(usize, String, u8, i8)> = schema
        .fields()
        .iter()
        .enumerate()
        .filter_map(|(idx, f)| match f.data_type() {
            DataType::Decimal128(p, s) => Some((idx, f.name().clone(), *p, *s)),
            _ => None,
        })
        .collect();

    let mut columns: Vec<RealColumn> = decimal_fields
        .iter()
        .map(|(_, name, p, s)| RealColumn {
            name: name.clone(),
            precision: *p,
            scale: *s,
            values: Vec::new(),
        })
        .collect();

    for batch in iter {
        append_decimal_batch(&batch, &decimal_fields, &mut columns);
    }
    columns
}

fn append_decimal_batch(
    batch: &RecordBatch,
    decimal_fields: &[(usize, String, u8, i8)],
    columns: &mut [RealColumn],
) {
    for (col_pos, (idx, _, _, _)) in decimal_fields.iter().enumerate() {
        let arr = batch
            .column(*idx)
            .as_any()
            .downcast_ref::<Decimal128Array>()
            .expect("decimal128 column");
        for i in 0..arr.len() {
            columns[col_pos].values.push(arr.value(i));
        }
    }
}
