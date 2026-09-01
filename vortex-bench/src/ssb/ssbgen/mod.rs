// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Native SSB data generation.
//!
//! SSB has no Rust generator, so this is an in-house one. It is built on the `dbgen` stream
//! primitives the `tpchgen` crate already ports: SSB's C generator is TPC-H's `dbgen` with
//! different cardinalities and a different draw order, so the machinery is shared and only the
//! table definitions are new. It is self-contained enough to lift out as its own crate if there
//! is ever interest in upstreaming it; otherwise it is fine here.
//!
//! Output is byte-identical to the reference C generator (`eyalroz/ssb-dbgen` at
//! `ae1e254aa4d603d8ef1f44078e5abed011634b23`) for all five tables at SF 1 and SF 10. The
//! benchmark's expected row counts are only meaningful against reference data, so the `tests`
//! module pins a digest per table.
//!
//! # Reference details this relies on
//!
//! A few reference behaviors are load-bearing and easy to mistake for mistakes here. Each is
//! documented where it is implemented:
//!
//! * The date table occupies TPC-H's `ORDER` slot in `tdefs[]`, so generating it advances the
//!   order streams — see [`lineorder`].
//! * `supplier` and `customer` share the city and phone streams and are generated in that order
//!   — see [`customer`].
//! * `p_name` permutes colors the way the `dbgen` revision SSB forked does, which is not the way
//!   [`tpchgen::random::RandomStringSequence`] does — see [`part`].
//! * Addresses draw characters through that revision's `UnifInt`, which evaluates its range in
//!   64-bit — see [`AlphaNumeric`].
//! * `part` sizes the dimension with a base-2 log and `lo_partkey`'s range with a natural log —
//!   see [`part_row_count`] and [`part_key_max`].
//! * `d_dayofweek` runs one day ahead of the real weekday — see [`dwdate`].
//!
//! # Timezone
//!
//! The reference derives its calendar with `localtime()`, so the date table depends on the host
//! timezone — west of UTC-7 it shifts a day, and DST transitions vary it further. `d_year` is
//! what the queries filter on, so the calendar here is frozen to GMT: 1992-01-01 through
//! 1998-12-31, which is what CI has been generating all along.

pub mod arrow;
pub mod customer;
pub mod dwdate;
pub mod lineorder;
pub mod part;
pub mod supplier;
#[cfg(test)]
mod tests;

use tpchgen::distribution::Distribution;
use tpchgen::distribution::Distributions;
use tpchgen::random::RowRandomInt;
use vortex::error::VortexExpect;

pub use crate::ssb::ssbgen::customer::Customer;
pub use crate::ssb::ssbgen::customer::CustomerGenerator;
pub use crate::ssb::ssbgen::dwdate::Dwdate;
pub use crate::ssb::ssbgen::dwdate::DwdateGenerator;
pub use crate::ssb::ssbgen::lineorder::Lineorder;
pub use crate::ssb::ssbgen::lineorder::LineorderGenerator;
pub use crate::ssb::ssbgen::part::Part;
pub use crate::ssb::ssbgen::part::PartGenerator;
pub use crate::ssb::ssbgen::supplier::Supplier;
pub use crate::ssb::ssbgen::supplier::SupplierGenerator;

/// Rows per unit of scale factor for each SSB table, from the reference `tdefs[]`.
///
/// `part` and `dwdate` do not scale linearly; see [`part_row_count`] and [`DWDATE_ROWS`].
pub const PART_SCALE_BASE: i64 = 200_000;
/// Rows per unit of scale factor in `supplier` (TPC-H uses 10_000).
pub const SUPPLIER_SCALE_BASE: i64 = 2_000;
/// Rows per unit of scale factor in `customer` (TPC-H uses 150_000).
pub const CUSTOMER_SCALE_BASE: i64 = 30_000;
/// Orders per unit of scale factor. Each order expands to 1-7 `lineorder` rows.
///
/// The reference `tdefs[]` entry reads 150_000 and is multiplied by `ORDERS_PER_CUST` (10) at
/// startup, "after init"; this is the product.
pub const ORDER_SCALE_BASE: i64 = 1_500_000;
/// The date dimension is a fixed 2557-day calendar, 1992-01-01 through 1998-12-31.
pub const DWDATE_ROWS: i64 = 2557;

/// Number of `part` rows at `scale_factor`: `200_000 * ⌊1 + log₂ SF⌋`.
///
/// This is not the range `lo_partkey` is drawn from; see [`part_key_max`].
#[expect(
    clippy::cast_possible_truncation,
    reason = "truncating the double is the reference generator's `(long)` cast"
)]
pub fn part_row_count(scale_factor: f64) -> i64 {
    PART_SCALE_BASE * (1.0 + scale_factor.log2()).floor() as i64
}

/// Upper bound on `lo_partkey`: `200_000 * (⌊ln SF⌋ + 1)`.
///
/// The reference uses a natural log here and a base-2 log for the row count, so from SF 3 up the
/// fact table references only part of the dimension (at SF 10: 800_000 rows, keys to 600_000).
#[expect(
    clippy::cast_possible_truncation,
    reason = "truncating the double is the reference generator's `(long)` cast"
)]
pub fn part_key_max(scale_factor: f64) -> i64 {
    PART_SCALE_BASE * (scale_factor.ln().floor() as i64 + 1)
}

/// Number of `supplier` rows at `scale_factor`.
#[expect(
    clippy::cast_possible_truncation,
    reason = "truncating the double is the reference generator's `(long)` cast"
)]
pub fn supplier_row_count(scale_factor: f64) -> i64 {
    (SUPPLIER_SCALE_BASE as f64 * scale_factor) as i64
}

/// Number of `customer` rows at `scale_factor`.
#[expect(
    clippy::cast_possible_truncation,
    reason = "truncating the double is the reference generator's `(long)` cast"
)]
pub fn customer_row_count(scale_factor: f64) -> i64 {
    (CUSTOMER_SCALE_BASE as f64 * scale_factor) as i64
}

/// Number of orders at `scale_factor`. The `lineorder` row count is data dependent: each order
/// expands to 1-7 lines drawn uniformly, so it averages four rows per order (6,001,173 rows for
/// 1,500,000 orders at SF 1).
#[expect(
    clippy::cast_possible_truncation,
    reason = "truncating the double is the reference generator's `(long)` cast"
)]
pub fn order_count(scale_factor: f64) -> i64 {
    (ORDER_SCALE_BASE as f64 * scale_factor) as i64
}

/// Smallest supported scale factor.
///
/// Below 1 the reference's own `part` formulas stop making sense: `⌊1 + log₂ SF⌋` goes negative,
/// which would empty the dimension while the fact table kept drawing keys from an inverted range.
/// The C generator sidesteps this by clamping every table to at least one row; rather than invent
/// a meaning for fractional scales, this rejects them.
pub const MIN_SCALE_FACTOR: f64 = 1.0;

/// Reject a scale factor this generator cannot represent, before anything is written.
///
/// Two bounds. The lower one is [`MIN_SCALE_FACTOR`]. The upper one comes from the schema: the
/// reference DDL types every numeric column as a 32-bit `INTEGER`, and `lo_orderkey` is sparse
/// (roughly four times the order index), so it is the first column to leave `i32` — at around
/// SF 358. Without this check that overflow surfaces as a panic partway through writing a
/// multi-gigabyte file.
pub fn validate_scale_factor(scale_factor: f64) -> anyhow::Result<()> {
    if !scale_factor.is_finite() {
        anyhow::bail!("ssb: scale factor must be a finite number, got {scale_factor}");
    }
    if scale_factor < MIN_SCALE_FACTOR {
        anyhow::bail!(
            "ssb: scale factors below {MIN_SCALE_FACTOR} are not supported (got {scale_factor}); \
             the reference `part` cardinality formula is undefined there",
        );
    }
    let max_order_key = lineorder::order_key(order_count(scale_factor));
    if max_order_key > i64::from(i32::MAX) {
        anyhow::bail!(
            "ssb: scale factor {scale_factor} needs a {max_order_key} lo_orderkey, which exceeds \
             the INTEGER column the reference schema declares; the maximum supported is about {}",
            max_supported_scale_factor(),
        );
    }
    Ok(())
}

/// Largest scale factor whose sparse `lo_orderkey` still fits the schema's `INTEGER`, rounded
/// down to a whole scale factor.
pub fn max_supported_scale_factor() -> i64 {
    (1i64..)
        .take_while(|sf| lineorder::order_key(order_count(*sf as f64)) <= i64::from(i32::MAX))
        .last()
        .unwrap_or(1)
}

/// Width of the fixed-width nation prefix in a city name, from `#define CITY_FIX 10`: nine
/// characters of nation name, space-padded, plus one digit.
const CITY_FIX: usize = 10;

/// Build an SSB city name: the nation name truncated or space-padded to nine characters,
/// followed by a digit drawn from `P_CITY_SD`.
///
/// ```text
/// MOROCCO  6
/// UNITED ST3
/// ```
fn city_name(nation: &str, pick: i32) -> String {
    let mut city = String::with_capacity(CITY_FIX);
    for (i, c) in nation.chars().take(CITY_FIX - 1).enumerate() {
        debug_assert!(i < CITY_FIX - 1);
        city.push(c);
    }
    while city.len() < CITY_FIX - 1 {
        city.push(' ');
    }
    city.push_str(&pick.to_string());
    city
}

/// A distribution's size, for the bounded draws that index into it.
fn distribution_size(distribution: &Distribution) -> i32 {
    i32::try_from(distribution.size()).vortex_expect("ssb: distribution does not fit in an i32")
}

/// The `regions` entry a `nations` entry joins to. In `dists.dss` each nation's weight *is* its
/// region index.
fn region_of(distributions: &'static Distributions, nation_index: usize) -> &'static str {
    let region = distributions.nations().get_weight(nation_index) as usize;
    distributions.regions().get_value(region)
}

/// `a_rnd`: a random alphanumeric string of a randomly chosen length, as `V_STR` produces for
/// `c_address` and `s_address`.
///
/// [`tpchgen::random::RandomAlphaNumeric`] cannot be reused here. Both draw a length and then one
/// value per five characters, but they differ on that character draw, `RANDOM(0, MAX_LONG)`:
/// TPC-H 2.x narrows the bounds to `int32_t` before computing `nHigh - nLow + 1`, so the range
/// overflows negative, which `tpchgen` reproduces. The revision SSB forked evaluates the range in
/// 64-bit and gets `2^31`. Which of the 64 alphabet characters each 6-bit group selects follows
/// from that.
#[derive(Debug)]
pub struct AlphaNumeric {
    random: RowRandomInt,
    min_length: i32,
    max_length: i32,
}

impl AlphaNumeric {
    /// The 64-character alphabet, indexed by 6-bit groups. Note the space and comma.
    const ALPHABET: &'static [u8; 64] =
        b"0123456789abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ,";
    /// `V_STR_LOW` and `V_STR_HGH`: lengths span 0.4x to 1.6x the average.
    const LOW_LENGTH_MULTIPLIER: f64 = 0.4;
    const HIGH_LENGTH_MULTIPLIER: f64 = 1.6;
    /// Draws per row reserved by `Seed[C_ADDR_SD].boundary`: one length plus one per five
    /// characters of the longest possible string.
    const USAGE_PER_ROW: i32 = 9;
    /// `MAX_LONG`, the upper bound of the character draw.
    const MAX_LONG: i64 = 0x7FFF_FFFF;
    /// `dM`, the modulus as a double.
    const MODULUS: f64 = 2147483647.0;

    /// Create a generator drawing from `seed`, with lengths centered on `average_length`.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "`V_STR` truncates the scaled average to an int"
    )]
    pub fn new(seed: i64, average_length: i32) -> Self {
        Self {
            random: RowRandomInt::new(seed, Self::USAGE_PER_ROW),
            min_length: (average_length as f64 * Self::LOW_LENGTH_MULTIPLIER) as i32,
            max_length: (average_length as f64 * Self::HIGH_LENGTH_MULTIPLIER) as i32,
        }
    }

    /// Draw the next string.
    pub fn next_value(&mut self) -> String {
        let length = self.random.next_int(self.min_length, self.max_length);
        let mut value = String::with_capacity(length as usize);
        let mut bits = 0;
        for i in 0..length {
            if i % 5 == 0 {
                bits = self.next_char_bits();
            }
            value.push(Self::ALPHABET[(bits & 0o77) as usize] as char);
            bits >>= 6;
        }
        value
    }

    /// `UnifInt(0, MAX_LONG, stream)` with a 64-bit range, as SSB's fork evaluates it.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "`UnifInt` truncates the scaled double, and the result is under 2^31 anyway"
    )]
    fn next_char_bits(&mut self) -> i64 {
        let seed = self.random.next_rand();
        ((seed as f64 / Self::MODULUS) * (Self::MAX_LONG + 1) as f64) as i64
    }

    /// Signal the end of a row, rewinding the stream to its per-row boundary.
    pub fn row_finished(&mut self) {
        self.random.row_finished();
    }
}

/// `dbgen`'s original `permute`, used by `agg_str` to pick `p_name`'s colors.
///
/// Permutes `0..count` in place by swapping each position with one drawn uniformly from the whole
/// range, consuming exactly `count` values from `stream`. Later TPC-H revisions draw the swap
/// index from `i..count`, which is what [`tpchgen::random::RandomStringSequence`] implements, so
/// the two select different colors from the same stream.
fn permute_indices(count: i32, random: &mut RowRandomInt) -> Vec<usize> {
    let mut permutation: Vec<usize> = (0..count as usize).collect();
    for i in 0..count {
        let source = random.next_int(0, count - 1);
        permutation.swap(source as usize, i as usize);
    }
    permutation
}
