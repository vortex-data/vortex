// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The `lineorder` fact table: TPC-H's `orders` and `lineitem` denormalized into one wide table.
//!
//! Each order still expands into 1-7 lines drawn uniformly, and each line carries its order's
//! date, priority and total price. TPC-H's `l_shipdate`, `l_receiptdate`, `l_returnflag`,
//! `l_linestatus`, `l_shipinstruct` and both comments are gone; `lo_revenue`, `lo_supplycost` and
//! `lo_ordtotalprice` are new, all derived rather than drawn.
//!
//! # Stream alignment
//!
//! Two things line up with the reference here:
//!
//! * The five order streams are advanced by [`DWDATE_ROWS`] rows before the first order. SSB's
//!   date table occupies TPC-H's `ORDER` slot in `tdefs[]`, and `dbgen` rewinds a table's streams
//!   once per row of that table, so generating the calendar advances them.
//! * The seven per-line streams are rewound to a boundary of 7 draws per *order* rather than per
//!   line, which is what [`RandomBoundedInt::row_finished`] does once per order below.
//!
//! `O_CLRK_SD` is drawn once per order by the reference and discarded, since SSB has no
//! `lo_clerk` column. Nothing else reads that stream, so the draw is omitted.

use std::fmt;

use tpchgen::dates::MIN_GENERATE_DATE;
use tpchgen::dates::TPCHDate;
use tpchgen::distribution::Distributions;
use tpchgen::generators::PartGeneratorIterator;
use tpchgen::random::RandomBoundedInt;
use tpchgen::random::RandomBoundedLong;
use tpchgen::random::RandomString;

use crate::ssb::ssbgen::DWDATE_ROWS;
use crate::ssb::ssbgen::customer_row_count;
use crate::ssb::ssbgen::order_count;
use crate::ssb::ssbgen::part_key_max;
use crate::ssb::ssbgen::supplier_row_count;

/// Seed of the order date stream, `O_ODATE_SD`.
const ORDER_DATE_SEED: i64 = 1066728069;
/// Seed of the customer key stream, `O_CKEY_SD`.
const CUSTOMER_KEY_SEED: i64 = 851767375;
/// Seed of the order priority stream, `O_PRIO_SD`.
const ORDER_PRIORITY_SEED: i64 = 591449447;
/// Seed of the line count stream, `O_LCNT_SD`.
const LINE_COUNT_SEED: i64 = 1434868289;
/// Seed of the part key stream, `L_PKEY_SD`.
const PART_KEY_SEED: i64 = 1808217256;
/// Seed of the supplier key stream, `L_SKEY_SD`.
const SUPPLIER_KEY_SEED: i64 = 2095021727;
/// Seed of the quantity stream, `L_QTY_SD`.
const QUANTITY_SEED: i64 = 209208115;
/// Seed of the discount stream, `L_DCNT_SD`.
const DISCOUNT_SEED: i64 = 554590007;
/// Seed of the tax stream, `L_TAX_SD`.
const TAX_SEED: i64 = 721958466;
/// Seed of the commit date stream, `L_CDTE_SD`.
const COMMIT_DATE_SEED: i64 = 904914315;
/// Seed of the ship mode stream, `L_SMODE_SD`.
const SHIP_MODE_SEED: i64 = 675466456;

/// Portion of customers with no orders: every key divisible by this is nudged aside.
const CUSTOMER_MORTALITY: i64 = 3;
/// Days of slack the reference leaves at the end of the calendar for ship and receipt dates,
/// `L_SDTE_MAX + L_RDTE_MAX`. SSB has neither column but keeps the narrower order date range.
const ORDER_DATE_SLACK: i32 = 121 + 30;

const LINE_COUNT_MIN: i32 = 1;
/// Also the per-order boundary of every per-line stream.
const LINE_COUNT_MAX: i32 = 7;
const QUANTITY_MIN: i32 = 1;
const QUANTITY_MAX: i32 = 50;
const DISCOUNT_MIN: i32 = 0;
const DISCOUNT_MAX: i32 = 10;
const TAX_MIN: i32 = 0;
const TAX_MAX: i32 = 8;
const COMMIT_DATE_MIN: i32 = 30;
const COMMIT_DATE_MAX: i32 = 90;
/// Scaled-integer money: all prices are in cents.
const PENNIES: i64 = 100;

/// A `lineorder` row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Lineorder {
    /// Sparse order key, shared by every line of an order.
    pub lo_orderkey: i64,
    /// `1..=7`, position of this line within its order.
    pub lo_linenumber: i32,
    pub lo_custkey: i64,
    pub lo_partkey: i64,
    pub lo_suppkey: i64,
    /// `yyyymmdd`, joining to `dwdate.d_datekey`.
    pub lo_orderdate: i32,
    pub lo_orderpriority: &'static str,
    /// Always zero. The reference generates it as an integer and never varies it.
    pub lo_shippriority: i32,
    pub lo_quantity: i32,
    /// Scaled-integer money, in cents.
    pub lo_extendedprice: i64,
    /// Total for the whole order, in cents, repeated on each of its lines.
    pub lo_ordtotalprice: i64,
    /// Percentage points, `0..=10`.
    pub lo_discount: i32,
    /// `lo_extendedprice * (100 - lo_discount) / 100`, in cents.
    pub lo_revenue: i64,
    /// In cents.
    pub lo_supplycost: i64,
    /// Percentage points, `0..=8`.
    pub lo_tax: i32,
    /// `yyyymmdd`, joining to `dwdate.d_datekey`.
    pub lo_commitdate: i32,
    pub lo_shipmode: &'static str,
}

impl fmt::Display for Lineorder {
    /// The reference generator's `.tbl` line for this row.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|",
            self.lo_orderkey,
            self.lo_linenumber,
            self.lo_custkey,
            self.lo_partkey,
            self.lo_suppkey,
            self.lo_orderdate,
            self.lo_orderpriority,
            self.lo_shippriority,
            self.lo_quantity,
            self.lo_extendedprice,
            self.lo_ordtotalprice,
            self.lo_discount,
            self.lo_revenue,
            self.lo_supplycost,
            self.lo_tax,
            self.lo_commitdate,
            self.lo_shipmode,
        )
    }
}

/// Generator for the `lineorder` fact table.
#[derive(Debug, Clone)]
pub struct LineorderGenerator {
    scale_factor: f64,
}

impl LineorderGenerator {
    /// Create a generator for `scale_factor`.
    pub fn new(scale_factor: f64) -> Self {
        Self { scale_factor }
    }

    /// Number of *orders* this generator expands. The row count is data dependent: each order
    /// yields 1-7 rows, uniformly, so expect roughly four times this.
    pub fn order_count(&self) -> i64 {
        order_count(self.scale_factor)
    }

    /// Iterate the rows, in order key then line number order.
    pub fn iter(&self) -> LineorderIterator {
        LineorderIterator::new(
            Distributions::static_default(),
            self.scale_factor,
            self.order_count(),
        )
    }
}

impl IntoIterator for LineorderGenerator {
    type Item = Lineorder;
    type IntoIter = LineorderIterator;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Iterator over [`Lineorder`] rows.
#[derive(Debug)]
pub struct LineorderIterator {
    order_date_random: RandomBoundedInt,
    customer_key_random: RandomBoundedLong,
    order_priority_random: RandomString<'static>,
    line_count_random: RandomBoundedInt,

    part_key_random: RandomBoundedLong,
    supplier_key_random: RandomBoundedLong,
    quantity_random: RandomBoundedInt,
    discount_random: RandomBoundedInt,
    tax_random: RandomBoundedInt,
    commit_date_random: RandomBoundedInt,
    ship_mode_random: RandomString<'static>,

    max_customer_key: i64,
    order_count: i64,
    order_index: i64,
    /// Lines of the current order, reversed so [`Iterator::next`] can pop from the end.
    pending: Vec<Lineorder>,
}

impl LineorderIterator {
    fn new(distributions: &'static Distributions, scale_factor: f64, order_count: i64) -> Self {
        let mut order_date_random = RandomBoundedInt::new(
            ORDER_DATE_SEED,
            MIN_GENERATE_DATE,
            MIN_GENERATE_DATE + tpchgen::dates::TOTAL_DATE_RANGE - ORDER_DATE_SLACK - 1,
        );
        let max_customer_key = customer_row_count(scale_factor);
        let mut customer_key_random =
            RandomBoundedLong::new(CUSTOMER_KEY_SEED, false, 1, max_customer_key);
        let mut order_priority_random =
            RandomString::new(ORDER_PRIORITY_SEED, distributions.order_priority());
        let mut line_count_random =
            RandomBoundedInt::new(LINE_COUNT_SEED, LINE_COUNT_MIN, LINE_COUNT_MAX);

        // The date dimension shares TPC-H's ORDER table slot, so emitting it advanced every
        // order stream once per calendar day. See the module docs.
        order_date_random.advance_rows(DWDATE_ROWS);
        customer_key_random.advance_rows(DWDATE_ROWS);
        order_priority_random.advance_rows(DWDATE_ROWS);
        line_count_random.advance_rows(DWDATE_ROWS);

        // Per-line streams are rewound once per order, to a boundary of LINE_COUNT_MAX draws.
        let seeds_per_row = LINE_COUNT_MAX;
        Self {
            order_date_random,
            customer_key_random,
            order_priority_random,
            line_count_random,
            part_key_random: RandomBoundedLong::new_with_seeds_per_row(
                PART_KEY_SEED,
                false,
                1,
                part_key_max(scale_factor),
                seeds_per_row,
            ),
            supplier_key_random: RandomBoundedLong::new_with_seeds_per_row(
                SUPPLIER_KEY_SEED,
                false,
                1,
                supplier_row_count(scale_factor),
                seeds_per_row,
            ),
            quantity_random: RandomBoundedInt::new_with_seeds_per_row(
                QUANTITY_SEED,
                QUANTITY_MIN,
                QUANTITY_MAX,
                seeds_per_row,
            ),
            discount_random: RandomBoundedInt::new_with_seeds_per_row(
                DISCOUNT_SEED,
                DISCOUNT_MIN,
                DISCOUNT_MAX,
                seeds_per_row,
            ),
            tax_random: RandomBoundedInt::new_with_seeds_per_row(
                TAX_SEED,
                TAX_MIN,
                TAX_MAX,
                seeds_per_row,
            ),
            commit_date_random: RandomBoundedInt::new_with_seeds_per_row(
                COMMIT_DATE_SEED,
                COMMIT_DATE_MIN,
                COMMIT_DATE_MAX,
                seeds_per_row,
            ),
            ship_mode_random: RandomString::new_with_expected_row_count(
                SHIP_MODE_SEED,
                distributions.ship_modes(),
                seeds_per_row,
            ),
            max_customer_key,
            order_count,
            order_index: 0,
            pending: Vec::with_capacity(LINE_COUNT_MAX as usize),
        }
    }

    /// Expand one order into its lines, leaving them in [`LineorderIterator::pending`].
    fn make_order(&mut self, order_index: i64) {
        let order_date = self.order_date_random.next_value();
        let lo_orderkey = order_key(order_index);

        // Customers whose key is divisible by CUSTOMER_MORTALITY place no orders.
        let mut customer_key = self.customer_key_random.next_value();
        let mut delta = 1;
        while customer_key % CUSTOMER_MORTALITY == 0 {
            customer_key += delta;
            customer_key = customer_key.min(self.max_customer_key);
            delta *= -1;
        }

        let lo_orderpriority = self.order_priority_random.next_value();
        let line_count = self.line_count_random.next_value();

        let mut ordtotalprice: i64 = 0;
        for line in 0..line_count {
            let partkey = self.part_key_random.next_value();
            let suppkey = self.supplier_key_random.next_value();
            let quantity = self.quantity_random.next_value();
            let discount = self.discount_random.next_value();
            let tax = self.tax_random.next_value();
            let commit_date = order_date + self.commit_date_random.next_value();
            let ship_mode = self.ship_mode_random.next_value();

            let retail_price = PartGeneratorIterator::calculate_part_price(partkey);
            let extended_price = retail_price * i64::from(quantity);
            let discounted_price = extended_price * (PENNIES - i64::from(discount));
            ordtotalprice += (discounted_price / PENNIES) * (PENNIES + i64::from(tax)) / PENNIES;

            self.pending.push(Lineorder {
                lo_orderkey,
                lo_linenumber: line + 1,
                lo_custkey: customer_key,
                lo_partkey: partkey,
                lo_suppkey: suppkey,
                lo_orderdate: datekey(order_date),
                lo_orderpriority,
                lo_shippriority: 0,
                lo_quantity: quantity,
                lo_extendedprice: extended_price,
                // Backfilled below, once every line of the order has been priced.
                lo_ordtotalprice: 0,
                lo_discount: discount,
                lo_revenue: discounted_price / PENNIES,
                lo_supplycost: 6 * retail_price / 10,
                lo_tax: tax,
                lo_commitdate: datekey(commit_date),
                lo_shipmode: ship_mode,
            });
        }

        for line in &mut self.pending {
            line.lo_ordtotalprice = ordtotalprice;
        }
        self.pending.reverse();

        self.order_date_random.row_finished();
        self.customer_key_random.row_finished();
        self.order_priority_random.row_finished();
        self.line_count_random.row_finished();
        self.part_key_random.row_finished();
        self.supplier_key_random.row_finished();
        self.quantity_random.row_finished();
        self.discount_random.row_finished();
        self.tax_random.row_finished();
        self.commit_date_random.row_finished();
        self.ship_mode_random.row_finished();
    }
}

impl Iterator for LineorderIterator {
    type Item = Lineorder;

    fn next(&mut self) -> Option<Self::Item> {
        while self.pending.is_empty() {
            if self.order_index >= self.order_count {
                return None;
            }
            self.order_index += 1;
            self.make_order(self.order_index);
        }
        self.pending.pop()
    }
}

/// `ez_sparse`: spread order keys out so they are not dense, leaving room for the refresh
/// functions SSB does not use. Identical to TPC-H's, with the update sequence fixed at zero.
///
/// Also used by [`validate_scale_factor`](super::validate_scale_factor), which rejects any scale
/// factor whose largest key would not fit the schema's `INTEGER`.
pub(super) fn order_key(order_index: i64) -> i64 {
    const SPARSE_BITS: i64 = 2;
    const SPARSE_KEEP: i64 = 3;

    let low_bits = order_index & ((1 << SPARSE_KEEP) - 1);
    (((order_index >> SPARSE_KEEP) << SPARSE_BITS) << SPARSE_KEEP) + low_bits
}

/// Convert a generated date into the `yyyymmdd` integer the fact table stores.
fn datekey(generated_date: i32) -> i32 {
    let (short_year, month, day) = TPCHDate::new(generated_date).to_ymd();
    (1900 + short_year) * 10000 + month * 100 + day
}
