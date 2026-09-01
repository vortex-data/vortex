// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The `customer` dimension.
//!
//! As with [`supplier`](super::supplier), the nation/region keys are denormalized into names and
//! a `c_city` is added; `c_acctbal` and `c_comment` are dropped.
//!
//! # Shared streams
//!
//! `customer` draws its city suffix from `P_CITY_SD` and its phone number from `C_PHNE_SD`, and so
//! does `supplier`. `dbgen` only rewinds a stream to its per-row boundary while generating the
//! table that stream is registered to, so neither is reset between the two: `supplier` is
//! generated first and consumes [`supplier_row_count`] cities and three times that many phone
//! draws, then `customer` continues both sequences. This iterator advances both streams past the
//! supplier table before its first row.

use std::fmt;

use tpchgen::distribution::Distributions;
use tpchgen::random::PhoneNumberInstance;
use tpchgen::random::RandomBoundedInt;
use tpchgen::random::RandomPhoneNumber;
use tpchgen::random::RandomString;

use crate::ssb::ssbgen::AlphaNumeric;
use crate::ssb::ssbgen::city_name;
use crate::ssb::ssbgen::customer_row_count;
use crate::ssb::ssbgen::distribution_size;
use crate::ssb::ssbgen::region_of;
use crate::ssb::ssbgen::supplier_row_count;

/// Seed of the address stream, `C_ADDR_SD`.
const ADDRESS_SEED: i64 = 881155353;
/// Seed of the nation stream, `C_NTRG_SD`.
const NATION_SEED: i64 = 1489529863;
/// Seed of the market segment stream, `C_MSEG_SD`.
const MARKET_SEGMENT_SEED: i64 = 1140279430;
/// Seed of the phone stream, `C_PHNE_SD`. Shared with [`supplier`](super::supplier).
const PHONE_SEED: i64 = 1521138112;
/// Seed of the city stream, `P_CITY_SD`. Shared with [`supplier`](super::supplier).
const CITY_SEED: i64 = 1495190827;

/// The city stream, shared by `customer` and `supplier`.
pub(super) fn city_random() -> RandomBoundedInt {
    RandomBoundedInt::new(CITY_SEED, 0, 9)
}

/// The phone stream, shared by `customer` and `supplier`.
pub(super) fn phone_random() -> RandomPhoneNumber {
    RandomPhoneNumber::new(PHONE_SEED)
}

/// A `customer` row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Customer {
    /// Primary key, `1..=`[`customer_row_count`].
    pub c_custkey: i64,
    /// `Customer#000000001`.
    pub c_name: String,
    pub c_address: String,
    /// Nine characters of nation name, space padded, plus a digit: `MOROCCO  6`.
    pub c_city: String,
    pub c_nation: &'static str,
    pub c_region: &'static str,
    pub c_phone: PhoneNumberInstance,
    pub c_mktsegment: &'static str,
}

impl fmt::Display for Customer {
    /// The reference generator's `.tbl` line for this row.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}|{}|{}|{}|{}|{}|{}|{}|",
            self.c_custkey,
            self.c_name,
            self.c_address,
            self.c_city,
            self.c_nation,
            self.c_region,
            self.c_phone,
            self.c_mktsegment,
        )
    }
}

/// Generator for the `customer` dimension.
#[derive(Debug, Clone)]
pub struct CustomerGenerator {
    scale_factor: f64,
}

impl CustomerGenerator {
    /// Create a generator for `scale_factor`.
    pub fn new(scale_factor: f64) -> Self {
        Self { scale_factor }
    }

    /// Number of rows this generator yields.
    pub fn row_count(&self) -> i64 {
        customer_row_count(self.scale_factor)
    }

    /// Iterate the rows, in primary key order.
    pub fn iter(&self) -> CustomerIterator {
        CustomerIterator::new(
            Distributions::static_default(),
            self.row_count(),
            supplier_row_count(self.scale_factor),
        )
    }
}

impl IntoIterator for CustomerGenerator {
    type Item = Customer;
    type IntoIter = CustomerIterator;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Iterator over [`Customer`] rows.
#[derive(Debug)]
pub struct CustomerIterator {
    distributions: &'static Distributions,
    address_random: AlphaNumeric,
    nation_random: RandomBoundedInt,
    city_random: RandomBoundedInt,
    phone_random: RandomPhoneNumber,
    market_segment_random: RandomString<'static>,

    row_count: i64,
    index: i64,
}

impl CustomerIterator {
    fn new(distributions: &'static Distributions, row_count: i64, supplier_rows: i64) -> Self {
        let mut city_random = city_random();
        let mut phone_random = phone_random();

        // Both streams carry over from the supplier table; see the module docs.
        city_random.advance_rows(supplier_rows);
        phone_random.advance_rows(supplier_rows);

        Self {
            distributions,
            address_random: AlphaNumeric::new(
                ADDRESS_SEED,
                super::supplier::ADDRESS_AVERAGE_LENGTH,
            ),
            nation_random: RandomBoundedInt::new(
                NATION_SEED,
                0,
                distribution_size(distributions.nations()) - 1,
            ),
            city_random,
            phone_random,
            market_segment_random: RandomString::new(
                MARKET_SEGMENT_SEED,
                distributions.market_segments(),
            ),
            row_count,
            index: 0,
        }
    }

    fn make_customer(&mut self, c_custkey: i64) -> Customer {
        let nation_index = self.nation_random.next_value() as usize;
        Customer {
            c_custkey,
            c_name: format!("Customer#{c_custkey:09}"),
            c_address: self.address_random.next_value(),
            c_city: city_name(
                self.distributions.nations().get_value(nation_index),
                self.city_random.next_value(),
            ),
            c_nation: self.distributions.nations().get_value(nation_index),
            c_region: region_of(self.distributions, nation_index),
            c_phone: self.phone_random.next_value(nation_index as i64),
            c_mktsegment: self.market_segment_random.next_value(),
        }
    }
}

impl Iterator for CustomerIterator {
    type Item = Customer;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.row_count {
            return None;
        }
        let customer = self.make_customer(self.index + 1);

        self.address_random.row_finished();
        self.nation_random.row_finished();
        self.city_random.row_finished();
        self.phone_random.row_finished();
        self.market_segment_random.row_finished();

        self.index += 1;
        Some(customer)
    }
}
