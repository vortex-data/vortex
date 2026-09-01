// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The `supplier` dimension.
//!
//! SSB denormalizes TPC-H's `s_nationkey` into `s_nation` and `s_region` *names* and adds
//! `s_city`, giving queries a three-level geography to filter on without a join. `s_acctbal` and
//! `s_comment` are dropped, so none of the Better Business Bureau comment machinery survives.
//!
//! Two streams here are shared with [`customer`](super::customer), which continues them; see
//! that module.

use std::fmt;

use tpchgen::distribution::Distributions;
use tpchgen::random::PhoneNumberInstance;
use tpchgen::random::RandomBoundedInt;
use tpchgen::random::RandomPhoneNumber;

use crate::ssb::ssbgen::AlphaNumeric;
use crate::ssb::ssbgen::city_name;
use crate::ssb::ssbgen::distribution_size;
use crate::ssb::ssbgen::region_of;
use crate::ssb::ssbgen::supplier_row_count;

/// Seed of the address stream, `S_ADDR_SD`.
pub(super) const ADDRESS_SEED: i64 = 706178559;
/// Seed of the nation stream, `S_NTRG_SD`.
pub(super) const NATION_SEED: i64 = 110356601;
/// Average address length. SSB shortens TPC-H's 25 to 15, so addresses run 6-24 characters.
pub(super) const ADDRESS_AVERAGE_LENGTH: i32 = 15;

/// A `supplier` row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Supplier {
    /// Primary key, `1..=`[`supplier_row_count`].
    pub s_suppkey: i64,
    /// `Supplier#000000001`.
    pub s_name: String,
    pub s_address: String,
    /// Nine characters of nation name, space padded, plus a digit: `PERU     9`.
    pub s_city: String,
    pub s_nation: &'static str,
    pub s_region: &'static str,
    pub s_phone: PhoneNumberInstance,
}

impl fmt::Display for Supplier {
    /// The reference generator's `.tbl` line for this row.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}|{}|{}|{}|{}|{}|{}|",
            self.s_suppkey,
            self.s_name,
            self.s_address,
            self.s_city,
            self.s_nation,
            self.s_region,
            self.s_phone,
        )
    }
}

/// Generator for the `supplier` dimension.
#[derive(Debug, Clone)]
pub struct SupplierGenerator {
    scale_factor: f64,
}

impl SupplierGenerator {
    /// Create a generator for `scale_factor`.
    pub fn new(scale_factor: f64) -> Self {
        Self { scale_factor }
    }

    /// Number of rows this generator yields.
    pub fn row_count(&self) -> i64 {
        supplier_row_count(self.scale_factor)
    }

    /// Iterate the rows, in primary key order.
    pub fn iter(&self) -> SupplierIterator {
        SupplierIterator::new(Distributions::static_default(), self.row_count())
    }
}

impl IntoIterator for SupplierGenerator {
    type Item = Supplier;
    type IntoIter = SupplierIterator;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Iterator over [`Supplier`] rows.
#[derive(Debug)]
pub struct SupplierIterator {
    distributions: &'static Distributions,
    address_random: AlphaNumeric,
    nation_random: RandomBoundedInt,
    city_random: RandomBoundedInt,
    phone_random: RandomPhoneNumber,

    row_count: i64,
    index: i64,
}

impl SupplierIterator {
    fn new(distributions: &'static Distributions, row_count: i64) -> Self {
        Self {
            distributions,
            address_random: AlphaNumeric::new(ADDRESS_SEED, ADDRESS_AVERAGE_LENGTH),
            nation_random: RandomBoundedInt::new(
                NATION_SEED,
                0,
                distribution_size(distributions.nations()) - 1,
            ),
            city_random: super::customer::city_random(),
            phone_random: super::customer::phone_random(),
            row_count,
            index: 0,
        }
    }

    fn make_supplier(&mut self, s_suppkey: i64) -> Supplier {
        let nation_index = self.nation_random.next_value() as usize;
        Supplier {
            s_suppkey,
            s_name: format!("Supplier#{s_suppkey:09}"),
            s_address: self.address_random.next_value(),
            s_city: city_name(
                self.distributions.nations().get_value(nation_index),
                self.city_random.next_value(),
            ),
            s_nation: self.distributions.nations().get_value(nation_index),
            s_region: region_of(self.distributions, nation_index),
            s_phone: self.phone_random.next_value(nation_index as i64),
        }
    }
}

impl Iterator for SupplierIterator {
    type Item = Supplier;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.row_count {
            return None;
        }
        let supplier = self.make_supplier(self.index + 1);

        self.address_random.row_finished();
        self.nation_random.row_finished();
        self.city_random.row_finished();
        self.phone_random.row_finished();

        self.index += 1;
        Some(supplier)
    }
}
