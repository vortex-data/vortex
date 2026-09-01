// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The `part` dimension.
//!
//! SSB replaces TPC-H's `p_brand` with a three-level hierarchy (`p_mfgr` ⊃ `p_category` ⊃
//! `p_brand1`, each a string prefix of the next) and splits the first color out of `p_name` into
//! its own `p_color` column, so queries can filter a color equality instead of a substring
//! match. `p_retailprice` and `p_comment` are dropped.

use std::fmt;

use tpchgen::distribution::Distributions;
use tpchgen::random::RandomBoundedInt;
use tpchgen::random::RandomString;
use tpchgen::random::RowRandomInt;

use crate::ssb::ssbgen::distribution_size;
use crate::ssb::ssbgen::part_row_count;
use crate::ssb::ssbgen::permute_indices;

/// A `part` row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Part {
    /// Primary key, `1..=`[`part_row_count`].
    pub p_partkey: i64,
    /// Two color names, space separated (the first color goes to [`Part::p_color`]).
    pub p_name: String,
    /// `MFGR#<m>`, `m` in `1..=5`.
    pub p_mfgr: String,
    /// [`Part::p_mfgr`] plus a category digit in `1..=5`, e.g. `MFGR#11`.
    pub p_category: String,
    /// [`Part::p_category`] plus a brand number in `1..=40`, e.g. `MFGR#1121`.
    pub p_brand1: String,
    /// A single color name.
    pub p_color: &'static str,
    pub p_type: &'static str,
    pub p_size: i32,
    pub p_container: &'static str,
}

impl fmt::Display for Part {
    /// The reference generator's `.tbl` line for this row.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}|{}|{}|{}|{}|{}|{}|{}|{}|",
            self.p_partkey,
            self.p_name,
            self.p_mfgr,
            self.p_category,
            self.p_brand1,
            self.p_color,
            self.p_type,
            self.p_size,
            self.p_container,
        )
    }
}

/// Generator for the `part` dimension.
#[derive(Debug, Clone)]
pub struct PartGenerator {
    scale_factor: f64,
}

impl PartGenerator {
    /// Colors drawn per row: one for `p_color` and two for `p_name`.
    const NAME_COLORS: usize = 3;
    const MANUFACTURER_MIN: i32 = 1;
    const MANUFACTURER_MAX: i32 = 5;
    const CATEGORY_MIN: i32 = 1;
    const CATEGORY_MAX: i32 = 5;
    const BRAND_MIN: i32 = 1;
    const BRAND_MAX: i32 = 40;
    const SIZE_MIN: i32 = 1;
    const SIZE_MAX: i32 = 50;

    /// Create a generator for `scale_factor`.
    pub fn new(scale_factor: f64) -> Self {
        Self { scale_factor }
    }

    /// Number of rows this generator yields.
    pub fn row_count(&self) -> i64 {
        part_row_count(self.scale_factor)
    }

    /// Iterate the rows, in primary key order.
    pub fn iter(&self) -> PartIterator {
        PartIterator::new(Distributions::static_default(), self.row_count())
    }
}

impl IntoIterator for PartGenerator {
    type Item = Part;
    type IntoIter = PartIterator;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Iterator over [`Part`] rows.
#[derive(Debug)]
pub struct PartIterator {
    distributions: &'static Distributions,
    /// `P_NAME_SD`. Held as a raw stream because SSB needs `dbgen`'s original full-range
    /// permutation rather than [`tpchgen::random::RandomStringSequence`]; the 92 draws per row
    /// match the reference `Seed[P_NAME_SD].boundary`.
    name_random: RowRandomInt,
    manufacturer_random: RandomBoundedInt,
    category_random: RandomBoundedInt,
    brand_random: RandomBoundedInt,
    type_random: RandomString<'static>,
    size_random: RandomBoundedInt,
    container_random: RandomString<'static>,

    row_count: i64,
    index: i64,
}

impl PartIterator {
    fn new(distributions: &'static Distributions, row_count: i64) -> Self {
        let colors = distribution_size(distributions.part_colors());
        Self {
            distributions,
            name_random: RowRandomInt::new(709314158, colors),
            manufacturer_random: RandomBoundedInt::new(
                1,
                PartGenerator::MANUFACTURER_MIN,
                PartGenerator::MANUFACTURER_MAX,
            ),
            category_random: RandomBoundedInt::new(
                637858759,
                PartGenerator::CATEGORY_MIN,
                PartGenerator::CATEGORY_MAX,
            ),
            brand_random: RandomBoundedInt::new(
                46831694,
                PartGenerator::BRAND_MIN,
                PartGenerator::BRAND_MAX,
            ),
            type_random: RandomString::new(1841581359, distributions.part_types()),
            size_random: RandomBoundedInt::new(
                1193163244,
                PartGenerator::SIZE_MIN,
                PartGenerator::SIZE_MAX,
            ),
            container_random: RandomString::new(727633698, distributions.part_containers()),
            row_count,
            index: 0,
        }
    }

    fn make_part(&mut self, p_partkey: i64) -> Part {
        let colors = self.distributions.part_colors();
        let permutation = permute_indices(distribution_size(colors), &mut self.name_random);
        let color = colors.get_value(permutation[0]);
        let name = permutation[1..PartGenerator::NAME_COLORS]
            .iter()
            .map(|&i| colors.get_value(i))
            .collect::<Vec<_>>()
            .join(" ");

        let p_mfgr = format!("MFGR#{}", self.manufacturer_random.next_value());
        let p_category = format!("{p_mfgr}{}", self.category_random.next_value());
        let p_brand1 = format!("{p_category}{}", self.brand_random.next_value());

        Part {
            p_partkey,
            p_name: name,
            p_mfgr,
            p_category,
            p_brand1,
            p_color: color,
            p_type: self.type_random.next_value(),
            p_size: self.size_random.next_value(),
            p_container: self.container_random.next_value(),
        }
    }
}

impl Iterator for PartIterator {
    type Item = Part;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.row_count {
            return None;
        }
        let part = self.make_part(self.index + 1);

        self.name_random.row_finished();
        self.manufacturer_random.row_finished();
        self.category_random.row_finished();
        self.brand_random.row_finished();
        self.type_random.row_finished();
        self.size_random.row_finished();
        self.container_random.row_finished();

        self.index += 1;
        Some(part)
    }
}
