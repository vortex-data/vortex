// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The `dwdate` dimension: a fixed 2557-row calendar covering 1992-01-01 through 1998-12-31.
//!
//! This is the one SSB table with no TPC-H analogue, and the only one that draws no random
//! values — every column is a function of the day. The table is named `date` in the reference
//! `.tbl` output; both DataFusion's and DuckDB's parsers reserve that word, so it is registered
//! as `dwdate`, which is what the reference load scripts call it for the same reason.
//!
//! Two reference behaviors this reproduces:
//!
//! * `d_dayofweek` and the two week-related flags run a day ahead of the real calendar: the
//!   reference computes the weekday as `(tm_wday + 1) % 7 + 1`, so 1992-01-01, a Wednesday, is
//!   labelled Thursday.
//! * The calendar comes from `localtime()` and so depends on the host timezone. Ours is frozen to
//!   GMT.

use std::fmt;

use tpchgen::dates::MIN_GENERATE_DATE;
use tpchgen::dates::TPCHDate;

use crate::ssb::ssbgen::DWDATE_ROWS;

/// Month names, as the reference spells them.
const MONTH_NAMES: [&str; 12] = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
];

/// Weekday names, indexed by the reference's shifted `d_daynuminweek - 1`.
const WEEKDAY_NAMES: [&str; 7] = [
    "Sunday",
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
];

/// Cumulative days before each month in a non-leap year.
const MONTH_DAY_START: [i32; 12] = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];

/// Days in each month of a non-leap year.
const MONTH_LENGTHS: [i32; 12] = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];

/// `(name, start month, start day, end month, end day)`, tested in this order with the first
/// match winning. The windows are wide: `Christmas` spans November and December.
const SEASONS: [(&str, i32, i32, i32, i32); 5] = [
    ("Christmas", 11, 1, 12, 31),
    ("Summer", 5, 1, 8, 31),
    ("Winter", 1, 1, 3, 31),
    ("Spring", 4, 1, 4, 30),
    ("Fall", 9, 1, 10, 31),
];

/// `(month, day)` of each flagged holiday.
const HOLIDAYS: [(i32, i32); 10] = [
    (12, 24),
    (1, 1),
    (2, 20),
    (4, 20),
    (5, 20),
    (7, 20),
    (8, 20),
    (9, 20),
    (10, 20),
    (11, 20),
];

/// Weekday index of 1992-01-01, the first day of the calendar, counting Sunday as 0. It was a
/// Wednesday.
const FIRST_DAY_OF_WEEK: i32 = 3;

/// Whether `year` is a leap year, by the reference's rule (which ignores the 400-year
/// correction; immaterial over 1992-1998).
fn is_leap(year: i32) -> bool {
    year % 4 == 0 && year % 100 != 0
}

/// A `dwdate` row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Dwdate {
    /// `yyyymmdd`, and the join key `lo_orderdate`/`lo_commitdate` reference.
    pub d_datekey: i32,
    /// `January 1, 1992`.
    pub d_date: String,
    /// Weekday name, one day ahead of the real calendar.
    pub d_dayofweek: &'static str,
    pub d_month: &'static str,
    pub d_year: i32,
    /// `yyyymm` as an integer.
    pub d_yearmonthnum: i32,
    /// `Jan1992`.
    pub d_yearmonth: String,
    /// `1..=7`, Sunday being 1, shifted a day ahead of the real calendar.
    pub d_daynuminweek: i32,
    pub d_daynuminmonth: i32,
    pub d_daynuminyear: i32,
    pub d_monthnuminyear: i32,
    pub d_weeknuminyear: i32,
    pub d_sellingseason: &'static str,
    pub d_lastdayinweekfl: i32,
    pub d_lastdayinmonthfl: i32,
    pub d_holidayfl: i32,
    pub d_weekdayfl: i32,
}

impl fmt::Display for Dwdate {
    /// The reference generator's `.tbl` line for this row.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|",
            self.d_datekey,
            self.d_date,
            self.d_dayofweek,
            self.d_month,
            self.d_year,
            self.d_yearmonthnum,
            self.d_yearmonth,
            self.d_daynuminweek,
            self.d_daynuminmonth,
            self.d_daynuminyear,
            self.d_monthnuminyear,
            self.d_weeknuminyear,
            self.d_sellingseason,
            self.d_lastdayinweekfl,
            self.d_lastdayinmonthfl,
            self.d_holidayfl,
            self.d_weekdayfl,
        )
    }
}

/// Generator for the `dwdate` dimension. The calendar does not scale.
#[derive(Debug, Clone, Default)]
pub struct DwdateGenerator;

impl DwdateGenerator {
    /// Create a generator.
    pub fn new() -> Self {
        Self
    }

    /// Number of rows this generator yields, always [`DWDATE_ROWS`].
    pub fn row_count(&self) -> i64 {
        DWDATE_ROWS
    }

    /// Iterate the calendar in date order.
    pub fn iter(&self) -> DwdateIterator {
        DwdateIterator {
            index: 0,
            row_count: self.row_count(),
        }
    }
}

impl IntoIterator for DwdateGenerator {
    type Item = Dwdate;
    type IntoIter = DwdateIterator;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Iterator over [`Dwdate`] rows.
#[derive(Debug)]
pub struct DwdateIterator {
    index: i32,
    row_count: i64,
}

impl Iterator for DwdateIterator {
    type Item = Dwdate;

    fn next(&mut self) -> Option<Self::Item> {
        if i64::from(self.index) >= self.row_count {
            return None;
        }
        let date = make_dwdate(self.index);
        self.index += 1;
        Some(date)
    }
}

/// Build the row for `offset` days after 1992-01-01.
fn make_dwdate(offset: i32) -> Dwdate {
    let (short_year, month, day) = TPCHDate::new(MIN_GENERATE_DATE + offset).to_ymd();
    let year = 1900 + short_year;

    // The reference's shifted weekday: one more than the true day of the week.
    let d_daynuminweek = (FIRST_DAY_OF_WEEK + offset + 1) % 7 + 1;
    let month_name = MONTH_NAMES[(month - 1) as usize];

    let leap_day = i32::from(is_leap(year) && month > 2);
    let d_daynuminyear = MONTH_DAY_START[(month - 1) as usize] + day + leap_day;

    let last_day_in_month = if month == 2 && is_leap(year) {
        29
    } else {
        MONTH_LENGTHS[(month - 1) as usize]
    };

    Dwdate {
        d_datekey: year * 10000 + month * 100 + day,
        d_date: format!("{month_name} {day}, {year}"),
        d_dayofweek: WEEKDAY_NAMES[(d_daynuminweek - 1) as usize],
        d_month: month_name,
        d_year: year,
        d_yearmonthnum: year * 100 + month,
        d_yearmonth: format!("{}{year}", &month_name[..3]),
        d_daynuminweek,
        d_daynuminmonth: day,
        d_daynuminyear,
        d_monthnuminyear: month,
        d_weeknuminyear: d_daynuminyear / 7 + 1,
        d_sellingseason: selling_season(month, day),
        d_lastdayinweekfl: i32::from(d_daynuminweek == 7),
        d_lastdayinmonthfl: i32::from(day == last_day_in_month),
        d_holidayfl: i32::from(HOLIDAYS.contains(&(month, day))),
        d_weekdayfl: i32::from(d_daynuminweek != 1 && d_daynuminweek != 7),
    }
}

/// The first [`SEASONS`] window containing `(month, day)`.
fn selling_season(month: i32, day: i32) -> &'static str {
    SEASONS
        .iter()
        .find(|(_, start_month, start_day, end_month, end_day)| {
            month >= *start_month && month <= *end_month && day >= *start_day && day <= *end_day
        })
        .map(|(name, ..)| *name)
        .unwrap_or("")
}
