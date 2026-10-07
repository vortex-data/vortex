// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The JSONBench queries as hand-written plans: which JSON paths each query reads, which rows it
//! keeps, and how it aggregates them.
//!
//! Every format feeds rows into the same [`Partial`] aggregation, so the formats differ only in
//! how they produce the extracted path values.

use std::fmt::Write;

use arrow_array::Array;
use arrow_array::Int64Array;
use arrow_array::StringViewArray;
use rustc_hash::FxHashMap;

/// `commit.collection` values Q2 keeps.
pub const Q2_COLLECTIONS: [&str; 3] = [
    "app.bsky.feed.post",
    "app.bsky.feed.repost",
    "app.bsky.feed.like",
];

/// The collection Q3 and Q4 keep.
pub const POST_COLLECTION: &str = "app.bsky.feed.post";

/// A JSONBench query, numbered from Q0 as in `sql/jsonbench.sql`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Query {
    /// Top event types.
    Q0,
    /// Top event types with unique users, over commit creations.
    Q1,
    /// Event counts by type and hour, over post, repost and like creations.
    Q2,
    /// The three users who posted first.
    Q3,
    /// The three users with the longest posting activity span.
    Q4,
}

impl Query {
    pub fn from_index(idx: usize) -> anyhow::Result<Self> {
        Ok(match idx {
            0 => Self::Q0,
            1 => Self::Q1,
            2 => Self::Q2,
            3 => Self::Q3,
            4 => Self::Q4,
            _ => anyhow::bail!("jsonbench has no query {idx}"),
        })
    }

    /// Whether the query keeps only `kind = 'commit' AND commit.operation = 'create'` rows.
    pub fn filters_commit_creates(self) -> bool {
        self != Self::Q0
    }

    /// The `commit.collection` values the query keeps, if it filters on them.
    pub fn collection_filter(self) -> Option<&'static [&'static str]> {
        match self {
            Self::Q0 | Self::Q1 => None,
            Self::Q2 => Some(&Q2_COLLECTIONS),
            Self::Q3 | Self::Q4 => Some(&[POST_COLLECTION]),
        }
    }

    /// The paths the aggregation reads, after filtering.
    pub fn outputs(self) -> Outputs {
        match self {
            Self::Q0 => Outputs {
                collection: true,
                did: false,
                time_us: false,
            },
            Self::Q1 => Outputs {
                collection: true,
                did: true,
                time_us: false,
            },
            Self::Q2 => Outputs {
                collection: true,
                did: false,
                time_us: true,
            },
            Self::Q3 | Self::Q4 => Outputs {
                collection: false,
                did: true,
                time_us: true,
            },
        }
    }

    /// Whether a row with these filter path values passes the query's filter.
    pub fn keeps(
        self,
        kind: Option<&str>,
        operation: Option<&str>,
        collection: Option<&str>,
    ) -> bool {
        if self.filters_commit_creates() && (kind != Some("commit") || operation != Some("create"))
        {
            return false;
        }
        self.collection_filter()
            .is_none_or(|allowed| collection.is_some_and(|c| allowed.contains(&c)))
    }
}

/// The path values a query aggregates over.
#[derive(Clone, Copy, Debug)]
pub struct Outputs {
    pub collection: bool,
    pub did: bool,
    pub time_us: bool,
}

/// The string at `idx` of an optional extracted column.
pub fn str_at(array: Option<&StringViewArray>, idx: usize) -> Option<&str> {
    array.and_then(|array| array.is_valid(idx).then(|| array.value(idx)))
}

/// The integer at `idx` of an optional extracted column.
pub fn i64_at(array: Option<&Int64Array>, idx: usize) -> Option<i64> {
    array.and_then(|array| array.is_valid(idx).then(|| array.value(idx)))
}

/// One filtered row, with the path values the query aggregates.
#[derive(Clone, Copy, Debug, Default)]
pub struct Row<'a> {
    pub collection: Option<&'a str>,
    pub did: Option<&'a str>,
    pub time_us: Option<i64>,
}

/// A hash map keyed by an optional string that looks keys up without allocating.
#[derive(Debug)]
pub struct StrMap<V> {
    values: FxHashMap<Box<str>, V>,
    null: Option<V>,
}

impl<V> Default for StrMap<V> {
    fn default() -> Self {
        Self {
            values: FxHashMap::default(),
            null: None,
        }
    }
}

impl<V> StrMap<V> {
    fn get_or_insert_with(&mut self, key: Option<&str>, default: impl FnOnce() -> V) -> &mut V {
        match key {
            None => self.null.get_or_insert_with(default),
            Some(key) => {
                if !self.values.contains_key(key) {
                    self.values.insert(Box::from(key), default());
                }
                self.values.get_mut(key).expect("just inserted")
            }
        }
    }

    fn into_iter(self) -> impl Iterator<Item = (Option<Box<str>>, V)> {
        self.values
            .into_iter()
            .map(|(key, value)| (Some(key), value))
            .chain(self.null.map(|value| (None, value)))
    }

    fn merge_with(
        &mut self,
        other: Self,
        mut merge: impl FnMut(&mut V, V),
        default: impl Fn() -> V,
    ) {
        for (key, value) in other.into_iter() {
            merge(self.get_or_insert_with(key.as_deref(), &default), value);
        }
    }
}

/// Partial aggregation state of one query over some rows. Partials of disjoint rows merge.
#[derive(Debug)]
pub enum Partial {
    Q0(StrMap<u64>),
    Q1(StrMap<(u64, StrMap<()>)>),
    Q2(StrMap<FxHashMap<Option<i64>, u64>>),
    Q3(StrMap<Option<i64>>),
    Q4(StrMap<(Option<i64>, Option<i64>)>),
}

fn min_opt(a: Option<i64>, b: Option<i64>) -> Option<i64> {
    match (a, b) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (a, b) => a.or(b),
    }
}

fn max_opt(a: Option<i64>, b: Option<i64>) -> Option<i64> {
    match (a, b) {
        (Some(a), Some(b)) => Some(a.max(b)),
        (a, b) => a.or(b),
    }
}

/// UTC hour of day of a microsecond Unix timestamp, matching `date_part('hour', to_timestamp(us /
/// 1000000))`.
fn hour_of_day(time_us: i64) -> i64 {
    (time_us / 1_000_000).div_euclid(3600).rem_euclid(24)
}

impl Partial {
    pub fn new(query: Query) -> Self {
        match query {
            Query::Q0 => Self::Q0(StrMap::default()),
            Query::Q1 => Self::Q1(StrMap::default()),
            Query::Q2 => Self::Q2(StrMap::default()),
            Query::Q3 => Self::Q3(StrMap::default()),
            Query::Q4 => Self::Q4(StrMap::default()),
        }
    }

    /// Aggregate one row that passed the query's filter.
    pub fn add(&mut self, row: Row<'_>) {
        match self {
            Self::Q0(counts) => *counts.get_or_insert_with(row.collection, || 0) += 1,
            Self::Q1(groups) => {
                let (count, users) = groups.get_or_insert_with(row.collection, Default::default);
                *count += 1;
                users.get_or_insert_with(row.did, || ());
            }
            Self::Q2(counts) => {
                let hours = counts.get_or_insert_with(row.collection, Default::default);
                *hours.entry(row.time_us.map(hour_of_day)).or_default() += 1;
            }
            Self::Q3(first) => {
                let min = first.get_or_insert_with(row.did, || None);
                *min = min_opt(*min, row.time_us);
            }
            Self::Q4(spans) => {
                let (min, max) = spans.get_or_insert_with(row.did, || (None, None));
                *min = min_opt(*min, row.time_us);
                *max = max_opt(*max, row.time_us);
            }
        }
    }

    pub fn merge(&mut self, other: Self) {
        match (self, other) {
            (Self::Q0(a), Self::Q0(b)) => a.merge_with(b, |a, b| *a += b, || 0),
            (Self::Q1(a), Self::Q1(b)) => a.merge_with(
                b,
                |(count, users), (other_count, other_users)| {
                    *count += other_count;
                    users.merge_with(other_users, |_, _| {}, || ());
                },
                Default::default,
            ),
            (Self::Q2(a), Self::Q2(b)) => a.merge_with(
                b,
                |hours, other_hours| {
                    for (hour, count) in other_hours {
                        *hours.entry(hour).or_default() += count;
                    }
                },
                Default::default,
            ),
            (Self::Q3(a), Self::Q3(b)) => a.merge_with(b, |a, b| *a = min_opt(*a, b), || None),
            (Self::Q4(a), Self::Q4(b)) => a.merge_with(
                b,
                |(min, max), (other_min, other_max)| {
                    *min = min_opt(*min, other_min);
                    *max = max_opt(*max, other_max);
                },
                || (None, None),
            ),
            _ => unreachable!("partials of different queries never merge"),
        }
    }

    /// The query result rows, ordered and limited like the SQL query, one row per line.
    pub fn finish(self) -> Vec<String> {
        let show = |s: &Option<Box<str>>| s.as_deref().unwrap_or("").to_string();
        let show_i = |v: Option<i64>| v.map(|v| v.to_string()).unwrap_or_default();
        match self {
            Self::Q0(counts) => {
                let mut rows: Vec<_> = counts.into_iter().collect();
                rows.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
                rows.iter()
                    .map(|(event, cnt)| format!("{} | {cnt}", show(event)))
                    .collect()
            }
            Self::Q1(groups) => {
                let mut rows: Vec<_> = groups
                    .into_iter()
                    .map(|(event, (cnt, users))| (event, cnt, users.values.len()))
                    .collect();
                rows.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
                rows.iter()
                    .map(|(event, cnt, users)| format!("{} | {cnt} | {users}", show(event)))
                    .collect()
            }
            Self::Q2(counts) => {
                let mut rows: Vec<_> = counts
                    .into_iter()
                    .flat_map(|(event, hours)| {
                        hours
                            .into_iter()
                            .map(move |(hour, cnt)| (event.clone(), hour, cnt))
                    })
                    .collect();
                rows.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
                rows.iter()
                    .map(|(event, hour, cnt)| {
                        format!("{} | {} | {cnt}", show(event), show_i(*hour))
                    })
                    .collect()
            }
            Self::Q3(first) => {
                let mut rows: Vec<_> = first.into_iter().collect();
                rows.sort_by(|a, b| {
                    // NULLS LAST, then user_id.
                    (a.1.is_none(), a.1)
                        .cmp(&(b.1.is_none(), b.1))
                        .then_with(|| a.0.cmp(&b.0))
                });
                rows.truncate(3);
                rows.iter()
                    .map(|(user, first)| format!("{} | {}", show(user), show_i(*first)))
                    .collect()
            }
            Self::Q4(spans) => {
                let mut rows: Vec<_> = spans
                    .into_iter()
                    .map(|(user, (min, max))| {
                        (user, min.zip(max).map(|(min, max)| (max - min) / 1000))
                    })
                    .collect();
                // DESC with NULLS FIRST (the SQL default for DESC), then user_id.
                rows.sort_by(|a, b| {
                    (a.1.is_some(), std::cmp::Reverse(a.1))
                        .cmp(&(b.1.is_some(), std::cmp::Reverse(b.1)))
                        .then_with(|| a.0.cmp(&b.0))
                });
                rows.truncate(3);
                rows.iter()
                    .map(|(user, span)| format!("{} | {}", show(user), show_i(*span)))
                    .collect()
            }
        }
    }
}

/// Renders result rows for `VX_BENCH_PRINT_RESULTS` comparisons.
pub fn render(rows: &[String]) -> String {
    let mut out = String::new();
    for row in rows {
        writeln!(out, "{row}").expect("writing to a String cannot fail");
    }
    out
}
