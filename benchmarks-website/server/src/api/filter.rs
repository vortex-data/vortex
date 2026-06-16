// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Filter chip universe — distinct engines and formats observed across the
//! fact tables, surfaced to the global filter bar.

use std::collections::BTreeSet;

use anyhow::Result;
use duckdb::Connection;

use super::dto::FilterUniverse;

/// Collect the set of distinct engines and formats observed across the fact
/// tables. Used by the landing page to seed the global filter bar's chip
/// universe, so adding a new engine or format in ingest automatically
/// surfaces a chip without a code change.
///
/// Engines come from `query_measurements` only — the other fact tables don't
/// record an engine. Formats are unioned across `query_measurements`,
/// `compression_times`, `compression_sizes`, and `random_access_times`;
/// `vector_search_runs` is intentionally excluded because its `flavor`
/// column is not a format in the same sense the chip filter is matching on.
pub fn collect_filter_universe(conn: &Connection) -> Result<FilterUniverse> {
    let mut engines: BTreeSet<String> = BTreeSet::new();
    let mut formats: BTreeSet<String> = BTreeSet::new();

    let mut stmt =
        conn.prepare("SELECT DISTINCT engine FROM query_measurements WHERE engine IS NOT NULL")?;
    for row in stmt.query_map([], |r| r.get::<_, String>(0))? {
        engines.insert(row?);
    }

    for sql in [
        "SELECT DISTINCT format FROM query_measurements   WHERE format IS NOT NULL",
        "SELECT DISTINCT format FROM compression_times    WHERE format IS NOT NULL",
        "SELECT DISTINCT format FROM compression_sizes    WHERE format IS NOT NULL",
        "SELECT DISTINCT format FROM random_access_times  WHERE format IS NOT NULL",
    ] {
        let mut stmt = conn.prepare(sql)?;
        for row in stmt.query_map([], |r| r.get::<_, String>(0))? {
            formats.insert(row?);
        }
    }

    Ok(FilterUniverse {
        engines: engines.into_iter().collect(),
        formats: formats.into_iter().collect(),
    })
}
