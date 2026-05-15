// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Server-side commit-window cap used by every chart query.
//!
//! Visual downsampling is the *client's* job (see
//! `static/chart-init.js`) — this module only decides how many of the
//! most recent commits to load from DuckDB.

use std::num::NonZeroU32;

use serde::Deserialize;

use super::dto::DEFAULT_COMMIT_WINDOW;

/// Server-side cap on how many of the most recent commits a chart includes.
///
/// `Last(n)` keeps the most recent `n` commits by `commits.timestamp`; `All`
/// returns every commit ever ingested.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CommitWindow {
    /// Keep the most recent `n` commits.
    Last(NonZeroU32),
    /// No cap.
    All,
}

impl Default for CommitWindow {
    fn default() -> Self {
        Self::Last(NonZeroU32::new(DEFAULT_COMMIT_WINDOW).expect("non-zero default"))
    }
}

impl CommitWindow {
    /// Parse the `?n=...` query string parameter. `None` and malformed values
    /// fall back to [`CommitWindow::default`]. `"all"` (any case) means
    /// unbounded. Numeric values are floored to `1` so `?n=0` becomes
    /// `?n=1`; there is no upper bound — large histories are kept as-is.
    /// Any further reduction in rendered point count happens client-side
    /// (see `static/chart-init.js` for the LTTB pass on the visible
    /// commit range).
    pub fn parse(raw: Option<&str>) -> Self {
        let Some(s) = raw else {
            return Self::default();
        };
        let trimmed = s.trim();
        if trimmed.eq_ignore_ascii_case("all") {
            return Self::All;
        }
        trimmed
            .parse::<u32>()
            .ok()
            .map(|v| v.max(1))
            .and_then(NonZeroU32::new)
            .map(Self::Last)
            .unwrap_or_default()
    }

    /// SQL fragment to splice into chart queries that filters `commits c` to
    /// just the most recent `n` commits. Empty for `All`. The placeholder is
    /// satisfied by [`Self::limit_param`] so the LIMIT value travels as a
    /// bound parameter rather than an interpolated integer.
    pub(crate) fn sql_filter(&self) -> &'static str {
        match self {
            Self::All => "",
            Self::Last(_) => {
                " AND c.commit_sha IN \
                 (SELECT commit_sha FROM commits ORDER BY timestamp DESC, commit_sha DESC LIMIT ?)"
            }
        }
    }

    /// Bound parameter for the `LIMIT ?` placeholder produced by
    /// [`Self::sql_filter`]. `None` for [`Self::All`] (no extra `?` to bind).
    pub(crate) fn limit_param(&self) -> Option<i64> {
        match self {
            Self::All => None,
            Self::Last(n) => Some(i64::from(n.get())),
        }
    }

    /// Render this window as the value the URL would carry (`"100"` /
    /// `"all"`). Used by the HTML toolbar to mark the active scope.
    pub fn url_value(&self) -> String {
        match self {
            Self::All => "all".into(),
            Self::Last(n) => n.get().to_string(),
        }
    }
}

/// Query string for `/api/chart/{slug}` and `/chart/{slug}`. Only `?n=`
/// affects the JSON response; per-chart UI state (Y axis, slider) is local
/// to `chart-init.js` and intentionally not in the URL.
#[derive(Debug, Default, Deserialize)]
pub struct ChartQuery {
    /// Commit window: `25`, `50`, `100`, `250`, `all`, etc.
    pub n: Option<String>,
}

impl ChartQuery {
    /// Resolved [`CommitWindow`] from the raw `n` parameter.
    pub fn window(&self) -> CommitWindow {
        CommitWindow::parse(self.n.as_deref())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn commit_window_parse_defaults() {
        let CommitWindow::Last(n) = CommitWindow::parse(None) else {
            panic!("default should be Last");
        };
        assert_eq!(n.get(), DEFAULT_COMMIT_WINDOW);
    }

    #[test]
    fn commit_window_parse_all() {
        assert!(matches!(
            CommitWindow::parse(Some("all")),
            CommitWindow::All
        ));
        assert!(matches!(
            CommitWindow::parse(Some("ALL")),
            CommitWindow::All
        ));
        assert!(matches!(
            CommitWindow::parse(Some(" all ")),
            CommitWindow::All
        ));
    }

    #[test]
    fn commit_window_parse_numeric() {
        let CommitWindow::Last(n) = CommitWindow::parse(Some("50")) else {
            panic!()
        };
        assert_eq!(n.get(), 50);
    }

    #[test]
    fn commit_window_parse_floors_zero_but_keeps_large_values() {
        // Large values are kept as-is — full history is no longer clamped
        // server-side. Visual downsampling happens client-side in
        // `static/chart-init.js`, on the currently visible commit range.
        let CommitWindow::Last(n) = CommitWindow::parse(Some("99999")) else {
            panic!()
        };
        assert_eq!(n.get(), 99_999);

        // 0 floors to 1 since the underlying type is `NonZeroU32`.
        let CommitWindow::Last(n) = CommitWindow::parse(Some("0")) else {
            panic!("floor of 0 should round to 1")
        };
        assert_eq!(n.get(), 1);
    }

    #[test]
    fn commit_window_parse_malformed_falls_back() {
        let CommitWindow::Last(n) = CommitWindow::parse(Some("banana")) else {
            panic!()
        };
        assert_eq!(n.get(), DEFAULT_COMMIT_WINDOW);
        let CommitWindow::Last(n) = CommitWindow::parse(Some("")) else {
            panic!()
        };
        assert_eq!(n.get(), DEFAULT_COMMIT_WINDOW);
    }

    #[test]
    fn commit_window_url_value() {
        assert_eq!(CommitWindow::default().url_value(), "100");
        assert_eq!(CommitWindow::All.url_value(), "all");
    }

    #[test]
    fn commit_window_sql_filter_shape() {
        let CommitWindow::Last(n) = CommitWindow::parse(Some("42")) else {
            panic!()
        };
        let f = CommitWindow::Last(n).sql_filter();
        // Bound placeholder, not an interpolated integer.
        assert!(f.contains("LIMIT ?"));
        assert!(!f.contains("42"));
        assert!(CommitWindow::All.sql_filter().is_empty());
    }

    #[test]
    fn commit_window_limit_param() {
        let CommitWindow::Last(n) = CommitWindow::parse(Some("42")) else {
            panic!()
        };
        assert_eq!(CommitWindow::Last(n).limit_param(), Some(42));
        assert_eq!(CommitWindow::All.limit_param(), None);
        assert_eq!(CommitWindow::default().limit_param(), Some(100));
    }
}
