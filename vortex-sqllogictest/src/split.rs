// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Splitting an `.slt` file into shared setup, per-query fragments and teardown.
//!
//! The generated-data suites (TPC-H, TPC-DS, ClickBench) are single files that
//! pull in one fragment per query with glob includes such as
//! `include ../results/*.slt.no`. [`split_glob_includes`] separates those
//! fragments from the records around them, so a runner can set up the database
//! once and then run the fragments concurrently.

use std::path::Path;

use sqllogictest::ColumnType;
use sqllogictest::Injected;
use sqllogictest::Record;

/// The records of one file matched by a top-level glob include.
#[derive(Debug, Clone, PartialEq)]
pub struct Fragment<T: ColumnType> {
    /// The fragment's file and parent directory, e.g. `results/q1.slt.no`.
    pub name: String,
    /// The fragment's records, including its `BeginInclude` and `EndInclude` markers.
    pub records: Vec<Record<T>>,
}

impl<T: ColumnType> Fragment<T> {
    /// Whether the fragment only runs queries, so it neither changes the database
    /// or session nor depends on the order it runs in relative to other fragments.
    pub fn is_read_only(&self) -> bool {
        self.records.iter().all(|record| {
            matches!(
                record,
                Record::Query { .. } | Record::Injected(_) | Record::Comment(_) | Record::Newline
            )
        })
    }
}

/// An `.slt` file split around its glob-included fragments.
#[derive(Debug, Clone, PartialEq)]
pub struct Split<T: ColumnType> {
    /// The records before the first fragment.
    pub prelude: Vec<Record<T>>,
    /// The fragments, in file order.
    pub fragments: Vec<Fragment<T>>,
    /// The records after the last fragment.
    pub epilogue: Vec<Record<T>>,
}

impl<T: ColumnType> Split<T> {
    /// The prelude records that set state held by the runner rather than the
    /// database, such as `let` variables and `control` settings. A fresh runner
    /// replays these to see the prelude the way the runner that ran it does.
    pub fn runner_state(&self) -> impl Iterator<Item = Record<T>> + '_ {
        self.prelude
            .iter()
            .filter(|record| {
                matches!(
                    record,
                    Record::Control(_) | Record::Let { .. } | Record::HashThreshold { .. }
                )
            })
            .cloned()
    }
}

/// Splits parsed records around the fragments of top-level glob includes.
///
/// Returns the records unchanged when the file has fewer than two fragments, or
/// when a record other than a fragment sits between two fragments: such a record
/// may set state that only the later fragments expect.
pub fn split_glob_includes<T: ColumnType>(
    records: Vec<Record<T>>,
) -> Result<Split<T>, Vec<Record<T>>> {
    let segments = segment(records.clone());
    let first = segments
        .iter()
        .position(|s| matches!(s, Segment::Fragment(_)));
    let last = segments
        .iter()
        .rposition(|s| matches!(s, Segment::Fragment(_)));
    let (Some(first), Some(last)) = (first, last) else {
        return Err(records);
    };
    if segments[first..=last]
        .iter()
        .any(|s| matches!(s, Segment::Shared(_)))
        || first == last
    {
        return Err(records);
    }

    let mut split = Split {
        prelude: Vec::new(),
        fragments: Vec::new(),
        epilogue: Vec::new(),
    };
    for (i, segment) in segments.into_iter().enumerate() {
        match segment {
            Segment::Fragment(fragment) => split.fragments.push(fragment),
            Segment::Shared(record) if i < first => split.prelude.push(record),
            Segment::Shared(record) => split.epilogue.push(record),
        }
    }
    Ok(split)
}

enum Segment<T: ColumnType> {
    Shared(Record<T>),
    Fragment(Fragment<T>),
}

/// Groups records into shared records and fragments of top-level glob includes.
///
/// The parser emits an `include` record followed, for each file its pattern
/// matches, by that file's records between `BeginInclude` and `EndInclude`.
fn segment<T: ColumnType>(records: Vec<Record<T>>) -> Vec<Segment<T>> {
    let mut segments = Vec::new();
    let mut depth = 0usize;
    let mut after_glob = false;
    let mut fragment: Option<Fragment<T>> = None;

    for record in records {
        let delta: isize = match &record {
            Record::Injected(Injected::BeginInclude(_)) => 1,
            Record::Injected(Injected::EndInclude(_)) => -1,
            _ => 0,
        };

        if let Some(open) = fragment.as_mut() {
            open.records.push(record);
            depth = depth.saturating_add_signed(delta);
            if depth == 0
                && let Some(done) = fragment.take()
            {
                segments.push(Segment::Fragment(done));
            }
            continue;
        }

        if depth == 0 {
            match &record {
                Record::Include { filename, .. } => {
                    after_glob = filename.contains(['*', '?', '[']);
                    // The include record itself does nothing when run.
                    continue;
                }
                Record::Injected(Injected::BeginInclude(name)) if after_glob => {
                    fragment = Some(Fragment {
                        name: fragment_name(name),
                        records: vec![record],
                    });
                    depth = 1;
                    continue;
                }
                Record::Injected(Injected::BeginInclude(_)) => {}
                _ => after_glob = false,
            }
        }

        depth = depth.saturating_add_signed(delta);
        segments.push(Segment::Shared(record));
    }

    segments
}

/// Names a fragment by its file and parent directory, e.g. `results/q1.slt.no`.
fn fragment_name(path: &str) -> String {
    let path = Path::new(path);
    let file = path.file_name().unwrap_or_default().to_string_lossy();
    match path.parent().and_then(Path::file_name) {
        Some(dir) => format!("{}/{file}", dir.to_string_lossy()),
        None => file.into_owned(),
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use datafusion_sqllogictest::DFColumnType;
    use sqllogictest::parse_file;

    use super::*;

    fn sql(records: &[Record<DFColumnType>]) -> Vec<String> {
        records
            .iter()
            .filter_map(|r| match r {
                Record::Statement { sql, .. } | Record::Query { sql, .. } => Some(sql.clone()),
                _ => None,
            })
            .collect()
    }

    fn write(dir: &Path, name: &str, body: &str) -> anyhow::Result<()> {
        let path = dir.join(name);
        fs::create_dir_all(path.parent().unwrap_or(dir))?;
        fs::write(path, body)?;
        Ok(())
    }

    const QUERY: &str = "query I\nSELECT 1\n----\n1\n";

    #[test]
    fn splits_setup_fragments_and_teardown() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let root = dir.path();
        write(root, "create.slt.no", "statement ok\nCREATE x\n")?;
        write(root, "drop.slt.no", "statement ok\nDROP x\n")?;
        write(root, "results/q1.slt.no", QUERY)?;
        write(root, "results/q2.slt.no", "statement ok\nSET y\n")?;
        write(root, "plans/q1.slt.no", QUERY)?;
        write(
            root,
            "suite.slt",
            "control substitution on\n\nlet V\nSELECT 1\n\ninclude ./create.slt.no\n\
             include ./results/*.slt.no\ninclude ./plans/*.slt.no\ninclude ./drop.slt.no\n",
        )?;

        let split = split_glob_includes(parse_file::<DFColumnType>(root.join("suite.slt"))?)
            .map_err(|_| anyhow::anyhow!("expected a split"))?;

        assert_eq!(sql(&split.prelude), ["CREATE x"]);
        assert_eq!(split.runner_state().count(), 2);
        assert_eq!(sql(&split.epilogue), ["DROP x"]);
        let fragments: Vec<_> = split
            .fragments
            .iter()
            .map(|f| (f.name.as_str(), f.is_read_only()))
            .collect();
        assert_eq!(
            fragments,
            [
                ("results/q1.slt.no", true),
                ("results/q2.slt.no", false),
                ("plans/q1.slt.no", true),
            ]
        );
        Ok(())
    }

    #[rstest::rstest]
    #[case::single_fragment("include ./plans/*.slt.no\n")]
    #[case::no_glob("include ./results/q1.slt.no\ninclude ./results/q2.slt.no\n")]
    #[case::record_between_fragments(
        "include ./results/q1.slt.no\ninclude ./results/*.slt.no\nstatement ok\nSET y\n\n\
         include ./plans/*.slt.no\n"
    )]
    fn keeps_files_it_cannot_split_whole(#[case] suite: &str) -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let root = dir.path();
        write(root, "results/q1.slt.no", QUERY)?;
        write(root, "results/q2.slt.no", QUERY)?;
        write(root, "plans/q1.slt.no", QUERY)?;
        write(root, "suite.slt", suite)?;

        let records = parse_file::<DFColumnType>(root.join("suite.slt"))?;
        assert_eq!(split_glob_includes(records.clone()), Err(records));
        Ok(())
    }
}
