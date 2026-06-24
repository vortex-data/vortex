// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Scratch-directory handling and output normalization.
//!
//! Tests reference a per-test working directory through the `${WORK_DIR}`
//! substitution variable. The runner sets `WORK_DIR` (via `Runner::set_var`) to a
//! unique directory beneath [`scratch_root`] for each test, and [`PathNormalizing`]
//! rewrites that path back to the `${WORK_DIR}` token in query output. Normalizing
//! at the source keeps comparisons stable and makes `--complete` write the portable
//! token rather than a machine-specific path.

use std::path::PathBuf;
use std::process::Command;
use std::sync::LazyLock;
use std::time::Duration;

use async_trait::async_trait;
use sqllogictest::DBOutput;
use sqllogictest::runner::AsyncDB;

/// The substitution variable tests use for their scratch directory: `${WORK_DIR}`.
pub const WORK_DIR_VAR: &str = "WORK_DIR";

/// The token that scratch paths are normalized to in query output. Matches the
/// `${WORK_DIR}` substitution variable so expected output reads naturally.
pub const WORK_DIR_TOKEN: &str = "${WORK_DIR}";

/// Root of the scratch directory used for test artifacts, inside this crate.
///
/// Each runner process uses a unique subdirectory beneath this root, and each test
/// a unique directory beneath that.
pub fn scratch_root() -> PathBuf {
    static SCRATCH_DIR: LazyLock<PathBuf> =
        LazyLock::new(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("scratch"));

    SCRATCH_DIR.clone()
}

/// Replaces every occurrence of `work_dir` in `cell` with [`WORK_DIR_TOKEN`].
pub fn normalize_work_dir(cell: &str, work_dir: &str) -> String {
    cell.replace(work_dir, WORK_DIR_TOKEN)
}

/// Wraps an [`AsyncDB`] and rewrites this test's scratch path in query output back
/// to `${WORK_DIR}`.
///
/// `--complete` writes raw query rows, so normalizing here (rather than only in the
/// comparison normalizer) is what keeps completed expected output stable.
pub struct PathNormalizing<D> {
    inner: D,
    work_dir: String,
}

impl<D> PathNormalizing<D> {
    /// Wraps `inner`, rewriting `work_dir` to `${WORK_DIR}` in its output.
    pub fn new(inner: D, work_dir: impl Into<String>) -> Self {
        Self {
            inner,
            work_dir: work_dir.into(),
        }
    }
}

#[async_trait]
impl<D: AsyncDB + Send> AsyncDB for PathNormalizing<D> {
    type Error = D::Error;
    type ColumnType = D::ColumnType;

    async fn run(&mut self, sql: &str) -> Result<DBOutput<Self::ColumnType>, Self::Error> {
        Ok(match self.inner.run(sql).await? {
            DBOutput::Rows { types, rows } => DBOutput::Rows {
                types,
                rows: rows
                    .into_iter()
                    .map(|row| {
                        row.iter()
                            .map(|cell| normalize_work_dir(cell, &self.work_dir))
                            .collect()
                    })
                    .collect(),
            },
            other => other,
        })
    }

    async fn shutdown(&mut self) {
        self.inner.shutdown().await
    }

    fn engine_name(&self) -> &str {
        self.inner.engine_name()
    }

    async fn sleep(dur: Duration) {
        D::sleep(dur).await
    }

    async fn run_command(command: Command) -> std::io::Result<std::process::Output> {
        D::run_command(command).await
    }

    fn error_sql_state(err: &Self::Error) -> Option<String> {
        D::error_sql_state(err)
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::WORK_DIR_TOKEN;
    use super::normalize_work_dir;

    #[rstest]
    #[case("no paths here", "no paths here")]
    #[case("plain 42", "plain 42")]
    fn leaves_unrelated_text_untouched(#[case] input: &str, #[case] expected: &str) {
        assert_eq!(
            normalize_work_dir(input, "/repo/scratch/123/df_create.slt"),
            expected
        );
    }

    #[test]
    fn rewrites_work_dir_to_token() {
        let work_dir = "/repo/scratch/123/df_create.slt";
        let cell = format!("output_url={work_dir}/sink/data1.vortex foo");
        assert_eq!(
            normalize_work_dir(&cell, work_dir),
            format!("output_url={WORK_DIR_TOKEN}/sink/data1.vortex foo")
        );
    }
}
