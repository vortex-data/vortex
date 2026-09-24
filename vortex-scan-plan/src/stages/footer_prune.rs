// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! File-statistics pruning: finish without output when the footer proves no row can match.

use vortex_array::expr::Expression;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_file::pruning::can_prune_file_stats;
use vortex_session::VortexSession;

use crate::io::IoConsumer;
use crate::io::IoRequestId;
use crate::io::IoResult;
use crate::next::Next;
use crate::planner::Planner;
use crate::planner::PlannerOutput;
use crate::planner::State;
use crate::planner::WorkScope;
use crate::stages::OpenedFile;

/// Decides from the footer alone whether the file can be skipped.
///
/// Missing or inconclusive statistics retain the file; only a proof of no match rejects it. An
/// empty file is rejected outright. The stage never requests IO.
pub struct FooterPrune {
    opened: Option<OpenedFile>,
    filter: Option<Expression>,
    session: VortexSession,
    next: Next<OpenedFile>,
}

impl FooterPrune {
    /// Creates the stage for `opened`, handing survivors to `next`.
    pub fn new(
        opened: OpenedFile,
        filter: Option<Expression>,
        session: VortexSession,
        next: Next<OpenedFile>,
    ) -> Self {
        Self {
            opened: Some(opened),
            filter,
            session,
            next,
        }
    }

    /// Whether the file statistics prove the filter false for every row.
    fn can_prune(&self, opened: &OpenedFile) -> VortexResult<bool> {
        let footer = &opened.footer;
        let Some(filter) = &self.filter else {
            return Ok(false);
        };
        let Some((stats, fields)) = footer
            .statistics()
            .zip(footer.dtype().as_struct_fields_opt())
        else {
            return Ok(false);
        };
        can_prune_file_stats(
            &filter.bind(footer.dtype())?,
            footer.row_count(),
            stats,
            fields,
            &self.session,
        )
    }
}

impl IoConsumer for FooterPrune {
    fn set_io_result(&mut self, _request: IoRequestId, _result: IoResult) {}
}

impl Planner for FooterPrune {
    fn state(&self) -> State {
        if self.opened.is_some() {
            State::NeedsCompute
        } else {
            State::Done
        }
    }

    fn compute(&mut self) -> VortexResult<PlannerOutput> {
        let Some(opened) = self.opened.take() else {
            vortex_bail!("FooterPrune: compute called after Done");
        };
        let row_count = opened.footer.row_count();
        if row_count == 0 || self.can_prune(&opened)? {
            return Ok(PlannerOutput::Done);
        }
        let scope = WorkScope {
            file_ordinal: 0,
            rows: 0..row_count,
        };
        Ok(PlannerOutput::Planner(scope, (self.next)(opened)?))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;
    use std::sync::atomic::Ordering;

    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::arrays::VarBinArray;
    use vortex_array::expr::col;
    use vortex_array::expr::gt;
    use vortex_array::expr::lit;
    use vortex_buffer::ByteBuffer;
    use vortex_buffer::buffer;

    use super::*;
    use crate::tests::fixtures::SESSION;
    use crate::tests::fixtures::open_buffer;
    use crate::tests::fixtures::recording_child;
    use crate::tests::fixtures::write_no_stats_test_file;
    use crate::tests::fixtures::write_test_file;

    fn columns() -> Vec<(&'static str, vortex_array::ArrayRef)> {
        vec![
            ("numbers", buffer![1u32, 2, 3, 4, 5, 6, 7, 8].into_array()),
            (
                "strings",
                VarBinArray::from(vec!["a", "b", "c", "d", "e", "f", "g", "h"]).into_array(),
            ),
        ]
    }

    fn opened(buffer: ByteBuffer) -> VortexResult<OpenedFile> {
        let footer = open_buffer(&buffer)?.footer().clone();
        Ok(OpenedFile {
            size: buffer.len() as u64,
            read: Arc::new(buffer),
            footer,
        })
    }

    fn prune(
        buffer: ByteBuffer,
        filter: Option<Expression>,
    ) -> VortexResult<(FooterPrune, Arc<AtomicBool>)> {
        let (next, invoked) = recording_child();
        Ok((
            FooterPrune::new(opened(buffer)?, filter, SESSION.clone(), next),
            invoked,
        ))
    }

    #[rstest]
    #[case::provably_false(Some(gt(col("numbers"), lit(100u32))), false)]
    #[case::possibly_true(Some(gt(col("numbers"), lit(3u32))), true)]
    #[case::no_filter(None, true)]
    fn prunes_by_file_statistics(
        #[case] filter: Option<Expression>,
        #[case] survives: bool,
    ) -> VortexResult<()> {
        let (mut stage, invoked) = prune(write_test_file(&columns())?, filter)?;
        assert_eq!(stage.state(), State::NeedsCompute);
        let output = stage.compute()?;
        if survives {
            assert!(
                matches!(&output, PlannerOutput::Planner(scope, _) if scope.rows == (0..8)),
                "{output:?}"
            );
        } else {
            assert!(matches!(output, PlannerOutput::Done), "{output:?}");
        }
        assert_eq!(invoked.load(Ordering::SeqCst), survives);
        assert_eq!(stage.state(), State::Done);
        Ok(())
    }

    #[test]
    fn missing_statistics_retain_the_file() -> VortexResult<()> {
        let (mut stage, invoked) = prune(
            write_no_stats_test_file(&columns())?,
            Some(gt(col("numbers"), lit(100u32))),
        )?;
        assert!(matches!(stage.compute()?, PlannerOutput::Planner(..)));
        assert!(invoked.load(Ordering::SeqCst));
        Ok(())
    }

    #[test]
    fn empty_file_finishes_without_a_child() -> VortexResult<()> {
        let empty = write_test_file(&[("numbers", buffer![0u32; 0].into_array())])?;
        let (mut stage, invoked) = prune(empty, None)?;
        assert!(matches!(stage.compute()?, PlannerOutput::Done));
        assert!(!invoked.load(Ordering::SeqCst));
        assert_eq!(stage.state(), State::Done);
        Ok(())
    }

    #[test]
    fn missing_column_surfaces_the_bind_error() -> VortexResult<()> {
        let (mut stage, invoked) = prune(
            write_test_file(&columns())?,
            Some(gt(col("missing"), lit(1u32))),
        )?;
        let err = stage.compute().err().map(|e| e.to_string());
        assert!(
            err.as_deref().is_some_and(|m| m.contains("missing")),
            "{err:?}"
        );
        assert!(!invoked.load(Ordering::SeqCst));
        Ok(())
    }
}
