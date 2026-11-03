// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use vortex::error::VortexResult;
use vortex::expr::BoundExpression;
use vortex::io::request::IoConsumer;
use vortex::io::request::IoRequestId;
use vortex::io::request::IoResult;
use vortex::scalar_fn::fns::dynamic::DynamicComparison;
use vortex::scan::planning::planner::Planner;
use vortex::scan::planning::planner::PlannerOutput;
use vortex::scan::planning::planner::State;

const INITIAL_ROWS: u64 = 8192;

/// Limits the startup path to V2 scans with a pushed dynamic comparison.
pub(crate) fn initial_budget(
    scan_v2: bool,
    filter: Option<&BoundExpression>,
    chunks: usize,
    workers: usize,
) -> VortexResult<usize> {
    if !scan_v2 || chunks == 0 {
        return Ok(0);
    }
    if filter
        .map(|filter| filter.contains::<DynamicComparison>())
        .transpose()?
        .unwrap_or(false)
    {
        Ok(chunks.saturating_mul(workers))
    } else {
        Ok(0)
    }
}

/// Reserves disjoint startup ranges, carrying unused budget across short files.
pub(super) fn initial_ranges(rows: Range<u64>, remaining: &AtomicUsize) -> Vec<Range<u64>> {
    let needed =
        usize::try_from((rows.end - rows.start).div_ceil(INITIAL_ROWS)).unwrap_or(usize::MAX);
    let mut previous = remaining.load(Ordering::Relaxed);
    let claimed = loop {
        let claimed = previous.min(needed);
        if claimed == 0 {
            break 0;
        }
        match remaining.compare_exchange_weak(
            previous,
            previous - claimed,
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => break claimed,
            Err(current) => previous = current,
        }
    };
    (0..claimed)
        .map(|index| {
            let start = rows.start + index as u64 * INITIAL_ROWS;
            start..start.saturating_add(INITIAL_ROWS).min(rows.end)
        })
        .collect()
}

/// Keeps an initial range in one root, including children the chunk planner normally spawns.
pub(super) struct InitialPlanner(pub Box<dyn Planner>);

impl IoConsumer for InitialPlanner {
    fn set_io_result(&mut self, request: IoRequestId, result: IoResult) {
        self.0.set_io_result(request, result);
    }
}

impl Planner for InitialPlanner {
    fn state(&self) -> State {
        self.0.state()
    }

    fn compute(&mut self) -> VortexResult<PlannerOutput> {
        Ok(match self.0.compute()? {
            PlannerOutput::Spawn(scope, child) | PlannerOutput::Planner(scope, child) => {
                PlannerOutput::Planner(scope, Box::new(Self(child)))
            }
            output => output,
        })
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::single_range_in_vec_init)]
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;

    use rstest::rstest;
    use vortex::dtype::DType;
    use vortex::dtype::PType;
    use vortex::error::VortexResult;
    use vortex::expr::dynamic;
    use vortex::expr::lit;
    use vortex::expr::root;
    use vortex::scalar_fn::fns::operators::CompareOperator;

    use super::initial_budget;
    use super::initial_ranges;

    #[rstest]
    #[case::v1_dynamic(false, true, 2, 0)]
    #[case::v2_static(true, false, 2, 0)]
    #[case::disabled(true, true, 0, 0)]
    #[case::v2_dynamic(true, true, 2, 16)]
    fn only_dynamic_v2_scans_use_startup(
        #[case] scan_v2: bool,
        #[case] is_dynamic: bool,
        #[case] chunks: usize,
        #[case] expected: usize,
    ) -> VortexResult<()> {
        let dtype = DType::Primitive(PType::I64, false.into());
        let filter = if is_dynamic {
            dynamic(CompareOperator::Lt, || None, dtype.clone(), true, root())
        } else {
            lit(true)
        }
        .bind(&dtype)?;
        assert_eq!(initial_budget(scan_v2, Some(&filter), chunks, 8)?, expected);
        assert_eq!(initial_budget(scan_v2, None, chunks, 8)?, 0);
        Ok(())
    }

    #[test]
    fn startup_budget_tiles_prefix_and_carries_across_files() {
        let budget = AtomicUsize::new(4);
        assert_eq!(initial_ranges(5..10, &budget), vec![5..10]);
        assert_eq!(
            initial_ranges(7..40_000, &budget),
            vec![7..8199, 8199..16391, 16391..24583]
        );
        assert!(initial_ranges(0..100, &budget).is_empty());
        assert_eq!(budget.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn startup_budget_handles_empty_and_overflow_boundary() {
        let budget = AtomicUsize::new(2);
        assert!(initial_ranges(5..5, &budget).is_empty());
        assert_eq!(
            initial_ranges(u64::MAX - 10..u64::MAX, &budget),
            vec![u64::MAX - 10..u64::MAX]
        );
        assert_eq!(budget.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn concurrent_files_share_one_budget() {
        let budget = AtomicUsize::new(16);
        let claimed = AtomicUsize::new(0);
        std::thread::scope(|scope| {
            for _ in 0..8 {
                scope.spawn(|| {
                    claimed.fetch_add(
                        initial_ranges(0..1_000_000, &budget).len(),
                        Ordering::Relaxed,
                    );
                });
            }
        });
        assert_eq!(claimed.load(Ordering::Relaxed), 16);
        assert_eq!(budget.load(Ordering::Relaxed), 0);
    }
}
