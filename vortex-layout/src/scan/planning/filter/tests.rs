// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use rstest::rstest;
use vortex_array::expr::checked_add;
use vortex_array::expr::gt;
use vortex_array::expr::lit;
use vortex_array::expr::lt;
use vortex_array::expr::or;
use vortex_array::expr::root;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_mask::Mask;
use vortex_scan::planning::next::Next;
use vortex_scan::planning::planner::Planner;
use vortex_scan::planning::planner::State;
use vortex_scan::planning::planner::WorkScope;

use super::FilterPlanner;
use super::FilterPlans;
use super::SelectedRows;
use crate::plan::EvalPlan;
use crate::plan::RowIdxPlan;
use crate::plan::TakePlan;
use crate::plan::exec::DecodeCache;
use crate::scan::filter::FilterExpr;
use crate::scan::planning::ScanPlans;
use crate::test::new_session;

#[rstest]
#[case::dense(&[0, 2, 4, 7], false, false, false, true, &[4, 7])]
#[case::sparse(&[7], false, false, false, false, &[7])]
#[case::sparse_dictionary(&[7], true, false, false, true, &[7])]
#[case::sparse_disjunction(&[7], false, false, true, true, &[7])]
#[case::fallible_dense(&[0, 1, 2, 3], false, true, false, false, &[3])]
fn predicate_selection(
    #[case] selected: &[usize],
    #[case] dictionary: bool,
    #[case] fallible: bool,
    #[case] disjunction: bool,
    #[case] evaluate_all: bool,
    #[case] expected: &[usize],
) -> VortexResult<()> {
    let source = RowIdxPlan::new(8).into_plan();
    let expression = if fallible {
        // Rows 4..8 overflow, so evaluating outside the selection would fail.
        gt(checked_add(root(), lit(u64::MAX - 3)), lit(u64::MAX - 1))
    } else if disjunction {
        or(gt(root(), lit(3_u64)), lt(root(), lit(1_u64)))
    } else {
        gt(root(), lit(3_u64))
    }
    .bind(source.dtype())?;
    let predicate = EvalPlan::try_new(expression.clone(), source.clone())?.into_plan();
    let predicate = if dictionary {
        TakePlan::new(source.clone(), predicate).into_plan()
    } else {
        predicate
    };
    let filters = FilterPlans::conjuncts(
        Arc::new(FilterExpr::from_conjuncts(vec![expression])),
        vec![predicate],
    )?;
    let next: Next<SelectedRows> = Arc::new(|_| vortex_bail!("test stops before handoff"));
    let mut planner = FilterPlanner::new(
        ScanPlans {
            session: new_session(),
            locations: Arc::from([]),
            projection: source,
            projection_starts: Arc::from([]),
            row_offset: 0,
            decoded: DecodeCache::default(),
            segments: None,
        },
        filters,
        WorkScope {
            file_ordinal: 0,
            rows: 0..8,
        },
        Mask::from_iter((0..8).map(|row| selected.contains(&row))),
        next,
    );
    planner.compute()?;
    assert_eq!(planner.evaluate_all, evaluate_all);
    for _ in 0..100 {
        if planner.remaining.iter().all(|pending| !pending) {
            let actual: Vec<_> = (0..8).filter(|&row| planner.mask.value(row)).collect();
            assert_eq!(actual, expected);
            return Ok(());
        }
        assert_eq!(planner.state(), State::NeedsCompute);
        planner.compute()?;
    }
    vortex_bail!("predicate did not finish")
}
