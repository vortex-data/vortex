// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use rstest::rstest;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability::NonNullable;
use vortex_array::dtype::PType;
use vortex_array::expr::checked_add;
use vortex_array::expr::gt;
use vortex_array::expr::lit;
use vortex_array::expr::lt;
use vortex_array::expr::or;
use vortex_array::expr::root;
use vortex_buffer::Alignment;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_mask::Mask;
use vortex_scan::planning::next::Next;
use vortex_scan::planning::planner::Planner;
use vortex_scan::planning::planner::State;
use vortex_scan::planning::planner::WorkScope;
use vortex_session::registry::ReadContext;

use super::FilterPlanner;
use super::FilterPlans;
use super::Keep;
use super::SelectedRows;
use crate::plan::ConcatPlan;
use crate::plan::EvalPlan;
use crate::plan::RowIdxPlan;
use crate::plan::SegmentScanPlan;
use crate::plan::TakePlan;
use crate::plan::exec::DecodeCache;
use crate::scan::filter::FilterExpr;
use crate::scan::planning::ScanPlans;
use crate::scan::planning::SegmentLocation;
use crate::segments::SegmentId;
use crate::test::new_session;

#[rstest]
#[case::disabled(Keep::True, 0, false, &[])]
#[case::one_segment(Keep::True, 4, false, &[0])]
#[case::skips_large_segment(Keep::True, 11, false, &[0, 2])]
#[case::two_segments(Keep::True, 16, false, &[0, 1])]
#[case::all_segments(Keep::True, 23, false, &[0, 1, 2])]
#[case::fragmented_excluded_chunk(Keep::True, 23, true, &[0, 2])]
#[case::zone_pruning(Keep::False, 23, false, &[])]
fn projection_prefetch_budget(
    #[case] keep: Keep,
    #[case] budget: usize,
    #[case] exclude_middle: bool,
    #[case] expected: &[usize],
) -> VortexResult<()> {
    let dtype = DType::Primitive(PType::I32, NonNullable);
    let chunks = (0..3)
        .map(|segment| {
            SegmentScanPlan::new(
                dtype.clone(),
                1000,
                SegmentId::from(segment),
                ReadContext::new([]),
                None,
            )
            .into_plan()
        })
        .collect();
    let locations: Arc<[_]> = [4, 12, 7, 1]
        .into_iter()
        .enumerate()
        .map(|(offset, length)| SegmentLocation {
            offset: offset as u64 * 100,
            length,
            alignment: Alignment::none(),
        })
        .collect();
    let filter = SegmentScanPlan::new(
        DType::Bool(NonNullable),
        3000,
        SegmentId::from(3),
        ReadContext::new([]),
        None,
    )
    .into_plan();
    let next: Next<SelectedRows> = Arc::new(|_| vortex_bail!("test stops before handoff"));
    let planner = FilterPlanner::with_keep(
        ScanPlans {
            session: new_session(),
            locations: Arc::clone(&locations),
            projection: ConcatPlan::try_new(dtype, chunks)?.into_plan(),
            projection_starts: Arc::from([0, 1000, 2000]),
            row_offset: 0,
            decoded: DecodeCache::default(),
        },
        FilterPlans::single(filter),
        keep,
        WorkScope {
            file_ordinal: 0,
            rows: 0..3000,
        },
        // More than 64 selected runs must still exclude the empty middle chunk.
        Mask::from_iter(
            (0..3000).map(|row| !exclude_middle || (row % 2 == 0 && !(1000..2000).contains(&row))),
        ),
        next,
    );
    let actual: Vec<_> = planner
        .prefetch_others(0, budget)?
        .iter()
        .map(|request| request.target)
        .collect();
    let expected: Vec<_> = expected.iter().map(|&id| locations[id].target()).collect();
    assert_eq!(actual, expected);
    Ok(())
}

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
    #[values(false, true)] initial_selection: bool,
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
    let selection = Mask::from_iter((0..8).map(|row| selected.contains(&row)));
    let mut planner = FilterPlanner::new(
        ScanPlans {
            session: new_session(),
            locations: Arc::from([]),
            projection: source,
            projection_starts: Arc::from([]),
            row_offset: 0,
            decoded: DecodeCache::default(),
        },
        filters,
        WorkScope {
            file_ordinal: 0,
            rows: 0..8,
        },
        if initial_selection {
            selection.clone()
        } else {
            Mask::new_true(8)
        },
        next,
    );
    // A preceding conjunct can narrow the selection without excluding segments at scan setup.
    planner.mask = selection;
    planner.compute()?;
    assert_eq!(planner.evaluate_all, evaluate_all && !initial_selection);
    for _ in 0..100 {
        if planner.remaining.iter().all(|pending| !pending) {
            let actual: Vec<_> = (0..8).filter(|&row| planner.mask.value(row)).collect();
            assert_eq!(actual, expected);
            assert_eq!(planner.trace_selection(), Some((1, expected.len())));
            return Ok(());
        }
        assert_eq!(planner.state(), State::NeedsCompute);
        planner.compute()?;
    }
    vortex_bail!("predicate did not finish")
}
