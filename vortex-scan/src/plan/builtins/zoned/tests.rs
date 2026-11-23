// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::IntoArray;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::aggregate_fn::AggregateFnVTableExt;
use vortex_array::aggregate_fn::NumericalAggregateOpts;
use vortex_array::aggregate_fn::fns::max::Max;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::StructFields;
use vortex_array::expr::gt;
use vortex_array::expr::lit;
use vortex_array::expr::root;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_expr::stats::rewrite::BoundExpressionPruneExt;
use vortex_mask::Mask;

use crate::execution::pipeline::synthetic::SyntheticPlan;
use crate::execution::zone_map::ZoneMap;
use crate::plan::*;
use crate::test::SESSION;

fn placeholder(dtype: DType, rows: u64) -> PlanRef {
    SyntheticPlan::new(dtype, rows, Vec::new(), |_, _, _| {
        unreachable!("proof test does not execute payloads")
    })
    .into_plan()
}

fn primitive(ptype: PType, nullability: Nullability) -> DType {
    DType::Primitive(ptype, nullability)
}

fn max_zones_dtype(dtype: &DType) -> VortexResult<(AggregateFnRef, DType)> {
    let max = Max.bind(NumericalAggregateOpts::skip_nans());
    let state = max
        .state_dtype(dtype)
        .ok_or_else(|| vortex_err!("max does not support {dtype}"))?;
    let table = DType::Struct(
        StructFields::from_iter([(max.to_string(), state.as_nullable())]),
        Nullability::NonNullable,
    );
    Ok((max, table))
}

/// Rebuilding a pruning plan over new zones discards the proof over the old ones.
#[test]
fn zoned_pruning_plan_with_new_zones_proves_them_again() -> VortexResult<()> {
    let dtype = primitive(PType::I32, Nullability::NonNullable);
    let (max, zones_dtype) = max_zones_dtype(&dtype)?;
    let zones = placeholder(zones_dtype.clone(), 2);
    let source = ZonedPlan::new(
        placeholder(dtype.clone(), 5),
        zones,
        3,
        vec![max.clone()].into(),
    )
    .into_plan();
    let falsifier = gt(root(), lit(5_i32))
        .bind(&dtype)?
        .falsify(&SESSION)?
        .ok_or_else(|| vortex_err!("filter has no falsifier"))?;
    let pruning = optimize(
        EvalPlan::try_new(falsifier.clone(), source)?.into_plan(),
        &SESSION,
    )?;
    let pruning = pruning
        .as_opt::<Zoned>()
        .ok_or_else(|| vortex_err!("predicate did not become a pruning plan"))?;
    let proof = pruning
        .pruned_zones()
        .ok_or_else(|| vortex_err!("pruning plan has no proof"))?;

    let zone_map = |bounds: [i32; 2]| {
        let table = StructArray::from_fields(&[(
            max.to_string().as_str(),
            PrimitiveArray::from_option_iter(bounds.map(Some)).into_array(),
        )])?;
        ZoneMap::try_new(dtype.clone(), table, vec![max.clone()].into(), 3, 5)
    };
    let original = Mask::from_iter([true, false]);
    assert_eq!(
        proof.init(zone_map([3, 9])?, &falsifier, &SESSION)?,
        original
    );

    let new_zones = placeholder(zones_dtype, 2);
    let rebuilt = pruning
        .clone()
        .into_plan()
        .with_slots(vec![None, Some(LazyPlanRef::ready(new_zones))])?;
    let rebuilt_proof = rebuilt
        .as_opt::<Zoned>()
        .and_then(|zoned| zoned.pruned_zones())
        .ok_or_else(|| vortex_err!("rebuilt plan has no proof"))?;
    assert!(rebuilt_proof.get(&falsifier, &SESSION)?.is_none());
    let updated = Mask::from_iter([false, true]);
    assert_eq!(
        rebuilt_proof.init(zone_map([9, 3])?, &falsifier, &SESSION)?,
        updated
    );
    assert_eq!(rebuilt_proof.get(&falsifier, &SESSION)?, Some(updated));
    assert_eq!(proof.get(&falsifier, &SESSION)?, Some(original));
    Ok(())
}
