// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::BuildHasher;
use std::sync::Arc;
use std::sync::Barrier;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::thread;

use rstest::rstest;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::expr::Expression;
use vortex_array::expr::get_item;
use vortex_array::expr::gt;
use vortex_array::expr::lit;
use vortex_array::expr::lt;
use vortex_array::expr::root;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use super::ExprKey;
use super::PlanCache;

fn ints_gt(value: i32) -> Expression {
    gt(get_item("ints", root()), lit(value))
}

fn table(ints: PType) -> DType {
    DType::struct_(
        [("ints", DType::Primitive(ints, Nullability::NonNullable))],
        Nullability::NonNullable,
    )
}

#[test]
fn separately_bound_equal_expressions_share_a_key() -> VortexResult<()> {
    let dtype = table(PType::I32);
    let lhs = ExprKey(ints_gt(5).bind(&dtype)?);
    let rhs = ExprKey(ints_gt(5).bind(&dtype)?);
    assert!(lhs == rhs);
    assert_eq!(
        rustc_hash::FxBuildHasher.hash_one(&lhs),
        rustc_hash::FxBuildHasher.hash_one(&rhs)
    );
    Ok(())
}

#[rstest]
#[case::literal(ints_gt(6), table(PType::I32))]
#[case::function(lt(get_item("ints", root()), lit(5i32)), table(PType::I32))]
#[case::dtype(get_item("ints", root()), table(PType::I64))]
fn different_expressions_have_different_keys(
    #[case] other: Expression,
    #[case] dtype: DType,
) -> VortexResult<()> {
    let i32s = table(PType::I32);
    for key in [ints_gt(5), get_item("ints", root())] {
        assert!(ExprKey(key.bind(&i32s)?) != ExprKey(other.bind(&dtype)?));
    }
    Ok(())
}

#[test]
fn concurrent_scans_initialize_equal_expressions_once() -> VortexResult<()> {
    let cache = PlanCache::default();
    let builds = AtomicUsize::new(0);
    let barrier = Barrier::new(8);
    let expressions = (0..8)
        .map(|_| ints_gt(5).bind(&table(PType::I32)))
        .collect::<VortexResult<Vec<_>>>()?;
    let plans = thread::scope(|scope| {
        let cache = &cache;
        let builds = &builds;
        let barrier = &barrier;
        let handles = expressions
            .iter()
            .map(|expression| {
                scope.spawn(move || {
                    barrier.wait();
                    cache.get_or_try_init(expression, || {
                        builds.fetch_add(1, Ordering::Relaxed);
                        Ok(Arc::new(42))
                    })
                })
            })
            .collect::<Vec<_>>();
        handles
            .into_iter()
            .map(|handle| handle.join().expect("scan preparation panicked"))
            .collect::<VortexResult<Vec<_>>>()
    })?;
    assert_eq!(builds.load(Ordering::Relaxed), 1);
    assert!(plans.iter().all(|plan| Arc::ptr_eq(plan, &plans[0])));
    Ok(())
}

#[test]
fn failed_plan_initialization_can_be_retried() -> VortexResult<()> {
    let cache = PlanCache::default();
    let expression = ints_gt(5).bind(&table(PType::I32))?;
    let failed = cache.get_or_try_init(&expression, || vortex_bail!("planning failed"));
    assert!(failed.is_err());
    assert_eq!(cache.get_or_try_init(&expression, || Ok(42))?, 42);
    assert_eq!(
        cache.get_or_try_init(&expression, || vortex_bail!("cached plan was rebuilt"))?,
        42
    );
    Ok(())
}

#[test]
fn different_expressions_keep_separate_plans() -> VortexResult<()> {
    let cache = PlanCache::default();
    let dtype = table(PType::I32);
    let first = ints_gt(5).bind(&dtype)?;
    let second = ints_gt(6).bind(&dtype)?;
    assert_eq!(cache.get_or_try_init(&first, || Ok(5))?, 5);
    assert_eq!(cache.get_or_try_init(&second, || Ok(6))?, 6);
    assert_eq!(
        cache.get_or_try_init(&first, || vortex_bail!("first plan was replaced"))?,
        5
    );
    Ok(())
}
