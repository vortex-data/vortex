// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Common Subplan Elimination pass.
//!
//! [`cse`] walks the plan tree, identifies any subtree that appears
//! more than once (by structural [`PartialEq`] + [`Hash`] of `dyn
//! LayoutPlan`), and rewrites the duplicates to share a single
//! materialised result via [`crate::v2::let_use::LetPlan`] /
//! [`crate::v2::let_use::UsePlan`].
//!
//! The pass is intentionally conservative:
//! - We only consider subtrees that appear at least twice. Single-use
//!   subtrees stay as-is.
//! - We skip trivial subtrees (already-shared `UsePlan`, the
//!   near-zero-cost `MaskSlicePlan` adapter) where wrapping in
//!   `Let`/`Use` would cost more than the duplication itself.
//! - Each shared subtree gets one `LetPlan` near the root, with the
//!   body containing the rewritten tree where every occurrence of
//!   the subtree is replaced by `UsePlan(id)`. We don't try to place
//!   the `Let` at the lowest common dominator yet — root-level
//!   placement is correct; finer placement is a future optimisation.
//!
//! The pass is idempotent: applying it twice produces the same plan
//! as applying it once.
//!
//! See `LAYOUT_PLAN.md` § Tee and CommonSubplanElimination.

use std::any::Any;
use std::sync::Arc;

use vortex_error::VortexResult;
use vortex_utils::aliases::hash_map::HashMap;

use crate::v2::let_use::LetId;
use crate::v2::let_use::LetPlan;
use crate::v2::let_use::UsePlan;
use crate::v2::mask_slice::MaskSlicePlan;
use crate::v2::plan::LayoutPlan;
use crate::v2::plan::LayoutPlanRef;
use crate::v2::plan::hash_plan;
use crate::v2::plan::plans_eq;
use crate::v2::plan::with_hash_cache;

/// Run common-subplan elimination on `plan`. Returns either the
/// original plan (if no sharing opportunities were found) or a
/// rewritten plan wrapped in one `LetPlan` per shared subtree.
pub fn cse(plan: LayoutPlanRef) -> VortexResult<LayoutPlanRef> {
    // The whole pass runs under a hash cache: subtree hashes are
    // memoised by `Arc` pointer so each unique node is hashed once,
    // not once per occurrence (pushdown produces N copies of the
    // same mask reference and without caching CSE is quadratic).
    with_hash_cache(|| cse_inner(plan))
}

fn cse_inner(plan: LayoutPlanRef) -> VortexResult<LayoutPlanRef> {
    // Step 1: walk the tree and count how many times each distinct
    // subtree appears. Keys are `LayoutPlanRef`s compared
    // structurally via `dyn LayoutPlan`'s `PartialEq` impl.
    let mut counts: HashMap<PlanKey, usize> = HashMap::new();
    walk(&plan, &mut counts);

    // Step 2: collect the subtrees that appear more than once and
    // are worth sharing. Today's heuristic skips already-shared or
    // adapter-thin plans where the `Let`/`Use` overhead would dwarf
    // the duplicated work.
    let shared: Vec<(LayoutPlanRef, LetId)> = counts
        .into_iter()
        .filter(|(_, count)| *count > 1)
        .filter(|(key, _)| !is_trivial(&key.0))
        .map(|(key, _)| (key.0, LetId::fresh()))
        .collect();

    if shared.is_empty() {
        return Ok(plan);
    }

    let lookup: HashMap<PlanKey, LetId> = shared
        .iter()
        .map(|(p, id)| (PlanKey(Arc::clone(p)), *id))
        .collect();

    // Step 3: rewrite the tree top-down. Each occurrence of a shared
    // subtree becomes `UsePlan(id)`; we don't recurse into shared
    // subtrees because the `LetPlan`'s source keeps the original.
    let body = rewrite(&plan, &lookup)?;

    // Step 4: wrap each shared subtree's source in a `LetPlan`.
    // Order doesn't matter for correctness — each `Let` only
    // publishes its own id.
    let mut wrapped = body;
    for (source, id) in shared {
        wrapped = Arc::new(LetPlan::with_id(id, source, wrapped));
    }
    Ok(wrapped)
}

/// Newtype around `LayoutPlanRef` that delegates `Hash + Eq` to the
/// inner `dyn LayoutPlan`'s structural impls. We go through the
/// `plans_eq` / `hash_plan` helpers (rather than relying on the std
/// `Arc` blankets) to keep the `Arc::ptr_eq` fast path and stay
/// consistent with the rest of the v2 codebase.
#[derive(Clone)]
struct PlanKey(LayoutPlanRef);

impl std::hash::Hash for PlanKey {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        hash_plan(&self.0, state);
    }
}

impl PartialEq for PlanKey {
    fn eq(&self, other: &Self) -> bool {
        plans_eq(&self.0, &other.0)
    }
}

impl Eq for PlanKey {}

/// "Don't bother sharing this — wrapping it in Let/Use would cost
/// more than the duplication." Today: already-shared `UsePlan` and
/// the very-thin `MaskSlicePlan` adapter. `LayoutPlan: DynEq: Any`,
/// so trait upcasting from `&dyn LayoutPlan` to `&dyn Any` is direct.
fn is_trivial(plan: &LayoutPlanRef) -> bool {
    let plan_ref: &dyn LayoutPlan = &**plan;
    let any: &dyn Any = plan_ref;
    any.is::<UsePlan>() || any.is::<MaskSlicePlan>()
}

/// Recursive walk that increments the count for each distinct subtree
/// encountered, then descends into children.
fn walk(plan: &LayoutPlanRef, counts: &mut HashMap<PlanKey, usize>) {
    *counts.entry(PlanKey(Arc::clone(plan))).or_insert(0) += 1;
    for child in plan.children() {
        walk(child, counts);
    }
}

/// Recursive rewrite. If `plan` is in `shared`, replace it with a
/// `UsePlan(id)`. Otherwise rebuild it with rewritten children via
/// `with_new_children` (or clone unchanged if it has no children).
fn rewrite(plan: &LayoutPlanRef, shared: &HashMap<PlanKey, LetId>) -> VortexResult<LayoutPlanRef> {
    if let Some(&id) = shared.get(&PlanKey(Arc::clone(plan))) {
        return Ok(Arc::new(UsePlan::new(
            id,
            plan.schema().clone(),
            plan_row_count(plan),
        )));
    }
    let new_children: Vec<LayoutPlanRef> = plan
        .children()
        .iter()
        .map(|c| rewrite(c, shared))
        .collect::<VortexResult<_>>()?;
    if new_children.is_empty() {
        return Ok(Arc::clone(plan));
    }
    Arc::clone(plan).with_new_children(new_children)
}

/// Sum of `plan`'s partition row counts — the value `UsePlan` reports
/// as its single partition's row range.
fn plan_row_count(plan: &LayoutPlanRef) -> u64 {
    (0..plan.partition_count())
        .filter_map(|i| plan.partition_stats(i).ok())
        .map(|s| s.row_count())
        .sum()
}

#[cfg(test)]
mod tests {
    use std::any::Any;
    use std::sync::Arc;
    use std::sync::OnceLock;

    use futures::stream;
    use vortex_array::IntoArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability::NonNullable;
    use vortex_array::dtype::PType;
    use vortex_array::stream::ArrayStreamAdapter;
    use vortex_array::stream::ArrayStreamExt;
    use vortex_array::stream::SendableArrayStream;
    use vortex_error::VortexResult;
    use vortex_error::vortex_bail;

    use super::cse;
    use crate::v2::let_use::LetPlan;
    use crate::v2::let_use::UsePlan;
    use crate::v2::plan::LayoutPlan;
    use crate::v2::plan::LayoutPlanRef;
    use crate::v2::plan::PartitionStats;
    use crate::v2::plan::hash_plan_slice;
    use crate::v2::plan::plan_slices_eq;
    use crate::v2::plan::plans_eq;
    use crate::v2::scan_ctx::ScanCtx;

    fn dtype() -> &'static DType {
        static D: OnceLock<DType> = OnceLock::new();
        D.get_or_init(|| DType::Primitive(PType::I32, NonNullable))
    }

    /// Synthetic leaf plan used only by these tests. `tag` + `chunks`
    /// drive `PartialEq + Hash`, so two `IdentPlan`s with matching
    /// tag/chunks compare structurally equal regardless of `Arc`
    /// identity.
    struct IdentPlan {
        tag: &'static str,
        chunks: Vec<Vec<i32>>,
        row_count: u64,
    }

    impl IdentPlan {
        fn new(tag: &'static str, chunks: Vec<Vec<i32>>) -> Arc<Self> {
            let row_count = chunks.iter().map(|c| c.len() as u64).sum();
            Arc::new(Self {
                tag,
                chunks,
                row_count,
            })
        }
    }

    impl PartialEq for IdentPlan {
        fn eq(&self, other: &Self) -> bool {
            self.tag == other.tag && self.chunks == other.chunks
        }
    }

    impl Eq for IdentPlan {}

    impl std::hash::Hash for IdentPlan {
        fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
            self.tag.hash(state);
            self.chunks.hash(state);
        }
    }

    impl LayoutPlan for IdentPlan {
        fn schema(&self) -> &DType {
            dtype()
        }
        fn partition_count(&self) -> usize {
            1
        }
        fn partition_stats(&self, _: usize) -> VortexResult<PartitionStats> {
            Ok(PartitionStats::for_range(0..self.row_count))
        }
        fn output_ordered(&self) -> bool {
            true
        }
        fn required_input_ordered(&self) -> Vec<bool> {
            vec![]
        }
        fn maintains_input_order(&self) -> Vec<bool> {
            vec![]
        }
        fn children(&self) -> &[LayoutPlanRef] {
            &[]
        }
        fn with_new_children(
            self: Arc<Self>,
            _children: Vec<LayoutPlanRef>,
        ) -> VortexResult<LayoutPlanRef> {
            Ok(self)
        }
        fn execute(
            &self,
            _row_range: std::ops::Range<u64>,
            _ctx: &ScanCtx,
        ) -> VortexResult<SendableArrayStream> {
            let arrays: Vec<_> = self
                .chunks
                .iter()
                .map(|c| Ok(PrimitiveArray::from_iter(c.iter().copied()).into_array()))
                .collect();
            Ok(ArrayStreamExt::boxed(ArrayStreamAdapter::new(
                dtype().clone(),
                stream::iter(arrays),
            )))
        }
    }

    /// A test-only N-ary container that exposes its children to
    /// `children()` so the CSE walker can find duplicates across them.
    /// Real two-child plans (`AndBoolStreamsPlan`, `FilterPlan`, etc.)
    /// require Bool schemas; this stays untyped.
    struct ContainerPlan {
        children: Vec<LayoutPlanRef>,
        row_count: u64,
    }

    impl ContainerPlan {
        fn new(children: Vec<LayoutPlanRef>) -> Arc<Self> {
            let row_count = children
                .first()
                .map(|c| {
                    (0..c.partition_count())
                        .filter_map(|i| c.partition_stats(i).ok())
                        .map(|s| s.row_count())
                        .sum()
                })
                .unwrap_or(0);
            Arc::new(Self {
                children,
                row_count,
            })
        }
    }

    impl PartialEq for ContainerPlan {
        fn eq(&self, other: &Self) -> bool {
            plan_slices_eq(&self.children, &other.children)
        }
    }

    impl Eq for ContainerPlan {}

    impl std::hash::Hash for ContainerPlan {
        fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
            hash_plan_slice(&self.children, state);
        }
    }

    impl LayoutPlan for ContainerPlan {
        fn schema(&self) -> &DType {
            dtype()
        }
        fn partition_count(&self) -> usize {
            1
        }
        fn partition_stats(&self, _: usize) -> VortexResult<PartitionStats> {
            Ok(PartitionStats::for_range(0..self.row_count))
        }
        fn output_ordered(&self) -> bool {
            true
        }
        fn required_input_ordered(&self) -> Vec<bool> {
            vec![true; self.children.len()]
        }
        fn maintains_input_order(&self) -> Vec<bool> {
            vec![true; self.children.len()]
        }
        fn children(&self) -> &[LayoutPlanRef] {
            &self.children
        }
        fn with_new_children(
            self: Arc<Self>,
            children: Vec<LayoutPlanRef>,
        ) -> VortexResult<LayoutPlanRef> {
            if children.len() != self.children.len() {
                vortex_bail!(
                    "ContainerPlan::with_new_children expected {} children, got {}",
                    self.children.len(),
                    children.len()
                );
            }
            Ok(Arc::new(Self {
                children,
                row_count: self.row_count,
            }))
        }
        fn execute(
            &self,
            _row_range: std::ops::Range<u64>,
            _ctx: &ScanCtx,
        ) -> VortexResult<SendableArrayStream> {
            // Only used for structural tests, never executed.
            unreachable!("ContainerPlan is structural-only")
        }
    }

    /// Walk a plan tree, counting how many distinct nodes downcast to
    /// `T`. Used by tests to assert structural rewrites.
    fn count_kind<T: Any>(root: &LayoutPlanRef) -> usize {
        let mut found = 0usize;
        let mut stack = vec![Arc::clone(root)];
        while let Some(p) = stack.pop() {
            let plan_ref: &dyn LayoutPlan = &*p;
            let any: &dyn Any = plan_ref;
            if any.is::<T>() {
                found += 1;
            }
            for c in p.children() {
                stack.push(Arc::clone(c));
            }
        }
        found
    }

    #[test]
    fn cse_dedupes_identical_subtrees() -> VortexResult<()> {
        // Two structurally-equal IdentPlans, both reachable as
        // children of one container so the CSE walker sees both.
        let a: LayoutPlanRef = IdentPlan::new("dup", vec![vec![1, 2, 3]]);
        let b: LayoutPlanRef = IdentPlan::new("dup", vec![vec![1, 2, 3]]);
        let original: LayoutPlanRef = ContainerPlan::new(vec![a, b]);

        let rewritten = cse(Arc::clone(&original))?;

        assert!(
            !plans_eq(&original, &rewritten),
            "CSE should have rewritten the tree containing duplicated subtrees"
        );
        // Both IdentPlan occurrences should have been replaced with
        // UsePlans. The container is unchanged structurally except
        // for its children.
        assert_eq!(count_kind::<UsePlan>(&rewritten), 2);
        // Exactly one new LetPlan was introduced by the pass.
        assert_eq!(count_kind::<LetPlan>(&rewritten), 1);
        // No IdentPlans remain in the rewritten body — they live
        // only inside the LetPlan's source.
        assert_eq!(count_kind::<IdentPlan>(&rewritten), 0);
        Ok(())
    }

    #[test]
    fn cse_is_noop_when_no_sharing() -> VortexResult<()> {
        let a: LayoutPlanRef = IdentPlan::new("alpha", vec![vec![1]]);
        let b: LayoutPlanRef = IdentPlan::new("beta", vec![vec![2]]);
        let plan: LayoutPlanRef = ContainerPlan::new(vec![a, b]);
        let rewritten = cse(Arc::clone(&plan))?;
        assert!(
            Arc::ptr_eq(&plan, &rewritten),
            "CSE must short-circuit when no subtree appears twice"
        );
        Ok(())
    }

    #[test]
    fn cse_is_idempotent() -> VortexResult<()> {
        let a: LayoutPlanRef = IdentPlan::new("dup", vec![vec![1, 2, 3]]);
        let b: LayoutPlanRef = IdentPlan::new("dup", vec![vec![1, 2, 3]]);
        let original: LayoutPlanRef = ContainerPlan::new(vec![a, b]);
        let once = cse(original)?;
        let twice = cse(Arc::clone(&once))?;
        assert!(
            plans_eq(&once, &twice),
            "CSE applied twice should equal CSE applied once"
        );
        Ok(())
    }

    #[test]
    fn cse_skips_already_use_plans() -> VortexResult<()> {
        // A container with two identical UsePlans should be a no-op:
        // `is_trivial` skips already-shared references.
        let id = crate::v2::let_use::LetId::fresh();
        let u1: LayoutPlanRef = Arc::new(UsePlan::new(id, dtype().clone(), 4));
        let u2: LayoutPlanRef = Arc::new(UsePlan::new(id, dtype().clone(), 4));
        let plan: LayoutPlanRef = ContainerPlan::new(vec![u1, u2]);
        let rewritten = cse(Arc::clone(&plan))?;
        assert!(
            Arc::ptr_eq(&plan, &rewritten),
            "CSE must skip duplicated UsePlans (already shared)"
        );
        Ok(())
    }
}
