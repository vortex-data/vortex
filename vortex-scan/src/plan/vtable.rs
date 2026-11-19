// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::borrow::Cow;
use std::fmt;
use std::fmt::Debug;
use std::ops::Range;

use vortex_array::DeserializeMetadata;
use vortex_array::SerializeMetadata;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_mask::Mask;
use vortex_session::VortexSession;
use vortex_session::registry::Id;

use crate::execution::pipeline::Chain;
use crate::execution::pipeline::Compiler;
use crate::plan::PlanRef;
use crate::plan::PlanSlots;
use crate::plan::typed::Plan;

/// A unique identifier for a plan operator.
pub type PlanId = Id;

/// Operator-specific behavior for a typed [`Plan`].
///
/// Common fields — dtype, row count, and child slots — are stored outside the erased operator
/// data. Implementations own only their operator-specific data, its metadata codec, and a callback
/// for refreshing cached data after generic slot replacement.
///
/// Each operator fixes what every slot position holds. An absent child is a `None` slot, so the
/// positions of the others never depend on which children are present.
///
/// Operators describe physical work over a row domain. Their identity and operator-specific data
/// do not depend on the source layout kind. A lazy slot may nevertheless own a hidden
/// source-layout reference used for on-demand lowering.
pub trait PlanVTable: 'static + Clone + Sized + Send + Sync + Debug {
    /// Operator-specific data, excluding children.
    ///
    /// Children belong in [`PlanParts::slots`](crate::plan::PlanParts::slots) so that
    /// traversal and rewriting can discover them generically.
    type PlanData: 'static + Send + Sync + Clone + Debug;

    /// Serialized form of [`PlanData`](Self::PlanData).
    type Metadata: SerializeMetadata + DeserializeMetadata + Debug;

    /// Returns the globally unique operator ID.
    fn id(&self) -> PlanId;

    /// Writes operator-specific fields after the plan's standard display summary.
    fn fmt(plan: &Plan<Self>, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let _ = (plan, formatter);
        Ok(())
    }

    /// Returns the serializable metadata for this operator.
    ///
    /// Returns `None` when the operator holds state that cannot be serialized.
    fn metadata(plan: &Plan<Self>) -> Option<Self::Metadata>;

    /// Refreshes cloned operator data after the slots are replaced.
    ///
    /// The plan layer clones [`PlanData`](Self::PlanData), replaces the slots externally, and
    /// invokes this callback. Implementations validate the new slots and update any derived
    /// values in `data`; they do not rebuild the plan itself.
    fn with_slots(
        plan: &Plan<Self>,
        slots: &PlanSlots,
        data: &mut Self::PlanData,
    ) -> VortexResult<()> {
        let _ = (plan, slots, data);
        Ok(())
    }

    /// Returns the display name of `slot`.
    fn slot_name(plan: &Plan<Self>, slot: usize) -> Cow<'_, str> {
        let _ = plan;
        Cow::Owned(format!("slot[{slot}]"))
    }

    /// Whether this node reads payload data and can apply a row mask at the source.
    ///
    /// Metadata-only pruning rejects these nodes outside pruning subplans. A Filter directly
    /// above such a node forwards its mask to the node's compiler instead of adding a mask stage.
    fn is_data_source(_plan: &Plan<Self>) -> bool {
        false
    }

    /// Applies an adapter-owned rewrite before optimizing this node's children.
    fn reduce(_plan: &Plan<Self>, _session: &VortexSession) -> VortexResult<Option<PlanRef>> {
        Ok(None)
    }

    /// Rewrites a parent using format-specific knowledge of this child.
    fn reduce_parent(
        _plan: &Plan<Self>,
        _parent: &PlanRef,
        _slot: usize,
        _session: &VortexSession,
    ) -> VortexResult<Option<PlanRef>> {
        Ok(None)
    }

    /// Compiles this plan over `rows` of its row domain, restricted to `mask`, into a chain:
    /// a source and the stages after it, producing the rows `mask` selects in row order, or
    /// `None` when that is no row. `rows` lies within the plan's row domain and `mask` is as long
    /// as `rows`.
    ///
    /// A plan compiles its children through `compiler` and either adds stages to a child's
    /// chain or joins several children's chains with a source of its own. Compiling does no IO.
    ///
    /// Pipelines have no absent inputs: a plan that joins its children compiles
    /// [`Plan::children`], the present slots in slot order, so inlet `i` of its source reads the
    /// `i`-th present slot.
    fn compile(
        plan: &Plan<Self>,
        rows: Range<u64>,
        mask: &Mask,
        compiler: &mut Compiler<'_>,
    ) -> VortexResult<Option<Chain>> {
        let _ = (rows, mask, compiler);
        vortex_bail!("Plan {} has no pipeline implementation", plan.vtable().id())
    }
}
