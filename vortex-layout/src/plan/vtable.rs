// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::borrow::Cow;
use std::fmt;
use std::fmt::Debug;

use vortex_array::DeserializeMetadata;
use vortex_array::SerializeMetadata;
use vortex_error::VortexResult;
use vortex_session::registry::Id;

use crate::plan::PlanChildren;
use crate::plan::typed::Plan;

/// A unique identifier for a plan operator.
pub type PlanId = Id;

/// Operator-specific behavior for a typed [`Plan`].
///
/// Common fields — dtype, row count, and children — are stored outside the erased operator data.
/// Implementations own only their operator-specific data, its metadata codec, and a callback for
/// refreshing cached data after generic child replacement.
///
/// Operators describe physical work over a row domain. Their identity and operator-specific data
/// do not depend on the source layout kind. The common lazy-child storage may nevertheless own a
/// hidden source-layout reference used for on-demand lowering.
pub trait PlanVTable: 'static + Clone + Sized + Send + Sync + Debug {
    /// Operator-specific data, excluding children.
    ///
    /// Children belong in [`PlanParts::children`](crate::plan::PlanParts::children) so that
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

    /// Refreshes cloned operator data after the common child container is replaced.
    ///
    /// The plan layer clones [`PlanData`](Self::PlanData), replaces the children externally, and
    /// invokes this callback. Implementations validate the new children and update any derived
    /// values in `data`; they do not rebuild the plan itself.
    fn with_children(
        plan: &Plan<Self>,
        children: &PlanChildren,
        data: &mut Self::PlanData,
    ) -> VortexResult<()> {
        let _ = (plan, children, data);
        Ok(())
    }

    /// Returns the display name of the child at `index`.
    fn child_name(plan: &Plan<Self>, index: usize) -> Cow<'_, str> {
        let _ = plan;
        Cow::Owned(format!("child[{index}]"))
    }
}
