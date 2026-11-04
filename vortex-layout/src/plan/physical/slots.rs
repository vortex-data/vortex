// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;
use std::ops::Deref;
use std::sync::Arc;

use once_cell::sync::OnceCell;
use smallvec::SmallVec;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;

use crate::LayoutRef;
use crate::plan::physical::PlanRef;

type SlotInitializer = dyn Fn() -> VortexResult<PlanRef> + 'static + Send + Sync;

/// A plan child that may be built on first access.
///
/// Clones share the built plan, so a child is built at most once however many plans hold its
/// slot. A failed build leaves the slot unbuilt, and the next access retries it.
#[derive(Clone)]
pub struct LazyPlanRef(Arc<LazyPlan<SlotInitializer>>);

/// The built plan and its initializer, in one allocation.
struct LazyPlan<F: ?Sized> {
    cell: OnceCell<PlanRef>,
    init: F, // must be last for unsized coercion
}

impl LazyPlanRef {
    /// A slot built by `init` on first access.
    pub fn lazy(init: impl Fn() -> VortexResult<PlanRef> + 'static + Send + Sync) -> Self {
        Self(Arc::new(LazyPlan {
            cell: OnceCell::new(),
            init,
        }))
    }

    /// A slot holding the already-built `plan`.
    pub fn ready(plan: PlanRef) -> Self {
        Self(Arc::new(LazyPlan {
            cell: OnceCell::with_value(plan),
            init: || Err(vortex_err!("A built plan slot was initialized again")),
        }))
    }

    /// Returns the plan, building it on first access.
    pub fn get(&self) -> VortexResult<&PlanRef> {
        self.0.cell.get_or_try_init(|| (self.0.init)())
    }

    /// Returns whether the plan has been built.
    pub fn is_initialized(&self) -> bool {
        self.0.cell.get().is_some()
    }
}

impl From<PlanRef> for LazyPlanRef {
    fn from(plan: PlanRef) -> Self {
        Self::ready(plan)
    }
}

impl fmt::Debug for LazyPlanRef {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0.cell.get() {
            Some(plan) => write!(formatter, "{plan}"),
            None => formatter.write_str("<unbuilt>"),
        }
    }
}

/// The fixed-position child slots of a plan.
///
/// Each operator defines what every slot position holds. A `None` slot is an absent child, such
/// as the validity of a non-nullable struct, so the positions of the other slots never shift.
#[derive(Clone, Default)]
pub struct PlanSlots(SmallVec<[Option<LazyPlanRef>; 4]>);

impl PlanSlots {
    /// One lazy plan slot per slot of `layout`, at the same position.
    pub(crate) fn from_layout(layout: &LayoutRef) -> Self {
        Self::from_layout_slots(layout, 0..layout.nslots())
    }

    /// One lazy plan slot per entry of `layout_slots`: plan slot `i` plans layout slot
    /// `layout_slots[i]`.
    ///
    /// This is where every layout maps its slots to plan slots. An absent layout slot is an
    /// absent plan slot; presence comes from the layout's metadata, so nothing is read or planned
    /// until a plan slot is first accessed.
    pub(crate) fn from_layout_slots(
        layout: &LayoutRef,
        layout_slots: impl IntoIterator<Item = usize>,
    ) -> Self {
        layout_slots
            .into_iter()
            .map(|slot| {
                layout.slot_type(slot)?;
                let layout = Arc::clone(layout);
                Some(LazyPlanRef::lazy(move || {
                    let child = layout
                        .slot(slot)?
                        .ok_or_else(|| vortex_err!("Layout child slot {slot} is absent"))?;
                    child.new_plan()
                }))
            })
            .collect()
    }

    /// Creates empty slots with room for `capacity` slots.
    pub fn with_capacity(capacity: usize) -> Self {
        Self(SmallVec::with_capacity(capacity))
    }

    /// Appends a slot.
    pub fn push(&mut self, slot: Option<LazyPlanRef>) {
        self.0.push(slot);
    }

    /// Returns the plan in `slot`, building it on first access, or `None` when the slot is
    /// absent. Errors when `slot` is out of bounds.
    pub fn get(&self, slot: usize) -> VortexResult<Option<&PlanRef>> {
        match self.0.get(slot) {
            Some(Some(lazy)) => lazy.get().map(Some),
            Some(None) => Ok(None),
            None => vortex_bail!(
                "Plan slot {slot} is out of bounds for {} slots",
                self.0.len()
            ),
        }
    }

    /// Returns the plan in `slot`, or an error when the slot is absent or out of bounds.
    pub fn required(&self, slot: usize) -> VortexResult<&PlanRef> {
        self.get(slot)?
            .ok_or_else(|| vortex_err!("Plan slot {slot} is absent"))
    }

    /// The present slots with their positions, in slot order, without building any child.
    ///
    /// This is the one place absent slots are dropped. Everything that walks a plan's children
    /// goes through it, with the slot position when it needs one.
    pub fn present(&self) -> impl Iterator<Item = (usize, &LazyPlanRef)> + '_ {
        self.0
            .iter()
            .enumerate()
            .filter_map(|(slot, lazy)| Some((slot, lazy.as_ref()?)))
    }

    /// The plans of the present slots in slot order, building each as it is visited.
    ///
    /// Pipelines have no absent inputs, so compiling a plan's children uses this: the `i`-th
    /// child is the `i`-th present slot.
    pub fn children(&self) -> impl Iterator<Item = VortexResult<&PlanRef>> + '_ {
        self.present().map(|(_, lazy)| lazy.get())
    }

    /// Returns an error when there are not exactly `expected` slots.
    pub(crate) fn check_len(&self, name: &str, expected: usize) -> VortexResult<()> {
        if self.0.len() != expected {
            vortex_bail!("{name} expects {expected} slots but got {}", self.0.len());
        }
        Ok(())
    }
}

impl Deref for PlanSlots {
    type Target = [Option<LazyPlanRef>];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl FromIterator<Option<LazyPlanRef>> for PlanSlots {
    fn from_iter<I: IntoIterator<Item = Option<LazyPlanRef>>>(iter: I) -> Self {
        Self(iter.into_iter().collect())
    }
}

/// Built slots, with `None` for an absent child.
impl FromIterator<Option<PlanRef>> for PlanSlots {
    fn from_iter<I: IntoIterator<Item = Option<PlanRef>>>(iter: I) -> Self {
        Self(
            iter.into_iter()
                .map(|plan| plan.map(LazyPlanRef::ready))
                .collect(),
        )
    }
}

impl From<Vec<Option<LazyPlanRef>>> for PlanSlots {
    fn from(slots: Vec<Option<LazyPlanRef>>) -> Self {
        Self(SmallVec::from_vec(slots))
    }
}

/// Built slots, every one present.
impl From<Vec<PlanRef>> for PlanSlots {
    fn from(children: Vec<PlanRef>) -> Self {
        // `PlanRef` and `Option<LazyPlanRef>` have the same layout, so this collect reuses the
        // vector's buffer, and `from_vec` keeps it when the slots do not fit inline.
        let slots: Vec<Option<LazyPlanRef>> = children
            .into_iter()
            .map(|child| Some(LazyPlanRef::ready(child)))
            .collect();
        Self(SmallVec::from_vec(slots))
    }
}

/// Built slots, every one present, without a temporary vector.
impl<const N: usize> From<[PlanRef; N]> for PlanSlots {
    fn from(children: [PlanRef; N]) -> Self {
        children.into_iter().map(Some).collect()
    }
}

impl fmt::Debug for PlanSlots {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PlanSlots")
            .field("len", &self.0.len())
            .field("present", &self.present().count())
            .field(
                "built",
                &self
                    .present()
                    .filter(|(_, lazy)| lazy.is_initialized())
                    .count(),
            )
            .finish()
    }
}
