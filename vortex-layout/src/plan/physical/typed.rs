// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::borrow::Cow;
use std::fmt;
use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;
use std::marker::PhantomData;
use std::ops::Deref;
use std::ops::Range;
use std::sync::Arc;

use vortex_array::SerializeMetadata;
use vortex_array::dtype::DType;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_mask::Mask;

use crate::plan::physical::PlanId;
use crate::plan::physical::PlanSlots;
use crate::plan::physical::PlanVTable;
use crate::plan::physical::display::PlanTreeDisplay;
use crate::plan::physical::pipeline::Chain;
use crate::plan::physical::pipeline::Compiler;

/// The combined allocation behind [`PlanRef`].
///
/// Common plan state is stored before the unsized `data` tail, so reading the operator ID, dtype,
/// row count, or slots does not dispatch through the operator vtable. Only `PlanData<V>` is
/// erased to [`DynPlan`].
struct PlanInner<D: ?Sized> {
    id: PlanId,
    dtype: DType,
    row_count: u64,
    slots: PlanSlots,
    data: D, // must be last for unsized coercion
}

/// Shared, erased handle to a plan operator.
#[derive(Clone)]
pub struct PlanRef(Arc<PlanInner<dyn DynPlan>>);

impl PlanRef {
    fn from_inner<V: PlanVTable>(inner: Arc<PlanInner<PlanData<V>>>) -> Self {
        let inner: Arc<PlanInner<dyn DynPlan>> = inner;
        Self(inner)
    }

    fn dyn_plan(&self) -> &dyn DynPlan {
        &self.0.data
    }

    /// Returns whether two references point at the same plan.
    pub fn ptr_eq(lhs: &Self, rhs: &Self) -> bool {
        Arc::ptr_eq(&lhs.0, &rhs.0)
    }

    /// Returns the operator ID.
    pub fn id(&self) -> PlanId {
        self.0.id
    }

    /// Returns the dtype produced by this plan.
    pub fn dtype(&self) -> &DType {
        &self.0.dtype
    }

    /// Returns the number of rows in this plan's row domain.
    pub fn row_count(&self) -> u64 {
        self.0.row_count
    }

    /// Returns the child slots without building any child.
    pub fn slots(&self) -> &PlanSlots {
        &self.0.slots
    }

    /// Returns the number of slots, present or absent, without building any child.
    pub fn nslots(&self) -> usize {
        self.0.slots.len()
    }

    /// Returns the plan in `slot`, building it on first access, or `None` when the slot is
    /// absent. Errors when `slot` is out of bounds.
    pub fn slot(&self, slot: usize) -> VortexResult<Option<PlanRef>> {
        Ok(self.0.slots.get(slot)?.cloned())
    }

    /// Returns the plan in `slot`, or an error when the slot is absent or out of bounds.
    pub fn slot_required(&self, slot: usize) -> VortexResult<PlanRef> {
        self.0.slots.required(slot).cloned()
    }

    /// The plans of the present slots in slot order, building each as it is visited. See
    /// [`PlanSlots::children`].
    pub fn children(&self) -> impl Iterator<Item = VortexResult<&PlanRef>> + '_ {
        self.0.slots.children()
    }

    /// Rebuilds this plan with `slots` stored outside its erased operator data.
    pub fn with_slots(&self, slots: impl Into<PlanSlots>) -> VortexResult<PlanRef> {
        self.dyn_plan().dyn_with_slots(self, slots.into())
    }

    /// Returns the display name of `slot`.
    pub fn slot_name(&self, slot: usize) -> Cow<'_, str> {
        self.dyn_plan().dyn_slot_name(self, slot)
    }

    /// Serializes operator-specific metadata, or `None` when the operator is not serializable.
    pub fn metadata(&self) -> Option<Vec<u8>> {
        self.dyn_plan().dyn_metadata(self)
    }

    /// Returns whether this plan uses vtable `V`.
    pub fn is<V: PlanVTable>(&self) -> bool {
        self.dyn_plan().as_any().is::<PlanData<V>>()
    }

    /// Downcasts this plan to vtable `V`.
    pub fn as_<V: PlanVTable>(&self) -> &Plan<V> {
        self.as_opt::<V>().vortex_expect("Failed to downcast")
    }

    /// Attempts to borrow this plan as a typed handle for vtable `V`.
    pub fn as_opt<V: PlanVTable>(&self) -> Option<&Plan<V>> {
        if !self.is::<V>() {
            return None;
        }

        // SAFETY: Plan<V> is transparent over PlanRef, and the type check above proves that its
        // erased tail contains PlanData<V>.
        Some(unsafe { &*(std::ptr::from_ref(self).cast::<Plan<V>>()) })
    }

    /// Compiles this plan over `rows` of its row domain, restricted to `mask`. See
    /// [`PlanVTable::compile`].
    pub fn compile(
        &self,
        rows: Range<u64>,
        mask: &Mask,
        compiler: &mut Compiler<'_>,
    ) -> VortexResult<Option<Chain>> {
        vortex_ensure!(
            rows.start <= rows.end && rows.end <= self.row_count(),
            "Compile rows {rows:?} exceed plan row count {}",
            self.row_count()
        );
        vortex_ensure!(
            mask.len() as u64 == rows.end - rows.start,
            "Compile mask length {} does not match rows {rows:?}",
            mask.len()
        );
        self.dyn_plan().dyn_compile(self, rows, mask, compiler)
    }

    /// Displays this plan and its descendants with the default plan extractors.
    pub fn display_tree(&self) -> PlanTreeDisplay<'_> {
        PlanTreeDisplay::default_display(self)
    }

    /// Creates a composable tree display with no extractors.
    pub fn tree_display_builder(&self) -> PlanTreeDisplay<'_> {
        PlanTreeDisplay::new(self)
    }
}

impl Display for PlanRef {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{}({}, rows={})",
            self.id(),
            self.dtype(),
            self.row_count()
        )?;
        self.dyn_plan().dyn_fmt(self, formatter)
    }
}

impl Debug for PlanRef {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Plan")
            .field("id", &self.0.id)
            .field("dtype", &self.0.dtype)
            .field("row_count", &self.0.row_count)
            .field("slots", &self.0.slots)
            .field("data", &&self.0.data)
            .finish()
    }
}

/// Pieces used to construct a typed plan.
pub struct PlanParts<V: PlanVTable> {
    /// The vtable identifying the operator.
    pub vtable: V,
    /// Logical dtype produced by the operator.
    pub dtype: DType,
    /// Number of rows in the operator's row domain.
    pub row_count: u64,
    /// Child slots, at the positions the operator defines.
    pub slots: PlanSlots,
    /// Operator-specific, non-child data.
    pub data: V::PlanData,
}

impl<V: PlanVTable> PlanParts<V> {
    /// Converts these parts into a typed plan.
    pub fn into_typed(self) -> Plan<V> {
        Plan::from_parts(self)
    }

    /// Erases these parts into a plan reference.
    pub fn into_plan(self) -> PlanRef {
        self.into_typed().into_plan()
    }
}

/// A plan keyed by identity: two keys are equal only when they hold the same plan, as
/// [`PlanRef::ptr_eq`] decides.
///
/// The key holds the plan, so its address cannot be freed and reused by another plan while the
/// key exists.
#[derive(Clone, Debug)]
pub(crate) struct PlanByPtr(pub(crate) PlanRef);

impl PartialEq for PlanByPtr {
    fn eq(&self, other: &Self) -> bool {
        PlanRef::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for PlanByPtr {}

impl Hash for PlanByPtr {
    fn hash<H: Hasher>(&self, state: &mut H) {
        Arc::as_ptr(&self.0.0).cast::<()>().hash(state);
    }
}

/// A typed, shared handle to a plan operator.
#[repr(transparent)]
pub struct Plan<V: PlanVTable> {
    inner: PlanRef,
    _vtable: PhantomData<V>,
}

impl<V: PlanVTable> Plan<V> {
    /// Constructs a plan from explicit parts.
    pub fn from_parts(parts: PlanParts<V>) -> Self {
        let inner = Arc::new(PlanInner {
            id: parts.vtable.id(),
            dtype: parts.dtype,
            row_count: parts.row_count,
            slots: parts.slots,
            data: PlanData {
                vtable: parts.vtable,
                data: parts.data,
            },
        });
        Self {
            inner: PlanRef::from_inner(inner),
            _vtable: PhantomData,
        }
    }

    fn typed_data(&self) -> &PlanData<V> {
        self.inner
            .dyn_plan()
            .as_any()
            .downcast_ref::<PlanData<V>>()
            .vortex_expect("Typed plan contains the wrong vtable")
    }

    /// Returns the vtable.
    pub fn vtable(&self) -> &V {
        &self.typed_data().vtable
    }

    /// Returns operator-specific data.
    pub fn data(&self) -> &V::PlanData {
        &self.typed_data().data
    }

    /// Returns the dtype produced by this plan.
    pub fn dtype(&self) -> &DType {
        self.inner.dtype()
    }

    /// Returns the number of rows in this plan's row domain.
    pub fn row_count(&self) -> u64 {
        self.inner.row_count()
    }

    /// Returns the child slots without building any child.
    pub fn slots(&self) -> &PlanSlots {
        self.inner.slots()
    }

    /// Returns the number of slots, present or absent, without building any child.
    pub fn nslots(&self) -> usize {
        self.inner.nslots()
    }

    /// Returns the plan in `slot`, building it on first access, or `None` when the slot is
    /// absent.
    pub fn slot(&self, slot: usize) -> VortexResult<Option<PlanRef>> {
        self.inner.slot(slot)
    }

    /// Returns the plan in `slot`, or an error when the slot is absent or out of bounds.
    pub fn slot_required(&self, slot: usize) -> VortexResult<PlanRef> {
        self.inner.slot_required(slot)
    }

    /// The plans of the present slots in slot order. See [`PlanSlots::children`].
    pub fn children(&self) -> impl Iterator<Item = VortexResult<&PlanRef>> + '_ {
        self.inner.children()
    }

    /// Erases this typed plan into a shared reference.
    pub fn to_plan(&self) -> PlanRef {
        self.inner.clone()
    }

    /// Erases this typed plan into a shared reference.
    pub fn into_plan(self) -> PlanRef {
        self.inner
    }
}

impl<V: PlanVTable> Clone for Plan<V> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            _vtable: PhantomData,
        }
    }
}

impl<V: PlanVTable> Debug for Plan<V> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        Debug::fmt(&self.inner, formatter)
    }
}

impl<V: PlanVTable> Display for Plan<V> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        Display::fmt(&self.inner, formatter)
    }
}

impl<V: PlanVTable> Deref for Plan<V> {
    type Target = V::PlanData;

    fn deref(&self) -> &Self::Target {
        self.data()
    }
}

impl<V: PlanVTable> From<Plan<V>> for PlanRef {
    fn from(value: Plan<V>) -> Self {
        value.into_plan()
    }
}

/// A vtable value paired with its operator-specific plan data.
struct PlanData<V: PlanVTable> {
    vtable: V,
    data: V::PlanData,
}

impl<V: PlanVTable> Debug for PlanData<V> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PlanData")
            .field("vtable", &self.vtable)
            .field("data", &self.data)
            .finish()
    }
}

/// Erased operator-specific behavior stored in the unsized tail of a [`PlanRef`].
#[doc(hidden)]
pub trait DynPlan: 'static + Send + Sync + Debug {
    /// Returns this operator data as [`Any`] for downcasting.
    fn as_any(&self) -> &dyn Any;

    /// Formats operator-specific fields.
    fn dyn_fmt(&self, plan: &PlanRef, formatter: &mut Formatter<'_>) -> fmt::Result;

    /// Clones operator data, runs its slot-replacement callback, and rebuilds the common node.
    fn dyn_with_slots(&self, plan: &PlanRef, slots: PlanSlots) -> VortexResult<PlanRef>;

    /// Returns the display name of `slot`.
    fn dyn_slot_name<'a>(&'a self, plan: &'a PlanRef, slot: usize) -> Cow<'a, str>;

    /// Serializes operator-specific metadata, or `None` when the operator is not serializable.
    fn dyn_metadata(&self, plan: &PlanRef) -> Option<Vec<u8>>;

    /// Compiles this operator. See [`PlanVTable::compile`].
    fn dyn_compile(
        &self,
        plan: &PlanRef,
        rows: Range<u64>,
        mask: &Mask,
        compiler: &mut Compiler<'_>,
    ) -> VortexResult<Option<Chain>>;
}

impl<V: PlanVTable> DynPlan for PlanData<V> {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dyn_fmt(&self, plan: &PlanRef, formatter: &mut Formatter<'_>) -> fmt::Result {
        <V as PlanVTable>::fmt(plan.as_::<V>(), formatter)
    }

    fn dyn_with_slots(&self, plan: &PlanRef, slots: PlanSlots) -> VortexResult<PlanRef> {
        let mut data = self.data.clone();
        V::with_slots(plan.as_::<V>(), &slots, &mut data)?;
        Ok(PlanParts {
            vtable: self.vtable.clone(),
            dtype: plan.dtype().clone(),
            row_count: plan.row_count(),
            slots,
            data,
        }
        .into_plan())
    }

    fn dyn_slot_name<'a>(&'a self, plan: &'a PlanRef, slot: usize) -> Cow<'a, str> {
        V::slot_name(plan.as_::<V>(), slot)
    }

    fn dyn_metadata(&self, plan: &PlanRef) -> Option<Vec<u8>> {
        V::metadata(plan.as_::<V>()).map(SerializeMetadata::serialize)
    }

    fn dyn_compile(
        &self,
        plan: &PlanRef,
        rows: Range<u64>,
        mask: &Mask,
        compiler: &mut Compiler<'_>,
    ) -> VortexResult<Option<Chain>> {
        V::compile(plan.as_::<V>(), rows, mask, compiler)
    }
}
