// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::borrow::Cow;
use std::fmt;
use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;
use std::marker::PhantomData;
use std::ops::Deref;
use std::ops::Range;
use std::sync::Arc;

use vortex_array::MaskFuture;
use vortex_array::SerializeMetadata;
use vortex_array::dtype::DType;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::plan::PlanArrayFuture;
use crate::plan::PlanChildren;
use crate::plan::PlanExecutionContext;
use crate::plan::PlanId;
use crate::plan::PlanVTable;
use crate::plan::display::PlanTreeDisplay;

/// The combined allocation behind [`PlanRef`].
///
/// Common plan state is stored before the unsized `data` tail, so reading the operator ID, dtype,
/// row count, or children does not dispatch through the operator vtable. Only `PlanData<V>` is
/// erased to [`DynPlan`].
struct PlanInner<D: ?Sized> {
    id: PlanId,
    dtype: DType,
    row_count: u64,
    children: PlanChildren,
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

    /// Returns the common child container without initializing any child.
    pub fn children(&self) -> &PlanChildren {
        &self.0.children
    }

    /// Returns the number of children without initializing any child.
    pub fn child_count(&self) -> usize {
        self.0.children.len()
    }

    /// Returns the child at `index`, initializing it on first access.
    pub fn child(&self, index: usize) -> VortexResult<Option<PlanRef>> {
        self.0.children.get(index)
    }

    /// Returns the child at `index`, or an error when the index is out of bounds.
    pub fn child_required(&self, index: usize) -> VortexResult<PlanRef> {
        self.child(index)?
            .ok_or_else(|| vortex_err!("Missing plan child {index}"))
    }

    /// Rebuilds this plan with `children` stored outside its erased operator data.
    pub fn with_children(&self, children: impl Into<PlanChildren>) -> VortexResult<PlanRef> {
        self.dyn_plan().dyn_with_children(self, children.into())
    }

    /// Returns the display name of the child at `index`.
    pub fn child_name(&self, index: usize) -> Cow<'_, str> {
        self.dyn_plan().dyn_child_name(self, index)
    }

    /// Serializes operator-specific metadata, or `None` when the operator is not serializable.
    pub fn metadata(&self) -> Option<Vec<u8>> {
        self.dyn_plan().dyn_metadata(self)
    }

    /// Executes this plan over `row_range`, returning the values selected by `mask`.
    pub fn execute(
        &self,
        ctx: &PlanExecutionContext,
        row_range: &Range<u64>,
        mask: MaskFuture,
    ) -> VortexResult<PlanArrayFuture> {
        self.dyn_plan().dyn_execute(self, ctx, row_range, mask)
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
            .field("children", &self.0.children)
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
    /// Child operators, in stable logical order.
    pub children: PlanChildren,
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
            children: parts.children,
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

    /// Returns the common child container without initializing any child.
    pub fn children(&self) -> &PlanChildren {
        self.inner.children()
    }

    /// Returns a child, initializing it on first access.
    pub fn child(&self, index: usize) -> VortexResult<Option<PlanRef>> {
        self.inner.child(index)
    }

    /// Returns the child at `index`, or an error when the index is out of bounds.
    pub fn child_required(&self, index: usize) -> VortexResult<PlanRef> {
        self.inner.child_required(index)
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

    /// Clones operator data, runs its child-replacement callback, and rebuilds the common node.
    fn dyn_with_children(&self, plan: &PlanRef, children: PlanChildren) -> VortexResult<PlanRef>;

    /// Returns the display name of the child at `index`.
    fn dyn_child_name<'a>(&'a self, plan: &'a PlanRef, index: usize) -> Cow<'a, str>;

    /// Serializes operator-specific metadata, or `None` when the operator is not serializable.
    fn dyn_metadata(&self, plan: &PlanRef) -> Option<Vec<u8>>;

    /// Executes this plan over `row_range`, returning the values selected by `mask`.
    fn dyn_execute(
        &self,
        plan: &PlanRef,
        ctx: &PlanExecutionContext,
        row_range: &Range<u64>,
        mask: MaskFuture,
    ) -> VortexResult<PlanArrayFuture>;
}

impl<V: PlanVTable> DynPlan for PlanData<V> {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dyn_fmt(&self, plan: &PlanRef, formatter: &mut Formatter<'_>) -> fmt::Result {
        <V as PlanVTable>::fmt(plan.as_::<V>(), formatter)
    }

    fn dyn_with_children(&self, plan: &PlanRef, children: PlanChildren) -> VortexResult<PlanRef> {
        let mut data = self.data.clone();
        V::with_children(plan.as_::<V>(), &children, &mut data)?;
        Ok(PlanParts {
            vtable: self.vtable.clone(),
            dtype: plan.dtype().clone(),
            row_count: plan.row_count(),
            children,
            data,
        }
        .into_plan())
    }

    fn dyn_child_name<'a>(&'a self, plan: &'a PlanRef, index: usize) -> Cow<'a, str> {
        V::child_name(plan.as_::<V>(), index)
    }

    fn dyn_metadata(&self, plan: &PlanRef) -> Option<Vec<u8>> {
        V::metadata(plan.as_::<V>()).map(SerializeMetadata::serialize)
    }

    fn dyn_execute(
        &self,
        plan: &PlanRef,
        ctx: &PlanExecutionContext,
        row_range: &Range<u64>,
        mask: MaskFuture,
    ) -> VortexResult<PlanArrayFuture> {
        V::execute(plan.as_::<V>(), ctx, row_range, mask)
    }
}
