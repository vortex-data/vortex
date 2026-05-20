// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Type-erased aggregate function ([`AggregateFnRef`]).

use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;
use std::sync::Arc;

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_utils::debug_with::DebugWith;

use crate::aggregate_fn::AccumulatorRef;
use crate::aggregate_fn::AggregateFnId;
use crate::aggregate_fn::AggregateFnVTable;
use crate::aggregate_fn::GroupedAccumulatorRef;
use crate::aggregate_fn::options::AggregateFnOptions;
use crate::aggregate_fn::typed::AggregateFnInner;
use crate::aggregate_fn::typed::DynAggregateFn;
use crate::dtype::DType;

/// A type-erased aggregate function, pairing a vtable with bound options behind a trait object.
///
/// This stores an [`AggregateFnVTable`] and its options behind an `Arc<dyn DynAggregateFn>`,
/// allowing heterogeneous storage and dispatch.
///
/// Use [`super::AggregateFn::new()`] to construct, and [`super::AggregateFn::erased()`] to
/// obtain an [`AggregateFnRef`].
#[derive(Clone)]
pub struct AggregateFnRef(pub(super) Arc<dyn DynAggregateFn>);

impl AggregateFnRef {
    /// Returns the ID of this aggregate function.
    pub fn id(&self) -> AggregateFnId {
        self.0.id()
    }

    /// Returns whether the aggregate function is of the given vtable type.
    pub fn is<V: AggregateFnVTable>(&self) -> bool {
        self.0.as_any().is::<AggregateFnInner<V>>()
    }

    /// Returns the typed options for this aggregate function if it matches the given vtable type.
    pub fn as_opt<V: AggregateFnVTable>(&self) -> Option<&V::Options> {
        self.downcast_inner::<V>().map(|inner| &inner.options)
    }

    /// Returns a reference to the typed vtable if it matches the given vtable type.
    pub fn vtable_ref<V: AggregateFnVTable>(&self) -> Option<&V> {
        self.downcast_inner::<V>().map(|inner| &inner.vtable)
    }

    /// Downcast the inner to the concrete `AggregateFnInner<V>`.
    fn downcast_inner<V: AggregateFnVTable>(&self) -> Option<&AggregateFnInner<V>> {
        self.0.as_any().downcast_ref::<AggregateFnInner<V>>()
    }

    /// Returns the typed options for this aggregate function if it matches the given vtable type.
    ///
    /// # Panics
    ///
    /// Panics if the vtable type does not match.
    pub fn as_<V: AggregateFnVTable>(&self) -> &V::Options {
        self.as_opt::<V>()
            .vortex_expect("Aggregate function options type mismatch")
    }

    /// The type-erased options for this aggregate function.
    pub fn options(&self) -> AggregateFnOptions<'_> {
        AggregateFnOptions { inner: &*self.0 }
    }

    /// Compute the return [`DType`] per group given the input element type.
    ///
    /// Returns `None` if the input dtype is not supported by the aggregate function.
    pub fn return_dtype(&self, input_dtype: &DType) -> Option<DType> {
        self.0.return_dtype(input_dtype)
    }

    /// DType of the intermediate accumulator state.
    ///
    /// Returns `None` if the input dtype is not supported by the aggregate function.
    pub fn state_dtype(&self, input_dtype: &DType) -> Option<DType> {
        self.0.state_dtype(input_dtype)
    }

    /// Create an accumulator for streaming aggregation.
    pub fn accumulator(&self, input_dtype: &DType) -> VortexResult<AccumulatorRef> {
        self.0.accumulator(input_dtype)
    }

    /// Create a grouped accumulator for grouped streaming aggregation.
    pub fn accumulator_grouped(&self, input_dtype: &DType) -> VortexResult<GroupedAccumulatorRef> {
        self.0.accumulator_grouped(input_dtype)
    }
}

impl Debug for AggregateFnRef {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AggregateFnRef")
            .field("vtable", &self.0.id())
            .field("options", &DebugWith(|fmt| self.0.options_debug(fmt)))
            .finish()
    }
}

impl Display for AggregateFnRef {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}(", self.0.id())?;
        self.0.options_display(f)?;
        write!(f, ")")
    }
}

impl PartialEq for AggregateFnRef {
    fn eq(&self, other: &Self) -> bool {
        self.0.id() == other.0.id() && self.0.options_eq(other.0.options_any())
    }
}
impl Eq for AggregateFnRef {}

impl Hash for AggregateFnRef {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.id().hash(state);
        self.0.options_hash(state);
    }
}
