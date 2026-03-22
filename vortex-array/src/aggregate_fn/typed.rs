// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Typed and inner representations of aggregate functions.
//!
//! - [`AggregateFn<V>`]: The public typed wrapper, parameterized by a concrete
//!   [`AggregateFnVTable`].
//! - [`AggregateFnInner<V>`]: The private inner struct that holds the vtable + options.
//! - [`DynAggregateFn`]: The private sealed trait for type-erased dispatch (bound, options in
//!   self).

use std::any::Any;
use std::fmt;
use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;
use std::sync::Arc;

use vortex_error::VortexResult;

use crate::aggregate_fn::Accumulator;
use crate::aggregate_fn::AccumulatorRef;
use crate::aggregate_fn::AggregateFnId;
use crate::aggregate_fn::AggregateFnRef;
use crate::aggregate_fn::AggregateFnVTable;
use crate::aggregate_fn::GroupedAccumulator;
use crate::aggregate_fn::GroupedAccumulatorRef;
use crate::dtype::DType;

/// An object-safe, sealed trait for bound aggregate function dispatch.
///
/// Options are stored inside the implementing [`AggregateFnInner<V>`], not passed externally.
/// This is the sole trait behind [`AggregateFnRef`]'s `Arc<dyn DynAggregateFn>`.
pub(super) trait DynAggregateFn: 'static + Send + Sync + super::sealed::Sealed {
    fn as_any(&self) -> &dyn Any;
    fn id(&self) -> AggregateFnId;
    fn options_any(&self) -> &dyn Any;

    fn coerce_args(&self, input_dtype: &DType) -> VortexResult<DType>;
    fn return_dtype(&self, input_dtype: &DType) -> Option<DType>;
    fn state_dtype(&self, input_dtype: &DType) -> Option<DType>;
    fn accumulator(&self, input_dtype: &DType) -> VortexResult<AccumulatorRef>;
    fn accumulator_grouped(&self, input_dtype: &DType) -> VortexResult<GroupedAccumulatorRef>;

    fn options_serialize(&self) -> VortexResult<Option<Vec<u8>>>;
    fn options_eq(&self, other_options: &dyn Any) -> bool;
    fn options_hash(&self, hasher: &mut dyn Hasher);
    fn options_display(&self, f: &mut Formatter<'_>) -> fmt::Result;
    fn options_debug(&self, f: &mut Formatter<'_>) -> fmt::Result;
}

/// The private inner representation of a bound aggregate function, pairing a vtable with its
/// options.
///
/// This is the sole implementor of [`DynAggregateFn`], enabling [`AggregateFnRef`] to safely
/// downcast back to the concrete vtable type via [`Any`].
pub(super) struct AggregateFnInner<V: AggregateFnVTable> {
    pub(super) vtable: V,
    pub(super) options: V::Options,
}

impl<V: AggregateFnVTable> DynAggregateFn for AggregateFnInner<V> {
    #[inline(always)]
    fn as_any(&self) -> &dyn Any {
        self
    }

    #[inline(always)]
    fn id(&self) -> AggregateFnId {
        V::id(&self.vtable)
    }

    fn options_any(&self) -> &dyn Any {
        &self.options
    }

    fn coerce_args(&self, input_dtype: &DType) -> VortexResult<DType> {
        V::coerce_args(&self.vtable, &self.options, input_dtype)
    }

    fn return_dtype(&self, input_dtype: &DType) -> Option<DType> {
        V::return_dtype(&self.vtable, &self.options, input_dtype)
    }

    fn state_dtype(&self, input_dtype: &DType) -> Option<DType> {
        V::partial_dtype(&self.vtable, &self.options, input_dtype)
    }

    fn accumulator(&self, input_dtype: &DType) -> VortexResult<AccumulatorRef> {
        Ok(Box::new(Accumulator::try_new(
            self.vtable.clone(),
            self.options.clone(),
            input_dtype.clone(),
        )?))
    }

    fn accumulator_grouped(&self, input_dtype: &DType) -> VortexResult<GroupedAccumulatorRef> {
        Ok(Box::new(GroupedAccumulator::try_new(
            self.vtable.clone(),
            self.options.clone(),
            input_dtype.clone(),
        )?))
    }

    fn options_serialize(&self) -> VortexResult<Option<Vec<u8>>> {
        V::serialize(&self.vtable, &self.options)
    }

    fn options_eq(&self, other_options: &dyn Any) -> bool {
        other_options
            .downcast_ref::<V::Options>()
            .is_some_and(|o| self.options == *o)
    }

    fn options_hash(&self, mut hasher: &mut dyn Hasher) {
        self.options.hash(&mut hasher);
    }

    fn options_display(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Display::fmt(&self.options, f)
    }

    fn options_debug(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Debug::fmt(&self.options, f)
    }
}

/// A typed aggregate function instance, parameterized by a concrete [`AggregateFnVTable`].
///
/// You can construct one via [`new()`], and erase the type with [`erased()`] to obtain an
/// [`AggregateFnRef`].
///
/// [`new()`]: AggregateFn::new
/// [`erased()`]: AggregateFn::erased
pub struct AggregateFn<V: AggregateFnVTable>(pub(super) Arc<AggregateFnInner<V>>);

impl<V: AggregateFnVTable> AggregateFn<V> {
    /// Create a new typed aggregate function instance.
    pub fn new(vtable: V, options: V::Options) -> Self {
        Self(Arc::new(AggregateFnInner { vtable, options }))
    }

    /// Returns a reference to the vtable.
    pub fn vtable(&self) -> &V {
        &self.0.vtable
    }

    /// Returns a reference to the options.
    pub fn options(&self) -> &V::Options {
        &self.0.options
    }

    /// Erase the concrete type information, returning a type-erased [`AggregateFnRef`].
    pub fn erased(self) -> AggregateFnRef {
        AggregateFnRef(self.0)
    }
}
