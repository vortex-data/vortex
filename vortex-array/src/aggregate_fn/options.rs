// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::fmt::Debug;
use std::fmt::Display;
use std::hash::Hash;
use std::hash::Hasher;

use vortex_error::VortexResult;

use crate::aggregate_fn::typed::DynAggregateFn;

/// An opaque handle to aggregate function options.
pub struct AggregateFnOptions<'a> {
    pub(super) inner: &'a dyn DynAggregateFn,
}

impl Display for AggregateFnOptions<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.inner.options_display(f)
    }
}

impl Debug for AggregateFnOptions<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.inner.options_debug(f)
    }
}

impl PartialEq for AggregateFnOptions<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.inner.id() == other.inner.id() && self.inner.options_eq(other.inner.options_any())
    }
}
impl Eq for AggregateFnOptions<'_> {}

impl Hash for AggregateFnOptions<'_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.inner.id().hash(state);
        self.inner.options_hash(state);
    }
}

impl AggregateFnOptions<'_> {
    /// Serialize the options to a byte vector.
    pub fn serialize(&self) -> VortexResult<Option<Vec<u8>>> {
        self.inner.options_serialize()
    }
}

impl<'a> AggregateFnOptions<'a> {
    /// Return the underlying `Any` reference.
    pub fn as_any(&self) -> &'a dyn Any {
        self.inner.options_any()
    }
}
