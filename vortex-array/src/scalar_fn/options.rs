// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::fmt::Debug;
use std::fmt::Display;
use std::hash::Hash;
use std::hash::Hasher;

use vortex_error::VortexResult;

use crate::scalar_fn::typed::DynScalarFn;

/// An opaque handle to expression options.
pub struct ScalarFnOptions<'a> {
    pub(super) inner: &'a dyn DynScalarFn,
}

impl Display for ScalarFnOptions<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.inner.options_display(f)
    }
}

impl Debug for ScalarFnOptions<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.inner.options_debug(f)
    }
}

impl PartialEq for ScalarFnOptions<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.inner.id() == other.inner.id() && self.inner.options_eq(other.inner.options_any())
    }
}
impl Eq for ScalarFnOptions<'_> {}

impl Hash for ScalarFnOptions<'_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.inner.id().hash(state);
        self.inner.options_hash(state);
    }
}

impl ScalarFnOptions<'_> {
    /// Serialize the options to a byte vector.
    pub fn serialize(&self) -> VortexResult<Option<Vec<u8>>> {
        self.inner.options_serialize()
    }
}

impl<'a> ScalarFnOptions<'a> {
    /// Return the underlying `Any` reference.
    pub fn as_any(&self) -> &'a dyn Any {
        self.inner.options_any()
    }
}
