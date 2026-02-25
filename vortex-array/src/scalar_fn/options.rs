// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::fmt::Debug;
use std::fmt::Display;
use std::hash::Hash;
use std::hash::Hasher;

use vortex_error::VortexResult;

use crate::scalar_fn::ScalarFnPlugin;

/// An opaque handle to expression options.
pub struct ScalarFnOptions<'a> {
    pub(crate) vtable: &'a ScalarFnPlugin,
    pub(crate) options: &'a dyn Any,
}

impl Display for ScalarFnOptions<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.vtable.as_dyn().options_display(self.options, f)
    }
}

impl Debug for ScalarFnOptions<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.vtable.as_dyn().options_debug(self.options, f)
    }
}

impl PartialEq for ScalarFnOptions<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.vtable == other.vtable && self.vtable.as_dyn().options_eq(self.options, other.options)
    }
}
impl Eq for ScalarFnOptions<'_> {}

impl Hash for ScalarFnOptions<'_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.vtable.hash(state);
        self.vtable.as_dyn().options_hash(self.options, state);
    }
}

impl ScalarFnOptions<'_> {
    /// Serialize the options to a byte vector.
    pub fn serialize(&self) -> VortexResult<Option<Vec<u8>>> {
        self.vtable.as_dyn().options_serialize(self.options)
    }
}

impl<'a> ScalarFnOptions<'a> {
    /// Return the underlying `Any` reference.
    pub fn as_any(&self) -> &'a dyn Any {
        self.options
    }
}
