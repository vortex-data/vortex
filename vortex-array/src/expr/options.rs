// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::fmt::Debug;
use std::fmt::Display;
use std::hash::Hash;
use std::hash::Hasher;

use vortex_error::VortexResult;

use crate::expr::ExprVTable;

/// An opaque handle to expression options.
pub struct ExpressionOptions<'a> {
    pub(super) vtable: &'a ExprVTable,
    pub(super) options: &'a dyn Any,
}

impl Display for ExpressionOptions<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.vtable.as_dyn().options_display(self.options, f)
    }
}

impl Debug for ExpressionOptions<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.vtable.as_dyn().options_debug(self.options, f)
    }
}

impl PartialEq for ExpressionOptions<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.vtable == other.vtable && self.vtable.as_dyn().options_eq(self.options, other.options)
    }
}
impl Eq for ExpressionOptions<'_> {}

impl Hash for ExpressionOptions<'_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.vtable.hash(state);
        self.vtable.as_dyn().options_hash(self.options, state);
    }
}

impl ExpressionOptions<'_> {
    /// Serialize the options to a byte vector.
    pub fn serialize(&self) -> VortexResult<Option<Vec<u8>>> {
        self.vtable.as_dyn().options_serialize(self.options)
    }
}

impl<'a> ExpressionOptions<'a> {
    /// Return the underlying `Any` reference.
    pub fn as_any(&self) -> &'a dyn Any {
        self.options
    }
}
