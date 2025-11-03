// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::display::DisplayFormat;
use std::any::Any;
use std::fmt::{Debug, Formatter};
use std::hash::{Hash, Hasher};
use vortex_utils::dyn_traits::{DynEq, DynHash};

/// Trait for instance data of a Vortex expression.
pub trait ExprInstance: 'static + Send + Sync + Debug + DynEq + DynHash {
    fn as_any(&self) -> &dyn Any;
    fn fmt_as(&self, df: DisplayFormat, f: &mut Formatter) -> std::fmt::Result;
}

impl Hash for dyn ExprInstance + '_ {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.dyn_hash(state);
    }
}

impl PartialEq for dyn ExprInstance + '_ {
    fn eq(&self, other: &Self) -> bool {
        self.dyn_eq(other.as_any())
    }
}
impl Eq for dyn ExprInstance + '_ {}

impl ExprInstance for () {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn fmt_as(&self, _df: DisplayFormat, f: &mut Formatter) -> std::fmt::Result {
        write!(f, "")
    }
}
