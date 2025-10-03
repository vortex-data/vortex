// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::{Hash, Hasher};

use crate::arrays::VarBinArray;
use crate::operator::{OperatorEq, OperatorHash};
use crate::vtable::ValidityHelper;

impl OperatorHash for VarBinArray {
    fn operator_hash<H: Hasher>(&self, state: &mut H) {
        self.dtype.hash(state);
        self.bytes().operator_hash(state);
        self.offsets().operator_hash(state);
        self.validity().operator_hash(state);
    }
}

impl OperatorEq for VarBinArray {
    fn operator_eq(&self, other: &Self) -> bool {
        self.dtype == other.dtype
            && self.bytes().operator_eq(other.bytes())
            && self.offsets().operator_eq(other.offsets())
            && self.validity().operator_eq(other.validity())
    }
}

// TODO(ngates): impl Operator
