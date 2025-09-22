// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::VarBinArray;
use crate::operator::OperatorHash;
use crate::vtable::ValidityHelper;
use std::hash::{Hash, Hasher};

impl Hash for VarBinArray {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.dtype.hash(state);
        OperatorHash(self.bytes()).hash(state);
        OperatorHash(self.offsets()).hash(state);
        OperatorHash(self.validity()).hash(state);
    }
}

impl PartialEq for VarBinArray {
    fn eq(&self, other: &Self) -> bool {
        self.dtype == other.dtype
            && OperatorHash(self.bytes()) == OperatorHash(other.bytes())
            && OperatorHash(self.offsets()) == OperatorHash(other.offsets())
            && OperatorHash(self.validity()) == OperatorHash(other.validity())
    }
}
impl Eq for VarBinArray {}

// TODO(ngates): impl Operator
