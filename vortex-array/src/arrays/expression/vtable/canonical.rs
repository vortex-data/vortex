// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect;

use crate::Array;
use crate::Canonical;
use crate::arrays::expression::ExpressionArray;
use crate::arrays::expression::ExpressionVTable;
use crate::vtable::CanonicalVTable;

impl CanonicalVTable<ExpressionVTable> for ExpressionVTable {
    fn canonicalize(array: &ExpressionArray) -> Canonical {
        array
            .expression
            .evaluate(&array.to_array())
            .vortex_expect("Canonicalize should be fallible!")
            .to_canonical()
    }
}
