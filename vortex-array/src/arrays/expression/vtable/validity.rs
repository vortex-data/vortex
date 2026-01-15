// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::arrays::expression::ExpressionArray;
use crate::arrays::expression::ExpressionVTable;
use crate::validity::Validity;
use crate::vtable::ValidityVTable;

impl ValidityVTable<ExpressionVTable> for ExpressionVTable {
    fn is_valid(array: &ExpressionArray, index: usize) -> bool {}

    fn all_valid(array: &ExpressionArray) -> bool {
        todo!()
    }

    fn all_invalid(array: &ExpressionArray) -> bool {
        todo!()
    }

    fn validity(array: &ExpressionArray) -> VortexResult<Validity> {
        todo!()
    }
}
