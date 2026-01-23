// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_scalar::Scalar;

use crate::arrays::FixedSizeListArray;
use crate::arrays::FixedSizeListVTable;
use crate::vtable::OperationsVTable;

impl OperationsVTable<FixedSizeListVTable> for FixedSizeListVTable {
    fn scalar_at(array: &FixedSizeListArray, index: usize) -> VortexResult<Scalar> {
        // By the preconditions we know that the list scalar is not null.
        let list = array.fixed_size_list_elements_at(index)?;
        let children_elements: Vec<Scalar> = (0..list.len())
            .map(|i| list.scalar_at(i))
            .collect::<VortexResult<_>>()?;

        debug_assert_eq!(children_elements.len(), array.list_size() as usize);

        Ok(Scalar::fixed_size_list(
            list.dtype().clone(),
            children_elements,
            array.dtype().nullability(),
        ))
    }
}
