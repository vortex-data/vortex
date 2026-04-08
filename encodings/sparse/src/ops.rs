// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::scalar::Scalar;
use vortex_array::vtable::OperationsVTable;
use vortex_error::VortexResult;

use crate::Sparse;

impl OperationsVTable<Sparse> for Sparse {
    fn scalar_at(
        array: ArrayView<'_, Sparse>,
        index: usize,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        Ok(array
            .patches()
            .get_patched(index)?
            .unwrap_or_else(|| array.fill_scalar().clone()))
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::IntoArray;
    use vortex_array::ToCanonical;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_buffer::buffer;

    use crate::Sparse;

    #[test]
    fn slice_partially_invalid() {
        let values = buffer![0u64].into_array();
        let indices = buffer![0u8].into_array();

        let sparse = Sparse::try_new(indices, values, 1000, 999u64.into()).unwrap();
        let sliced = sparse.slice(0..1000).unwrap();
        let mut expected = vec![999u64; 1000];
        expected[0] = 0;

        let values = sliced.to_primitive();
        assert_arrays_eq!(values, PrimitiveArray::from_iter(expected));
    }
}
