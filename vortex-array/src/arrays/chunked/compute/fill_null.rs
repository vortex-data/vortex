// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::ChunkedArray;
use crate::arrays::ChunkedVTable;
use crate::builtins::ArrayBuiltins;
use crate::expr::FillNullReduce;
use crate::scalar::Scalar;

impl FillNullReduce for ChunkedVTable {
    fn fill_null(array: &ChunkedArray, fill_value: &Scalar) -> VortexResult<Option<ArrayRef>> {
        let new_chunks = array
            .chunks()
            .iter()
            .map(|c| c.fill_null(fill_value.clone()))
            .collect::<VortexResult<Vec<_>>>()?;

        // SAFETY: wrapping each chunk in ScalarFnArray preserves the same DType across all chunks.
        Ok(Some(
            unsafe { ChunkedArray::new_unchecked(new_chunks, fill_value.dtype().clone()) }
                .into_array(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::BitBuffer;
    use vortex_dtype::DType;
    use vortex_dtype::Nullability;

    use crate::array::Array;
    use crate::arrays::BoolArray;
    use crate::arrays::ChunkedArray;
    use crate::builtins::ArrayBuiltins;
    use crate::scalar::Scalar;
    use crate::validity::Validity;

    #[test]
    fn fill_null_chunks() {
        let chunked = ChunkedArray::try_new(
            vec![
                BoolArray::new(BitBuffer::new_set(5), Validity::AllInvalid).to_array(),
                BoolArray::new(BitBuffer::new_set(5), Validity::AllValid).to_array(),
            ],
            DType::Bool(Nullability::Nullable),
        )
        .unwrap();

        let filled = chunked.to_array().fill_null(Scalar::from(false)).unwrap();
        assert_eq!(*filled.dtype(), DType::Bool(Nullability::NonNullable));
    }
}
