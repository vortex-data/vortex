// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Chunked;
use crate::arrays::ChunkedArray;
use crate::arrays::chunked::ChunkedArrayExt;
use crate::builtins::ArrayBuiltins;
use crate::scalar::Scalar;
use crate::scalar_fn::fns::fill_null::FillNullReduce;

impl FillNullReduce for Chunked {
    fn fill_null(
        array: ArrayView<'_, Chunked>,
        fill_value: &Scalar,
    ) -> VortexResult<Option<ArrayRef>> {
        let chunks = array
            .iter_chunks()
            .map(|c| c.fill_null(fill_value.clone()));
        chunks.process_results(|chunks| {
            // SAFETY: filling nulls gives every chunk the fill value's dtype.
            Some(unsafe {
                ChunkedArray::new_unchecked_sized(
                    chunks,
                    fill_value.dtype().clone(),
                    array.nchunks(),
                )
                .into_array()
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::BitBuffer;

    use crate::IntoArray;
    use crate::arrays::BoolArray;
    use crate::arrays::ChunkedArray;
    use crate::builtins::ArrayBuiltins;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::scalar::Scalar;
    use crate::validity::Validity;

    #[test]
    fn fill_null_chunks() {
        let chunked = ChunkedArray::try_new(
            vec![
                BoolArray::new(BitBuffer::new_set(5), Validity::AllInvalid).into_array(),
                BoolArray::new(BitBuffer::new_set(5), Validity::AllValid).into_array(),
            ],
            DType::Bool(Nullability::Nullable),
        )
        .unwrap();

        let filled = chunked.into_array().fill_null(Scalar::from(false)).unwrap();
        assert_eq!(*filled.dtype(), DType::Bool(Nullability::NonNullable));
    }
}
