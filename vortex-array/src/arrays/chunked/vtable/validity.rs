// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use vortex_error::VortexResult;

use crate::Array;
use crate::IntoArray;
use crate::arrays::ChunkedArray;
use crate::arrays::ChunkedVTable;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::validity::Validity;
use crate::vtable::ValidityVTable;

impl ValidityVTable<ChunkedVTable> for ChunkedVTable {
    fn validity(array: &ChunkedArray) -> VortexResult<Validity> {
        let validities: Vec<Validity> =
            array.chunks().iter().map(|c| c.validity()).try_collect()?;

        match validities.first() {
            // If there are no chunks, return the array's dtype nullability
            None => return Ok(array.dtype().nullability().into()),
            // If all chunks have the same non-array validity, return that validity directly
            // We skip Validity::Array since equality is very expensive.
            Some(first) if !matches!(first, Validity::Array(_)) => {
                let target = std::mem::discriminant(first);
                if validities
                    .iter()
                    .all(|v| std::mem::discriminant(v) == target)
                {
                    return Ok(first.clone());
                }
            }
            _ => {
                // Array validity or mixed validities, proceed to build the validity array
            }
        }

        Ok(Validity::Array(
            unsafe {
                ChunkedArray::new_unchecked(
                    validities
                        .into_iter()
                        .zip(array.chunks())
                        .map(|(v, chunk)| v.to_array(chunk.len()))
                        .collect(),
                    DType::Bool(Nullability::NonNullable),
                )
            }
            .into_array(),
        ))
    }
}
