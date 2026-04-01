// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::dtype::FieldNames;
use vortex_array::validity::Validity;
use vortex_error::VortexResult;

use crate::fixtures::FlatLayoutFixture;

pub struct ChunkedFixture;

impl FlatLayoutFixture for ChunkedFixture {
    fn name(&self) -> &str {
        "chunked.vortex"
    }

    fn description(&self) -> &str {
        "ChunkedArray with variable-size chunks containing nullable data"
    }

    fn expected_encodings(&self) -> Vec<ArrayId> {
        vec![Primitive::ID]
    }

    fn build(&self) -> VortexResult<ArrayRef> {
        // Variable chunk sizes: 500, 1000, 250
        let chunk_sizes: [u32; 3] = [500, 1000, 250];
        let mut offset = 0u32;

        let chunks = chunk_sizes
            .iter()
            .map(|&size| {
                let ids: PrimitiveArray = (offset..offset + size).collect();
                let nullable_vals = PrimitiveArray::from_option_iter(
                    (offset..offset + size)
                        .map(|i| if i % 7 == 0 { None } else { Some(i as i64 * 3) }),
                );
                offset += size;
                Ok(StructArray::try_new(
                    FieldNames::from(["id", "nullable_val"]),
                    vec![ids.into_array(), nullable_vals.into_array()],
                    size as usize,
                    Validity::NonNullable,
                )?
                .into_array())
            })
            .collect::<VortexResult<Vec<_>>>()?;

        Ok(ChunkedArray::from_iter(chunks).into_array())
    }
}
