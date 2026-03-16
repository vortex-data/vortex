// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::dtype::FieldNames;
use vortex_array::validity::Validity;
use vortex_array::vtable::ArrayId;
use vortex_error::VortexResult;

use crate::fixtures::ArrayFixture;

pub struct ChunkedFixture;

impl ArrayFixture for ChunkedFixture {
    fn name(&self) -> &str {
        "chunked.vortex"
    }

    fn description(&self) -> &str {
        "ChunkedArray with 3 chunks of 1000 rows each containing deterministic u32 values"
    }

    fn expected_encodings(&self) -> Vec<ArrayId> {
        vec![Primitive::ID]
    }

    fn build(&self) -> VortexResult<ArrayRef> {
        let value_gen = |chunk_idx| {
            let values: Vec<u32> = (0u32..1000).map(|i| chunk_idx * 1000 + i).collect();
            let primitives =
                PrimitiveArray::new(vortex_buffer::Buffer::from(values), Validity::NonNullable);
            Ok(StructArray::try_new(
                FieldNames::from(["id"]),
                vec![primitives.into_array()],
                1000,
                Validity::NonNullable,
            )?
            .into_array())
        };

        Ok(
            ChunkedArray::from_iter((0u32..3).map(value_gen).collect::<VortexResult<Vec<_>>>()?)
                .into_array(),
        )
    }
}
