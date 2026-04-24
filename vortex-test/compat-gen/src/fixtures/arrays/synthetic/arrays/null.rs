// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::ArrayVTable;
use vortex_array::IntoArray;
use vortex_array::arrays::Null;
use vortex_array::arrays::NullArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::dtype::FieldNames;
use vortex_array::validity::Validity;
use vortex_buffer::buffer;
use vortex_error::VortexResult;

use crate::fixtures::FlatLayoutFixture;

pub struct NullFixture;

impl FlatLayoutFixture for NullFixture {
    fn name(&self) -> &str {
        "null.vortex"
    }

    fn description(&self) -> &str {
        "All-null column using NullArray alongside an integer column"
    }

    fn expected_encodings(&self) -> Vec<ArrayId> {
        vec![Null.id()]
    }

    fn build(&self) -> VortexResult<ArrayRef> {
        let null_col = NullArray::new(10);
        let int_col = PrimitiveArray::new(
            buffer![0i32, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            Validity::NonNullable,
        );

        let arr = StructArray::try_new(
            FieldNames::from(["nulls", "ids"]),
            vec![null_col.into_array(), int_col.into_array()],
            10,
            Validity::NonNullable,
        )?;
        Ok(arr.into_array())
    }
}
