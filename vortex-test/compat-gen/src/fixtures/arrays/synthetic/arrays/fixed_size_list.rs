// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::FixedSizeList;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::dtype::FieldNames;
use vortex_array::validity::Validity;
use vortex_buffer::buffer;
use vortex_error::VortexResult;

use crate::fixtures::FlatLayoutFixture;

pub struct FixedSizeListFixture;

impl FlatLayoutFixture for FixedSizeListFixture {
    fn name(&self) -> &str {
        "fixed_size_list.vortex"
    }

    fn description(&self) -> &str {
        "Fixed-size list arrays (e.g. 3-element vectors)"
    }

    fn expected_encodings(&self) -> Vec<ArrayId> {
        vec![FixedSizeList::ID]
    }

    fn build(&self) -> VortexResult<ArrayRef> {
        // 4 vectors of 3 f64 each
        let elements = PrimitiveArray::new(
            buffer![
                1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0
            ],
            Validity::NonNullable,
        );
        let fsl = FixedSizeListArray::try_new(elements.into_array(), 3, Validity::NonNullable, 4)?;

        // Nullable FSL: 4 vectors of 2 i32, second entry is null
        let nullable_elements = PrimitiveArray::new(
            buffer![10i32, 20, 0, 0, 50, 60, 70, 80],
            Validity::NonNullable,
        );
        let nullable_fsl = FixedSizeListArray::try_new(
            nullable_elements.into_array(),
            2,
            Validity::from_iter([true, false, true, true]),
            4,
        )?;

        let arr = StructArray::try_new(
            FieldNames::from(["vectors", "nullable_vectors"]),
            vec![fsl.into_array(), nullable_fsl.into_array()],
            4,
            Validity::NonNullable,
        )?;
        Ok(arr.into_array())
    }
}
