// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex::array::ArrayId;
use vortex::array::ArrayRef;
use vortex::array::ArrayVTable;
use vortex::array::IntoArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::StructArray;
use vortex::array::dtype::FieldNames;
use vortex::array::validity::Validity;
use vortex::encodings::pco::Pco;
use vortex::error::VortexResult;
use vortex_array::ExecutionCtx;

use super::N;
use crate::fixtures::FlatLayoutFixture;

/// 8-bit Pco patterns.
///
/// Kept separate from [`PcoFixture`](super::pco::PcoFixture) because a published fixture's
/// schema is frozen: `check --mode superset` decodes each stored file and compares it against
/// a freshly built one of the same name, so adding fields to an existing fixture fails against
/// every version already in the store. A new file is simply skipped by older versions.
pub struct Pco8BitFixture;

impl FlatLayoutFixture for Pco8BitFixture {
    fn name(&self) -> &str {
        "pco_8bit.vortex"
    }

    fn description(&self) -> &str {
        "8-bit integer patterns for Pco encoding"
    }

    fn expected_encodings(&self) -> Vec<ArrayId> {
        vec![Pco.id()]
    }

    fn build(&self, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
        let gradient_u8: PrimitiveArray = (0..N).map(|i| (i % 251) as u8).collect();
        let saturated_u8: PrimitiveArray = (0..N)
            .map(|i| if i % 64 == 0 { u8::MAX } else { 0 })
            .collect();
        let negative_i8: PrimitiveArray = (0..N).map(|i| (-128 + (i % 256) as i32) as i8).collect();
        let nullable_i8 = PrimitiveArray::from_option_iter(
            (0..N).map(|i| (i % 5 != 0).then_some((-64 + (i % 129) as i32) as i8)),
        );

        let arr = StructArray::try_new(
            FieldNames::from(["gradient_u8", "saturated_u8", "negative_i8", "nullable_i8"]),
            vec![
                Pco::from_primitive(gradient_u8.as_view(), 8, 0, ctx)?.into_array(),
                Pco::from_primitive(saturated_u8.as_view(), 8, 0, ctx)?.into_array(),
                Pco::from_primitive(negative_i8.as_view(), 8, 0, ctx)?.into_array(),
                Pco::from_primitive(nullable_i8.as_view(), 8, 0, ctx)?.into_array(),
            ],
            N,
            Validity::NonNullable,
        )?;

        Ok(arr.into_array())
    }
}
