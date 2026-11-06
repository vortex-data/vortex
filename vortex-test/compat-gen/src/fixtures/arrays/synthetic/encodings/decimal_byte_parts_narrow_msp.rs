// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compatibility coverage for stored MSP widths, including a wider legacy column.
//!
//! Narrow MSPs use the normal writer. A fixture writer widens the precision-4 MSP on the wire
//! while [`FlatLayoutFixture::build`] keeps a valid logical array.

use vortex::VortexSessionDefault;
use vortex::array::ArrayId;
use vortex::array::ArrayRef;
use vortex::array::ArrayVTable;
use vortex::array::IntoArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::StructArray;
use vortex::array::arrays::WidenArray;
use vortex::array::dtype::DType;
use vortex::array::dtype::DecimalDType;
use vortex::array::dtype::Nullability;
use vortex::array::dtype::PType;
use vortex::array::validity::Validity;
use vortex::buffer::Buffer;
use vortex::encodings::decimal_byte_parts::DecimalByteParts;
use vortex::encodings::decimal_byte_parts::DecimalBytePartsPlugin;
use vortex::encodings::decimal_byte_parts::decimal_byte_parts_v1_id;
use vortex::encodings::decimal_byte_parts::decimal_byte_parts_v2_id;
use vortex::error::VortexExpect;
use vortex::error::VortexResult;
use vortex::session::VortexSession;
use vortex_array::ArrayDeserialization;
use vortex_array::ArrayPlugin;
use vortex_array::ArraySerialization;
use vortex_array::ExecutionCtx;
use vortex_array::VortexSessionExecute;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::session::ArraySessionExt;

use super::N;
use crate::fixtures::FlatLayoutFixture;

/// DecimalByteParts columns whose MSP width differs from the canonical width of the decimal dtype.
///
/// The `decimal_byte_parts.vortex` fixture covers only `i64` MSPs.
pub struct DecimalBytePartsNarrowMspFixture;

impl FlatLayoutFixture for DecimalBytePartsNarrowMspFixture {
    fn name(&self) -> &str {
        "decimal_byte_parts_narrow_msp.vortex"
    }

    fn description(&self) -> &str {
        "DecimalByteParts arrays with an MSP narrower or wider than the canonical decimal width"
    }

    fn expected_encodings(&self) -> Vec<ArrayId> {
        vec![ArrayVTable::id(&DecimalByteParts)]
    }

    fn writer_session(&self) -> VortexSession {
        let session = VortexSession::default();
        session.arrays().register(LegacyMspWriter);
        session
    }

    fn build(&self, _ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
        let max_p18 = 10i64.pow(18) - 1;

        let msp_i8_p10_s2 = evenly_spaced(i8::MIN, i8::MAX).into_array();
        let msp_i16_p10_s2_gap = evenly_spaced(10_000i16, i16::MAX).into_array();
        let msp_i32_p18_s2 = evenly_spaced(-999_999_999i32, 999_999_999).into_array();
        let msp_i32_p18_s2_gap = evenly_spaced(1_000_000_000i32, i32::MAX).into_array();
        let msp_i64_p38_s2 = evenly_spaced(-max_p18, max_p18).into_array();
        let msp_i64_p4_s2_wide = evenly_spaced(-9_999i16, 9_999).into_array();

        // The null slots hold `i32::MAX`, a nonzero value. The remainder 3 keeps the first and the
        // last rows valid, so the valid values still reach both bounds.
        let is_null = |idx: usize| idx % 7 == 3;
        let nullable_values: Buffer<i32> = evenly_spaced(-999_999_999i32, 999_999_999)
            .into_iter()
            .enumerate()
            .map(|(idx, value)| if is_null(idx) { i32::MAX } else { value })
            .collect();
        let nullable_validity = Validity::from_iter((0..N).map(|idx| !is_null(idx)));
        let msp_i32_p18_s2_nullable =
            PrimitiveArray::new(nullable_values, nullable_validity).into_array();

        let arr = StructArray::try_from_iter([
            (
                "msp_i8_p10_s2",
                DecimalByteParts::try_new(
                    WidenArray::try_new(
                        msp_i8_p10_s2,
                        DType::Primitive(PType::I64, Nullability::NonNullable),
                    )?
                    .into_array(),
                    DecimalDType::new(10, 2),
                )?,
            ),
            (
                "msp_i16_p10_s2_gap",
                DecimalByteParts::try_new(
                    WidenArray::try_new(
                        msp_i16_p10_s2_gap,
                        DType::Primitive(PType::I64, Nullability::NonNullable),
                    )?
                    .into_array(),
                    DecimalDType::new(10, 2),
                )?,
            ),
            (
                "msp_i32_p18_s2",
                DecimalByteParts::try_new(
                    WidenArray::try_new(
                        msp_i32_p18_s2,
                        DType::Primitive(PType::I64, Nullability::NonNullable),
                    )?
                    .into_array(),
                    DecimalDType::new(18, 2),
                )?,
            ),
            (
                "msp_i32_p18_s2_gap",
                DecimalByteParts::try_new(
                    WidenArray::try_new(
                        msp_i32_p18_s2_gap,
                        DType::Primitive(PType::I64, Nullability::NonNullable),
                    )?
                    .into_array(),
                    DecimalDType::new(18, 2),
                )?,
            ),
            (
                "msp_i64_p38_s2",
                DecimalByteParts::try_new(msp_i64_p38_s2, DecimalDType::new(38, 2))?,
            ),
            (
                "msp_i32_p18_s2_nullable",
                DecimalByteParts::try_new(
                    WidenArray::try_new(
                        msp_i32_p18_s2_nullable,
                        DType::Primitive(PType::I64, Nullability::Nullable),
                    )?
                    .into_array(),
                    DecimalDType::new(18, 2),
                )?,
            ),
            (
                "msp_i64_p4_s2_wide",
                DecimalByteParts::try_new(msp_i64_p4_s2_wide, DecimalDType::new(4, 2))?,
            ),
        ])?;

        Ok(arr.into_array())
    }
}

/// Returns `N` values from `min` to `max`, with both bounds included.
///
/// The steps are as even as integer division allows. When the range holds fewer than `N` values,
/// some values repeat, and every value of the range appears.
fn evenly_spaced<T>(min: T, max: T) -> Buffer<T>
where
    T: Copy + Into<i128> + TryFrom<i128>,
{
    let (min, max) = (min.into(), max.into());
    let last = (N - 1) as i128;

    (0..=last)
        .map(|idx| {
            T::try_from(min + (max - min) * idx / last)
                .ok()
                .vortex_expect("an evenly spaced value lies between min and max")
        })
        .collect()
}

/// Writes the wider legacy column without constructing a DBP that violates its MSP invariant.
struct LegacyMspWriter;

impl ArrayPlugin for LegacyMspWriter {
    fn id(&self) -> ArrayId {
        ArrayVTable::id(&DecimalByteParts)
    }

    fn read_ids(&self) -> Vec<ArrayId> {
        vec![]
    }

    fn write_ids(&self) -> Vec<ArrayId> {
        vec![decimal_byte_parts_v1_id(), decimal_byte_parts_v2_id()]
    }

    fn serialize(
        &self,
        array: &ArrayRef,
        session: &VortexSession,
    ) -> VortexResult<Option<ArraySerialization>> {
        let Some(mut written) = DecimalBytePartsPlugin.serialize(array, session)? else {
            return Ok(None);
        };
        if array.dtype().as_decimal_opt() == Some(&DecimalDType::new(4, 2)) {
            let mut ctx = session.create_execution_ctx();
            written.children[0] = written.children[0]
                .cast(DType::Primitive(PType::I64, Nullability::NonNullable))?
                .execute::<PrimitiveArray>(&mut ctx)?
                .into_array();
            // Frozen v1 metadata: field 1 is the MSP PType, and field 2 defaults to zero.
            written.metadata = vec![8, PType::I64 as u8];
        }

        Ok(Some(written))
    }

    fn deserialize(
        &self,
        _parts: ArrayDeserialization<'_>,
        _session: &VortexSession,
    ) -> VortexResult<ArrayRef> {
        vortex::error::vortex_bail!("the legacy MSP fixture adapter only writes arrays")
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::Struct;
    use vortex_array::arrays::struct_::StructArrayExt;
    use vortex_array::dtype::PType;
    use vortex_array::session::ArraySessionExt;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use super::DecimalBytePartsNarrowMspFixture;
    use crate::fixtures::FlatLayoutFixture;

    #[test]
    fn writer_preserves_stored_msp_widths() -> VortexResult<()> {
        let fixture = DecimalBytePartsNarrowMspFixture;
        let array = fixture.build_written(&mut VortexSession::empty().create_execution_ctx())?;
        let fields = array.as_::<Struct>();
        let session = fixture.writer_session();
        for (name, ptype) in [
            ("msp_i8_p10_s2", PType::I8),
            ("msp_i16_p10_s2_gap", PType::I16),
            ("msp_i32_p18_s2", PType::I32),
            ("msp_i64_p4_s2_wide", PType::I64),
        ] {
            let field = fields.unmasked_field_by_name(name)?;
            let written = session
                .array_serialize(field)?
                .expect("fixture DBP is serializable");
            assert_eq!(written.metadata, vec![8, ptype as u8]);
            assert_eq!(written.children[0].dtype().as_ptype(), ptype);
        }

        Ok(())
    }
}
