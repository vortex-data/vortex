// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Expected values for files that older writers stored at noncanonical decimal widths.
//!
//! Field names record the original storage widths. Current writers use the canonical width for
//! each precision. Published files keep the original widths and null payloads.

use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::ArrayVTable;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::Decimal;
use vortex_array::arrays::DecimalArray;
use vortex_array::arrays::StructArray;
use vortex_array::dtype::BigCast;
use vortex_array::dtype::DecimalDType;
use vortex_array::match_each_decimal_value_type;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::fixtures::FlatLayoutFixture;

/// The number of rows in every legacy decimal fixture.
const ROWS: usize = 5;

/// The expected values of a column in a published decimal fixture.
struct LegacyDecimalColumn {
    name: &'static str,
    decimal_dtype: DecimalDType,
    slots: [i128; ROWS],
    validity: Option<[bool; ROWS]>,
}

impl LegacyDecimalColumn {
    fn build(&self) -> VortexResult<ArrayRef> {
        let values_type = self.decimal_dtype.canonical_values_type();
        let validity = match self.validity {
            Some(validity) => Validity::from_iter(validity),
            None => Validity::NonNullable,
        };
        match_each_decimal_value_type!(values_type, |T| {
            let values = self
                .slots
                .into_iter()
                .map(|slot| {
                    <T as BigCast>::from(slot).ok_or_else(|| {
                        vortex_err!("a slot value must fit in {values_type}, got {slot}")
                    })
                })
                .collect::<VortexResult<Buffer<T>>>()?;
            Ok(DecimalArray::try_new(values, self.decimal_dtype, validity)?.into_array())
        })
    }
}

/// Builds a non-nullable column.
fn column(
    name: &'static str,
    decimal_dtype: DecimalDType,
    slots: [i128; ROWS],
) -> LegacyDecimalColumn {
    LegacyDecimalColumn {
        name,
        decimal_dtype,
        slots,
        validity: None,
    }
}

/// Builds a nullable column.
fn nullable_column(
    name: &'static str,
    decimal_dtype: DecimalDType,
    slots: [i128; ROWS],
    validity: [bool; ROWS],
) -> LegacyDecimalColumn {
    LegacyDecimalColumn {
        name,
        decimal_dtype,
        slots,
        validity: Some(validity),
    }
}

/// Expected columns of a published legacy decimal file.
pub(super) struct LegacyDecimalFixture {
    name: &'static str,
    description: &'static str,
    columns: Vec<LegacyDecimalColumn>,
}

impl FlatLayoutFixture for LegacyDecimalFixture {
    fn name(&self) -> &str {
        self.name
    }

    fn description(&self) -> &str {
        self.description
    }

    fn expected_encodings(&self) -> Vec<ArrayId> {
        vec![Decimal.id()]
    }

    fn build(&self, _ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
        let fields = self
            .columns
            .iter()
            .map(|column| Ok((column.name, column.build()?)))
            .collect::<VortexResult<Vec<_>>>()?;
        Ok(StructArray::try_from_iter(fields)?.into_array())
    }
}

/// All legacy decimal fixtures.
pub(super) fn fixtures() -> [LegacyDecimalFixture; 5] {
    [
        narrow_storage(),
        gap_values(),
        high_scale(),
        wide_storage(),
        null_slots(),
    ]
}

/// Older releases stored these columns in narrower integers. Every value fits the decimal
/// precision of its original integer type.
fn narrow_storage() -> LegacyDecimalFixture {
    let max_p9 = 999_999_999;
    let max_p18 = 10i128.pow(18) - 1;
    let max_p38 = 10i128.pow(38) - 1;

    LegacyDecimalFixture {
        name: "decimal_narrow_storage.vortex",
        description: "Decimals that older releases stored in integers narrower than their \
                      precision requires",
        columns: vec![
            column("i8_p10_s2", DecimalDType::new(10, 2), [-99, -1, 0, 1, 99]),
            column(
                "i16_p18_s2",
                DecimalDType::new(18, 2),
                [-9_999, -1, 0, 1, 9_999],
            ),
            column(
                "i32_p18_s2",
                DecimalDType::new(18, 2),
                [-max_p9, -1, 0, 1, max_p9],
            ),
            column(
                "i64_p30_s2",
                DecimalDType::new(30, 2),
                [-max_p18, -1, 0, 1, max_p18],
            ),
            column(
                "i128_p40_s0",
                DecimalDType::new(40, 0),
                [-max_p38, -1, 0, 1, max_p38],
            ),
            column("i8_p76_s0", DecimalDType::new(76, 0), [-99, -1, 0, 1, 99]),
        ],
    }
}

/// Some values fit the original integer type but need more digits than its largest decimal
/// precision. For example, an older file stored 127 in an eight-bit integer.
fn gap_values() -> LegacyDecimalFixture {
    let i32_max: i128 = i32::MAX.into();
    let i64_max: i128 = i64::MAX.into();
    let ten_pow_18 = 10i128.pow(18);

    LegacyDecimalFixture {
        name: "decimal_gap_values.vortex",
        description: "Decimals with values that exceed the decimal precision of the narrow \
                      integers that older releases used",
        columns: vec![
            column(
                "i8_p10_s2_gap",
                DecimalDType::new(10, 2),
                [-128, 100, 101, 126, 127],
            ),
            column(
                "i8_p4_s2_gap",
                DecimalDType::new(4, 2),
                [-99, 0, 99, 100, 127],
            ),
            column(
                "i16_p10_s2_gap",
                DecimalDType::new(10, 2),
                [10_000, 10_001, 20_000, 32_766, 32_767],
            ),
            column(
                "i32_p18_s2_gap",
                DecimalDType::new(18, 2),
                [
                    1_000_000_000,
                    1_000_000_001,
                    1_500_000_000,
                    i32_max - 1,
                    i32_max,
                ],
            ),
            column(
                "i64_p38_s0_gap",
                DecimalDType::new(38, 0),
                [
                    ten_pow_18,
                    ten_pow_18 + 1,
                    5 * ten_pow_18,
                    i64_max - 1,
                    i64_max,
                ],
            ),
        ],
    }
}

/// Older releases stored these columns in narrower integers. The scale sets the minimum
/// precision the reader can give a child that holds the same values.
fn high_scale() -> LegacyDecimalFixture {
    LegacyDecimalFixture {
        name: "decimal_high_scale.vortex",
        description: "Decimals whose scale requires a wider integer than the one older \
                      releases stored",
        columns: vec![
            column("i8_p18_s10", DecimalDType::new(18, 10), [-99, -1, 0, 1, 99]),
            column("i8_p18_s5", DecimalDType::new(18, 5), [-99, -1, 0, 1, 99]),
            column(
                "i16_p38_s20",
                DecimalDType::new(38, 20),
                [-9_999, -1, 0, 1, 9_999],
            ),
            column("i8_p4_s4", DecimalDType::new(4, 4), [-99, -1, 0, 1, 99]),
        ],
    }
}

/// Older releases could retain a wide Arrow integer when a narrower type held the full precision.
fn wide_storage() -> LegacyDecimalFixture {
    let max_p18 = 10i128.pow(18) - 1;
    let max_p38 = 10i128.pow(38) - 1;

    LegacyDecimalFixture {
        name: "decimal_wide_storage.vortex",
        description: "Decimals that older releases stored in integers wider than their \
                      precision requires",
        columns: vec![
            column(
                "i128_p5_s2",
                DecimalDType::new(5, 2),
                [-99_999, -1, 0, 1, 99_999],
            ),
            column(
                "i128_p18_s2",
                DecimalDType::new(18, 2),
                [-max_p18, -1, 0, 1, max_p18],
            ),
            column(
                "i64_p4_s2",
                DecimalDType::new(4, 2),
                [-9_999, -1, 0, 1, 9_999],
            ),
            column("i32_p2_s1", DecimalDType::new(2, 1), [-99, -1, 0, 1, 99]),
            column(
                "i256_p38_s2",
                DecimalDType::new(38, 2),
                [-max_p38, -1, 0, 1, max_p38],
            ),
        ],
    }
}

/// Older files stored out-of-precision values in null slots. The canonical column retains its
/// raw payload, while the expected values for the other columns use zeros in null slots.
fn null_slots() -> LegacyDecimalFixture {
    let validity = [true, false, true, false, true];
    let i32_max: i128 = i32::MAX.into();

    LegacyDecimalFixture {
        name: "decimal_null_slots.vortex",
        description: "Nullable decimals with nonzero payloads in the null slots of older files",
        columns: vec![
            nullable_column(
                "canonical_i32_p9_s2",
                DecimalDType::new(9, 2),
                [-999_999_999, i32_max, 0, i32_max, 999_999_999],
                validity,
            ),
            nullable_column(
                "narrow_i8_p10_s2",
                DecimalDType::new(10, 2),
                [-99, 0, 0, 0, 99],
                validity,
            ),
            nullable_column(
                "narrow_i8_p4_s2",
                DecimalDType::new(4, 2),
                [-99, 0, 0, 0, 99],
                validity,
            ),
            nullable_column(
                "wide_i128_p5_s2",
                DecimalDType::new(5, 2),
                [-99_999, 0, 0, 0, 99_999],
                validity,
            ),
        ],
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::Decimal;
    use vortex_array::arrays::StructArray;
    use vortex_array::arrays::decimal::DecimalArrayExt;
    use vortex_array::arrays::struct_::StructArrayExt;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use super::null_slots;
    use crate::adapter;
    use crate::fixtures::FlatLayoutFixture;

    #[test]
    fn canonical_column_preserves_raw_null_slots() -> VortexResult<()> {
        let mut ctx = VortexSession::empty().create_execution_ctx();
        let fixture = null_slots().build(&mut ctx)?;
        let bytes = adapter::write_file_to_bytes(fixture)?;
        let file = adapter::read_file(bytes)?.execute::<StructArray>(&mut ctx)?;
        let column = file.unmasked_field_by_name("canonical_i32_p9_s2")?;

        assert_eq!(
            column.as_::<Decimal>().buffer::<i32>().as_slice(),
            &[-999_999_999, i32::MAX, 0, i32::MAX, 999_999_999],
        );
        Ok(())
    }
}
