// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compat fixtures that write decimal columns at a stored width other than the canonical width.
//!
//! Older writers stored one decimal dtype at any width that held its values, so old files hold
//! narrow, wide and Arrow-style forms. Each fixture here writes such forms with
//! [`FlatLayoutFixture::build_written`], with the raw value of every slot, including nonzero
//! values in null slots. [`FlatLayoutFixture::build`] returns the same logical values at the
//! canonical width, so `check` compares each old file with a canonical array.
//!
//! The overrides depend on [`DecimalArray::try_new`] accepting a stored width that differs from
//! the canonical width. Once it rejects such a width, `build_written` returns that error, and
//! `generate` and the round-trip test fail. The overrides must then be deleted. The published files
//! keep the old forms.

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
use vortex_array::dtype::DecimalType;
use vortex_array::match_each_decimal_value_type;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::fixtures::FlatLayoutFixture;

/// The number of rows in every legacy decimal fixture.
const ROWS: usize = 5;

/// One column of a legacy decimal fixture.
pub struct LegacyDecimalColumn {
    /// The field name of the column.
    pub name: &'static str,
    /// The logical dtype of the column.
    pub decimal_dtype: DecimalDType,
    /// The width at which the written form stores the column.
    pub stored_type: DecimalType,
    /// The raw value of every slot, including null slots.
    pub slots: [i128; ROWS],
    /// The validity of every slot, or `None` for a non-nullable column.
    pub validity: Option<[bool; ROWS]>,
}

impl LegacyDecimalColumn {
    /// Builds the column at its stored width with the raw slot values.
    fn written(&self) -> VortexResult<ArrayRef> {
        decimal_array(
            self.decimal_dtype,
            self.stored_type,
            self.slots,
            self.validity(),
        )
    }

    /// Builds the column at the canonical width.
    ///
    /// A column that is already stored at the canonical width keeps its raw slot values, so its
    /// expected form is its written form. A column that changes width gets zeros in its null
    /// slots, because a raw null-slot value need not fit the canonical width.
    fn expected(&self) -> VortexResult<ArrayRef> {
        let canonical_type = DecimalType::smallest_decimal_value_type(&self.decimal_dtype);
        if canonical_type == self.stored_type {
            return self.written();
        }

        let mut slots = self.slots;
        if let Some(validity) = self.validity {
            for (slot, valid) in slots.iter_mut().zip(validity) {
                if !valid {
                    *slot = 0;
                }
            }
        }

        decimal_array(self.decimal_dtype, canonical_type, slots, self.validity())
    }

    fn validity(&self) -> Validity {
        match self.validity {
            Some(validity) => Validity::from_iter(validity),
            None => Validity::NonNullable,
        }
    }
}

/// Builds a decimal array that stores `slots` at `values_type`.
fn decimal_array(
    decimal_dtype: DecimalDType,
    values_type: DecimalType,
    slots: [i128; ROWS],
    validity: Validity,
) -> VortexResult<ArrayRef> {
    match_each_decimal_value_type!(values_type, |T| {
        let values = slots
            .into_iter()
            .map(|slot| {
                <T as BigCast>::from(slot).ok_or_else(|| {
                    vortex_err!("a slot value must fit in {values_type}, got {slot}")
                })
            })
            .collect::<VortexResult<Buffer<T>>>()?;
        Ok(DecimalArray::try_new(values, decimal_dtype, validity)?.into_array())
    })
}

/// Builds a non-nullable column.
fn column(
    name: &'static str,
    decimal_dtype: DecimalDType,
    stored_type: DecimalType,
    slots: [i128; ROWS],
) -> LegacyDecimalColumn {
    LegacyDecimalColumn {
        name,
        decimal_dtype,
        stored_type,
        slots,
        validity: None,
    }
}

/// Builds a nullable column.
fn nullable_column(
    name: &'static str,
    decimal_dtype: DecimalDType,
    stored_type: DecimalType,
    slots: [i128; ROWS],
    validity: [bool; ROWS],
) -> LegacyDecimalColumn {
    LegacyDecimalColumn {
        name,
        decimal_dtype,
        stored_type,
        slots,
        validity: Some(validity),
    }
}

/// A compat fixture that stores its decimal columns at the widths of a [`LegacyDecimalColumn`].
pub struct LegacyDecimalFixture {
    name: &'static str,
    description: &'static str,
    columns: Vec<LegacyDecimalColumn>,
}

impl LegacyDecimalFixture {
    #[cfg(test)]
    pub fn columns(&self) -> &[LegacyDecimalColumn] {
        &self.columns
    }

    fn build_struct(
        &self,
        build_column: impl Fn(&LegacyDecimalColumn) -> VortexResult<ArrayRef>,
    ) -> VortexResult<ArrayRef> {
        let fields = self
            .columns
            .iter()
            .map(|column| Ok((column.name, build_column(column)?)))
            .collect::<VortexResult<Vec<_>>>()?;

        Ok(StructArray::try_from_iter(fields)?.into_array())
    }
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
        self.build_struct(LegacyDecimalColumn::expected)
    }

    fn build_written(&self, _ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
        self.build_struct(LegacyDecimalColumn::written)
    }
}

/// All legacy decimal fixtures.
pub fn fixtures() -> [LegacyDecimalFixture; 5] {
    [
        narrow_storage(),
        gap_values(),
        high_scale(),
        wide_storage(),
        null_slots(),
    ]
}

/// Columns stored narrower than the canonical width, with values that fit the stored width at the
/// precision of the stored width.
fn narrow_storage() -> LegacyDecimalFixture {
    let max_p9 = 999_999_999;
    let max_p18 = 10i128.pow(18) - 1;
    let max_p38 = 10i128.pow(38) - 1;

    LegacyDecimalFixture {
        name: "decimal_narrow_storage.vortex",
        description: "Decimal arrays stored narrower than the canonical width of their precision",
        columns: vec![
            column(
                "i8_p10_s2",
                DecimalDType::new(10, 2),
                DecimalType::I8,
                [-99, -1, 0, 1, 99],
            ),
            column(
                "i16_p18_s2",
                DecimalDType::new(18, 2),
                DecimalType::I16,
                [-9_999, -1, 0, 1, 9_999],
            ),
            column(
                "i32_p18_s2",
                DecimalDType::new(18, 2),
                DecimalType::I32,
                [-max_p9, -1, 0, 1, max_p9],
            ),
            column(
                "i64_p30_s2",
                DecimalDType::new(30, 2),
                DecimalType::I64,
                [-max_p18, -1, 0, 1, max_p18],
            ),
            column(
                "i128_p40_s0",
                DecimalDType::new(40, 0),
                DecimalType::I128,
                [-max_p38, -1, 0, 1, max_p38],
            ),
            column(
                "i8_p76_s0",
                DecimalDType::new(76, 0),
                DecimalType::I8,
                [-99, -1, 0, 1, 99],
            ),
        ],
    }
}

/// Columns stored narrower than the canonical width, with values that fit the stored width but
/// need a precision above the precision of the stored width.
fn gap_values() -> LegacyDecimalFixture {
    let i32_max: i128 = i32::MAX.into();
    let i64_max: i128 = i64::MAX.into();
    let ten_pow_18 = 10i128.pow(18);

    LegacyDecimalFixture {
        name: "decimal_gap_values.vortex",
        description: "Decimal arrays stored narrower than canonical, with values that need the \
                      precision of the next width",
        columns: vec![
            column(
                "i8_p10_s2_gap",
                DecimalDType::new(10, 2),
                DecimalType::I8,
                [-128, 100, 101, 126, 127],
            ),
            column(
                "i8_p4_s2_gap",
                DecimalDType::new(4, 2),
                DecimalType::I8,
                [-99, 0, 99, 100, 127],
            ),
            column(
                "i16_p10_s2_gap",
                DecimalDType::new(10, 2),
                DecimalType::I16,
                [10_000, 10_001, 20_000, 32_766, 32_767],
            ),
            column(
                "i32_p18_s2_gap",
                DecimalDType::new(18, 2),
                DecimalType::I32,
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
                DecimalType::I64,
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

/// Columns stored narrower than the canonical width, where the scale sets the smallest precision
/// that a narrower child can have.
fn high_scale() -> LegacyDecimalFixture {
    LegacyDecimalFixture {
        name: "decimal_high_scale.vortex",
        description: "Decimal arrays stored narrower than canonical, with a scale near the \
                      precision",
        columns: vec![
            column(
                "i8_p18_s10",
                DecimalDType::new(18, 10),
                DecimalType::I8,
                [-99, -1, 0, 1, 99],
            ),
            column(
                "i8_p18_s5",
                DecimalDType::new(18, 5),
                DecimalType::I8,
                [-99, -1, 0, 1, 99],
            ),
            column(
                "i16_p38_s20",
                DecimalDType::new(38, 20),
                DecimalType::I16,
                [-9_999, -1, 0, 1, 9_999],
            ),
            column(
                "i8_p4_s4",
                DecimalDType::new(4, 4),
                DecimalType::I8,
                [-99, -1, 0, 1, 99],
            ),
        ],
    }
}

/// Columns stored wider than the canonical width, as an import of an Arrow decimal type can store
/// them.
fn wide_storage() -> LegacyDecimalFixture {
    let max_p18 = 10i128.pow(18) - 1;
    let max_p38 = 10i128.pow(38) - 1;

    LegacyDecimalFixture {
        name: "decimal_wide_storage.vortex",
        description: "Decimal arrays stored wider than the canonical width of their precision",
        columns: vec![
            column(
                "i128_p5_s2",
                DecimalDType::new(5, 2),
                DecimalType::I128,
                [-99_999, -1, 0, 1, 99_999],
            ),
            column(
                "i128_p18_s2",
                DecimalDType::new(18, 2),
                DecimalType::I128,
                [-max_p18, -1, 0, 1, max_p18],
            ),
            column(
                "i64_p4_s2",
                DecimalDType::new(4, 2),
                DecimalType::I64,
                [-9_999, -1, 0, 1, 9_999],
            ),
            column(
                "i32_p2_s1",
                DecimalDType::new(2, 1),
                DecimalType::I32,
                [-99, -1, 0, 1, 99],
            ),
            column(
                "i256_p38_s2",
                DecimalDType::new(38, 2),
                DecimalType::I256,
                [-max_p38, -1, 0, 1, max_p38],
            ),
        ],
    }
}

/// Nullable columns with nonzero values in their null slots. Each null slot holds a value that is
/// outside the precision, or outside the precision of the narrowest child that the valid values
/// allow.
fn null_slots() -> LegacyDecimalFixture {
    let validity = [true, false, true, false, true];
    let i32_max: i128 = i32::MAX.into();
    let outside_p5 = 10i128.pow(30);

    LegacyDecimalFixture {
        name: "decimal_null_slots.vortex",
        description: "Nullable decimal arrays with nonzero values in null slots",
        columns: vec![
            nullable_column(
                "canonical_i32_p9_s2",
                DecimalDType::new(9, 2),
                DecimalType::I32,
                [-999_999_999, i32_max, 0, i32_max, 999_999_999],
                validity,
            ),
            nullable_column(
                "narrow_i8_p10_s2",
                DecimalDType::new(10, 2),
                DecimalType::I8,
                [-99, 127, 0, -128, 99],
                validity,
            ),
            nullable_column(
                "narrow_i8_p4_s2",
                DecimalDType::new(4, 2),
                DecimalType::I8,
                [-99, 127, 0, 127, 99],
                validity,
            ),
            nullable_column(
                "wide_i128_p5_s2",
                DecimalDType::new(5, 2),
                DecimalType::I128,
                [-99_999, outside_p5, 0, outside_p5, 99_999],
                validity,
            ),
        ],
    }
}
