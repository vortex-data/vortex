// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod arrays;
#[expect(clippy::cast_possible_truncation)]
mod encodings;

use crate::fixtures::FlatLayoutFixture;

/// All synthetic fixtures (arrays + encodings).
pub fn fixtures() -> Vec<Box<dyn FlatLayoutFixture>> {
    let mut fixtures = Vec::new();
    fixtures.extend(arrays::fixtures());
    fixtures.extend(encodings::fixtures());
    fixtures
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::Decimal;
    use vortex_array::arrays::DecimalArray;
    use vortex_array::arrays::Struct;
    use vortex_array::arrays::StructArray;
    use vortex_array::arrays::decimal::DecimalArrayExt;
    use vortex_array::arrays::struct_::StructArrayExt;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::BigCast;
    use vortex_array::dtype::DecimalType;
    use vortex_array::match_each_decimal_value_type;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use super::arrays::legacy_decimal_fixtures;
    use super::fixtures;
    use crate::adapter;
    use crate::fixtures::FlatLayoutFixture;
    use crate::fixtures::check_expected_encodings;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    /// Returns the raw value of every slot of `array`, including null slots.
    fn raw_slots(array: &DecimalArray) -> Vec<i128> {
        match_each_decimal_value_type!(array.values_type(), |T| {
            array
                .buffer::<T>()
                .iter()
                .map(|&value| {
                    <i128 as BigCast>::from(value)
                        .vortex_expect("every legacy slot value is an i128 in its column spec")
                })
                .collect()
        })
    }

    #[test]
    fn roundtrip_fixtures_to_bytes() -> VortexResult<()> {
        for fixture in fixtures() {
            let mut ctx = SESSION.create_execution_ctx();
            let expected = fixture.build(&mut ctx)?;
            let written = fixture.build_written(&mut ctx)?;
            assert_arrays_eq!(written, expected, &mut ctx);

            check_expected_encodings(&written, fixture.as_ref())?;
            adapter::compute_all_stats(&written, &mut ctx)?;
            let bytes = adapter::write_file_to_bytes(written)?;
            let roundtripped = adapter::read_file(bytes)?;
            assert_arrays_eq!(roundtripped, expected, &mut ctx);
        }

        Ok(())
    }

    #[test]
    fn fixtures_build_canonical_decimal_widths() -> VortexResult<()> {
        for fixture in fixtures() {
            let mut ctx = SESSION.create_execution_ctx();
            let expected = fixture.build(&mut ctx)?;
            for node in expected.depth_first_traversal() {
                let Some(decimal) = node.as_opt::<Decimal>() else {
                    continue;
                };
                let canonical_type =
                    DecimalType::smallest_decimal_value_type(&decimal.decimal_dtype());
                assert_eq!(
                    decimal.values_type(),
                    canonical_type,
                    "a decimal leaf of {} with dtype {}",
                    fixture.name(),
                    node.dtype(),
                );
            }
        }

        Ok(())
    }

    #[test]
    fn legacy_decimal_fixtures_keep_stored_widths() -> VortexResult<()> {
        for fixture in legacy_decimal_fixtures() {
            let mut ctx = SESSION.create_execution_ctx();
            let written = fixture.build_written(&mut ctx)?;
            let bytes = adapter::write_file_to_bytes(written.clone())?;
            let read_back = adapter::read_file(bytes)?.execute::<StructArray>(&mut ctx)?;
            let written = written.as_::<Struct>();

            for column in fixture.columns() {
                let context = format!("column {} of {}", column.name, fixture.name());

                // A typo in a column spec can make a column canonical, and the column then tests
                // nothing. The name states the intended stored width, and only a column whose
                // name starts with `canonical_` is meant to be stored at the canonical width.
                let precision = column.decimal_dtype.precision();
                let scale = column.decimal_dtype.scale();
                let spec_name = format!("{}_p{precision}_s{scale}", column.stored_type);
                assert!(
                    column.name.contains(&spec_name),
                    "{context}: the name must contain {spec_name}",
                );
                let canonical_type =
                    DecimalType::smallest_decimal_value_type(&column.decimal_dtype);
                assert_eq!(
                    column.stored_type == canonical_type,
                    column.name.starts_with("canonical_"),
                    "{context}: only a column named `canonical_*` has the canonical width \
                     {canonical_type}",
                );

                // A width above the canonical width can no longer be constructed, so such a
                // column is written at the canonical width with zeroed null slots.
                let written_type = column.written_type();
                let written_slots = if written_type == column.stored_type {
                    column.slots
                } else {
                    column.canonical_slots()
                };
                let written_field = written.unmasked_field_by_name(column.name)?;
                assert_eq!(
                    written_field.as_::<Decimal>().values_type(),
                    written_type,
                    "{context}",
                );

                let read_field = read_back
                    .unmasked_field_by_name(column.name)?
                    .clone()
                    .execute::<DecimalArray>(&mut ctx)?;
                assert_eq!(
                    read_field.values_type(),
                    written_type,
                    "{context} after a file round trip",
                );
                assert_eq!(
                    raw_slots(&read_field),
                    written_slots,
                    "{context} after a file round trip",
                );
            }
        }

        Ok(())
    }
}
