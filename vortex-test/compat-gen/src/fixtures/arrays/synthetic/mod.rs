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
    use vortex_array::arrays::decimal::DecimalArrayExt;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::DecimalType;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use super::fixtures;
    use crate::adapter;
    use crate::fixtures::check_expected_encodings;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

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
}
