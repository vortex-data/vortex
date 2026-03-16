// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod arrays;
#[allow(clippy::cast_possible_truncation)]
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
    use vortex_array::IntoArray;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::StructArray;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::FieldNames;
    use vortex_array::validity::Validity;

    use super::fixtures;
    use crate::adapter;
    use crate::fixtures::check_expected_encodings;

    fn boundary_length_array(len: usize) -> vortex_error::VortexResult<vortex_array::ArrayRef> {
        let ints = PrimitiveArray::from_iter((0..i32::try_from(len)?).map(|i| i - 17));
        let nullable_ints = PrimitiveArray::from_option_iter(
            (0..len as i64).map(|i| if i % 5 == 0 { None } else { Some(i * 3 - 7) }),
        );
        let bools = BoolArray::from_iter((0..len).map(|i| i % 3 == 0));
        let strings = VarBinViewArray::from_iter_nullable_str((0..len).map(|i| match i % 5 {
            0 => None,
            1 => Some(""),
            2 => Some("edge"),
            3 => Some("boundary-length-string"),
            _ => Some("zz"),
        }));

        Ok(StructArray::try_new(
            FieldNames::from(["ints", "nullable_ints", "bools", "strings"]),
            vec![
                ints.into_array(),
                nullable_ints.into_array(),
                bools.into_array(),
                strings.into_array(),
            ],
            len,
            Validity::NonNullable,
        )?
        .into_array())
    }

    #[test]
    fn roundtrip_fixtures_to_bytes() {
        for fixture in fixtures() {
            eprintln!("--- writing {} to bytes ---", fixture.name());
            let array = fixture.build().unwrap();
            check_expected_encodings(&array, fixture.as_ref()).unwrap();
            let bytes = adapter::write_file_to_bytes(array.clone()).unwrap();
            let roundtripped = adapter::read_file(bytes).unwrap();
            assert_arrays_eq!(array, roundtripped);
            eprintln!("  OK: {}", fixture.name());
        }
    }

    #[test]
    fn roundtrip_boundary_lengths_to_bytes() {
        const BOUNDARY_LENGTHS: [usize; 15] = [
            0, 1, 2, 31, 32, 63, 64, 127, 128, 255, 256, 511, 512, 1023, 1025,
        ];

        for len in BOUNDARY_LENGTHS {
            eprintln!(
                "--- writing shared boundary fixture length {} to bytes ---",
                len
            );
            let boundary_array = boundary_length_array(len).unwrap();
            if len == 0 {
                assert!(adapter::write_file_to_bytes(boundary_array).is_err());
                continue;
            }
            let bytes = adapter::write_file_to_bytes(boundary_array).unwrap();
            let _array = adapter::read_file(bytes).unwrap();
        }
    }
}
