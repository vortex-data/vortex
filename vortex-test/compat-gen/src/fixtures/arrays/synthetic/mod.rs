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
    use vortex_array::assert_arrays_eq;

    use super::fixtures;
    use crate::adapter;
    use crate::fixtures::check_expected_encodings;

    #[test]
    fn roundtrip_fixtures_to_bytes() {
        for fixture in fixtures() {
            let array = fixture.build().unwrap();
            check_expected_encodings(&array, fixture.as_ref()).unwrap();
            let bytes = adapter::write_file_to_bytes(array.clone()).unwrap();
            let roundtripped = adapter::read_file(bytes).unwrap();
            assert_arrays_eq!(array, roundtripped);
        }
    }
}
