// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#[macro_export]
macro_rules! assert_arrays_eq {
    ($expected:expr, $actual:expr) => {
        let expected: vortex_array::ArrayRef = $expected.into_array();
        let actual: vortex_array::ArrayRef = $actual.into_array();
        assert_eq!(expected.dtype(), actual.dtype());

        let expected_contents: Vec<_> = (0..expected.len())
            .map(|idx| expected.scalar_at(idx))
            .collect();
        let actual_contents: Vec<_> = (0..actual.len()).map(|idx| actual.scalar_at(idx)).collect();

        assert_eq!(expected_contents, actual_contents);
    };
}
