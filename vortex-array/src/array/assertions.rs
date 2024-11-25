#[macro_export]
macro_rules! assert_arrays_eq {
    ($expected:expr, $actual:expr) => {
        let expected: $crate::ArrayData = $expected.into_array();
        let actual: $crate::ArrayData = $actual.into_array();
        assert_eq!(expected.dtype(), actual.dtype());

        let expected_contents = (0..expected.len())
            .map(|idx| scalar_at(&expected, idx))
            .collect::<VortexResult<Vec<_>>>()
            .unwrap();
        let actual_contents = (0..actual.len())
            .map(|idx| scalar_at(&expected, idx))
            .collect::<VortexResult<Vec<_>>>()
            .unwrap();

        assert_eq!(expected_contents, actual_contents);
    };
}
