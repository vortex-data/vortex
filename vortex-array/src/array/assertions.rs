#[macro_export]
macro_rules! assert_arrays_eq {
    ($expected:expr, $actual:expr) => {
        let expected: $crate::ArrayData = $expected.into();
        let actual: $crate::ArrayData = $actual.into();
        assert_eq!(expected.dtype(), actual.dtype());

        let expected_contents = (0..expected.len())
            .map(|idx| scalar_at(&expected, idx).map(|x| x.into_value()))
            .collect::<VortexResult<Vec<_>>>()
            .unwrap();
        let actual_contents = (0..actual.len())
            .map(|idx| scalar_at(&expected, idx).map(|x| x.into_value()))
            .collect::<VortexResult<Vec<_>>>()
            .unwrap();

        assert_eq!(expected_contents, actual_contents);
    };
}
