// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::arrays::Extension;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::builtins::ArrayFunctions;
use crate::zip::ZipReduce;

/// Zips the storage arrays of two extension arrays and wraps the result in the extension type.
impl ZipReduce for Extension {
    fn zip(
        if_true: ArrayView<'_, Extension>,
        if_false: &ArrayRef,
        mask: &ArrayRef,
    ) -> VortexResult<Option<ArrayRef>> {
        let Some(if_false) = if_false.as_opt::<Extension>() else {
            return Ok(None);
        };

        vortex_ensure!(
            if_true
                .ext_dtype()
                .eq_ignore_nullability(if_false.ext_dtype()),
            "zip requires if_true and if_false to share an extension type, got {} and {}",
            if_true.dtype(),
            if_false.dtype()
        );

        let storage = mask.zip(
            if_true.storage_array().clone(),
            if_false.storage_array().clone(),
        )?;
        let ext_dtype = if_true
            .ext_dtype()
            .with_nullability(storage.dtype().nullability());

        // Zip selects values from inputs with the same extension dtype.
        Ok(Some(
            ExtensionArray::try_new_structural(ext_dtype, storage)?.into_array(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::Extension;
    use vortex_array::arrays::ExtensionArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::Nullability;
    use vortex_array::extension::datetime::TimeUnit;
    use vortex_array::extension::datetime::Timestamp;
    use vortex_error::VortexResult;
    use vortex_mask::Mask;
    use vortex_session::VortexSession;

    use crate::builtins::ArrayFunctions;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty();
        vortex_array::initialize(&session);
        crate::initialize(&session);
        session
    });

    fn timestamps(
        values: impl IntoIterator<Item = Option<i64>>,
        nullability: Nullability,
    ) -> ArrayRef {
        let storage = match nullability {
            Nullability::NonNullable => PrimitiveArray::from_iter(
                values
                    .into_iter()
                    .map(|value| value.expect("non-nullable timestamps have no nulls")),
            ),
            Nullability::Nullable => PrimitiveArray::from_option_iter(values),
        };
        ExtensionArray::try_new(
            Timestamp::new(TimeUnit::Milliseconds, nullability).erased(),
            storage.into_array(),
            &mut SESSION.create_execution_ctx(),
        )
        .expect("timestamp storage is valid")
        .into_array()
    }

    /// A fragmented mask selects timestamps and their nulls from the chosen side, and the result
    /// is nullable when either side is nullable.
    #[rstest]
    #[case::non_nullable(Nullability::NonNullable, Nullability::NonNullable)]
    #[case::nullable_if_true(Nullability::Nullable, Nullability::NonNullable)]
    #[case::nullable_if_false(Nullability::NonNullable, Nullability::Nullable)]
    #[case::nullable(Nullability::Nullable, Nullability::Nullable)]
    fn zip_timestamps(
        #[case] true_nullability: Nullability,
        #[case] false_nullability: Nullability,
    ) -> VortexResult<()> {
        let len = 100i64;
        let true_value = |i: i64| (!true_nullability.is_nullable() || i % 7 != 0).then_some(i);
        let false_value =
            |i: i64| (!false_nullability.is_nullable() || i % 5 != 0).then_some(1_000 + i);
        let selected = |i: i64| i % 3 == 0;

        let if_true = timestamps((0..len).map(true_value), true_nullability);
        let if_false = timestamps((0..len).map(false_value), false_nullability);
        let mask = Mask::from_iter((0..len).map(selected));

        let mut ctx = SESSION.create_execution_ctx();
        let result = mask
            .into_array()
            .zip(if_true, if_false)?
            .execute::<ArrayRef>(&mut ctx)?;
        assert!(result.is::<Extension>());

        let expected = timestamps(
            (0..len).map(|i| {
                if selected(i) {
                    true_value(i)
                } else {
                    false_value(i)
                }
            }),
            true_nullability | false_nullability,
        );
        assert_arrays_eq!(result, expected, &mut ctx);
        Ok(())
    }
}
