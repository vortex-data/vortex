// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_error::VortexResult;

use super::BloomPartial;

pub(super) fn accumulate_extension(
    array: &ExtensionArray,
    partial: &mut BloomPartial,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let storage = array.storage_array().clone();
    let canonical = storage.execute::<Canonical>(ctx)?;

    super::accumulate_canonical(&canonical, partial, ctx)
}

#[cfg(test)]
mod tests {
    use vortex_array::IntoArray;
    use vortex_array::arrays::ExtensionArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::extension::datetime::TimeUnit;
    use vortex_array::extension::datetime::Timestamp;
    use vortex_array::scalar::Scalar;
    use vortex_error::VortexResult;

    use crate::layouts::zoned::aggregates::bloom_filter::test_utils::build_filter;
    use crate::layouts::zoned::aggregates::bloom_filter::test_utils::setup;

    #[test]
    fn hashes_extension_values_through_storage() -> VortexResult<()> {
        let ctx = setup()?;
        let ext_dtype = Timestamp::new(TimeUnit::Milliseconds, Nullability::NonNullable).erased();
        let bloom_filter = build_filter(
            ExtensionArray::new(
                ext_dtype.clone(),
                PrimitiveArray::from_iter([1_000i64, 2_000, 3_000]).into_array(),
            )
            .into_array(),
            DType::Extension(ext_dtype.clone()),
            ctx,
        )?;

        for value in [1_000i64, 2_000, 3_000] {
            let scalar = Scalar::extension_ref(
                ext_dtype.clone(),
                Scalar::primitive(value, Nullability::NonNullable),
            );
            assert!(bloom_filter.contains_scalar(&scalar)?);
        }

        let absent = Scalar::extension_ref(
            ext_dtype,
            Scalar::primitive(4_000i64, Nullability::NonNullable),
        );
        assert!(!bloom_filter.contains_scalar(&absent)?);
        Ok(())
    }

    #[test]
    fn ignores_null_extension_values() -> VortexResult<()> {
        let ctx = setup()?;
        let ext_dtype = Timestamp::new(TimeUnit::Milliseconds, Nullability::Nullable).erased();
        let bloom_filter = build_filter(
            ExtensionArray::new(
                ext_dtype.clone(),
                PrimitiveArray::from_option_iter([Some(1_000i64), None, Some(3_000)]).into_array(),
            )
            .into_array(),
            DType::Extension(ext_dtype.clone()),
            ctx,
        )?;

        let present = Scalar::extension_ref(
            ext_dtype.clone(),
            Scalar::primitive(1_000i64, Nullability::Nullable),
        );
        let null_slot_value =
            Scalar::extension_ref(ext_dtype, Scalar::primitive(0i64, Nullability::Nullable));
        assert!(bloom_filter.contains_scalar(&present)?);
        assert!(!bloom_filter.contains_scalar(&null_slot_value)?);
        Ok(())
    }
}
