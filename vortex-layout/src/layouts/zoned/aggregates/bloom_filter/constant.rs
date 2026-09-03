// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::arrays::ConstantArray;
use vortex_array::scalar::Scalar;
use vortex_error::VortexResult;

use super::BloomPartial;

/// Accumulates a constant sharing the same [Scalar]-based hash path used
/// for Bloom-filter pruning.
///
/// Constant accumulation implementation is different from canonicals,
/// in the sense that canonicals' value extraction happens outside
/// the [BloomPartial], while for a [Scalar]/`Constant`,
/// value extraction happens inside the partial/filter.
///
/// This is because constant accumulation and pruning both use [Scalar],
/// so a common place for both is the [BloomPartial].
pub(super) fn accumulate_constant(
    constant: &ConstantArray,
    partial: &mut BloomPartial,
) -> VortexResult<()> {
    let scalar: &Scalar = constant.scalar();
    partial.insert_scalar(scalar)?;

    Ok(())
}

#[cfg(test)]
mod tests {

    use vortex_array::aggregate_fn::AggregateFnVTable;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::extension::datetime::TimeUnit;
    use vortex_array::extension::datetime::Timestamp;
    use vortex_array::scalar::Scalar;
    use vortex_error::VortexResult;

    use crate::layouts::zoned::aggregates::bloom_filter::BloomFilter;
    use crate::layouts::zoned::aggregates::bloom_filter::BloomOptions;
    use crate::layouts::zoned::aggregates::bloom_filter::constant::accumulate_constant;

    #[test]
    fn nulls_are_omitted() {
        let bloom = BloomFilter;
        let mut zone_partial = bloom
            .empty_partial(
                &BloomOptions::default(),
                &DType::Primitive(PType::I32, Nullability::Nullable),
            )
            .unwrap();

        assert!(
            accumulate_constant(
                &ConstantArray::new(
                    Scalar::null(DType::Primitive(PType::I32, Nullability::Nullable)),
                    1,
                ),
                &mut zone_partial,
            )
            .is_ok(),
            "expected to return Ok() for null scalars"
        )
    }

    #[test]
    fn null_always_returns_false() {
        let bloom = BloomFilter;
        let zone_partial = bloom
            .empty_partial(
                &BloomOptions::default(),
                &DType::Primitive(PType::I32, Nullability::Nullable),
            )
            .unwrap();

        assert!(
            !zone_partial
                .contains_scalar(&Scalar::null(DType::Primitive(
                    PType::I32,
                    Nullability::Nullable
                )))
                .expect("to return valid bool"),
            "expected to return false for null scalars"
        )
    }

    #[test]
    fn valid_extension_is_a_member() -> VortexResult<()> {
        let ext_dtype = Timestamp::new(TimeUnit::Milliseconds, Nullability::NonNullable).erased();
        let scalar = Scalar::extension_ref(
            ext_dtype.clone(),
            Scalar::primitive(1_000i64, Nullability::NonNullable),
        );
        let bloom = BloomFilter;
        let mut zone_partial =
            bloom.empty_partial(&BloomOptions::default(), &DType::Extension(ext_dtype))?;

        accumulate_constant(&ConstantArray::new(scalar.clone(), 1), &mut zone_partial)?;

        assert!(zone_partial.contains_scalar(&scalar)?);
        Ok(())
    }
}
