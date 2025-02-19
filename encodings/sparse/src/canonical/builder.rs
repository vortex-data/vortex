use std::mem::MaybeUninit;

use vortex_array::builders::{BoolBuilder, PrimitiveBuilder};
use vortex_array::variants::PrimitiveArrayTrait;
use vortex_array::IntoArrayVariant;
use vortex_dtype::{match_each_integer_ptype, NativePType};
use vortex_error::{VortexError, VortexResult};
use vortex_scalar::PValue;

use crate::SparseArray;

#[allow(clippy::cast_possible_truncation)]
pub(super) fn canonicalize_primitive_into<T: NativePType + TryFrom<PValue, Error = VortexError>>(
    sparse: &SparseArray,
    builder: &mut PrimitiveBuilder<T>,
) -> VortexResult<()> {
    // Scatter the fill value into the output buffer
    let mut values_uninit = builder.uninit_values(sparse.len());
    if let Some(fill_value) = sparse.fill_scalar().as_primitive().typed_value() {
        values_uninit.fill(MaybeUninit::new(fill_value));
    } else {
        // fill value is NULL, leave the slots with uninitialized values.
    }

    let patches = sparse.resolved_patches()?;
    let indices = patches.indices().clone().into_primitive()?;
    let values = patches.values().clone().into_primitive()?;

    fn scatter_values<T>(index: usize, value: T, values_uninit: &mut [MaybeUninit<T>]) {
        values_uninit[index] = MaybeUninit::new(value);
    };

    let scatter_values_with_nulls =
        |index: usize, value: T, values_uninit: &mut [MaybeUninit<T>]| {
            values_uninit[index] = MaybeUninit::new(value);
            // Offset the index using our builtin buffer.
            builder.nulls.set_bit()
        };

    // Get access to the validity function of the patch values instead.
    let scatter_values = |index: usize, value: T, values_uninit: &mut [MaybeUninit<T>]| {
        values_uninit[index] = MaybeUninit::new(value);
    };

    match_each_integer_ptype!(indices.ptype(), |$I| {
        let indices = indices.as_slice::<$I>();
        for (&index, &value) in indices.iter().zip(values.as_slice::<T>()) {
            builder.values[index as usize] = value;
        }
    });

    builder.patch(sparse.resolved_patches()?, 0)?;

    // If the array is nullable, and we have some sparse NULLs spread throughout the code,
    // we can create a new null buffer and set it directly.
    if sparse.dtype().is_nullable() {
        sparse.patches().values()
    }

    // Set the validity from the sparse array.
    builder.nulls.append_validity_mask(sparse.validity_mask()?);

    Ok(())
}

/// Canonicalize a set of bools into a builder.
///
/// The builder must be properly sized before being used.
pub(super) fn canonicalize_bool_into(
    sparse: &SparseArray,
    builder: &mut BoolBuilder,
) -> VortexResult<()> {
    builder.inner.append_n(
        sparse.len(),
        sparse.fill_scalar().as_bool().value().unwrap_or_default(),
    );

    let patches = sparse.resolved_patches()?;
    let indices = patches.indices().clone().into_primitive()?;
    let values = patches.values().clone().into_bool()?;

    // Scatter the values into the output buffer
    match_each_integer_ptype!(indices.ptype(), |$I| {
        let indices = indices.as_slice::<$I>();
        for (&index, value) in indices.iter().zip(values.boolean_buffer().into_iter()) {
            builder.inner.set_bit(index as usize, value);
        }
    });

    // Set the validity from the sparse array.
    builder.nulls.append_validity_mask(sparse.validity_mask()?);

    Ok(())
}

// TODO(aduffy): support for string, binary, float, struct, list

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use vortex_array::arrays::BoolArray;
    use vortex_array::builders::{ArrayBuilder, BoolBuilder, PrimitiveBuilder};
    use vortex_array::{IntoArray, IntoCanonical};
    use vortex_buffer::{buffer, buffer_mut, Buffer};
    use vortex_dtype::Nullability;
    use vortex_scalar::Scalar;

    use crate::SparseArray;

    #[test]
    fn test_primitive() {
        let len = 10;
        let indices: Buffer<u64> = buffer![0u64, 2, 4, 6, 8];

        let values = buffer_mut![18u64, 19, 20, 21, 22];

        let sparse = SparseArray::try_new(
            indices.into_array(),
            values.into_array(),
            len,
            Scalar::primitive(u64::MAX, Nullability::NonNullable),
        )
        .unwrap();

        let mut result = PrimitiveBuilder::<u64>::with_capacity(Nullability::NonNullable, len);
        sparse.canonicalize_into(&mut result).unwrap();

        assert_eq!(
            result.finish_into_primitive().as_slice::<u64>(),
            &[
                18,
                u64::MAX,
                19,
                u64::MAX,
                20,
                u64::MAX,
                21,
                u64::MAX,
                22,
                u64::MAX,
            ]
        );
    }

    #[test]
    fn test_bool() {
        let len = 10;
        let indices: Buffer<u64> = buffer![0u64, 2, 4, 6, 8];

        let values = BoolArray::from_iter(vec![true, true, true, true, true]);

        let sparse = SparseArray::try_new(
            indices.into_array(),
            values.into_array(),
            len,
            Scalar::bool(false, Nullability::NonNullable),
        )
        .unwrap();

        let mut result = BoolBuilder::with_capacity(Nullability::NonNullable, len);
        sparse.canonicalize_into(&mut result).unwrap();

        let bools = BoolArray::try_from(result.finish())
            .unwrap()
            .boolean_buffer();
        assert_eq!(
            bools.into_iter().collect_vec(),
            vec![true, false, true, false, true, false, true, false, true, false]
        );
    }
}
