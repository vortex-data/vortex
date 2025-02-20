use std::mem::MaybeUninit;

use vortex_array::arrays::{BoolArray, PrimitiveArray};
use vortex_array::builders::{BoolBuilder, PrimitiveBuilder, UninitBool, UninitPrimitive};
use vortex_array::validity::Validity;
use vortex_array::variants::PrimitiveArrayTrait;
use vortex_array::IntoArrayVariant;
use vortex_dtype::{match_each_integer_ptype, NativePType};
use vortex_error::{VortexError, VortexExpect, VortexResult};
use vortex_mask::Mask;
use vortex_scalar::PValue;

use crate::SparseArray;

#[allow(clippy::cast_possible_truncation)]
pub(super) fn canonicalize_primitive_into<T: NativePType + TryFrom<PValue, Error = VortexError>>(
    sparse: &SparseArray,
    builder: &mut PrimitiveBuilder<T>,
) -> VortexResult<()> {
    let fill_value: Option<T> = sparse.fill_scalar().as_primitive().typed_value();

    // Prepare the null buffer so we can set ranges of it.
    // If the fill value is NULL, we initialize the null buffer with false.
    //
    // If the fill value is non-NULL but the array is nullable, we initialize the null buffer with
    //  true, and will patch in the false bits as we write nulls.
    if fill_value.is_none() {
        builder.append_mask(Mask::AllFalse(sparse.len()));
    } else if sparse.dtype().is_nullable() {
        builder.append_mask(Mask::AllTrue(sparse.len()));
    }

    let mut values_uninit = builder.uninit_range(sparse.len());

    // If fill value is non-null, fill the buffer with it. If it is NULL, we leave the slots
    // uninitialized, as their values are not semantically meaningful.
    if let Some(value) = fill_value {
        values_uninit.fill(MaybeUninit::new(value));
    }

    let patches = sparse.resolved_patches()?;
    let indices = patches.indices().clone().into_primitive()?;
    let values = patches.values().clone().into_primitive()?;

    fn write_value<T, const VALUE: bool, const NULLS: bool>(
        index: usize,
        sparse_index: usize,
        value: T,
        values_uninit: &mut UninitPrimitive<T>,
        values: &PrimitiveArray,
    ) {
        if VALUE {
            values_uninit[index] = MaybeUninit::new(value);
        }

        if NULLS {
            // We use our final value bit instead.
            values_uninit.set_valid_bit(
                index,
                values
                    .is_valid(sparse_index)
                    .vortex_expect("values.is_valid"),
            );
        }
    }

    let writer = match values.validity() {
        // Just write values. We don't need to write false bits
        Validity::NonNullable | Validity::AllValid => write_value::<T, true, false>,
        // Write nulls, do not write values
        Validity::AllInvalid => write_value::<T, false, true>,
        // Write nulls and values.
        Validity::Array(_) => write_value::<T, true, true>,
    };

    match_each_integer_ptype!(indices.ptype(), |$I| {
        let indices = indices.as_slice::<$I>();
        for (sparse_index, (&index, &value)) in indices.iter().zip(values.as_slice::<T>()).enumerate() {
            writer(index as usize, sparse_index, value, &mut values_uninit, &values);
        }
    });

    values_uninit.finish();

    Ok(())
}

/// Canonicalize a set of bools into a builder.
///
/// The builder must be properly sized before being used.
pub(super) fn canonicalize_bool_into(
    sparse: &SparseArray,
    builder: &mut BoolBuilder,
) -> VortexResult<()> {
    let fill_value = sparse.fill_scalar().as_bool().value();

    if fill_value.is_none() {
        builder.append_mask(Mask::AllFalse(sparse.len()));
    } else if sparse.dtype().is_nullable() {
        builder.append_mask(Mask::AllTrue(sparse.len()));
    }

    let mut values_uninit = builder.uninit_range(sparse.len());

    // If the fill value is non-NULL.
    if let Some(value) = fill_value {
        values_uninit.set_all(value);
    }

    fn write_bool<const VALUE: bool, const NULLS: bool>(
        index: usize,
        sparse_index: usize,
        value: bool,
        uninit: &mut UninitBool,
        values: &BoolArray,
    ) {
        if VALUE {
            uninit.set_bit(index, value);
        }

        if NULLS {
            uninit.set_valid_bit(
                index,
                values
                    .is_valid(sparse_index)
                    .vortex_expect("values.is_valid"),
            );
        }
    }

    let patches = sparse.resolved_patches()?;
    let indices = patches.indices().clone().into_primitive()?;
    let values = patches.values().clone().into_bool()?;

    let writer = match values.validity() {
        // No nulls present, just write values
        Validity::NonNullable | Validity::AllValid => write_bool::<true, false>,
        // Write nulls, do not write values
        Validity::AllInvalid => write_bool::<false, true>,
        // Write nulls and values.
        Validity::Array(_) => write_bool::<true, true>,
    };

    // Scatter the patch values into the output buffer.
    match_each_integer_ptype!(indices.ptype(), |$I| {
        let indices = indices.as_slice::<$I>();
        for (sparse_index, (&index, value)) in indices.iter().zip(values.boolean_buffer().into_iter()).enumerate() {
            writer(index as usize, sparse_index, value, &mut values_uninit, &values);
        }
    });

    values_uninit.finish();

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
