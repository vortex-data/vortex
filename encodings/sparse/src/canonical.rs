use vortex_array::array::{BoolArray, BooleanBuffer, ConstantArray, PrimitiveArray};
use vortex_array::patches::Patches;
use vortex_array::validity::Validity;
use vortex_array::vtable::CanonicalVTable;
use vortex_array::{Canonical, IntoCanonical};
use vortex_buffer::buffer;
use vortex_dtype::{match_each_native_ptype, DType, NativePType, Nullability, PType};
use vortex_error::{VortexError, VortexResult};
use vortex_scalar::Scalar;

use crate::{SparseArray, SparseEncoding};

impl CanonicalVTable<SparseArray> for SparseEncoding {
    fn into_canonical(&self, array: SparseArray) -> VortexResult<Canonical> {
        let resolved_patches = array.resolved_patches()?;
        if resolved_patches.num_patches() == 0 {
            return ConstantArray::new(array.fill_scalar(), array.len()).into_canonical();
        }

        if matches!(array.dtype(), DType::Bool(_)) {
            canonicalize_sparse_bools(resolved_patches, &array.fill_scalar())
        } else {
            let ptype = PType::try_from(resolved_patches.values().dtype())?;
            match_each_native_ptype!(ptype, |$P| {
                canonicalize_sparse_primitives::<$P>(
                    resolved_patches,
                    &array.fill_scalar(),
                )
            })
        }
    }
}

fn canonicalize_sparse_bools(patches: Patches, fill_value: &Scalar) -> VortexResult<Canonical> {
    let (fill_bool, validity) = if fill_value.is_null() {
        (false, Validity::AllInvalid)
    } else {
        (
            fill_value.try_into()?,
            if patches.dtype().nullability() == Nullability::NonNullable {
                Validity::NonNullable
            } else {
                Validity::AllValid
            },
        )
    };

    let bools = BoolArray::try_new(
        if fill_bool {
            BooleanBuffer::new_set(patches.array_len())
        } else {
            BooleanBuffer::new_unset(patches.array_len())
        },
        validity,
    )?;

    bools.patch(patches).map(Canonical::Bool)
}

fn canonicalize_sparse_primitives<
    T: NativePType + for<'a> TryFrom<&'a Scalar, Error = VortexError>,
>(
    patches: Patches,
    fill_value: &Scalar,
) -> VortexResult<Canonical> {
    let (primitive_fill, validity) = if fill_value.is_null() {
        (T::default(), Validity::AllInvalid)
    } else {
        (
            fill_value.try_into()?,
            if patches.dtype().nullability() == Nullability::NonNullable {
                Validity::NonNullable
            } else {
                Validity::AllValid
            },
        )
    };

    let parray = PrimitiveArray::new(buffer![primitive_fill; patches.array_len()], validity);

    parray.patch(patches).map(Canonical::Primitive)
}

#[cfg(test)]
mod test {
    use rstest::rstest;
    use vortex_array::array::{BoolArray, BooleanBufferBuilder, PrimitiveArray};
    use vortex_array::validity::Validity;
    use vortex_array::{IntoArray, IntoArrayVariant};
    use vortex_buffer::buffer;
    use vortex_dtype::{DType, Nullability, PType};
    use vortex_error::VortexExpect;
    use vortex_scalar::Scalar;

    use crate::SparseArray;

    #[rstest]
    #[case(Some(true))]
    #[case(Some(false))]
    #[case(None)]
    fn test_sparse_bool(#[case] fill_value: Option<bool>) {
        let indices = buffer![0u64, 1, 7].into_array();
        let values = bool_array_from_nullable_vec(vec![Some(true), None, Some(false)], fill_value)
            .into_array();
        let sparse_bools =
            SparseArray::try_new(indices, values, 10, Scalar::from(fill_value)).unwrap();
        assert_eq!(sparse_bools.dtype(), &DType::Bool(Nullability::Nullable));

        let flat_bools = sparse_bools.into_bool().unwrap();
        let expected = bool_array_from_nullable_vec(
            vec![
                Some(true),
                None,
                fill_value,
                fill_value,
                fill_value,
                fill_value,
                fill_value,
                Some(false),
                fill_value,
                fill_value,
            ],
            fill_value,
        );

        assert_eq!(flat_bools.boolean_buffer(), expected.boolean_buffer());
        assert_eq!(flat_bools.validity(), expected.validity());

        assert!(flat_bools.boolean_buffer().value(0));
        assert!(flat_bools.validity().is_valid(0).unwrap());
        assert_eq!(
            flat_bools.boolean_buffer().value(1),
            fill_value.unwrap_or_default()
        );
        assert!(!flat_bools.validity().is_valid(1).unwrap());
        assert_eq!(
            flat_bools.validity().is_valid(2).unwrap(),
            fill_value.is_some()
        );
        assert!(!flat_bools.boolean_buffer().value(7));
        assert!(flat_bools.validity().is_valid(7).unwrap());
    }

    fn bool_array_from_nullable_vec(
        bools: Vec<Option<bool>>,
        fill_value: Option<bool>,
    ) -> BoolArray {
        let mut buffer = BooleanBufferBuilder::new(bools.len());
        let mut validity = BooleanBufferBuilder::new(bools.len());
        for maybe_bool in bools {
            buffer.append(maybe_bool.unwrap_or_else(|| fill_value.unwrap_or_default()));
            validity.append(maybe_bool.is_some());
        }
        BoolArray::try_new(buffer.finish(), Validity::from(validity.finish()))
            .vortex_expect("Validity length cannot mismatch")
    }

    #[rstest]
    #[case(Some(0i32))]
    #[case(Some(-1i32))]
    #[case(None)]
    fn test_sparse_primitive(#[case] fill_value: Option<i32>) {
        use vortex_scalar::Scalar;

        let indices = buffer![0u64, 1, 7].into_array();
        let values = PrimitiveArray::from_option_iter([Some(0i32), None, Some(1)]).into_array();
        let sparse_ints =
            SparseArray::try_new(indices, values, 10, Scalar::from(fill_value)).unwrap();
        assert_eq!(
            *sparse_ints.dtype(),
            DType::Primitive(PType::I32, Nullability::Nullable)
        );

        let flat_ints = sparse_ints.into_primitive().unwrap();
        let expected = PrimitiveArray::from_option_iter([
            Some(0i32),
            None,
            fill_value,
            fill_value,
            fill_value,
            fill_value,
            fill_value,
            Some(1),
            fill_value,
            fill_value,
        ]);

        assert_eq!(flat_ints.byte_buffer(), expected.byte_buffer());
        assert_eq!(flat_ints.validity(), expected.validity());

        assert_eq!(flat_ints.as_slice::<i32>()[0], 0);
        assert!(flat_ints.validity().is_valid(0).unwrap());
        assert_eq!(flat_ints.as_slice::<i32>()[1], 0);
        assert!(!flat_ints.validity().is_valid(1).unwrap());
        assert_eq!(
            flat_ints.as_slice::<i32>()[2],
            fill_value.unwrap_or_default()
        );
        assert_eq!(
            flat_ints.validity().is_valid(2).unwrap(),
            fill_value.is_some()
        );
        assert_eq!(flat_ints.as_slice::<i32>()[7], 1);
        assert!(flat_ints.validity().is_valid(7).unwrap());
    }
}
