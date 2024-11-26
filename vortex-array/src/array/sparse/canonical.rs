use arrow_buffer::{ArrowNativeType, BooleanBuffer};
use vortex_dtype::{match_each_native_ptype, DType, NativePType, Nullability};
use vortex_error::{VortexError, VortexResult};
use vortex_scalar::Scalar;

use crate::array::primitive::PrimitiveArray;
use crate::array::sparse::SparseArray;
use crate::array::BoolArray;
use crate::validity::Validity;
use crate::variants::PrimitiveArrayTrait;
use crate::{ArrayDType, ArrayLen, Canonical, IntoArrayVariant, IntoCanonical};

impl IntoCanonical for SparseArray {
    fn into_canonical(self) -> VortexResult<Canonical> {
        // Resolve our indices into a vector of usize applying the offset
        let indices = self.resolved_indices();

        if matches!(self.dtype(), DType::Bool(_)) {
            let values = self.values().into_bool()?;
            canonicalize_sparse_bools(values, &indices, self.len(), &self.fill_scalar())
        } else {
            let values = self.values().into_primitive()?;
            match_each_native_ptype!(values.ptype(), |$P| {
                canonicalize_sparse_primitives::<$P>(
                    values,
                    &indices,
                    self.len(),
                    &self.fill_scalar(),
                )
            })
        }
    }
}

fn canonicalize_sparse_bools(
    values: BoolArray,
    indices: &[usize],
    len: usize,
    fill_value: &Scalar,
) -> VortexResult<Canonical> {
    let (fill_bool, validity) = if fill_value.is_null() {
        (false, Validity::AllInvalid)
    } else {
        (
            fill_value.try_into()?,
            if values.dtype().nullability() == Nullability::NonNullable {
                Validity::NonNullable
            } else {
                Validity::AllValid
            },
        )
    };

    let bools = BoolArray::try_new(
        if fill_bool {
            BooleanBuffer::new_set(len)
        } else {
            BooleanBuffer::new_unset(len)
        },
        validity,
    )?;
    let patched = bools.patch(indices, values)?;
    Ok(Canonical::Bool(patched))
}

fn canonicalize_sparse_primitives<
    T: NativePType + for<'a> TryFrom<&'a Scalar, Error = VortexError> + ArrowNativeType,
>(
    values: PrimitiveArray,
    indices: &[usize],
    len: usize,
    fill_value: &Scalar,
) -> VortexResult<Canonical> {
    let values_validity = values.validity();
    let (primitive_fill, validity) = if fill_value.is_null() {
        (T::default(), Validity::AllInvalid)
    } else {
        (
            fill_value.try_into()?,
            if values_validity == Validity::NonNullable {
                Validity::NonNullable
            } else {
                Validity::AllValid
            },
        )
    };

    let parray = PrimitiveArray::from_vec(vec![primitive_fill; len], validity);
    let patched = parray.patch(indices, values.maybe_null_slice::<T>(), values_validity)?;
    Ok(Canonical::Primitive(patched))
}

#[cfg(test)]
mod test {
    use arrow_buffer::BooleanBufferBuilder;
    use rstest::rstest;
    use vortex_dtype::{DType, Nullability, PType};
    use vortex_error::VortexExpect;
    use vortex_scalar::Scalar;

    use crate::array::sparse::SparseArray;
    use crate::array::{BoolArray, PrimitiveArray};
    use crate::validity::Validity;
    use crate::{ArrayDType, IntoArrayData, IntoCanonical};

    #[rstest]
    #[case(Some(true))]
    #[case(Some(false))]
    #[case(None)]
    fn test_sparse_bool(#[case] fill_value: Option<bool>) {
        let indices = vec![0u64, 1, 7].into_array();
        let values = bool_array_from_nullable_vec(vec![Some(true), None, Some(false)], fill_value)
            .into_array();
        let sparse_bools =
            SparseArray::try_new(indices, values, 10, Scalar::from(fill_value)).unwrap();
        assert_eq!(*sparse_bools.dtype(), DType::Bool(Nullability::Nullable));

        let flat_bools = sparse_bools.into_canonical().unwrap().into_bool().unwrap();
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
        assert!(flat_bools.validity().is_valid(0));
        assert_eq!(
            flat_bools.boolean_buffer().value(1),
            fill_value.unwrap_or_default()
        );
        assert!(!flat_bools.validity().is_valid(1));
        assert_eq!(flat_bools.validity().is_valid(2), fill_value.is_some());
        assert!(!flat_bools.boolean_buffer().value(7));
        assert!(flat_bools.validity().is_valid(7));
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

        let indices = vec![0u64, 1, 7].into_array();
        let values =
            PrimitiveArray::from_nullable_vec(vec![Some(0i32), None, Some(1)]).into_array();
        let sparse_ints =
            SparseArray::try_new(indices, values, 10, Scalar::from(fill_value)).unwrap();
        assert_eq!(
            *sparse_ints.dtype(),
            DType::Primitive(PType::I32, Nullability::Nullable)
        );

        let flat_ints = sparse_ints
            .into_canonical()
            .unwrap()
            .into_primitive()
            .unwrap();
        let expected = PrimitiveArray::from_nullable_vec(vec![
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

        assert_eq!(flat_ints.buffer(), expected.buffer());
        assert_eq!(flat_ints.validity(), expected.validity());

        assert_eq!(flat_ints.maybe_null_slice::<i32>()[0], 0);
        assert!(flat_ints.validity().is_valid(0));
        assert_eq!(flat_ints.maybe_null_slice::<i32>()[1], 0);
        assert!(!flat_ints.validity().is_valid(1));
        assert_eq!(
            flat_ints.maybe_null_slice::<i32>()[2],
            fill_value.unwrap_or_default()
        );
        assert_eq!(flat_ints.validity().is_valid(2), fill_value.is_some());
        assert_eq!(flat_ints.maybe_null_slice::<i32>()[7], 1);
        assert!(flat_ints.validity().is_valid(7));
    }
}
