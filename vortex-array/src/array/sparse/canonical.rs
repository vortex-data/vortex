use arrow_buffer::BooleanBufferBuilder;
use vortex_dtype::{match_each_native_ptype, DType, NativePType};
use vortex_error::{VortexError, VortexResult};
use vortex_scalar::ScalarValue;

use crate::array::primitive::PrimitiveArray;
use crate::array::sparse::SparseArray;
use crate::array::BoolArray;
use crate::validity::Validity;
use crate::variants::PrimitiveArrayTrait;
use crate::{ArrayDType, Canonical, IntoArrayVariant, IntoCanonical};

impl IntoCanonical for SparseArray {
    fn into_canonical(self) -> VortexResult<Canonical> {
        // Resolve our indices into a vector of usize, applying the offset
        let indices = self.resolved_indices();

        if matches!(self.dtype(), DType::Bool(_)) {
            canonicalize_sparse_bools(
                self.values().into_bool()?,
                &indices,
                self.len(),
                self.fill_value(),
            )
        } else {
            let values = self.values().into_primitive()?;
            match_each_native_ptype!(values.ptype(), |$P| {
                canonicalize_sparse_primitives(
                    values.maybe_null_slice::<$P>(),
                    values.validity(),
                    &indices,
                    self.len(),
                    self.fill_value()
                )
            })
        }
    }
}

fn canonicalize_sparse_bools(
    values_array: BoolArray,
    indices: &[usize],
    len: usize,
    fill_value: &ScalarValue,
) -> VortexResult<Canonical> {
    let fill_bool = if fill_value.is_null() {
        false
    } else {
        fill_value.try_into()?
    };

    let values_validity = values_array.validity();
    let values = values_array.boolean_buffer();

    // pre-fill both values and validity based on fill_value
    // this optimizes performance for the common case where indices.len() < len / 2
    let mut flat_bools = BooleanBufferBuilder::new(len);
    flat_bools.append_n(len, fill_bool);
    let mut validity_buffer = BooleanBufferBuilder::new(len);
    validity_buffer.append_n(len, !fill_value.is_null());

    // patch in the actual values and validity
    for (i, idx) in indices.iter().enumerate() {
        flat_bools.set_bit(*idx, values.value(i));
        validity_buffer.set_bit(*idx, values_validity.is_valid(i));
    }

    BoolArray::try_new(
        flat_bools.finish(),
        Validity::from(validity_buffer.finish()),
    )
    .map(Canonical::Bool)
}

fn canonicalize_sparse_primitives<
    T: NativePType + for<'a> TryFrom<&'a ScalarValue, Error = VortexError>,
>(
    values: &[T],
    values_validity: Validity,
    indices: &[usize],
    len: usize,
    fill_value: &ScalarValue,
) -> VortexResult<Canonical> {
    let primitive_fill = if fill_value.is_null() {
        T::default()
    } else {
        fill_value.try_into()?
    };
    let mut result = vec![primitive_fill; len];
    let mut validity = BooleanBufferBuilder::new(len);
    validity.append_n(len, !fill_value.is_null());

    for (i, idx) in indices.iter().enumerate() {
        result[*idx] = values[i];
        validity.set_bit(*idx, values_validity.is_valid(i));
    }

    Ok(Canonical::Primitive(PrimitiveArray::from_vec(
        result,
        Validity::from(validity.finish()),
    )))
}

#[cfg(test)]
mod test {
    use arrow_buffer::BooleanBufferBuilder;
    use rstest::rstest;
    use vortex_dtype::{DType, Nullability, PType};
    use vortex_error::VortexExpect as _;
    use vortex_scalar::ScalarValue;

    use crate::array::sparse::SparseArray;
    use crate::array::{BoolArray, PrimitiveArray};
    use crate::validity::Validity;
    use crate::{ArrayDType, IntoArray, IntoCanonical};

    #[rstest]
    #[case(Some(true))]
    #[case(Some(false))]
    #[case(None)]
    fn test_sparse_bool(#[case] fill_value: Option<bool>) {
        let indices = vec![0u64, 1, 7].into_array();
        let values = bool_array_from_nullable_vec(vec![Some(true), None, Some(false)], fill_value)
            .into_array();
        let sparse_bools =
            SparseArray::try_new(indices, values, 10, ScalarValue::from(fill_value)).unwrap();
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
            .vortex_expect("Failed to create BoolArray from nullable vec")
    }

    #[rstest]
    #[case(Some(0i32))]
    #[case(Some(-1i32))]
    #[case(None)]
    fn test_sparse_primitive(#[case] fill_value: Option<i32>) {
        let indices = vec![0u64, 1, 7].into_array();
        let values =
            PrimitiveArray::from_nullable_vec(vec![Some(0i32), None, Some(1)]).into_array();
        let sparse_ints =
            SparseArray::try_new(indices, values, 10, ScalarValue::from(fill_value)).unwrap();
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
