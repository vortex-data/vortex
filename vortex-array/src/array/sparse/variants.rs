use itertools::{EitherOrBoth, Itertools as _};
use vortex_dtype::field::Field;
use vortex_dtype::DType;
use vortex_error::{vortex_err, vortex_panic, VortexExpect, VortexResult};
use vortex_scalar::StructScalar;

use crate::array::sparse::SparseArray;
use crate::variants::{
    ArrayVariants, BinaryArrayTrait, BoolArrayTrait, ExtensionArrayTrait, ListArrayTrait,
    NullArrayTrait, PrimitiveArrayTrait, StructArrayTrait, Utf8ArrayTrait,
};
use crate::{Array, ArrayDType, IntoArray};

/// Sparse arrays support all DTypes
impl ArrayVariants for SparseArray {
    fn as_null_array(&self) -> Option<&dyn NullArrayTrait> {
        matches!(self.dtype(), DType::Null).then_some(self)
    }

    fn as_bool_array(&self) -> Option<&dyn BoolArrayTrait> {
        matches!(self.dtype(), DType::Bool(_)).then_some(self)
    }

    fn as_primitive_array(&self) -> Option<&dyn PrimitiveArrayTrait> {
        matches!(self.dtype(), DType::Primitive(..)).then_some(self)
    }

    fn as_utf8_array(&self) -> Option<&dyn Utf8ArrayTrait> {
        matches!(self.dtype(), DType::Utf8(_)).then_some(self)
    }

    fn as_binary_array(&self) -> Option<&dyn BinaryArrayTrait> {
        matches!(self.dtype(), DType::Binary(_)).then_some(self)
    }

    fn as_struct_array(&self) -> Option<&dyn StructArrayTrait> {
        matches!(self.dtype(), DType::Struct(..)).then_some(self)
    }

    fn as_list_array(&self) -> Option<&dyn ListArrayTrait> {
        matches!(self.dtype(), DType::List(..)).then_some(self)
    }

    fn as_extension_array(&self) -> Option<&dyn ExtensionArrayTrait> {
        matches!(self.dtype(), DType::Extension(..)).then_some(self)
    }
}

impl NullArrayTrait for SparseArray {}

impl BoolArrayTrait for SparseArray {
    fn invert(&self) -> VortexResult<Array> {
        let inverted_fill = self.fill_value().as_bool()?.map(|v| !v);
        SparseArray::try_new(
            self.indices(),
            self.values().with_dyn(|a| {
                a.as_bool_array()
                    .ok_or_else(|| vortex_err!("Not a bool array"))
                    .and_then(|b| b.invert())
            })?,
            self.len(),
            inverted_fill.into(),
        )
        .map(|a| a.into_array())
    }

    fn maybe_null_indices_iter<'a>(&'a self) -> Box<dyn Iterator<Item = usize> + 'a> {
        let sparse_to_real = self.resolved_indices();
        match self
            .fill_value()
            .as_bool()
            .vortex_expect("sparse bool array fill value must be bool")
        {
            Some(true) => {
                let unset_values = self.values().with_dyn(|values| {
                    values
                        .as_bool_array()
                        .vortex_expect("values of sparse bool array must be bools")
                        .invert()
                        .vortex_expect("bools must be invertible")
                });
                let unset_indices = unset_values.with_dyn(|unset_values| {
                    unset_values
                        .as_bool_array()
                        .vortex_expect("inverted bool array is still bool array")
                        .maybe_null_indices_iter()
                        .map(|unset_sparse_index| sparse_to_real[unset_sparse_index])
                        .collect::<Vec<usize>>()
                });
                Box::new(
                    (0..self.len())
                        .merge_join_by(unset_indices, Ord::cmp)
                        .filter_map(|joined| match joined {
                            EitherOrBoth::Left(i) => Some(i),
                            EitherOrBoth::Both(..) => None,
                            EitherOrBoth::Right(j) => vortex_panic!(
                                "sparse array index greater than len: {} {}",
                                j,
                                self.len()
                            ),
                        }),
                )
            }
            Some(false) | None => {
                let set_indices = self.values().with_dyn(|values| {
                    values
                        .as_bool_array()
                        .vortex_expect("values of sparse bool array must be bools")
                        .maybe_null_indices_iter()
                        .map(|set_sparse_index| sparse_to_real[set_sparse_index])
                        .collect::<Vec<usize>>()
                });

                Box::new(set_indices.into_iter())
            }
        }
    }

    fn maybe_null_slices_iter(&self) -> Box<dyn Iterator<Item = (usize, usize)>> {
        todo!()
    }
}

impl PrimitiveArrayTrait for SparseArray {}

impl Utf8ArrayTrait for SparseArray {}

impl BinaryArrayTrait for SparseArray {}

impl StructArrayTrait for SparseArray {
    fn field(&self, idx: usize) -> Option<Array> {
        let values = self
            .values()
            .with_dyn(|s| s.as_struct_array().and_then(|s| s.field(idx)))?;
        let scalar = StructScalar::try_new(self.dtype(), self.fill_value())
            .ok()?
            .field_by_idx(idx)?;

        Some(
            SparseArray::try_new_with_offset(
                self.indices().clone(),
                values,
                self.len(),
                self.indices_offset(),
                scalar.value().clone(),
            )
            .ok()?
            .into_array(),
        )
    }

    fn project(&self, projection: &[Field]) -> VortexResult<Array> {
        let values = self.values().with_dyn(|s| {
            s.as_struct_array()
                .ok_or_else(|| vortex_err!("Chunk was not a StructArray"))?
                .project(projection)
        })?;
        let scalar = StructScalar::try_new(self.dtype(), self.fill_value())?.project(projection)?;

        SparseArray::try_new_with_offset(
            self.indices().clone(),
            values,
            self.len(),
            self.indices_offset(),
            scalar.value().clone(),
        )
        .map(|a| a.into_array())
    }
}

impl ListArrayTrait for SparseArray {}

impl ExtensionArrayTrait for SparseArray {
    fn storage_array(&self) -> Array {
        SparseArray::try_new_with_offset(
            self.indices().clone(),
            self.values()
                .with_dyn(|a| a.as_extension_array_unchecked().storage_array()),
            self.len(),
            self.indices_offset(),
            self.fill_value().clone(),
        )
        .vortex_expect("Failed to create new sparse array")
        .into_array()
    }
}

#[cfg(test)]
mod tests {
    use vortex_scalar::ScalarValue;

    use crate::array::{BoolArray, PrimitiveArray, SparseArray};
    use crate::{Array, IntoArray, IntoArrayVariant};

    #[test]
    fn invert_bools_non_null_fill() {
        let sparse_bools = SparseArray::try_new(
            PrimitiveArray::from(vec![0u64]).into_array(),
            BoolArray::from(vec![false]).into_array(),
            2,
            true.into(),
        )
        .unwrap()
        .into_array();
        let inverted = sparse_bools
            .with_dyn(|a| a.as_bool_array_unchecked().invert())
            .unwrap();
        assert_eq!(
            inverted
                .into_bool()
                .unwrap()
                .boolean_buffer()
                .iter()
                .collect::<Vec<_>>(),
            vec![true, false]
        );
    }

    fn sparse_array<T>(indices: Vec<u64>, values: Vec<T>, len: usize, fill: T) -> Array
    where
        Array: From<Vec<T>>,
        ScalarValue: From<T>,
    {
        SparseArray::try_new(
            From::<Vec<u64>>::from(indices),
            Array::from(values),
            len,
            ScalarValue::from(fill),
        )
        .unwrap()
        .into_array()
    }

    fn maybe_null_indices(array: &Array) -> Vec<usize> {
        array.with_dyn(|array| {
            array
                .as_bool_array()
                .unwrap()
                .maybe_null_indices_iter()
                .collect::<Vec<_>>()
        })
    }

    #[test]
    fn nonnullable_sparse_bool_false_fill_no_true_values() {
        let array = sparse_array(vec![2], vec![false], 10, false);
        assert_eq!(maybe_null_indices(&array), Vec::<usize>::new());
    }

    #[test]
    fn nonnullable_sparse_bool_false_fill_one_true_value() {
        let array = sparse_array(vec![2, 8], vec![false, true], 10, false);
        assert_eq!(maybe_null_indices(&array), vec![8]);
    }

    #[test]
    fn nonnullable_sparse_bool_true_fill_no_true_values() {
        let array = sparse_array(vec![2], vec![false], 10, true);
        assert_eq!(maybe_null_indices(&array), vec![0, 1, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn nonnullable_sparse_bool_true_fill_one_true_value() {
        let array = sparse_array(vec![2, 8], vec![false, true], 10, true);
        assert_eq!(maybe_null_indices(&array), vec![0, 1, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn sparse_bool_null_fill_no_true_values() {
        let array = sparse_array(vec![2, 3], vec![Some(false), None], 10, None);
        assert_eq!(maybe_null_indices(&array), Vec::<usize>::new());
    }

    #[test]
    fn sparse_bool_null_fill_one_true_value() {
        let array = sparse_array(vec![2, 3, 8], vec![Some(false), None, Some(true)], 10, None);
        assert_eq!(maybe_null_indices(&array), vec![8]);
    }

    #[test]
    fn sparse_bool_true_fill() {
        let array = sparse_array(
            vec![2, 3, 8],
            vec![Some(false), None, Some(true)],
            10,
            Some(true),
        );
        assert_eq!(maybe_null_indices(&array), vec![0, 1, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn sparse_bool_false_fill_no_true_values() {
        let array = sparse_array(vec![2, 3], vec![Some(false), None], 10, Some(false));
        assert_eq!(maybe_null_indices(&array), Vec::<usize>::new());
    }

    #[test]
    fn sparse_bool_false_fill_one_true_value() {
        let array = sparse_array(
            vec![2, 3, 8],
            vec![Some(false), None, Some(true)],
            10,
            Some(false),
        );
        assert_eq!(maybe_null_indices(&array), vec![8]);
    }
}
