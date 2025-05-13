use vortex_error::VortexResult;
use vortex_mask::{AllOr, Mask};
use vortex_scalar::Scalar;

use crate::arrays::{ConstantArray, ConstantVTable};
use crate::builders::builder_with_capacity;
use crate::compute::{TakeKernel, TakeKernelAdapter};
use crate::{Array, ArrayRef, IntoArray, register_kernel};

impl TakeKernel for ConstantVTable {
    fn take(&self, array: &ConstantArray, indices: &dyn Array) -> VortexResult<ArrayRef> {
        match indices.validity_mask()?.boolean_buffer() {
            AllOr::All => {
                let scalar = Scalar::new(
                    array
                        .scalar()
                        .dtype()
                        .union_nullability(indices.dtype().nullability()),
                    array.scalar().value().clone(),
                );
                Ok(ConstantArray::new(scalar, indices.len()).into_array())
            }
            AllOr::None => Ok(ConstantArray::new(
                Scalar::null(
                    array
                        .dtype()
                        .union_nullability(indices.dtype().nullability()),
                ),
                indices.len(),
            )
            .into_array()),
            AllOr::Some(v) => {
                let arr = ConstantArray::new(array.scalar().clone(), indices.len()).into_array();

                if array.scalar().is_null() {
                    return Ok(arr);
                }

                let mut result_builder =
                    builder_with_capacity(&array.dtype().as_nullable(), indices.len());
                result_builder.extend_from_array(&arr)?;
                result_builder.set_validity(Mask::from_buffer(v.clone()));
                Ok(result_builder.finish())
            }
        }
    }
}

register_kernel!(TakeKernelAdapter(ConstantVTable).lift());

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_dtype::Nullability;
    use vortex_mask::AllOr;

    use crate::arrays::{ConstantArray, PrimitiveArray};
    use crate::compute::take;
    use crate::validity::Validity;
    use crate::{Array, IntoArray, ToCanonical};

    #[test]
    fn take_nullable_indices() {
        let array = ConstantArray::new(42, 10).to_array();
        let taken = take(
            &array,
            &PrimitiveArray::new(
                buffer![0, 5, 7],
                Validity::from_iter(vec![false, true, false]),
            )
            .into_array(),
        )
        .unwrap();
        let valid_indices: &[usize] = &[1usize];
        assert_eq!(
            &array.dtype().with_nullability(Nullability::Nullable),
            taken.dtype()
        );
        assert_eq!(
            taken.to_primitive().unwrap().as_slice::<i32>(),
            &[42, 42, 42]
        );
        assert_eq!(
            taken.validity_mask().unwrap().indices(),
            AllOr::Some(valid_indices)
        );
    }

    #[test]
    fn take_all_valid_indices() {
        let array = ConstantArray::new(42, 10).to_array();
        let taken = take(
            &array,
            &PrimitiveArray::new(buffer![0, 5, 7], Validity::AllValid).into_array(),
        )
        .unwrap();
        assert_eq!(
            &array.dtype().with_nullability(Nullability::Nullable),
            taken.dtype()
        );
        assert_eq!(
            taken.to_primitive().unwrap().as_slice::<i32>(),
            &[42, 42, 42]
        );
        assert_eq!(taken.validity_mask().unwrap().indices(), AllOr::All);
    }
}
