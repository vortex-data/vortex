use vortex_array::compute::{scalar_at, ScalarAtFn};
use vortex_error::VortexResult;
use vortex_scalar::Scalar;

use crate::{RunEndArray, RunEndEncoding};

impl ScalarAtFn<RunEndArray> for RunEndEncoding {
    fn scalar_at(&self, array: &RunEndArray, index: usize) -> VortexResult<Scalar> {
        scalar_at(array.values(), array.find_physical_index(index)?)
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::compute::scalar_at;
    use vortex_array::IntoArray;

    use crate::RunEndArray;

    #[test]
    fn ree_scalar_at_end() {
        let scalar = scalar_at(
            RunEndArray::encode(
                PrimitiveArray::from_iter([1, 1, 1, 4, 4, 4, 2, 2, 5, 5, 5, 5]).into_array(),
            )
            .unwrap()
            .as_ref(),
            11,
        )
        .unwrap();
        assert_eq!(scalar, 5.into());
    }
}
