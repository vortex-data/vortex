use vortex_error::VortexResult;

use crate::array::VarBinViewArray;
use crate::compute::{arrow_compare, CompareFn, Operator};
use crate::ArrayData;

impl CompareFn for VarBinViewArray {
    // TODO(ngates): this implementation is arguably the same for _all_ canonical encodings.
    //  Maybe the entry-point function should handle this?
    fn compare(&self, other: &ArrayData, operator: Operator) -> VortexResult<Option<ArrayData>> {
        // If the RHS is a constant, we know we should use Arrow kernels.
        if other.is_constant() {
            return arrow_compare(self.as_ref(), other, operator).map(Some);
        }

        // Otherwise, continue with the fall back
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use vortex_dtype::Nullability;
    use vortex_scalar::Scalar;

    use crate::array::{ConstantArray, VarBinViewArray};
    use crate::compute::{compare, Operator};
    use crate::{ArrayLen, IntoArrayVariant};

    #[test]
    fn basic_test() {
        let arr = VarBinViewArray::from_iter_nullable_str([
            Some("one"),
            Some("two"),
            Some("three"),
            Some("four"),
            Some("five"),
            Some("six"),
        ]);

        let s = Scalar::utf8("seven".to_string(), Nullability::Nullable);

        let constant_array = ConstantArray::new(s, arr.len());

        let r = compare(&arr, &constant_array, Operator::Eq)
            .unwrap()
            .into_bool()
            .unwrap();

        assert!(r.boolean_buffer().iter().all(|v| !v));
    }
}
