// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::Canonical;
use crate::arrays::expr::ExprArray;
use crate::arrays::expr::ExprVTable;
use crate::vtable::CanonicalVTable;

impl CanonicalVTable<ExprVTable> for ExprVTable {
    fn canonicalize(array: &ExprArray) -> VortexResult<Canonical> {
        array.expr.evaluate(&array.child)?.to_canonical()
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_dtype::DType;
    use vortex_dtype::Nullability::NonNullable;
    use vortex_dtype::PType;

    use crate::Array;
    use crate::IntoArray;
    use crate::arrays::expr::ExprArray;
    use crate::arrays::primitive::PrimitiveArray;
    use crate::assert_arrays_eq;
    use crate::expr::binary::checked_add;
    use crate::expr::literal::lit;
    use crate::validity::Validity;

    #[test]
    fn test_expr_array_canonicalize() {
        let child = PrimitiveArray::new(buffer![1i32, 2, 3], Validity::NonNullable).into_array();

        // This expression doesn't use the child, but demonstrates the ExprArray mechanics
        let expr = checked_add(lit(10), lit(5));

        let dtype = DType::Primitive(PType::I32, NonNullable);
        let expr_array = ExprArray::try_new(child, expr, dtype).unwrap();

        let actual = expr_array.to_canonical().unwrap().into_array();

        let expect = (0..3).map(|_| 15i32).collect::<PrimitiveArray>();
        assert_arrays_eq!(expect, actual);
    }
}
