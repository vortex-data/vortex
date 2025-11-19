// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect;

use crate::Canonical;
use crate::arrays::expr::{ExprArray, ExprVTable};
use crate::vtable::CanonicalVTable;

impl CanonicalVTable<ExprVTable> for ExprVTable {
    fn canonicalize(array: &ExprArray) -> Canonical {
        // Evaluate the expression on the child array and canonicalize the result
        array
            .expr
            .evaluate(&array.child)
            .vortex_expect("Failed to evaluate expression")
            .to_canonical()
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_dtype::Nullability::NonNullable;
    use vortex_dtype::{DType, PType};

    use crate::arrays::expr::ExprArray;
    use crate::arrays::primitive::PrimitiveArray;
    use crate::expr::binary::checked_add;
    use crate::expr::literal::lit;
    use crate::validity::Validity;
    use crate::{Array, IntoArray};

    #[test]
    fn test_expr_array_canonicalize() {
        let child = PrimitiveArray::new(buffer![1i32, 2, 3], Validity::NonNullable).into_array();

        // Create an expression: lit(10) + lit(5) = 15
        // This expression doesn't use the child, but demonstrates the ExprArray mechanics
        let expr = checked_add(lit(10), lit(5));

        // Create ExprArray with the expression
        let dtype = DType::Primitive(PType::I32, NonNullable);
        let expr_array = ExprArray::try_new(child, expr, dtype).unwrap();
        let array = expr_array.into_array();

        // Test canonicalize - should evaluate the expression
        let canonical = array.to_canonical();

        // The result should be a primitive array with value 15 repeated for each element
        let canonical_array = canonical.as_ref();
        assert_eq!(canonical_array.len(), 3);

        println!("a {}", array.display_tree());

        // Extract the primitive array from the Canonical enum
        if let crate::Canonical::Primitive(primitive) = canonical {
            assert_eq!(primitive.buffer::<i32>().as_slice(), &[15i32, 15, 15]);
        } else {
            panic!("Expected Canonical::Primitive variant");
        }
    }
}
