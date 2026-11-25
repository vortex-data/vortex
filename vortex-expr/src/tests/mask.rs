// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability::Nullable;
use vortex_array::dtype::PType;
use vortex_array::scalar::Scalar;
use vortex_error::VortexExpect;

use crate::bound;
use crate::lit;
use crate::mask;

#[test]
fn test_simplify() {
    let input_expr = lit(42u32);
    let true_mask_expr = lit(true);
    let false_mask_expr = lit(false);

    let mask_true_expr = mask(input_expr.clone(), true_mask_expr);
    let simplified_true = mask_true_expr
        .bind(&DType::Null)
        .and_then(|expr| expr.optimize())
        .vortex_expect("Simplification");
    assert_eq!(
        &simplified_true,
        &input_expr.bind(&DType::Null).vortex_expect("bind")
    );

    let mask_false_expr = mask(input_expr, false_mask_expr);
    let simplified_false = mask_false_expr
        .bind(&DType::Null)
        .and_then(|expr| expr.optimize())
        .vortex_expect("Simplification");
    let expected_null_expr = bound::lit(Scalar::null(DType::Primitive(PType::U32, Nullable)));
    assert_eq!(&simplified_false, &expected_null_expr);
}
