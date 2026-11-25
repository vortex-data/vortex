// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::scalar_fn::builtin::literal::Literal;

use crate::eq;
use crate::lit;
use crate::root;

#[test]
fn expr_contains() {
    let expression = &eq(root(), lit(3u64));
    assert!(expression.contains::<Literal>().unwrap());
    let expression = root();
    assert!(!expression.contains::<Literal>().unwrap());
}
