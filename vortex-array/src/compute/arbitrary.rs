// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use arbitrary::Arbitrary;
use arbitrary::Unstructured;

use crate::scalar_fn::CompareOperator;

impl<'a> Arbitrary<'a> for CompareOperator {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        Ok(match u.int_in_range(0..=5)? {
            0 => CompareOperator::Eq,
            1 => CompareOperator::NotEq,
            2 => CompareOperator::Gt,
            3 => CompareOperator::Gte,
            4 => CompareOperator::Lt,
            5 => CompareOperator::Lte,
            _ => unreachable!(),
        })
    }
}
