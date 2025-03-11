use arbitrary::{Arbitrary, Unstructured};

use crate::compute::Operator;

impl<'a> Arbitrary<'a> for Operator {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        Ok(match u.int_in_range(0..=5)? {
            0 => Operator::Eq,
            1 => Operator::NotEq,
            2 => Operator::Gt,
            3 => Operator::Gte,
            4 => Operator::Lt,
            5 => Operator::Lte,
            _ => unreachable!(),
        })
    }
}
