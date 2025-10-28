// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::BitAnd;

use vortex_buffer::{BitBuffer, BufferMut};
use vortex_vector::{BoolVector, VectorOps};

use crate::comparison::{
    Compare, Equal, GreaterThan, GreaterThanOrEqual, LessThan, LessThanOrEqual, NotEqual,
};

impl<Op> Compare<Op> for &BoolVector
where
    Op: BitComparisonOperator,
{
    type Output = BoolVector;

    fn compare(self, rhs: Self) -> Self::Output {
        let validity = self.validity().bitand(rhs.validity());

        let lhs = self.bits().chunks();
        let rhs = rhs.bits().chunks();

        // Reserve one extra chunk to account for partial padding chunk at the end.
        let mut buffer = BufferMut::<u64>::with_capacity(lhs.chunk_len() + 1);
        buffer.extend(
            lhs.iter_padded()
                .zip(rhs.iter_padded())
                .map(|(a_chunk, b_chunk)| Op::apply(&a_chunk, &b_chunk)),
        );
        let bits = BitBuffer::new(buffer.freeze().into_byte_buffer(), self.len());

        BoolVector::new(bits, validity)
    }
}

pub trait BitComparisonOperator {
    fn apply(a: &u64, b: &u64) -> u64;
}

impl BitComparisonOperator for Equal {
    fn apply(a: &u64, b: &u64) -> u64 {
        !(a ^ b)
    }
}
impl BitComparisonOperator for NotEqual {
    fn apply(a: &u64, b: &u64) -> u64 {
        a ^ b
    }
}
impl BitComparisonOperator for LessThan {
    fn apply(a: &u64, b: &u64) -> u64 {
        (!a) & b
    }
}
impl BitComparisonOperator for LessThanOrEqual {
    fn apply(a: &u64, b: &u64) -> u64 {
        !(a & (!b))
    }
}
impl BitComparisonOperator for GreaterThan {
    fn apply(a: &u64, b: &u64) -> u64 {
        a & (!b)
    }
}
impl BitComparisonOperator for GreaterThanOrEqual {
    fn apply(a: &u64, b: &u64) -> u64 {
        !((!a) & b)
    }
}
