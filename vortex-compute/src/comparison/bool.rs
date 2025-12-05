// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::BitAnd;

use vortex_buffer::BitBuffer;
use vortex_buffer::BufferMut;
use vortex_vector::VectorOps;
use vortex_vector::bool::BoolScalar;
use vortex_vector::bool::BoolVector;

use crate::comparison::Compare;
use crate::comparison::Equal;
use crate::comparison::GreaterThan;
use crate::comparison::GreaterThanOrEqual;
use crate::comparison::LessThan;
use crate::comparison::LessThanOrEqual;
use crate::comparison::NotEqual;

impl Compare<Equal> for BoolScalar {
    type Output = BoolScalar;

    fn compare(self, rhs: Self) -> Self::Output {
        BoolScalar::new(self.value().zip(rhs.value()).map(|(l, r)| l == r))
    }
}

impl Compare<NotEqual> for BoolScalar {
    type Output = BoolScalar;

    fn compare(self, rhs: Self) -> Self::Output {
        BoolScalar::new(self.value().zip(rhs.value()).map(|(l, r)| l != r))
    }
}

impl Compare<LessThan> for BoolScalar {
    type Output = BoolScalar;

    fn compare(self, rhs: Self) -> Self::Output {
        BoolScalar::new(self.value().zip(rhs.value()).map(|(l, r)| !l && r))
    }
}

impl Compare<LessThanOrEqual> for BoolScalar {
    type Output = BoolScalar;

    fn compare(self, rhs: Self) -> Self::Output {
        BoolScalar::new(self.value().zip(rhs.value()).map(|(l, r)| !l || r))
    }
}

impl Compare<GreaterThan> for BoolScalar {
    type Output = BoolScalar;

    fn compare(self, rhs: Self) -> Self::Output {
        BoolScalar::new(self.value().zip(rhs.value()).map(|(l, r)| l && !r))
    }
}

impl Compare<GreaterThanOrEqual> for BoolScalar {
    type Output = BoolScalar;

    fn compare(self, rhs: Self) -> Self::Output {
        BoolScalar::new(self.value().zip(rhs.value()).map(|(l, r)| l || !r))
    }
}

impl<Op> Compare<Op> for BoolVector
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
