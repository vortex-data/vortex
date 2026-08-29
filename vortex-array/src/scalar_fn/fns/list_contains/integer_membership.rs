// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::BitBuffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::ArrayRef;
use crate::ArrayView;
use crate::IntoArray;
use crate::arrays::BoolArray;
use crate::arrays::Primitive;
use crate::dtype::IntegerPType;
use crate::dtype::NativePType;
use crate::dtype::Nullability;

const MAX_DENSE_SPAN: usize = 4_096;

/// A prepared integer set for constant-list membership kernels.
///
/// The set sorts and deduplicates lists with more than four members. It builds a byte table when
/// the member span fits the bounded table.
pub struct IntegerMembership<T> {
    members: Box<[T]>,
    dense: Option<DenseIntegerMembership>,
}

impl<T: IntegerPType> IntegerMembership<T> {
    /// Prepares a membership set from integer values.
    pub fn new(mut members: Vec<T>) -> Self {
        if members.len() > 4 {
            members.sort_unstable();
            members.dedup();
        }
        let dense = DenseIntegerMembership::try_new(&members);

        Self {
            members: members.into_boxed_slice(),
            dense,
        }
    }

    /// Returns the normalized members.
    pub fn members(&self) -> &[T] {
        &self.members
    }

    /// Returns true when this set uses a dense lookup table.
    pub fn uses_dense_table(&self) -> bool {
        self.dense.is_some()
    }

    /// Tests membership through the selected lookup representation.
    pub fn contains(&self, value: T) -> bool {
        self.dense.as_ref().map_or_else(
            || {
                if self.members.len() <= 4 {
                    self.members.contains(&value)
                } else {
                    self.members.binary_search(&value).is_ok()
                }
            },
            |dense| dense.contains(value),
        )
    }

    /// Evaluates this set against a primitive array of the same integer type.
    pub fn evaluate_primitive(
        &self,
        element: ArrayView<'_, Primitive>,
        nullability: Nullability,
    ) -> VortexResult<ArrayRef> {
        vortex_ensure!(
            element.ptype() == T::PTYPE,
            "Membership type {} does not match array type {}",
            T::PTYPE,
            element.ptype(),
        );
        let values = element.as_slice::<T>();
        let bits = match self.members() {
            [] => BitBuffer::new_unset(values.len()),
            [member] => collect_direct(values, move |value| value.is_eq(*member)),
            [first, second] => collect_direct(values, move |value| {
                value.is_eq(*first) | value.is_eq(*second)
            }),
            [first, second, third] => collect_direct(values, move |value| {
                value.is_eq(*first) | value.is_eq(*second) | value.is_eq(*third)
            }),
            [first, second, third, fourth] => collect_direct(values, move |value| {
                value.is_eq(*first)
                    | value.is_eq(*second)
                    | value.is_eq(*third)
                    | value.is_eq(*fourth)
            }),
            _ => collect_many(values, self),
        };

        Ok(BoolArray::new(bits, element.validity()?.union_nullability(nullability)).into_array())
    }
}

fn collect_direct<T: NativePType>(values: &[T], mut predicate: impl FnMut(T) -> bool) -> BitBuffer {
    BitBuffer::collect_bool_multiversioned(values.len(), |index| {
        // SAFETY: collect_bool_multiversioned visits each valid index once.
        predicate(unsafe { *values.get_unchecked(index) })
    })
}

fn collect_many<T: IntegerPType>(values: &[T], membership: &IntegerMembership<T>) -> BitBuffer {
    if let Some(dense) = membership.dense.as_ref() {
        return BitBuffer::collect_bool(values.len(), |index| {
            // SAFETY: collect_bool visits each valid index once.
            let value = unsafe { *values.get_unchecked(index) };
            dense.contains(value)
        });
    }

    BitBuffer::collect_bool(values.len(), |index| {
        // SAFETY: collect_bool visits each valid index once.
        let value = unsafe { *values.get_unchecked(index) };
        membership.contains(value)
    })
}

/// A bounded byte table for dense integer membership.
struct DenseIntegerMembership {
    minimum: i128,
    table: Box<[u8]>,
}

impl DenseIntegerMembership {
    fn try_new<T: IntegerPType>(members: &[T]) -> Option<Self> {
        if members.len() <= 4 {
            return None;
        }

        let minimum = members[0].to_i128()?;
        let maximum = members[members.len() - 1].to_i128()?;
        let span = usize::try_from(maximum - minimum + 1).ok()?;
        if span > MAX_DENSE_SPAN {
            return None;
        }

        let mut table = vec![0u8; span];
        for member in members {
            let index = usize::try_from(
                member.to_i128().vortex_expect("integer converts to i128") - minimum,
            )
            .vortex_expect("member lies inside the dense span");
            table[index] = 1;
        }

        Some(Self {
            minimum,
            table: table.into_boxed_slice(),
        })
    }

    /// Tests whether the table contains an integer value.
    fn contains<T: IntegerPType>(&self, value: T) -> bool {
        let offset = value.to_i128().vortex_expect("integer converts to i128") - self.minimum;
        usize::try_from(offset)
            .ok()
            .and_then(|offset| self.table.get(offset))
            .copied()
            .unwrap_or(0)
            != 0
    }
}

#[cfg(test)]
mod tests {
    use super::IntegerMembership;

    #[test]
    fn normalizes_large_unsorted_duplicates() {
        let membership = IntegerMembership::new(vec![7i32, 3, 7, 1, 9, 3, 1]);

        assert_eq!(membership.members(), &[1, 3, 7, 9]);
        assert!(membership.contains(1));
        assert!(membership.contains(3));
        assert!(membership.contains(7));
        assert!(!membership.contains(5));
    }

    #[test]
    fn dense_table_span_boundary() {
        let at_limit = IntegerMembership::new(vec![0i32, 1, 2, 3, 4_095]);
        let above_limit = IntegerMembership::new(vec![0i32, 1, 2, 3, 4_096]);

        assert!(at_limit.uses_dense_table());
        assert!(!above_limit.uses_dense_table());
    }

    #[test]
    fn integer_extremes_do_not_overflow() {
        let signed = IntegerMembership::new(vec![i64::MAX, 0, i64::MIN, -1, 1]);
        assert!(signed.contains(i64::MIN));
        assert!(signed.contains(i64::MAX));
        assert!(!signed.uses_dense_table());

        let unsigned = IntegerMembership::new(vec![u64::MAX, 0, 1, 2, 3]);
        assert!(unsigned.contains(0));
        assert!(unsigned.contains(u64::MAX));
        assert!(!unsigned.uses_dense_table());
    }
}
