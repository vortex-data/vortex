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
use crate::dtype::DType;
use crate::dtype::IntegerPType;
use crate::dtype::NativePType;
use crate::dtype::Nullability;
use crate::scalar::Scalar;

/// A prepared integer set for constant-list membership kernels.
///
/// The set sorts and deduplicates its members.
#[doc(hidden)]
pub struct IntegerMembership<T> {
    members: Box<[T]>,
    non_null_source_len: usize,
    source_list: Scalar,
}

impl<T: IntegerPType> IntegerMembership<T> {
    fn new(mut members: Vec<T>, source_list: Scalar) -> Self {
        let non_null_source_len = members.len();
        members.sort_unstable();
        members.dedup();
        Self {
            members: members.into_boxed_slice(),
            non_null_source_len,
            source_list,
        }
    }

    /// Extracts an integer set from a compatible constant list.
    pub fn try_from_constant_list(
        list: &ArrayRef,
        element_dtype: &DType,
    ) -> VortexResult<Option<Self>> {
        let Some(list_scalar) = list.as_constant() else {
            return Ok(None);
        };
        let DType::List(member_dtype, _) = list.dtype() else {
            return Ok(None);
        };
        if !member_dtype.eq_ignore_nullability(element_dtype) {
            return Ok(None);
        }
        let Some(elements) = list_scalar.as_list().elements() else {
            return Ok(None);
        };

        let members = elements
            .iter()
            .map(|value| {
                value
                    .as_primitive_opt()
                    .vortex_expect("list member type was checked")
                    .try_typed_value::<T>()
            })
            .collect::<VortexResult<Vec<Option<T>>>>()?
            .into_iter()
            .flatten()
            .collect();
        Ok(Some(Self::new(members, list_scalar)))
    }

    /// Returns the prepared members.
    pub fn members(&self) -> &[T] {
        &self.members
    }

    /// Returns the number of non-null source members before deduplication.
    #[doc(hidden)]
    pub fn non_null_source_len(&self) -> usize {
        self.non_null_source_len
    }

    pub(crate) fn source_list(&self) -> &Scalar {
        &self.source_list
    }

    /// Tests whether the prepared set contains `value`.
    pub(crate) fn contains(&self, value: T) -> bool {
        self.members.binary_search(&value).is_ok()
    }

    /// Evaluates this set against a primitive array of the same integer type.
    pub(crate) fn evaluate_primitive(
        self,
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
            _ => collect_many(values, &self),
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
    BitBuffer::collect_bool(values.len(), |index| {
        // SAFETY: collect_bool visits each valid index once.
        let value = unsafe { *values.get_unchecked(index) };
        membership.contains(value)
    })
}
