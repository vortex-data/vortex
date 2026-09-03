// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::arrays::Constant;
use crate::dtype::DType;
use crate::dtype::IntegerPType;

/// Maximum source members for the direct integer membership kernel.
#[doc(hidden)]
pub const MAX_DIRECT_INTEGER_MEMBERS: usize = 4;

/// A prepared integer set for constant-list membership kernels.
///
/// The set sorts and deduplicates its members.
#[doc(hidden)]
pub struct IntegerMembership<T> {
    members: Box<[T]>,
}

impl<T: IntegerPType> IntegerMembership<T> {
    fn new(mut members: Vec<T>) -> Self {
        members.sort_unstable();
        members.dedup();
        Self {
            members: members.into_boxed_slice(),
        }
    }

    /// Extracts an integer set from a compatible constant list.
    pub fn try_from_constant_list(
        list: &ArrayRef,
        element_dtype: &DType,
    ) -> VortexResult<Option<Self>> {
        let Some(list_array) = list.as_opt::<Constant>() else {
            return Ok(None);
        };
        let DType::List(member_dtype, _) = list.dtype() else {
            return Ok(None);
        };
        if !member_dtype.eq_ignore_nullability(element_dtype) {
            return Ok(None);
        }
        let Some(elements) = list_array.scalar().as_list().values() else {
            return Ok(None);
        };
        if elements.len() > MAX_DIRECT_INTEGER_MEMBERS {
            return Ok(None);
        }

        let members = elements
            .iter()
            .filter_map(|value| value.as_ref())
            .map(|value| {
                // The validated list scalar stores primitive values of `member_dtype`.
                value.as_primitive().cast::<T>()
            })
            .collect::<VortexResult<Vec<T>>>()?;
        Ok(Some(Self::new(members)))
    }

    /// Returns the prepared members.
    pub fn members(&self) -> &[T] {
        &self.members
    }
}
