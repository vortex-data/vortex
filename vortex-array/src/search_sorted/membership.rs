// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cmp::Ordering;

use vortex_buffer::BitBufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Bool;
use crate::arrays::BoolArray;
use crate::arrays::Decimal;
use crate::arrays::DecimalArray;
use crate::arrays::Primitive;
use crate::arrays::PrimitiveArray;
use crate::arrays::VarBinView;
use crate::arrays::VarBinViewArray;
use crate::arrays::bool::BoolArrayExt;
use crate::arrays::varbinview::BinaryView;
use crate::buffer::BufferHandle;
use crate::dtype::BigCast;
use crate::dtype::DType;
use crate::dtype::DecimalType;
use crate::dtype::NativePType;
use crate::dtype::i256;
use crate::match_each_native_ptype;

/// Direction of a sorted array.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum SortedDirection {
    Ascending,
    Descending,
}

/// Placement of nulls in a sorted array.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum SortedNulls {
    First,
    Last,
}

/// Complete ordering contract for a [`SortedArray`].
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct SortedOrder {
    pub direction: SortedDirection,
    pub nulls: SortedNulls,
}

/// Whether null is a member of a set that contains null.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum NullEquality {
    Equal,
    Unequal,
}

/// A canonical array whose ordering has been validated once.
///
/// The wrapper retains the canonical member representation and validity mask,
/// so repeated probe batches never canonicalize or validate the full member
/// set again.
#[derive(Clone, Debug)]
pub struct SortedArray {
    array: ArrayRef,
    validity: Mask,
    order: SortedOrder,
}

impl SortedArray {
    pub fn try_new(
        array: ArrayRef,
        order: SortedOrder,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Self> {
        let array = canonicalize_supported(array, ctx)?;
        let validity = array.validity()?.execute_mask(array.len(), ctx)?;
        validate_array_order(&array, &validity, order)?;
        Ok(Self {
            array,
            validity,
            order,
        })
    }

    pub fn array(&self) -> &ArrayRef {
        &self.array
    }

    pub fn order(&self) -> SortedOrder {
        self.order
    }

    pub fn len(&self) -> usize {
        self.array.len()
    }

    pub fn is_empty(&self) -> bool {
        self.array.is_empty()
    }
}

/// Return a mask selecting every sorted `values` entry present in `members`.
///
/// `values` and `members` must have the same logical Bool, Primitive,
/// Decimal, Binary, or UTF-8 dtype; outer nullability may differ. The probe
/// batch is canonicalized and its ordering is validated during the call.
/// Members are validated only by [`SortedArray::try_new`]. Two binary
/// searches narrow the member range to the probe batch's first and last
/// values before one linear merge.
pub fn sorted_membership_mask(
    values: &ArrayRef,
    members: &SortedArray,
    null_equality: NullEquality,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Mask> {
    vortex_ensure!(
        values.dtype().eq_ignore_nullability(members.array.dtype()),
        "sorted membership requires matching logical dtypes, got {} and {}",
        values.dtype(),
        members.array.dtype()
    );
    let values = canonicalize_supported(values.clone(), ctx)?;
    let validity = values.validity()?.execute_mask(values.len(), ctx)?;
    membership_dispatch(
        &values,
        &validity,
        &members.array,
        &members.validity,
        members.order,
        null_equality,
    )
}

fn canonicalize_supported(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
    Ok(match array.dtype() {
        DType::Bool(_) => array.execute::<BoolArray>(ctx)?.into_array(),
        DType::Primitive(..) => array.execute::<PrimitiveArray>(ctx)?.into_array(),
        DType::Decimal(..) => array.execute::<DecimalArray>(ctx)?.into_array(),
        DType::Utf8(_) | DType::Binary(_) => array.execute::<VarBinViewArray>(ctx)?.into_array(),
        dtype => {
            vortex_ensure!(
                false,
                "sorted membership does not support dtype {dtype}; expected Bool, Primitive, Decimal, Binary, or UTF-8"
            );
            unreachable!()
        }
    })
}

fn validate_array_order(array: &ArrayRef, validity: &Mask, order: SortedOrder) -> VortexResult<()> {
    match array.dtype() {
        DType::Bool(_) => {
            let values = array
                .as_typed::<Bool>()
                .vortex_expect("canonical boolean array");
            let bits = values.to_bit_buffer();
            validate_order(array.len(), validity, order, |left, right| {
                bits.value(left).cmp(&bits.value(right))
            })
        }
        DType::Primitive(ptype, _) => match_each_native_ptype!(*ptype, |T| {
            let array = array
                .as_typed::<Primitive>()
                .vortex_expect("canonical primitive array");
            let values = array.as_slice::<T>();
            validate_order(array.len(), validity, order, |left, right| {
                values[left].total_compare(values[right])
            })
        }),
        DType::Decimal(..) => {
            let array = array
                .as_typed::<Decimal>()
                .vortex_expect("canonical decimal array");
            let values = DecimalValues::new(&array);
            validate_order(array.len(), validity, order, |left, right| {
                values.value(left).cmp(&values.value(right))
            })
        }
        DType::Utf8(_) | DType::Binary(_) => {
            let array = array
                .as_typed::<VarBinView>()
                .vortex_expect("canonical variable-width array");
            let values = VarBinValues::new(&array);
            validate_order(array.len(), validity, order, |left, right| {
                values.value(left).cmp(values.value(right))
            })
        }
        _ => unreachable!("canonicalize_supported rejects other dtypes"),
    }
}

fn membership_dispatch(
    values: &ArrayRef,
    value_validity: &Mask,
    members: &ArrayRef,
    member_validity: &Mask,
    order: SortedOrder,
    null_equality: NullEquality,
) -> VortexResult<Mask> {
    match values.dtype() {
        DType::Bool(_) => {
            let values = values
                .as_typed::<Bool>()
                .vortex_expect("canonical boolean values")
                .to_bit_buffer();
            let members = members
                .as_typed::<Bool>()
                .vortex_expect("canonical boolean members")
                .to_bit_buffer();
            membership_core(
                values.len(),
                value_validity,
                members.len(),
                member_validity,
                order,
                null_equality,
                |left, right| values.value(left).cmp(&values.value(right)),
                |member, value| members.value(member).cmp(&values.value(value)),
                |member, value| members.value(member) == values.value(value),
            )
        }
        DType::Primitive(ptype, _) => match_each_native_ptype!(*ptype, |T| {
            let value_array = values
                .as_typed::<Primitive>()
                .vortex_expect("canonical primitive values");
            let member_array = members
                .as_typed::<Primitive>()
                .vortex_expect("canonical primitive members");
            let values = value_array.as_slice::<T>();
            let members = member_array.as_slice::<T>();
            membership_core(
                values.len(),
                value_validity,
                members.len(),
                member_validity,
                order,
                null_equality,
                |left, right| values[left].total_compare(values[right]),
                |member, value| members[member].total_compare(values[value]),
                |member, value| members[member].is_eq(values[value]),
            )
        }),
        DType::Decimal(..) => {
            let value_array = values
                .as_typed::<Decimal>()
                .vortex_expect("canonical decimal values");
            let member_array = members
                .as_typed::<Decimal>()
                .vortex_expect("canonical decimal members");
            let values = DecimalValues::new(&value_array);
            let members = DecimalValues::new(&member_array);
            membership_core(
                values.len(),
                value_validity,
                members.len(),
                member_validity,
                order,
                null_equality,
                |left, right| values.value(left).cmp(&values.value(right)),
                |member, value| members.value(member).cmp(&values.value(value)),
                |member, value| members.value(member) == values.value(value),
            )
        }
        DType::Utf8(_) | DType::Binary(_) => {
            let value_array = values
                .as_typed::<VarBinView>()
                .vortex_expect("canonical variable-width values");
            let member_array = members
                .as_typed::<VarBinView>()
                .vortex_expect("canonical variable-width members");
            let values = VarBinValues::new(&value_array);
            let members = VarBinValues::new(&member_array);
            membership_core(
                values.len(),
                value_validity,
                members.len(),
                member_validity,
                order,
                null_equality,
                |left, right| values.value(left).cmp(values.value(right)),
                |member, value| members.value(member).cmp(values.value(value)),
                |member, value| members.value(member) == values.value(value),
            )
        }
        _ => unreachable!("canonicalize_supported rejects other dtypes"),
    }
}

#[allow(clippy::too_many_arguments)]
fn membership_core(
    values_len: usize,
    value_validity: &Mask,
    members_len: usize,
    member_validity: &Mask,
    order: SortedOrder,
    null_equality: NullEquality,
    value_cmp: impl Fn(usize, usize) -> Ordering,
    member_value_cmp: impl Fn(usize, usize) -> Ordering,
    member_value_eq: impl Fn(usize, usize) -> bool,
) -> VortexResult<Mask> {
    validate_order(values_len, value_validity, order, &value_cmp)?;
    if values_len == 0 || members_len == 0 {
        return Ok(Mask::new_false(values_len));
    }

    let compare = |member: usize, value: usize| {
        entry_cmp(
            member_validity.value(member),
            value_validity.value(value),
            order,
            || member_value_cmp(member, value),
        )
    };
    let lower = partition_point(members_len, |member| compare(member, 0).is_lt());
    let upper = lower
        + partition_point(members_len - lower, |offset| {
            !compare(lower + offset, values_len - 1).is_gt()
        });

    let mut selected = BitBufferMut::with_capacity(values_len);
    let mut member = lower;
    for value in 0..values_len {
        while member < upper && compare(member, value).is_lt() {
            member += 1;
        }
        let keep = if member == upper || !compare(member, value).is_eq() {
            false
        } else if value_validity.value(value) {
            member_value_eq(member, value)
        } else {
            null_equality == NullEquality::Equal
        };
        selected.append(keep);
    }
    Ok(Mask::from_buffer(selected.freeze()))
}

fn validate_order(
    len: usize,
    validity: &Mask,
    order: SortedOrder,
    value_cmp: impl Fn(usize, usize) -> Ordering,
) -> VortexResult<()> {
    for index in 1..len {
        vortex_ensure!(
            !entry_cmp(
                validity.value(index - 1),
                validity.value(index),
                order,
                || value_cmp(index - 1, index),
            )
            .is_gt(),
            "sorted membership input violates its {:?} order at index {index}",
            order
        );
    }
    Ok(())
}

fn entry_cmp(
    left_valid: bool,
    right_valid: bool,
    order: SortedOrder,
    value_cmp: impl FnOnce() -> Ordering,
) -> Ordering {
    match (left_valid, right_valid) {
        (false, false) => Ordering::Equal,
        (false, true) => match order.nulls {
            SortedNulls::First => Ordering::Less,
            SortedNulls::Last => Ordering::Greater,
        },
        (true, false) => match order.nulls {
            SortedNulls::First => Ordering::Greater,
            SortedNulls::Last => Ordering::Less,
        },
        (true, true) => match order.direction {
            SortedDirection::Ascending => value_cmp(),
            SortedDirection::Descending => value_cmp().reverse(),
        },
    }
}

fn partition_point(len: usize, mut predicate: impl FnMut(usize) -> bool) -> usize {
    let mut left = 0;
    let mut right = len;
    while left < right {
        let middle = left + (right - left) / 2;
        if predicate(middle) {
            left = middle + 1;
        } else {
            right = middle;
        }
    }
    left
}

struct VarBinValues<'a> {
    views: &'a [BinaryView],
    buffers: &'a [BufferHandle],
}

impl<'a> VarBinValues<'a> {
    fn new(array: &'a ArrayView<'a, VarBinView>) -> Self {
        Self {
            views: array.views(),
            buffers: array.data_buffers(),
        }
    }

    fn len(&self) -> usize {
        self.views.len()
    }

    fn value(&self, index: usize) -> &[u8] {
        let view = &self.views[index];
        if view.is_inlined() {
            view.as_inlined().value()
        } else {
            let reference = view.as_view();
            &self.buffers[reference.buffer_index as usize].as_host()[reference.as_range()]
        }
    }
}

enum DecimalValues {
    I8(vortex_buffer::Buffer<i8>),
    I16(vortex_buffer::Buffer<i16>),
    I32(vortex_buffer::Buffer<i32>),
    I64(vortex_buffer::Buffer<i64>),
    I128(vortex_buffer::Buffer<i128>),
    I256(vortex_buffer::Buffer<i256>),
}

impl DecimalValues {
    fn new(array: &ArrayView<'_, Decimal>) -> Self {
        match array.values_type() {
            DecimalType::I8 => Self::I8(array.buffer::<i8>()),
            DecimalType::I16 => Self::I16(array.buffer::<i16>()),
            DecimalType::I32 => Self::I32(array.buffer::<i32>()),
            DecimalType::I64 => Self::I64(array.buffer::<i64>()),
            DecimalType::I128 => Self::I128(array.buffer::<i128>()),
            DecimalType::I256 => Self::I256(array.buffer::<i256>()),
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::I8(values) => values.len(),
            Self::I16(values) => values.len(),
            Self::I32(values) => values.len(),
            Self::I64(values) => values.len(),
            Self::I128(values) => values.len(),
            Self::I256(values) => values.len(),
        }
    }

    fn value(&self, index: usize) -> i256 {
        match self {
            Self::I8(values) => <i256 as BigCast>::from(values[index]),
            Self::I16(values) => <i256 as BigCast>::from(values[index]),
            Self::I32(values) => <i256 as BigCast>::from(values[index]),
            Self::I64(values) => <i256 as BigCast>::from(values[index]),
            Self::I128(values) => <i256 as BigCast>::from(values[index]),
            Self::I256(values) => Some(values[index]),
        }
        .vortex_expect("every decimal storage value widens to i256")
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use vortex_buffer::Buffer;

    use super::*;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::arrays::DecimalArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::VarBinViewArray;
    use crate::dtype::DecimalDType;
    use crate::validity::Validity;

    const ASC: SortedOrder = SortedOrder {
        direction: SortedDirection::Ascending,
        nulls: SortedNulls::First,
    };

    #[test]
    fn primitive_membership_handles_duplicates_and_range_narrowing() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let members = PrimitiveArray::from_iter(0_i64..100_000).into_array();
        let members = SortedArray::try_new(members, ASC, &mut ctx)?;
        let values =
            PrimitiveArray::from_iter([49_999_i64, 50_000, 50_000, 75_000, 100_001]).into_array();
        let mask = sorted_membership_mask(&values, &members, NullEquality::Unequal, &mut ctx)?;
        assert_eq!(
            mask.iter().collect::<Vec<_>>(),
            [true, true, true, true, false]
        );
        Ok(())
    }

    #[test]
    fn boolean_and_empty_membership() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let members = SortedArray::try_new(
            BoolArray::from_iter([false, true]).into_array(),
            ASC,
            &mut ctx,
        )?;
        let values = BoolArray::from_iter([false, false, true]).into_array();
        assert!(
            sorted_membership_mask(&values, &members, NullEquality::Unequal, &mut ctx)?.all_true()
        );

        let empty = SortedArray::try_new(
            PrimitiveArray::from_iter(Vec::<i32>::new()).into_array(),
            ASC,
            &mut ctx,
        )?;
        let values = PrimitiveArray::from_iter([1_i32, 2]).into_array();
        assert!(
            sorted_membership_mask(&values, &empty, NullEquality::Unequal, &mut ctx)?.all_false()
        );
        Ok(())
    }

    #[test]
    fn float_membership_uses_vortex_bitwise_identity() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let members = SortedArray::try_new(
            PrimitiveArray::from_iter([-0.0_f64, 1.0, f64::NAN]).into_array(),
            ASC,
            &mut ctx,
        )?;
        let values = PrimitiveArray::from_iter([-0.0_f64, 0.0, 1.0, f64::NAN]).into_array();
        assert_eq!(
            sorted_membership_mask(&values, &members, NullEquality::Unequal, &mut ctx)?
                .iter()
                .collect::<Vec<_>>(),
            [true, false, true, true]
        );
        Ok(())
    }

    #[test]
    fn member_range_is_narrowed_before_linear_merge() -> VortexResult<()> {
        let members = (0_i64..1_000_000).collect::<Vec<_>>();
        let values = (500_000_i64..500_010).collect::<Vec<_>>();
        let comparisons = Cell::new(0_usize);
        let mask = membership_core(
            values.len(),
            &Mask::new_true(values.len()),
            members.len(),
            &Mask::new_true(members.len()),
            ASC,
            NullEquality::Unequal,
            |left, right| values[left].cmp(&values[right]),
            |member, value| {
                comparisons.set(comparisons.get() + 1);
                members[member].cmp(&values[value])
            },
            |member, value| members[member] == values[value],
        )?;
        assert!(mask.all_true());
        assert!(
            comparisons.get() < 100,
            "range narrowing performed {} comparisons",
            comparisons.get()
        );
        Ok(())
    }

    #[test]
    fn descending_null_semantics_are_explicit() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let order = SortedOrder {
            direction: SortedDirection::Descending,
            nulls: SortedNulls::Last,
        };
        let members = PrimitiveArray::from_option_iter([Some(9_i32), Some(3), None]).into_array();
        let members = SortedArray::try_new(members, order, &mut ctx)?;
        let values =
            PrimitiveArray::from_option_iter([Some(10_i32), Some(9), Some(3), None]).into_array();
        assert_eq!(
            sorted_membership_mask(&values, &members, NullEquality::Equal, &mut ctx)?
                .iter()
                .collect::<Vec<_>>(),
            [false, true, true, true]
        );
        assert_eq!(
            sorted_membership_mask(&values, &members, NullEquality::Unequal, &mut ctx)?
                .iter()
                .collect::<Vec<_>>(),
            [false, true, true, false]
        );
        Ok(())
    }

    #[test]
    fn decimal_membership_accepts_mixed_physical_widths() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let dtype = DecimalDType::new(20, 2);
        let members = DecimalArray::new(
            Buffer::from(vec![-100_i128, 0, 250]),
            dtype,
            Validity::NonNullable,
        )
        .into_array();
        let members = SortedArray::try_new(members, ASC, &mut ctx)?;
        let values = DecimalArray::new(
            Buffer::from(vec![
                <i256 as BigCast>::from(-100_i128).vortex_expect("test decimal fits"),
                <i256 as BigCast>::from(1_i128).vortex_expect("test decimal fits"),
                <i256 as BigCast>::from(250_i128).vortex_expect("test decimal fits"),
            ]),
            dtype,
            Validity::NonNullable,
        )
        .into_array();
        assert_eq!(
            sorted_membership_mask(&values, &members, NullEquality::Unequal, &mut ctx)?
                .iter()
                .collect::<Vec<_>>(),
            [true, false, true]
        );
        Ok(())
    }

    #[test]
    fn varbin_membership_reads_inline_and_external_values() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let members =
            VarBinViewArray::from_iter_bin([b"alpha".as_slice(), b"external-value-0001", b"omega"])
                .into_array();
        let members = SortedArray::try_new(members, ASC, &mut ctx)?;
        let values = VarBinViewArray::from_iter_bin([
            b"alpha".as_slice(),
            b"external-value-0000",
            b"external-value-0001",
            b"omega",
        ])
        .into_array();
        assert_eq!(
            sorted_membership_mask(&values, &members, NullEquality::Unequal, &mut ctx)?
                .iter()
                .collect::<Vec<_>>(),
            [true, false, true, true]
        );
        Ok(())
    }

    #[test]
    fn rejects_unsorted_members_and_values() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let unsorted = PrimitiveArray::from_iter([2_i32, 1]).into_array();
        assert!(SortedArray::try_new(unsorted, ASC, &mut ctx).is_err());

        let members = SortedArray::try_new(
            PrimitiveArray::from_iter([1_i32, 2]).into_array(),
            ASC,
            &mut ctx,
        )?;
        let values = PrimitiveArray::from_iter([2_i32, 1]).into_array();
        assert!(
            sorted_membership_mask(&values, &members, NullEquality::Unequal, &mut ctx).is_err()
        );
        Ok(())
    }

    #[test]
    fn rejects_mismatched_and_unsupported_dtypes() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let members = SortedArray::try_new(
            PrimitiveArray::from_iter([1_i32]).into_array(),
            ASC,
            &mut ctx,
        )?;
        let wrong = PrimitiveArray::from_iter([1_i64]).into_array();
        assert!(sorted_membership_mask(&wrong, &members, NullEquality::Unequal, &mut ctx).is_err());
        Ok(())
    }
}
