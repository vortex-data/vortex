// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Choosing the fill value of a sparse array: the most common valid value of the input.

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::DecimalArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::bool::BoolArrayExt;
use vortex_array::arrays::varbinview::BinaryView;
use vortex_array::dtype::DType;
use vortex_array::match_each_decimal_value_type;
use vortex_array::scalar::Scalar;
use vortex_array::stats::majority::MajorityKey;
use vortex_array::stats::majority::majority_vote;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_mask::AllOr;
use vortex_mask::Mask;

/// Returns the dominant valid value of `array`, whose validity is `mask`.
///
/// That is the value filling more than half of the valid positions whenever there is one, found
/// by a Boyer–Moore vote. Otherwise it is some valid value, which still makes a correct, if poorly
/// compressed, fill. At least one value must be valid.
pub(crate) fn dominant_value(
    array: &ArrayRef,
    mask: &Mask,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Scalar> {
    let nullability = array.dtype().nullability();
    match array.dtype() {
        DType::Primitive(..) => {
            let primitive = array.clone().execute::<PrimitiveArray>(ctx)?;
            let (top, _) = primitive
                .top_value(ctx)?
                .vortex_expect("array has a valid value");
            Ok(Scalar::primitive_value(top, top.ptype(), nullability))
        }
        DType::Bool(_) => {
            let bools = array.clone().execute::<BoolArray>(ctx)?;
            let valid_true = match mask.bit_buffer() {
                AllOr::All => bools.to_bit_buffer().true_count(),
                AllOr::None => unreachable!("array has a valid value"),
                AllOr::Some(validity) => (&bools.to_bit_buffer() & validity).true_count(),
            };
            let valid_false = mask.true_count() - valid_true;
            Ok(Scalar::bool(valid_true > valid_false, nullability))
        }
        DType::Decimal(..) => {
            let decimal = array.clone().execute::<DecimalArray>(ctx)?;
            let index = match_each_decimal_value_type!(decimal.values_type(), |D| {
                let values = decimal.buffer::<D>();
                let (top, _) =
                    majority_vote(&values, mask).vortex_expect("array has a valid value");
                first_valid_index_of(&values, mask, top)
            });
            decimal.into_array().execute_scalar(index, ctx)
        }
        DType::Utf8(_) | DType::Binary(_) => {
            let bytes = array.clone().execute::<VarBinViewArray>(ctx)?;
            let index = view_vote_index(&bytes, mask);
            bytes.into_array().execute_scalar(index, ctx)
        }
        dtype => vortex_bail!(
            "cannot choose a fill value for a {dtype} array, so one must be given explicitly"
        ),
    }
}

/// A string view compared by the group of values it may hold.
///
/// An inline view holds its whole value, zero-padded, so two inline views are in one group
/// exactly when their values are equal. A reference view is grouped by its
/// [`head`](BinaryView::head), the value's length and first four bytes. Every value lies in exactly
/// one group, so a value filling more than half of an array also makes its group a majority,
/// which the vectorized vote finds from the views alone.
#[derive(Clone, Copy, Default)]
#[repr(transparent)]
struct ViewGroup(BinaryView);

impl MajorityKey for ViewGroup {
    #[inline]
    fn key_eq(self, other: Self) -> bool {
        // Equal heads mean equal lengths, so `self` alone tells whether both are inline.
        self.0.head() == other.0.head()
            && (!self.0.is_inlined() || self.0.as_u128() == other.0.as_u128())
    }
}

/// Returns a valid index holding the dominant value of `bytes`.
///
/// The vectorized vote over [`ViewGroup`]s finds the majority value's group. An inline group is a
/// single value. A group of longer values shares a length and prefix, and a scalar vote over its
/// members, comparing their bytes, finds the majority within it, which is the majority overall.
fn view_vote_index(bytes: &VarBinViewArray, mask: &Mask) -> usize {
    let views = bytes.views();
    // SAFETY: `ViewGroup` is a transparent wrapper around `BinaryView`.
    let groups: &[ViewGroup] =
        unsafe { std::slice::from_raw_parts(views.as_ptr().cast(), views.len()) };
    let (group, _) = majority_vote(groups, mask).vortex_expect("array has a valid value");
    if group.0.is_inlined() {
        return first_valid_index_of(groups, mask, group);
    }

    let buffers: Vec<&[u8]> = (0..bytes.data_buffers().len())
        .map(|index| bytes.buffer(index).as_slice())
        .collect();
    let value = |index: usize| views[index].bytes(&buffers);
    let mut candidate = None;
    let mut surplus = 0usize;
    let members =
        (0..views.len()).filter(|&index| mask.value(index) && groups[index].key_eq(group));
    for index in members {
        match candidate {
            Some(current) if value(current) == value(index) => surplus += 1,
            _ if surplus == 0 => {
                candidate = Some(index);
                surplus = 1;
            }
            _ => surplus -= 1,
        }
    }
    candidate.vortex_expect("the group has a valid member")
}

/// Returns the first valid index holding `value`, which must occur at one.
fn first_valid_index_of<T: MajorityKey>(values: &[T], mask: &Mask, value: T) -> usize {
    values
        .iter()
        .enumerate()
        .position(|(index, &candidate)| candidate.key_eq(value) && mask.value(index))
        .vortex_expect("the majority occurs at a valid index")
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::StructArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::DecimalDType;
    use vortex_array::dtype::Nullability;
    use vortex_array::scalar::DecimalValue;
    use vortex_session::VortexSession;

    use super::*;
    use crate::Sparse;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty();
        crate::initialize(&session);
        session
    });

    const LONG: &str = "a string too long to inline in a view";

    /// Nulls fill most rows, but the fill is the most common valid value.
    fn nullable_bools() -> ArrayRef {
        BoolArray::from_iter((0..100).map(|index| match index % 10 {
            0 => Some(false),
            1..4 => Some(true),
            _ => None,
        }))
        .into_array()
    }

    fn nullable_strings() -> ArrayRef {
        VarBinViewArray::from_iter_nullable_str((0..100).map(|index| match index % 10 {
            0 => Some("rare"),
            1..4 => Some(LONG),
            _ => None,
        }))
        .into_array()
    }

    fn binary() -> ArrayRef {
        VarBinViewArray::from_iter_bin([b"x", b"y", b"x", b"z", b"x"]).into_array()
    }

    fn decimals() -> ArrayRef {
        DecimalArray::from_option_iter(
            (0..100i64)
                .map(|index| (index % 7 != 0).then_some(if index % 5 == 0 { index } else { 1_25 })),
            DecimalDType::new(10, 2),
        )
        .into_array()
    }

    #[rstest]
    #[case::bool(nullable_bools(), Scalar::bool(true, Nullability::Nullable))]
    #[case::utf8(nullable_strings(), Scalar::utf8(LONG, Nullability::Nullable))]
    #[case::binary(binary(), Scalar::binary(b"x".to_vec(), Nullability::NonNullable))]
    #[case::decimal(
        decimals(),
        Scalar::decimal(
            DecimalValue::I64(1_25),
            DecimalDType::new(10, 2),
            Nullability::Nullable
        )
    )]
    fn encodes_with_majority_value(
        #[case] array: ArrayRef,
        #[case] expected_fill: Scalar,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let encoded = Sparse::encode(&array, None, &mut ctx)?;
        let sparse = encoded
            .as_opt::<Sparse>()
            .vortex_expect("the input has values other than the fill");
        assert_eq!(sparse.fill_scalar(), &expected_fill);
        assert_arrays_eq!(encoded, array, &mut ctx);
        Ok(())
    }

    /// Long strings that share a length and prefix fall in one view group, which the vote finds
    /// before comparing bytes. Without a majority, any valid fill still encodes correctly.
    #[rstest]
    #[case::majority_in_group(55, 30, Some("same-prefix-value-AAAA"))]
    #[case::group_without_majority(40, 35, None)]
    #[case::no_majority_group(20, 20, None)]
    fn long_strings_sharing_a_head(
        #[case] first_percent: usize,
        #[case] second_percent: usize,
        #[case] expected: Option<&str>,
    ) -> VortexResult<()> {
        let array = VarBinViewArray::from_iter_str((0..100).map(|index| {
            if index < first_percent {
                "same-prefix-value-AAAA".to_string()
            } else if index < first_percent + second_percent {
                "same-prefix-value-BBBB".to_string()
            } else {
                format!("other-{index}")
            }
        }))
        .into_array();
        let mut ctx = SESSION.create_execution_ctx();
        let encoded = Sparse::encode(&array, None, &mut ctx)?;
        let sparse = encoded
            .as_opt::<Sparse>()
            .vortex_expect("the input has values other than the fill");
        if let Some(expected) = expected {
            assert_eq!(
                sparse.fill_scalar(),
                &Scalar::utf8(expected, Nullability::NonNullable)
            );
        }
        assert_arrays_eq!(encoded, array, &mut ctx);
        Ok(())
    }

    #[test]
    fn unsupported_dtype_needs_explicit_fill() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let array = StructArray::from_fields(&[("a", binary())])?.into_array();
        assert!(Sparse::encode(&array, None, &mut ctx).is_err());
        Ok(())
    }
}
