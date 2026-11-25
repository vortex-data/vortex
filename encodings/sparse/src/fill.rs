// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Choosing the fill value of a sparse array: the most common valid value of the input.

use std::cmp::Reverse;
use std::hash::Hash;

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::DecimalArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::bool::BoolArrayExt;
use vortex_array::dtype::DType;
use vortex_array::match_each_decimal_value_type;
use vortex_array::scalar::Scalar;
use vortex_array::stats::majority::MajorityKey;
use vortex_array::stats::majority::sampled_majority;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_mask::AllOr;
use vortex_mask::Mask;
use vortex_utils::aliases::hash_map::HashMap;

/// Returns the most common valid value of `array`, whose validity is `mask`.
///
/// At least one value must be valid.
pub(crate) fn most_common_value(
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
                match sampled_majority(&values, mask) {
                    Some((top, _)) => first_valid_index_of(&values, mask, top),
                    None => top_index(mask, |index| values[index]),
                }
            });
            decimal.into_array().execute_scalar(index, ctx)
        }
        DType::Utf8(_) | DType::Binary(_) => {
            let bytes = array.clone().execute::<VarBinViewArray>(ctx)?;
            let index = top_index(mask, |index| bytes.bytes_at(index));
            bytes.into_array().execute_scalar(index, ctx)
        }
        dtype => vortex_bail!(
            "cannot choose a fill value for a {dtype} array, so one must be given explicitly"
        ),
    }
}

/// Returns the first valid index holding `value`, which must occur at one.
fn first_valid_index_of<T: MajorityKey>(values: &[T], mask: &Mask, value: T) -> usize {
    values
        .iter()
        .enumerate()
        .position(|(index, &candidate)| candidate.key_eq(value) && mask.value(index))
        .vortex_expect("the majority occurs at a valid index")
}

/// Returns the first index of the most common key among the valid indices.
///
/// This is the scalar form of the vote in [`vortex_array::stats::majority`], for keys that do not
/// fit in a vector lane, such as byte strings: a Boyer–Moore vote finds a value filling more than
/// half of the valid indices with comparisons alone, which is the common case for a sparse array.
/// Without one, the keys are counted in a hash map.
fn top_index<K: Eq + Hash>(mask: &Mask, key: impl Fn(usize) -> K) -> usize {
    let valid_indices = || -> Box<dyn Iterator<Item = usize> + '_> {
        match mask.indices() {
            AllOr::All => Box::new(0..mask.len()),
            AllOr::None => unreachable!("array has a valid value"),
            AllOr::Some(indices) => Box::new(indices.iter().copied()),
        }
    };

    let mut candidate = None;
    let mut surplus = 0usize;
    for index in valid_indices() {
        let value = key(index);
        match &candidate {
            Some((_, current)) if *current == value => surplus += 1,
            _ if surplus == 0 => {
                candidate = Some((index, value));
                surplus = 1;
            }
            _ => surplus -= 1,
        }
    }
    let (_, candidate_key) = candidate.vortex_expect("array has a valid value");
    let mut first = None;
    let mut count = 0;
    for index in valid_indices() {
        if key(index) == candidate_key {
            first.get_or_insert(index);
            count += 1;
        }
    }
    if count > mask.true_count() / 2 {
        return first.vortex_expect("the candidate occurs");
    }

    let mut counts: HashMap<K, (usize, usize)> = HashMap::default();
    for index in valid_indices() {
        counts.entry(key(index)).or_insert((index, 0)).1 += 1;
    }
    counts
        .into_values()
        .max_by_key(|&(first, count)| (count, Reverse(first)))
        .map(|(first, _)| first)
        .vortex_expect("array has a valid value")
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

    /// No value fills more than half of the rows, so the fill comes from counting.
    fn plurality_binary() -> ArrayRef {
        VarBinViewArray::from_iter_bin([b"x", b"y", b"z", b"x", b"y", b"x"]).into_array()
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
    #[case::binary_plurality(plurality_binary(), Scalar::binary(b"x".to_vec(), Nullability::NonNullable))]
    #[case::decimal(
        decimals(),
        Scalar::decimal(
            DecimalValue::I64(1_25),
            DecimalDType::new(10, 2),
            Nullability::Nullable
        )
    )]
    fn encodes_with_most_common_value(
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

    #[test]
    fn unsupported_dtype_needs_explicit_fill() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let array = StructArray::from_fields(&[("a", plurality_binary())])?.into_array();
        assert!(Sparse::encode(&array, None, &mut ctx).is_err());
        Ok(())
    }
}
