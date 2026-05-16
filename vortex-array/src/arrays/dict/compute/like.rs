// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use super::Dict;
use super::DictArray;
use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::ConstantArray;
use crate::arrays::dict::DictArrayExt;
use crate::arrays::dict::DictArraySlotsExt;
use crate::arrays::dict::compute::between::reduce_sorted_between;
use crate::arrays::scalar_fn::ScalarFnFactoryExt;
use crate::dtype::DType;
use crate::optimizer::ArrayOptimizer;
use crate::scalar::Scalar;
use crate::scalar_fn::fns::between::BetweenOptions;
use crate::scalar_fn::fns::between::StrictComparison;
use crate::scalar_fn::fns::like::Like;
use crate::scalar_fn::fns::like::LikeOptions;
use crate::scalar_fn::fns::like::LikeReduce;

impl LikeReduce for Dict {
    fn like(
        array: ArrayView<'_, Dict>,
        pattern: &ArrayRef,
        options: LikeOptions,
    ) -> VortexResult<Option<ArrayRef>> {
        // Sorted-dict prefix fast path: a `prefix%` constant pattern (no other metachars,
        // not negated, not case-insensitive) is equivalent to `value BETWEEN prefix AND
        // next_after(prefix)` over the dict. Hand it to the sorted-between reduce so it
        // collapses to a codes-domain compare or a Mask constant.
        if !options.negated
            && !options.case_insensitive
            && array.has_sorted_values()
            && !array.values().dtype().is_nullable()
            && let Some(pattern_const) = pattern.as_constant()
            && let Some(prefix) = like_prefix(&pattern_const)
            && let Some(upper) = next_after(prefix)
        {
            let lower_scalar = make_string_scalar(array.values().dtype(), prefix);
            let upper_scalar = make_string_scalar(array.values().dtype(), &upper);
            let opts = BetweenOptions {
                lower_strict: StrictComparison::NonStrict,
                upper_strict: StrictComparison::Strict,
            };
            if let Some(rewrite) =
                reduce_sorted_between(array, &lower_scalar, &upper_scalar, &opts)?
            {
                return Ok(Some(rewrite));
            }
        }

        // Existing fallback: if we have more values than codes, it is faster to canonicalize
        // first.
        if array.values().len() > array.codes().len() {
            return Ok(None);
        }
        if let Some(pattern) = pattern.as_constant() {
            let pattern = ConstantArray::new(pattern, array.values().len()).into_array();

            let values = Like
                .try_new_array(pattern.len(), options, [array.values().clone(), pattern])?
                .optimize()?;

            // SAFETY: LIKE preserves the len of the values, so codes are still pointing at
            //  valid positions.
            // Preserve all_values_referenced since codes are unchanged.
            unsafe {
                Ok(Some(
                    DictArray::new_unchecked(array.codes().clone(), values)
                        .set_all_values_referenced(array.has_all_values_referenced())
                        .into_array(),
                ))
            }
        } else {
            Ok(None)
        }
    }
}

/// Extract the literal prefix from a `prefix%` LIKE pattern. Returns `None` if the
/// pattern is not a constant string, has any other `%` or `_`, contains escapes, or
/// doesn't end in `%`.
fn like_prefix(pattern: &Scalar) -> Option<&[u8]> {
    let bytes = match pattern.dtype() {
        DType::Utf8(_) => pattern.as_utf8_opt()?.value()?.as_bytes(),
        DType::Binary(_) => pattern.as_binary_opt()?.value()?.as_slice(),
        _ => return None,
    };
    if !bytes.ends_with(b"%") {
        return None;
    }
    let prefix = &bytes[..bytes.len() - 1];
    // Reject patterns with embedded wildcards or escapes — we'd have to fall back to a
    // post-filter on the candidates, which we don't bother with here.
    if prefix
        .iter()
        .any(|&b| b == b'%' || b == b'_' || b == b'\\')
    {
        return None;
    }
    Some(prefix)
}

/// Compute the smallest byte sequence strictly greater than `prefix` such that any string
/// starting with `prefix` is `< next_after(prefix)`. Returns `None` if `prefix` is all
/// `0xFF` bytes (no representable upper bound — caller must fall back).
fn next_after(prefix: &[u8]) -> Option<Vec<u8>> {
    let mut out = prefix.to_vec();
    while let Some(last) = out.last_mut() {
        if *last < 0xFF {
            *last += 1;
            return Some(out);
        }
        out.pop();
    }
    None
}

fn make_string_scalar(dtype: &DType, bytes: &[u8]) -> Scalar {
    match dtype {
        DType::Utf8(n) => Scalar::utf8(
            std::str::from_utf8(bytes).unwrap_or_default(),
            *n,
        ),
        DType::Binary(n) => {
            let buf = vortex_buffer::ByteBuffer::from(bytes.to_vec());
            Scalar::binary(buf, *n)
        }
        _ => vortex_error::vortex_panic!("expected Utf8/Binary dtype, got {dtype}"),
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use crate::IntoArray;
    use crate::arrays::BoolArray;
    use crate::arrays::DictArray;
    use crate::arrays::VarBinArray;
    use crate::arrays::dict::compute::like::ConstantArray;
    use crate::arrays::scalar_fn::ScalarFnFactoryExt;
    use crate::assert_arrays_eq;
    use crate::optimizer::ArrayOptimizer;
    use crate::scalar_fn::fns::like::Like;
    use crate::scalar_fn::fns::like::LikeOptions;

    #[test]
    fn like_reduce_dict() -> VortexResult<()> {
        let dict = DictArray::try_new(
            buffer![0u8, 1, 0, 2].into_array(),
            VarBinArray::from(vec!["hello", "world", "help"]).into_array(),
        )?
        .into_array();

        let pattern = ConstantArray::new("hello%", 4).into_array();
        let result = Like
            .try_new_array(4, LikeOptions::default(), [dict, pattern])?
            .optimize()?;

        assert_arrays_eq!(result, BoolArray::from_iter([true, false, true, false]));
        Ok(())
    }

    #[test]
    fn sorted_dict_like_prefix_string() -> VortexResult<()> {
        use crate::Canonical;
        use crate::LEGACY_SESSION;
        use crate::VortexSessionExecute;
        use crate::builders::dict::dict_encode_sorted;
        use crate::dtype::DType;
        use crate::dtype::Nullability;

        let arr = VarBinArray::from_iter(
            [
                Some("apple"),
                Some("banana"),
                Some("apricot"),
                Some("avocado"),
                Some("blueberry"),
                Some("apple"),
            ],
            DType::Utf8(Nullability::NonNullable),
        )
        .into_array();
        let dict = dict_encode_sorted(&arr)?.into_array();
        let pattern = ConstantArray::new("ap%", dict.len()).into_array();
        let result = Like
            .try_new_array(dict.len(), LikeOptions::default(), [dict, pattern])?
            .optimize()?;

        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let canon = result.execute::<Canonical>(&mut ctx)?.into_array();
        // apple, banana, apricot, avocado, blueberry, apple
        // ap%   :   T     F        T        F          F      T
        assert_arrays_eq!(
            canon,
            BoolArray::from_iter([true, false, true, false, false, true])
        );
        Ok(())
    }

    #[test]
    fn sorted_dict_like_no_match() -> VortexResult<()> {
        use crate::Canonical;
        use crate::LEGACY_SESSION;
        use crate::VortexSessionExecute;
        use crate::builders::dict::dict_encode_sorted;
        use crate::dtype::DType;
        use crate::dtype::Nullability;

        let arr = VarBinArray::from_iter(
            [Some("apple"), Some("banana"), Some("cherry")],
            DType::Utf8(Nullability::NonNullable),
        )
        .into_array();
        let dict = dict_encode_sorted(&arr)?.into_array();
        let pattern = ConstantArray::new("xyz%", dict.len()).into_array();
        let result = Like
            .try_new_array(dict.len(), LikeOptions::default(), [dict, pattern])?
            .optimize()?;
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let canon = result.execute::<Canonical>(&mut ctx)?.into_array();
        assert_arrays_eq!(canon, BoolArray::from_iter([false, false, false]));
        Ok(())
    }
}
