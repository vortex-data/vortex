// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_session::registry::CachedId;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::arrays::VarBinViewArray;
use crate::builders::ArrayBuilder;
use crate::builders::VarBinViewBuilder;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::scalar::Scalar;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::fns::lower::is_lowercase_ascii;

/// Which side of the [`LowerBound`] interval to compute.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LowerBoundKind {
    Min,
    Max,
}

impl Display for LowerBoundKind {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Min => write!(f, "min"),
            Self::Max => write!(f, "max"),
        }
    }
}

/// Bound on `lower(x)` for every `x` in the byte-ordered interval `[min, max]`, or null when no
/// useful bound is known.
///
/// Stats pruning substitutes this for the min/max of a [`Lower`] expression. Lowercasing is not
/// monotonic, even on ASCII (`'Z' < 'a'` but `'z' > 'a'`), so the bound is derived from the
/// prefix every value in the interval must share.
///
/// [`Lower`]: crate::scalar_fn::fns::lower::Lower
#[derive(Clone)]
pub struct LowerBound;

impl ScalarFnVTable for LowerBound {
    type Options = LowerBoundKind;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.lower_bound");
        *ID
    }

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Exact(2)
    }

    fn child_name(&self, _options: &Self::Options, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("min"),
            1 => ChildName::from("max"),
            _ => unreachable!("Invalid child index {child_idx} for lower_bound()"),
        }
    }

    fn return_dtype(&self, _options: &Self::Options, arg_dtypes: &[DType]) -> VortexResult<DType> {
        let (min, max) = (&arg_dtypes[0], &arg_dtypes[1]);
        if !min.is_utf8() || !max.is_utf8() {
            vortex_bail!("lower_bound() requires Utf8 arguments, got {min} and {max}");
        }
        Ok(DType::Utf8(Nullability::Nullable))
    }

    fn execute(
        &self,
        kind: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let dtype = DType::Utf8(Nullability::Nullable);
        let (min, max) = (args.get(0)?, args.get(1)?);
        if let (Some(min), Some(max)) = (min.as_constant(), max.as_constant()) {
            let mut scratch = String::new();
            let value = match (min.as_utf8().value(), max.as_utf8().value()) {
                (Some(min), Some(max)) => bound(min.as_str(), max.as_str(), *kind, &mut scratch)
                    .map(|value| Scalar::utf8(value, Nullability::Nullable)),
                _ => None,
            };
            let value = value.unwrap_or_else(|| Scalar::null(dtype));
            return Ok(ConstantArray::new(value, args.row_count()).into_array());
        }

        let min = min.execute::<VarBinViewArray>(ctx)?;
        let max = max.execute::<VarBinViewArray>(ctx)?;
        let min_valid = min.validity()?.execute_mask(min.len(), ctx)?;
        let max_valid = max.validity()?.execute_mask(max.len(), ctx)?;
        let (min_views, max_views) = (min.resolved_views(), max.resolved_views());
        let mut builder =
            VarBinViewBuilder::with_capacity_in(dtype, min.len(), ctx.allocator().clone());
        let mut scratch = String::new();
        for i in 0..min.len() {
            if !min_valid.value(i) || !max_valid.value(i) {
                builder.append_null();
                continue;
            }
            let low = std::str::from_utf8(min_views.bytes(i))
                .map_err(|e| vortex_err!("lower_bound() min is not valid UTF-8: {e}"))?;
            let high = std::str::from_utf8(max_views.bytes(i))
                .map_err(|e| vortex_err!("lower_bound() max is not valid UTF-8: {e}"))?;
            match bound(low, high, *kind, &mut scratch) {
                Some(value) => builder.append_value(value),
                None => builder.append_null(),
            }
        }
        Ok(builder.finish_into_varbinview().into_array())
    }

    fn is_strict(&self, _options: &Self::Options) -> bool {
        true
    }

    fn is_infallible(&self, _options: &Self::Options) -> bool {
        true
    }
}

/// Computes the [`LowerBound`] of `[low, high]` into `scratch`.
fn bound<'a>(
    low: &str,
    high: &str,
    kind: LowerBoundKind,
    scratch: &'a mut String,
) -> Option<&'a str> {
    scratch.clear();
    if low == high {
        if is_lowercase_ascii(low.as_bytes()) {
            scratch.push_str(low);
        } else {
            scratch.push_str(&low.to_lowercase());
        }
        return Some(scratch);
    }

    let common = low
        .bytes()
        .zip(high.bytes())
        .take_while(|(low, high)| low == high)
        .count();
    let prefix = &low.as_bytes()[..common];
    if !prefix.is_ascii() {
        return None;
    }
    // Every value in the interval starts with this ASCII prefix. ASCII lowercase does not
    // depend on the characters that follow, unlike Unicode mappings such as the final sigma.
    scratch.extend(
        prefix
            .iter()
            .map(|&byte| char::from(byte.to_ascii_lowercase())),
    );
    match (low.as_bytes().get(common), high.as_bytes().get(common)) {
        // Every value's next byte lies in `low..=high`, and all of them are ASCII. Bound the
        // lowercase of every byte in that range, not just the endpoints, because ASCII lowercase
        // is not monotonic across ranges such as `'Z'..='a'`.
        (Some(&low), Some(&high)) if high.is_ascii() => {
            let lowered = (low..=high).map(|byte| byte.to_ascii_lowercase());
            let byte = match kind {
                LowerBoundKind::Min => lowered.min()?,
                LowerBoundKind::Max => lowered.max()? + 1,
            };
            scratch.push(char::from(byte));
        }
        // `low` is the prefix itself, so every value extends it.
        (None, Some(_)) => {
            if kind == LowerBoundKind::Max {
                // The successor of the prefix exceeds every extension of it.
                let last = scratch.pop()?;
                scratch.push(char::from_u32(u32::from(last) + 1)?);
            }
        }
        _ => return None,
    }
    Some(scratch)
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::LowerBoundKind;
    use super::bound;

    #[rstest]
    #[case::equal_ascii("ABC", "ABC", Some("abc"), Some("abc"))]
    #[case::equal_unicode("ΟΣ", "ΟΣ", Some("ος"), Some("ος"))]
    #[case::shared_prefix(
        "Namespace-000",
        "Namespace-001",
        Some("namespace-000"),
        Some("namespace-002")
    )]
    #[case::non_monotonic_range("Z", "a", Some("["), Some("{"))]
    #[case::uppercase_range("A", "Z", Some("a"), Some("{"))]
    #[case::descending_lowercase("K", "z", Some("["), Some("{"))]
    #[case::prefix_of_max("prefix", "prefix-long", Some("prefix"), Some("prefiy"))]
    #[case::empty_prefix("", "abc", Some(""), None)]
    #[case::delete_successor("a\u{7f}", "a\u{7f}b", Some("a\u{7f}"), Some("a\u{80}"))]
    #[case::max_ascii_byte("a\u{7e}", "a\u{7f}", Some("a\u{7e}"), Some("a\u{80}"))]
    #[case::non_ascii_prefix("éa", "éb", None, None)]
    #[case::non_ascii_divergence("a", "é", None, None)]
    fn bounds(
        #[case] low: &str,
        #[case] high: &str,
        #[case] min: Option<&str>,
        #[case] max: Option<&str>,
    ) {
        let mut scratch = String::new();
        assert_eq!(
            bound(low, high, LowerBoundKind::Min, &mut scratch).map(str::to_owned),
            min.map(str::to_owned)
        );
        assert_eq!(
            bound(low, high, LowerBoundKind::Max, &mut scratch).map(str::to_owned),
            max.map(str::to_owned)
        );
    }
}
