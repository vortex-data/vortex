// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::BitBuffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::BoolArray;
use crate::arrays::ConstantArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::VarBinViewArray;
use crate::builders::ArrayBuilder;
use crate::builders::VarBinViewBuilder;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::scalar::Scalar;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::EmptyOptions;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;

/// Unicode string operations with literal, rather than SQL wildcard, search arguments.
#[derive(Clone, Copy)]
pub enum StringFn {
    /// Whole-string Unicode lowercase, including contextual final sigma.
    Lower,
    /// Equality after whole-string Unicode lowercase, without materializing the lowered array.
    LowerEquals,
    /// Whether a UTF-8 min/max interval excludes a lowercased equality target, or null if unknown.
    LowerBoundExcludes,
    /// Conservative minimum of lowercase over a UTF-8 min/max interval, or null if unknown.
    LowerMinBound,
    /// Conservative maximum of lowercase over a UTF-8 min/max interval, or null if unknown.
    LowerMaxBound,
    /// One-based character position, or zero when absent. Overflow produces null.
    Strpos,
    /// The 64-bit position variant for large strings.
    Strpos64,
    /// Whether the input starts with the search string.
    StartsWith,
    /// Whether the input ends with the search string.
    EndsWith,
}

impl ScalarFnVTable for StringFn {
    type Options = EmptyOptions;

    fn id(&self) -> ScalarFnId {
        match self {
            Self::Lower => {
                static ID: CachedId = CachedId::new("vortex.lower");
                *ID
            }
            Self::LowerEquals => {
                static ID: CachedId = CachedId::new("vortex.lower_equals");
                *ID
            }
            Self::LowerBoundExcludes => {
                static ID: CachedId = CachedId::new("vortex.lower_bound_excludes");
                *ID
            }
            Self::LowerMinBound => {
                static ID: CachedId = CachedId::new("vortex.lower_min_bound");
                *ID
            }
            Self::LowerMaxBound => {
                static ID: CachedId = CachedId::new("vortex.lower_max_bound");
                *ID
            }
            Self::Strpos => {
                static ID: CachedId = CachedId::new("vortex.strpos");
                *ID
            }
            Self::Strpos64 => {
                static ID: CachedId = CachedId::new("vortex.strpos64");
                *ID
            }
            Self::StartsWith => {
                static ID: CachedId = CachedId::new("vortex.starts_with");
                *ID
            }
            Self::EndsWith => {
                static ID: CachedId = CachedId::new("vortex.ends_with");
                *ID
            }
        }
    }

    fn serialize(&self, _options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(vec![]))
    }

    fn deserialize(
        &self,
        _metadata: &[u8],
        _session: &VortexSession,
    ) -> VortexResult<Self::Options> {
        Ok(EmptyOptions)
    }

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Exact(match self {
            Self::Lower => 1,
            Self::LowerBoundExcludes => 3,
            _ => 2,
        })
    }

    fn child_name(&self, _options: &Self::Options, child_idx: usize) -> ChildName {
        match child_idx {
            0 => "input".into(),
            1 => "search".into(),
            2 if matches!(self, Self::LowerBoundExcludes) => "target".into(),
            _ => unreachable!("invalid string function child index"),
        }
    }

    fn return_dtype(&self, _options: &Self::Options, args: &[DType]) -> VortexResult<DType> {
        vortex_ensure!(
            args.iter().all(DType::is_utf8),
            "{} requires UTF-8 arguments",
            self.id()
        );
        let nullable = Nullability::from(args.iter().any(DType::is_nullable));
        Ok(match self {
            Self::Lower => DType::Utf8(nullable),
            Self::LowerBoundExcludes => DType::Bool(Nullability::Nullable),
            Self::LowerMinBound | Self::LowerMaxBound => DType::Utf8(Nullability::Nullable),
            Self::Strpos => DType::Primitive(PType::I32, Nullability::Nullable),
            Self::Strpos64 => DType::Primitive(PType::I64, nullable),
            Self::LowerEquals | Self::StartsWith | Self::EndsWith => DType::Bool(nullable),
        })
    }

    fn execute(
        &self,
        options: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let input = args.get(0)?;
        if matches!(self, Self::Lower)
            && let Some(scalar) = input.as_constant()
        {
            let lowered = match scalar.as_utf8().value() {
                Some(value) => Scalar::utf8(value.to_lowercase(), input.dtype().nullability()),
                None => Scalar::null(input.dtype().clone()),
            };
            return Ok(ConstantArray::new(lowered, args.row_count()).into_array());
        }
        if matches!(self, Self::LowerEquals)
            && let Some(value) = input.as_constant()
        {
            let search = args.get(1)?;
            if let Some(search) = search.as_constant() {
                let nullable =
                    Nullability::from(value.dtype().is_nullable() || search.dtype().is_nullable());
                let result = match (value.as_utf8().value(), search.as_utf8().value()) {
                    (Some(value), Some(search)) => {
                        Scalar::bool(value.to_lowercase() == search.as_str(), nullable)
                    }
                    _ => Scalar::null(DType::Bool(nullable)),
                };
                return Ok(ConstantArray::new(result, args.row_count()).into_array());
            }
        }
        let input = input.execute::<VarBinViewArray>(ctx)?;
        let input_validity = input.validity()?.execute_mask(input.len(), ctx)?;
        let input_views = input.resolved_views();
        if matches!(self, Self::Lower) {
            if (0..input.len()).all(|i| {
                !input_validity.value(i)
                    || input_views
                        .bytes(i)
                        .iter()
                        .all(|byte| byte.is_ascii() && !byte.is_ascii_uppercase())
            }) {
                return Ok(input.into_array());
            }
            let values = (0..input.len()).map(|i| {
                input_validity.value(i).then(|| {
                    // Lowercasing a whole string preserves contextual Unicode mappings.
                    std::str::from_utf8(input_views.bytes(i))
                        .ok()
                        .vortex_expect("valid UTF-8 array")
                        .to_lowercase()
                })
            });
            return Ok(VarBinViewArray::from_iter(values, input.dtype().clone()).into_array());
        }

        let search = args.get(1)?;
        let dtypes = [input.dtype().clone(), search.dtype().clone()];
        if matches!(self, Self::LowerEquals)
            && let Some(scalar) = search.as_constant()
        {
            let Some(search) = scalar.as_utf8().value() else {
                return Ok(ConstantArray::new(
                    Scalar::null(DType::Bool(Nullability::Nullable)),
                    input.len(),
                )
                .into_array());
            };
            let bits = BitBuffer::collect_bool_in(
                input.len(),
                |i| {
                    if !input_validity.value(i) {
                        return false;
                    }
                    let value = std::str::from_utf8(input_views.bytes(i))
                        .ok()
                        .vortex_expect("valid UTF-8 array");
                    if value
                        .bytes()
                        .all(|byte| byte.is_ascii() && !byte.is_ascii_uppercase())
                    {
                        value == search.as_str()
                    } else {
                        value.to_lowercase() == search.as_str()
                    }
                },
                ctx.allocator().clone(),
            );
            return Ok(BoolArray::try_new(
                bits,
                input
                    .validity()?
                    .union_nullability(Nullability::from(dtypes.iter().any(DType::is_nullable))),
            )?
            .into_array());
        }
        let search = search.execute::<VarBinViewArray>(ctx)?;
        let search_validity = search.validity()?.execute_mask(search.len(), ctx)?;
        let search_views = search.resolved_views();
        let pairs = (0..input.len()).map(|i| {
            (input_validity.value(i) && search_validity.value(i)).then(|| {
                (
                    std::str::from_utf8(input_views.bytes(i))
                        .ok()
                        .vortex_expect("valid UTF-8 array"),
                    std::str::from_utf8(search_views.bytes(i))
                        .ok()
                        .vortex_expect("valid UTF-8 array"),
                )
            })
        });
        if matches!(self, Self::LowerBoundExcludes) {
            let target = args.get(2)?;
            let constant = target.as_constant();
            let target_array = if constant.is_none() {
                Some(target.clone().execute::<VarBinViewArray>(ctx)?)
            } else {
                None
            };
            let target_views = target_array.as_ref().map(|array| array.resolved_views());
            let target_validity = target_array
                .as_ref()
                .map(|array| array.validity()?.execute_mask(array.len(), ctx))
                .transpose()?;
            let mut bound = String::new();
            return Ok(BoolArray::from_iter(pairs.enumerate().map(|(i, pair)| {
                let (low, high) = pair?;
                let target = match &constant {
                    Some(scalar) => scalar.as_utf8().value()?.as_str(),
                    None => {
                        if !target_validity
                            .as_ref()
                            .vortex_expect("variable target validity")
                            .value(i)
                        {
                            return None;
                        }
                        std::str::from_utf8(
                            target_views
                                .as_ref()
                                .vortex_expect("variable target views")
                                .bytes(i),
                        )
                        .ok()
                        .vortex_expect("valid UTF-8 array")
                    }
                };
                if low == high {
                    return Some(
                        if low
                            .bytes()
                            .all(|byte| byte.is_ascii() && !byte.is_ascii_uppercase())
                        {
                            low != target
                        } else {
                            low.to_lowercase() != target
                        },
                    );
                }
                let below = lower_bound(low, high, false, &mut bound).map(|bound| bound > target);
                if below == Some(true) {
                    return Some(true);
                }
                let above = lower_bound(low, high, true, &mut bound).map(|bound| bound < target);
                if above == Some(true) {
                    return Some(true);
                }
                below.and(above).map(|_| false)
            }))
            .into_array());
        }
        let dtype = self.return_dtype(options, &dtypes)?;
        let result = match self {
            Self::Lower | Self::LowerBoundExcludes => unreachable!(),
            Self::LowerEquals => BoolArray::from_iter(
                pairs.map(|pair| pair.map(|(value, search)| value.to_lowercase() == search)),
            )
            .into_array(),
            Self::LowerMinBound | Self::LowerMaxBound => {
                // A constant interval whose value is already lowercase needs no new views.
                if pairs.clone().enumerate().all(|(i, pair)| match pair {
                    Some((low, high)) => {
                        low == high
                            && low
                                .bytes()
                                .all(|byte| byte.is_ascii() && !byte.is_ascii_uppercase())
                    }
                    None => !input_validity.value(i),
                }) {
                    return input.into_array().cast(dtype);
                }
                let mut builder = VarBinViewBuilder::with_capacity_in(
                    dtype.clone(),
                    input.len(),
                    ctx.allocator().clone(),
                );
                let mut bound = String::new();
                for pair in pairs {
                    match pair.and_then(|(low, high)| {
                        lower_bound(low, high, matches!(self, Self::LowerMaxBound), &mut bound)
                    }) {
                        Some(value) => builder.append_value(value),
                        None => builder.append_null(),
                    }
                }
                builder.finish_into_varbinview().into_array()
            }
            Self::Strpos | Self::Strpos64 => {
                let positions = pairs.into_iter().map(|pair| {
                    pair.map(|(value, search)| {
                        value
                            .find(search)
                            .map_or(0, |offset| value[..offset].chars().count() + 1)
                    })
                });
                if matches!(self, Self::Strpos) {
                    PrimitiveArray::from_option_iter(
                        positions.map(|position| position.and_then(|p| i32::try_from(p).ok())),
                    )
                    .into_array()
                } else {
                    PrimitiveArray::from_option_iter(
                        positions.map(|position| position.and_then(|p| i64::try_from(p).ok())),
                    )
                    .into_array()
                }
            }
            Self::StartsWith => BoolArray::from_iter(
                pairs
                    .into_iter()
                    .map(|pair| pair.map(|(value, search)| value.starts_with(search))),
            )
            .into_array(),
            Self::EndsWith => BoolArray::from_iter(
                pairs
                    .into_iter()
                    .map(|pair| pair.map(|(value, search)| value.ends_with(search))),
            )
            .into_array(),
        };
        result.cast(dtype)
    }

    fn is_strict(&self, _options: &Self::Options) -> bool {
        true
    }

    fn is_infallible(&self, _options: &Self::Options) -> bool {
        true
    }
}

fn lower_bound<'a>(low: &str, high: &str, upper: bool, bound: &'a mut String) -> Option<&'a str> {
    bound.clear();
    if low == high {
        if low.is_ascii() {
            bound.extend(
                low.bytes()
                    .map(|byte| char::from(byte.to_ascii_lowercase())),
            );
        } else {
            bound.push_str(&low.to_lowercase());
        }
        return Some(bound);
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
    // Every value in the interval shares this ASCII prefix. Its lowercase is independent
    // of any suffix, unlike Unicode contextual mappings such as Greek final sigma.
    bound.reserve(prefix.len() + 2);
    bound.extend(
        prefix
            .iter()
            .map(|byte| char::from(byte.to_ascii_lowercase())),
    );
    match (low.as_bytes().get(common), high.as_bytes().get(common)) {
        (Some(&low), Some(&high)) if low.is_ascii() && high.is_ascii() => {
            // Bound all possible first differing characters, not just the endpoints:
            // ASCII lowercase is not monotonic across ranges such as 'Z'..='a'.
            let letters = (low..=high).map(|byte| byte.to_ascii_lowercase());
            let byte = if upper {
                letters.max()? + 1
            } else {
                letters.min()?
            };
            bound.push(char::from(byte));
        }
        (None, Some(_)) => {
            if upper {
                // The successor prefix is above every possible suffix, even Unicode ones.
                let last = bound.pop()?;
                bound.push(char::from_u32(u32::from(last) + 1)?);
            }
        }
        _ => return None,
    }
    Some(bound)
}

#[cfg(test)]
mod tests {
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use super::StringFn;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::arrays::Constant;
    use crate::arrays::ConstantArray;
    use crate::arrays::DictArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::RunEnd;
    use crate::arrays::VarBinViewArray;
    use crate::assert_arrays_eq;
    use crate::expr::Expression;
    use crate::expr::eq;
    use crate::expr::lit;
    use crate::expr::proto::ExprSerializeProtoExt;
    use crate::expr::root;
    use crate::scalar::Scalar;
    use crate::scalar_fn::EmptyOptions;
    use crate::scalar_fn::ScalarFnVTableExt;
    use crate::scalar_fn::session::ScalarFnSession;

    #[test]
    fn unicode_lower_roundtrip() -> VortexResult<()> {
        let session = VortexSession::empty().with::<ScalarFnSession>();
        let expression = StringFn::Lower.try_new_expr(EmptyOptions, [root()])?;
        let expression = Expression::from_proto(&expression.serialize_proto()?, &session)?;
        let input = VarBinViewArray::from_iter_nullable_str([
            Some("GrEaT-LaKeS"),
            Some("ΟΣ"),
            Some("ΟΣΑ"),
            Some("İȾ"),
            None,
            Some(""),
        ])
        .into_array();
        assert_arrays_eq!(
            input.apply(&expression)?,
            VarBinViewArray::from_iter_nullable_str([
                Some("great-lakes"),
                Some("ος"),
                Some("οσα"),
                Some("i\u{307}ⱦ"),
                None,
                Some("")
            ]),
            &mut session.create_execution_ctx()
        );
        assert_arrays_eq!(
            VarBinViewArray::from_iter_nullable_str([Some("great-lakes"), None, Some("ΟΣ")])
                .into_array()
                .apply(&expression)?,
            VarBinViewArray::from_iter_nullable_str([Some("great-lakes"), None, Some("ος")]),
            &mut session.create_execution_ctx()
        );
        let unchanged =
            VarBinViewArray::from_iter_nullable_str([Some("great-lakes"), None, Some("")])
                .into_array();
        assert_arrays_eq!(
            unchanged.clone().apply(&expression)?,
            unchanged,
            &mut session.create_execution_ctx()
        );
        Ok(())
    }

    #[test]
    fn unicode_lower_preserves_runs() -> VortexResult<()> {
        let session = VortexSession::empty();
        let mut ctx = session.create_execution_ctx();
        let values = DictArray::try_new(
            PrimitiveArray::from_iter([0u8, 1, 2, 0]).into_array(),
            VarBinViewArray::from_iter_nullable_str([Some("ΟΣ"), None, Some("İȾ")]).into_array(),
        )?
        .into_array();
        let input = RunEnd::try_new_offset_length(
            PrimitiveArray::from_iter([3u32, 5, 7, 10]).into_array(),
            values,
            1,
            8,
            &mut ctx,
        )?
        .into_array();
        let expression = StringFn::Lower.try_new_expr(EmptyOptions, [root()])?;
        let result = input
            .apply(&expression)?
            .execute_until::<RunEnd>(&mut ctx)?;
        assert!(result.is::<RunEnd>());
        assert_arrays_eq!(
            result,
            VarBinViewArray::from_iter_nullable_str([
                Some("ος"),
                Some("ος"),
                None,
                None,
                Some("i\u{307}ⱦ"),
                Some("i\u{307}ⱦ"),
                Some("ος"),
                Some("ος"),
            ]),
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn unicode_lower_preserves_constants() -> VortexResult<()> {
        let mut ctx = VortexSession::empty().create_execution_ctx();
        let expression = StringFn::Lower.try_new_expr(EmptyOptions, [root()])?;
        for (input, expected) in [(Some("ΟΣ"), Some("ος")), (None, None)] {
            let result = ConstantArray::new(Scalar::from(input), 1024)
                .into_array()
                .apply(&expression)?
                .execute_until::<Constant>(&mut ctx)?;
            assert!(result.is::<Constant>());
            assert_arrays_eq!(
                result,
                ConstantArray::new(Scalar::from(expected), 1024),
                &mut ctx
            );
            let equality = ConstantArray::new(Scalar::from(input), 1024)
                .into_array()
                .apply(&eq(expression.clone(), lit("ος")))?
                .execute_until::<Constant>(&mut ctx)?;
            assert!(equality.is::<Constant>());
            assert_arrays_eq!(
                equality,
                ConstantArray::new(Scalar::from(expected.map(|_| true)), 1024),
                &mut ctx
            );
        }
        Ok(())
    }
}
