// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::BoolArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::VarBinViewArray;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
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
        Arity::Exact(if matches!(self, Self::Lower) { 1 } else { 2 })
    }

    fn child_name(&self, _options: &Self::Options, child_idx: usize) -> ChildName {
        match child_idx {
            0 => "input".into(),
            1 => "search".into(),
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
            Self::Strpos => DType::Primitive(PType::I32, Nullability::Nullable),
            Self::Strpos64 => DType::Primitive(PType::I64, nullable),
            Self::StartsWith | Self::EndsWith => DType::Bool(nullable),
        })
    }

    fn execute(
        &self,
        options: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let input = args.get(0)?;
        let mut dtypes = vec![input.dtype().clone()];
        let input = input.execute::<VarBinViewArray>(ctx)?;
        let input_validity = input.validity()?.execute_mask(input.len(), ctx)?;
        let input_views = input.resolved_views();
        if matches!(self, Self::Lower) {
            let values = (0..input.len()).map(|i| {
                input_validity.value(i).then(|| {
                    // Lowercasing a whole string preserves contextual Unicode mappings.
                    std::str::from_utf8(input_views.bytes(i))
                        .ok()
                        .vortex_expect("valid UTF-8 array")
                        .to_lowercase()
                })
            });
            return Ok(
                VarBinViewArray::from_iter(values, self.return_dtype(options, &dtypes)?)
                    .into_array(),
            );
        }

        let search = args.get(1)?;
        dtypes.push(search.dtype().clone());
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
        let dtype = self.return_dtype(options, &dtypes)?;
        let result = match self {
            Self::Lower => unreachable!(),
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

#[cfg(test)]
mod tests {
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use super::StringFn;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::arrays::VarBinViewArray;
    use crate::assert_arrays_eq;
    use crate::expr::Expression;
    use crate::expr::proto::ExprSerializeProtoExt;
    use crate::expr::root;
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
        Ok(())
    }
}
