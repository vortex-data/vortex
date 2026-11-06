// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The `text_match(text, query)` predicate that the full-text index answers.

use std::fmt::Formatter;

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::expr::Expression;
use vortex_array::expr::display::ExprDisplay;
use vortex_array::expr::lit;
use vortex_array::scalar_fn::Arity;
use vortex_array::scalar_fn::ChildName;
use vortex_array::scalar_fn::EmptyOptions;
use vortex_array::scalar_fn::ExecutionArgs;
use vortex_array::scalar_fn::ScalarFnId;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_array::scalar_fn::ScalarFnVTableExt;
use vortex_buffer::BitBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure_eq;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;
use vortex_utils::aliases::hash_set::HashSet;

use crate::tokenize::tokenize;

/// `text_match(text, query)`: true when every token of `query` is also a token of `text`.
///
/// Tokens come from [`tokenize`]: lowercase runs of alphanumeric characters. A query without
/// tokens matches every non-null row. A null text or query gives null.
#[derive(Clone, Debug)]
pub struct TextMatch;

/// Build a `text_match(text, query)` expression with a literal query.
pub fn text_match(text: Expression, query: &str) -> Expression {
    TextMatch.new_expr(EmptyOptions, [text, lit(query)])
}

/// Whether `text` holds every token in `query_tokens`.
fn matches(text: &str, query_tokens: &HashSet<String>) -> bool {
    if query_tokens.is_empty() {
        return true;
    }
    let text_tokens: HashSet<String> = tokenize(text).collect();
    query_tokens.iter().all(|token| text_tokens.contains(token))
}

impl ScalarFnVTable for TextMatch {
    type Options = EmptyOptions;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.index.text_match");
        *ID
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
        Arity::Exact(2)
    }

    fn child_name(&self, _options: &Self::Options, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("text"),
            _ => ChildName::from("query"),
        }
    }

    fn fmt_sql(
        &self,
        _options: &Self::Options,
        expr: &dyn ExprDisplay,
        f: &mut Formatter<'_>,
    ) -> std::fmt::Result {
        write!(
            f,
            "text_match({}, {})",
            expr.display_child(0),
            expr.display_child(1)
        )
    }

    fn return_dtype(&self, _options: &Self::Options, args: &[DType]) -> VortexResult<DType> {
        if !args[0].is_utf8() || !args[1].is_utf8() {
            vortex_bail!(
                "text_match requires utf8 arguments, got {} and {}",
                args[0],
                args[1]
            );
        }
        Ok(DType::Bool(Nullability::from(
            args[0].is_nullable() || args[1].is_nullable(),
        )))
    }

    fn execute(
        &self,
        _options: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let text = args.get(0)?;
        let query = args.get(1)?;
        vortex_ensure_eq!(text.len(), query.len());
        let nullability =
            Nullability::from(text.dtype().is_nullable() || query.dtype().is_nullable());

        let texts = text.execute::<VarBinViewArray>(ctx)?;
        let validity = texts.validity()?.union_nullability(nullability);

        let bits: BitBuffer = if let Some(constant) = query.as_constant() {
            let Some(query) = constant.as_utf8().value() else {
                vortex_bail!("text_match does not support a null query literal")
            };
            let query_tokens: HashSet<String> = tokenize(query.as_str()).collect();
            (0..texts.len())
                .map(|idx| {
                    let bytes = texts.bytes_at(idx);
                    matches(&String::from_utf8_lossy(bytes.as_slice()), &query_tokens)
                })
                .collect()
        } else {
            let queries = query.execute::<VarBinViewArray>(ctx)?;
            (0..texts.len())
                .map(|idx| {
                    let query = queries.bytes_at(idx);
                    let query_tokens: HashSet<String> =
                        tokenize(&String::from_utf8_lossy(query.as_slice())).collect();
                    let bytes = texts.bytes_at(idx);
                    matches(&String::from_utf8_lossy(bytes.as_slice()), &query_tokens)
                })
                .collect()
        };
        Ok(BoolArray::new(bits, validity).into_array())
    }

    fn is_strict(&self, _options: &Self::Options) -> bool {
        true
    }

    fn is_infallible(&self, _options: &Self::Options) -> bool {
        true
    }
}
