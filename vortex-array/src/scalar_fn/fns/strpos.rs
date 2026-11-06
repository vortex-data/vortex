// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use memchr::memmem;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::VarBinViewArray;
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

/// One-based character position of the first occurrence of `search` in `input`, or zero when
/// absent, matching SQL `strpos`. An empty `search` is found at position one.
#[derive(Clone)]
pub struct Strpos;

impl ScalarFnVTable for Strpos {
    type Options = EmptyOptions;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.strpos");
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
            0 => ChildName::from("input"),
            1 => ChildName::from("search"),
            _ => unreachable!("Invalid child index {child_idx} for strpos()"),
        }
    }

    fn return_dtype(&self, _options: &Self::Options, arg_dtypes: &[DType]) -> VortexResult<DType> {
        let (input, search) = (&arg_dtypes[0], &arg_dtypes[1]);
        if !input.is_utf8() || !search.is_utf8() {
            vortex_bail!("strpos() requires Utf8 arguments, got {input} and {search}");
        }
        Ok(DType::Primitive(
            PType::I64,
            (input.is_nullable() || search.is_nullable()).into(),
        ))
    }

    fn execute(
        &self,
        _options: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let input = args.get(0)?;
        let search = args.get(1)?;
        let nullability =
            Nullability::from(input.dtype().is_nullable() || search.dtype().is_nullable());
        let input = input.execute::<VarBinViewArray>(ctx)?;
        let haystack = input.resolved_views();

        if let Some(search) = search.as_constant() {
            let Some(needle) = search.as_utf8().value() else {
                return Ok(ConstantArray::new(
                    Scalar::null(DType::Primitive(PType::I64, Nullability::Nullable)),
                    args.row_count(),
                )
                .into_array());
            };
            let finder = memmem::Finder::new(needle.as_bytes());
            let positions: Buffer<i64> = (0..haystack.len())
                .map(|i| position(haystack.bytes(i), finder.find(haystack.bytes(i))))
                .collect();
            let validity = input.validity()?.union_nullability(nullability);
            return Ok(PrimitiveArray::new(positions, validity).into_array());
        }

        let search = search.execute::<VarBinViewArray>(ctx)?;
        let needles = search.resolved_views();
        let positions: Buffer<i64> = (0..haystack.len())
            .map(|i| {
                let bytes = haystack.bytes(i);
                position(bytes, memmem::find(bytes, needles.bytes(i)))
            })
            .collect();
        let validity = input
            .validity()?
            .and(search.validity()?)?
            .union_nullability(nullability);
        Ok(PrimitiveArray::new(positions, validity).into_array())
    }

    fn is_strict(&self, _options: &Self::Options) -> bool {
        true
    }

    fn is_infallible(&self, _options: &Self::Options) -> bool {
        true
    }
}

/// Converts a byte offset into a one-based character position, or zero when not found.
fn position(haystack: &[u8], offset: Option<usize>) -> i64 {
    let Some(offset) = offset else {
        return 0;
    };
    // Utf8 arrays hold valid UTF-8, so each character starts with exactly one byte that is not
    // a continuation byte (`0b10xx_xxxx`).
    let chars = haystack[..offset]
        .iter()
        .filter(|&&byte| (byte as i8) >= -0x40)
        .count();
    // A Vortex string length is at most `u32::MAX` bytes, so the position fits in an `i64`.
    chars as i64 + 1
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::StructArray;
    use crate::arrays::VarBinViewArray;
    use crate::assert_arrays_eq;
    use crate::expr::Expression;
    use crate::expr::get_item;
    use crate::expr::lit;
    use crate::expr::proto::ExprSerializeProtoExt;
    use crate::expr::root;
    use crate::expr::strpos;
    use crate::scalar_fn::session::ScalarFnSession;

    static SESSION: LazyLock<VortexSession> =
        LazyLock::new(|| VortexSession::empty().with::<ScalarFnSession>());

    #[rstest]
    #[case::ascii("xxhello", "hello", 3)]
    #[case::after_multibyte("é📊h", "h", 3)]
    #[case::absent("abc", "x", 0)]
    #[case::empty_needle("abc", "", 1)]
    #[case::empty_haystack("", "", 1)]
    fn constant_needle(
        #[case] haystack: &str,
        #[case] needle: &str,
        #[case] expected: i64,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let input = VarBinViewArray::from_iter_nullable_str([Some(haystack), None]).into_array();
        assert_arrays_eq!(
            input.apply(&strpos(root(), lit(needle)))?,
            PrimitiveArray::from_option_iter([Some(expected), None]),
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn column_needle() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let input = StructArray::from_fields(&[
            (
                "s",
                VarBinViewArray::from_iter_nullable_str([
                    Some("é📊hello"),
                    Some("abc"),
                    None,
                    Some("abc"),
                ])
                .into_array(),
            ),
            (
                "needle",
                VarBinViewArray::from_iter_nullable_str([Some("h"), Some("x"), Some("a"), None])
                    .into_array(),
            ),
        ])?
        .into_array();
        assert_arrays_eq!(
            input.apply(&strpos(get_item("s", root()), get_item("needle", root())))?,
            PrimitiveArray::from_option_iter([Some(3i64), Some(0), None, None]),
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn roundtrips_through_proto() -> VortexResult<()> {
        let expr = strpos(root(), lit("x"));
        assert_eq!(
            Expression::from_proto(&expr.serialize_proto()?, &SESSION)?,
            expr
        );
        Ok(())
    }
}
