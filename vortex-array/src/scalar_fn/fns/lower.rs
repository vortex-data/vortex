// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::arrays::VarBinViewArray;
use crate::builders::ArrayBuilder;
use crate::builders::VarBinViewBuilder;
use crate::dtype::DType;
use crate::scalar::Scalar;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::EmptyOptions;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;

/// Unicode lowercase of each element of a Utf8 array.
///
/// Each string is lowercased as a whole, so context-dependent mappings such as the Greek final
/// sigma (`ΟΣ` → `ος`) match SQL `lower`.
#[derive(Clone)]
pub struct Lower;

impl ScalarFnVTable for Lower {
    type Options = EmptyOptions;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.lower");
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
        Arity::Exact(1)
    }

    fn child_name(&self, _options: &Self::Options, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("input"),
            _ => unreachable!("Invalid child index {child_idx} for lower()"),
        }
    }

    fn return_dtype(&self, _options: &Self::Options, arg_dtypes: &[DType]) -> VortexResult<DType> {
        match &arg_dtypes[0] {
            dtype @ DType::Utf8(_) => Ok(dtype.clone()),
            other => vortex_bail!("lower() requires Utf8, got {other}"),
        }
    }

    fn execute(
        &self,
        _options: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let input = args.get(0)?;
        if let Some(scalar) = input.as_constant() {
            let lowered = match scalar.as_utf8().value() {
                Some(value) => Scalar::utf8(value.to_lowercase(), input.dtype().nullability()),
                None => Scalar::null(input.dtype().clone()),
            };
            return Ok(ConstantArray::new(lowered, args.row_count()).into_array());
        }
        lower(input.execute::<VarBinViewArray>(ctx)?, ctx)
    }

    fn is_strict(&self, _options: &Self::Options) -> bool {
        true
    }

    fn is_infallible(&self, _options: &Self::Options) -> bool {
        true
    }
}

fn lower(input: VarBinViewArray, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
    let views = input.resolved_views();
    let validity = input.validity()?.execute_mask(input.len(), ctx)?;
    // Label-like columns are often already lowercase, so avoid rebuilding their buffers.
    if (0..input.len()).all(|i| !validity.value(i) || is_lowercase_ascii(views.bytes(i))) {
        return Ok(input.into_array());
    }

    let mut builder = VarBinViewBuilder::with_capacity_in(
        input.dtype().clone(),
        input.len(),
        ctx.allocator().clone(),
    );
    let mut ascii = Vec::new();
    for i in 0..input.len() {
        if !validity.value(i) {
            builder.append_null();
            continue;
        }
        let bytes = views.bytes(i);
        if bytes.is_ascii() {
            ascii.clear();
            ascii.extend(bytes.iter().map(u8::to_ascii_lowercase));
            builder.append_value(&ascii);
        } else {
            let value = std::str::from_utf8(bytes)
                .map_err(|e| vortex_err!("lower() input is not valid UTF-8: {e}"))?;
            builder.append_value(value.to_lowercase());
        }
    }
    Ok(builder.finish_into_varbinview().into_array())
}

/// Returns whether `bytes` is ASCII without uppercase letters, i.e. unchanged by [`Lower`].
pub(crate) fn is_lowercase_ascii(bytes: &[u8]) -> bool {
    bytes
        .iter()
        .all(|byte| byte.is_ascii() && !byte.is_ascii_uppercase())
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::ArrayRef;
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
    use crate::expr::lower;
    use crate::expr::proto::ExprSerializeProtoExt;
    use crate::expr::root;
    use crate::scalar::Scalar;
    use crate::scalar_fn::session::ScalarFnSession;

    static SESSION: LazyLock<VortexSession> =
        LazyLock::new(|| VortexSession::empty().with::<ScalarFnSession>());

    fn strings(values: &[Option<&str>]) -> ArrayRef {
        VarBinViewArray::from_iter_nullable_str(values.iter().copied()).into_array()
    }

    #[rstest]
    #[case::ascii(&[Some("GrEaT-LaKeS"), None, Some("")], &[Some("great-lakes"), None, Some("")])]
    #[case::final_sigma(&[Some("ΟΣ"), Some("ΟΣΑ")], &[Some("ος"), Some("οσα")])]
    #[case::expanding(&[Some("İȾ")], &[Some("i\u{307}ⱦ")])]
    #[case::already_lowercase(&[Some("great-lakes"), None], &[Some("great-lakes"), None])]
    fn lowercases(
        #[case] input: &[Option<&str>],
        #[case] expected: &[Option<&str>],
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        assert_arrays_eq!(
            strings(input).apply(&lower(root()))?,
            strings(expected),
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn roundtrips_through_proto() -> VortexResult<()> {
        let expr = lower(root());
        assert_eq!(
            Expression::from_proto(&expr.serialize_proto()?, &SESSION)?,
            expr
        );
        Ok(())
    }

    #[test]
    fn preserves_runs_and_dictionary() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values = DictArray::try_new(
            PrimitiveArray::from_iter([0u8, 1, 2, 0]).into_array(),
            strings(&[Some("ΟΣ"), None, Some("İȾ")]),
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
        let result = input
            .apply(&lower(root()))?
            .execute_until::<RunEnd>(&mut ctx)?;
        assert!(result.is::<RunEnd>());
        assert_arrays_eq!(
            result,
            strings(&[
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

    #[rstest]
    #[case(Some("ΟΣ"), Some("ος"))]
    #[case(None, None)]
    fn preserves_constants(
        #[case] input: Option<&str>,
        #[case] expected: Option<&str>,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let result = ConstantArray::new(Scalar::from(input), 1024)
            .into_array()
            .apply(&lower(root()))?
            .execute_until::<Constant>(&mut ctx)?;
        assert!(result.is::<Constant>());
        assert_arrays_eq!(
            result,
            ConstantArray::new(Scalar::from(expected), 1024),
            &mut ctx
        );
        Ok(())
    }
}
