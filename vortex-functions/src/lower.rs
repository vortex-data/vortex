// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::AnyColumnar;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::builders::ArrayBuilder;
use vortex_array::builders::VarBinViewBuilder;
use vortex_array::dtype::DType;
use vortex_array::expr::BoundExpression;
use vortex_array::require_input;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::Arity;
use vortex_array::scalar_fn::ChildName;
use vortex_array::scalar_fn::EmptyOptions;
use vortex_array::scalar_fn::ExecutionArgs;
use vortex_array::scalar_fn::ScalarFnExecutionResult;
use vortex_array::scalar_fn::ScalarFnId;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_array::stats::Stat;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_expr::ScalarFnExprExt;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::lower_bound::LowerBound;
use crate::lower_bound::LowerBoundKind;

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

    fn execute_step(
        &self,
        options: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ScalarFnExecutionResult> {
        require_input!(args, 0 => AnyColumnar);
        self.execute(options, args, ctx)
            .map(ScalarFnExecutionResult::done)
    }

    fn is_strict(&self, _options: &Self::Options) -> bool {
        true
    }

    fn is_infallible(&self, _options: &Self::Options) -> bool {
        true
    }

    /// Bounds `lower(input)` from the byte-ordered min/max of `input`.
    fn stat_expression(
        &self,
        _options: &Self::Options,
        expr: &BoundExpression,
        stat: Stat,
        child_stat: &dyn Fn(&BoundExpression, Stat) -> Option<BoundExpression>,
    ) -> Option<BoundExpression> {
        let input = expr.child(0);
        let kind = match stat {
            Stat::Min => LowerBoundKind::Min,
            Stat::Max => LowerBoundKind::Max,
            // Lowercasing maps nulls to nulls and values to values.
            Stat::NullCount => return child_stat(input, Stat::NullCount),
            _ => return None,
        };
        LowerBound
            .try_new_bound_expr(
                kind,
                [child_stat(input, Stat::Min)?, child_stat(input, Stat::Max)?],
            )
            .ok()
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
    use std::ops::Range;
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::Constant;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::arrays::DictArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::RunEnd;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::scalar::Scalar;
    use vortex_error::VortexResult;
    use vortex_expr::ArrayExprExt;
    use vortex_expr::Expression;
    use vortex_expr::proto::ExprSerializeProtoExt;
    use vortex_expr::root;
    use vortex_session::VortexSession;

    use crate::expr::lower;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty();
        vortex_array::initialize(&session);
        crate::initialize(&session);
        session
    });

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

    #[rstest]
    #[case::whole(0..8)]
    #[case::sliced(1..7)]
    fn preserves_runs_and_dictionary(#[case] rows: Range<usize>) -> VortexResult<()> {
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
        .into_array()
        .slice(rows.clone())?;
        let result = input
            .apply(&lower(root()))?
            .execute_until::<RunEnd>(&mut ctx)?;
        assert!(result.is::<RunEnd>());
        assert_arrays_eq!(
            result,
            strings(
                &[
                    Some("ος"),
                    Some("ος"),
                    None,
                    None,
                    Some("i\u{307}ⱦ"),
                    Some("i\u{307}ⱦ"),
                    Some("ος"),
                    Some("ος"),
                ][rows]
            ),
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
