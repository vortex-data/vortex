// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod kernel;

use std::fmt::Formatter;

pub use kernel::*;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_mask::AllOr;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use crate::Array;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::builders::ArrayBuilder;
use crate::builders::builder_with_capacity;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::expr::Expression;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::EmptyOptions;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::SimplifyCtx;
use crate::scalar_fn::fns::literal::Literal;

/// An expression that conditionally selects between two arrays based on a boolean mask.
///
/// For each position `i`, `result[i] = if mask[i] then if_true[i] else if_false[i]`.
///
/// Null values in the mask are treated as false (selecting `if_false`). This follows
/// SQL semantics (DuckDB, Trino) where a null condition falls through to the ELSE branch,
/// rather than Arrow's `if_else` which propagates null conditions to the output.
#[derive(Clone)]
pub struct Zip;

impl ScalarFnVTable for Zip {
    type Options = EmptyOptions;

    fn id(&self) -> ScalarFnId {
        ScalarFnId::from("vortex.zip")
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
        Arity::Exact(3)
    }

    fn child_name(&self, _options: &Self::Options, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("if_true"),
            1 => ChildName::from("if_false"),
            2 => ChildName::from("mask"),
            _ => unreachable!("Invalid child index {} for Zip expression", child_idx),
        }
    }

    fn fmt_sql(
        &self,
        _options: &Self::Options,
        expr: &Expression,
        f: &mut Formatter<'_>,
    ) -> std::fmt::Result {
        write!(f, "zip(")?;
        expr.child(0).fmt_sql(f)?;
        write!(f, ", ")?;
        expr.child(1).fmt_sql(f)?;
        write!(f, ", ")?;
        expr.child(2).fmt_sql(f)?;
        write!(f, ")")
    }

    fn return_dtype(&self, _options: &Self::Options, arg_dtypes: &[DType]) -> VortexResult<DType> {
        vortex_ensure!(
            arg_dtypes[0].eq_ignore_nullability(&arg_dtypes[1]),
            "zip requires if_true and if_false to have the same base type, got {} and {}",
            arg_dtypes[0],
            arg_dtypes[1]
        );
        vortex_ensure!(
            matches!(arg_dtypes[2], DType::Bool(_)),
            "zip requires mask to be a boolean type, got {}",
            arg_dtypes[2]
        );
        Ok(arg_dtypes[0]
            .clone()
            .union_nullability(arg_dtypes[1].nullability()))
    }

    fn execute(
        &self,
        _options: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let if_true = args.get(0)?;
        let if_false = args.get(1)?;
        let mask_array = args.get(2)?;

        let mask = mask_array.try_to_mask_fill_null_false()?;

        let return_dtype = if_true
            .dtype()
            .clone()
            .union_nullability(if_false.dtype().nullability());

        if mask.all_true() {
            return if_true.cast(return_dtype)?.execute(ctx);
        }

        let return_dtype = if_true
            .dtype()
            .clone()
            .union_nullability(if_false.dtype().nullability());

        if mask.all_false() {
            return if_false.cast(return_dtype)?.execute(ctx);
        }

        if !if_true.is_canonical() || !if_false.is_canonical() {
            let if_true = if_true.execute::<ArrayRef>(ctx)?;
            let if_false = if_false.execute::<ArrayRef>(ctx)?;
            return if_true.zip(if_false, mask.into_array());
        }

        zip_impl(&if_true, &if_false, &mask)
    }

    fn simplify(
        &self,
        _options: &Self::Options,
        expr: &Expression,
        _ctx: &dyn SimplifyCtx,
    ) -> VortexResult<Option<Expression>> {
        let Some(mask_lit) = expr.child(2).as_opt::<Literal>() else {
            return Ok(None);
        };

        if let Some(mask_val) = mask_lit.as_bool().value() {
            if mask_val {
                return Ok(Some(expr.child(0).clone()));
            } else {
                return Ok(Some(expr.child(1).clone()));
            }
        }

        Ok(None)
    }

    fn is_null_sensitive(&self, _options: &Self::Options) -> bool {
        true
    }

    fn is_fallible(&self, _options: &Self::Options) -> bool {
        false
    }
}

pub(crate) fn zip_impl(
    if_true: &ArrayRef,
    if_false: &ArrayRef,
    mask: &Mask,
) -> VortexResult<ArrayRef> {
    assert_eq!(
        if_true.len(),
        if_false.len(),
        "zip requires arrays to have the same size"
    );

    let return_type = if_true
        .dtype()
        .clone()
        .union_nullability(if_false.dtype().nullability());
    zip_impl_with_builder(
        if_true,
        if_false,
        mask,
        builder_with_capacity(&return_type, if_true.len()),
    )
}

fn zip_impl_with_builder(
    if_true: &ArrayRef,
    if_false: &ArrayRef,
    mask: &Mask,
    mut builder: Box<dyn ArrayBuilder>,
) -> VortexResult<ArrayRef> {
    match mask.slices() {
        AllOr::All => Ok(if_true.to_array()),
        AllOr::None => Ok(if_false.to_array()),
        AllOr::Some(slices) => {
            for (start, end) in slices {
                builder.extend_from_array(&if_false.slice(builder.len()..*start)?);
                builder.extend_from_array(&if_true.slice(*start..*end)?);
            }
            if builder.len() < if_false.len() {
                builder.extend_from_array(&if_false.slice(builder.len()..if_false.len())?);
            }
            Ok(builder.finish())
        }
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::cast::AsArray;
    use arrow_select::zip::zip as arrow_zip;
    use vortex_buffer::buffer;
    use vortex_mask::Mask;

    use crate::Array;
    use crate::IntoArray;
    use crate::arrays::ConstantArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::StructArray;
    use crate::arrays::VarBinViewVTable;
    use crate::arrow::IntoArrowArray;
    use crate::assert_arrays_eq;
    use crate::builders::ArrayBuilder;
    use crate::builders::BufferGrowthStrategy;
    use crate::builders::VarBinViewBuilder;
    use crate::builtins::ArrayBuiltins;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::expr::lit;
    use crate::expr::root;
    use crate::expr::zip_expr;
    use crate::scalar::Scalar;

    #[test]
    fn dtype() {
        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let expr = zip_expr(root(), lit(0i32), lit(true));
        let result_dtype = expr.return_dtype(&dtype).unwrap();
        assert_eq!(
            result_dtype,
            DType::Primitive(PType::I32, Nullability::NonNullable)
        );
    }

    #[test]
    fn test_display() {
        let expr = zip_expr(root(), lit(0i32), lit(true));
        assert_eq!(expr.to_string(), "zip($, 0i32, true)");
    }

    #[test]
    fn test_zip_basic() {
        let mask = Mask::from_iter([true, false, false, true, false]);
        let if_true = buffer![10, 20, 30, 40, 50].into_array();
        let if_false = buffer![1, 2, 3, 4, 5].into_array();

        let result = if_true.zip(if_false, mask.into_array()).unwrap();
        let expected = buffer![10, 2, 3, 40, 5].into_array();

        assert_arrays_eq!(result, expected);
    }

    #[test]
    fn test_zip_all_true() {
        let mask = Mask::new_true(4);
        let if_true = buffer![10, 20, 30, 40].into_array();
        let if_false =
            PrimitiveArray::from_option_iter([Some(1), Some(2), Some(3), None]).into_array();

        let result = if_true.zip(if_false.clone(), mask.into_array()).unwrap();
        let expected =
            PrimitiveArray::from_option_iter([Some(10), Some(20), Some(30), Some(40)]).into_array();

        assert_arrays_eq!(result, expected);

        // result must be nullable even if_true was not
        assert_eq!(result.dtype(), if_false.dtype())
    }

    #[test]
    #[should_panic]
    fn test_invalid_lengths() {
        let mask = Mask::new_false(4);
        let if_true = buffer![10, 20, 30].into_array();
        let if_false = buffer![1, 2, 3, 4].into_array();

        let _result = if_true.zip(if_false, mask.into_array()).unwrap();
    }

    #[test]
    fn test_fragmentation() {
        let len = 100;

        let const1 = ConstantArray::new(
            Scalar::utf8("hello_this_is_a_longer_string", Nullability::Nullable),
            len,
        )
        .to_array();

        let const2 = ConstantArray::new(
            Scalar::utf8("world_this_is_another_string", Nullability::Nullable),
            len,
        )
        .to_array();

        let indices: Vec<usize> = (0..len).step_by(2).collect();
        let mask = Mask::from_indices(len, indices);
        let mask_array = mask.into_array();

        let result = const1.zip(const2.clone(), mask_array.clone()).unwrap();

        insta::assert_snapshot!(result.display_tree(), @r"
        root: vortex.varbinview(utf8?, len=100) nbytes=1.66 kB (100.00%) [all_valid]
          metadata: EmptyMetadata
          buffer: buffer_0 host 29 B (align=1) (1.75%)
          buffer: buffer_1 host 28 B (align=1) (1.69%)
          buffer: views host 1.60 kB (align=16) (96.56%)
        ");

        // test wrapped in a struct
        let wrapped1 = StructArray::try_from_iter([("nested", const1)])
            .unwrap()
            .to_array();
        let wrapped2 = StructArray::try_from_iter([("nested", const2)])
            .unwrap()
            .to_array();

        let wrapped_result = wrapped1.zip(wrapped2, mask_array).unwrap();
        insta::assert_snapshot!(wrapped_result.display_tree(), @r"
        root: vortex.struct({nested=utf8?}, len=100) nbytes=1.66 kB (100.00%)
          metadata: EmptyMetadata
          nested: vortex.varbinview(utf8?, len=100) nbytes=1.66 kB (100.00%) [all_valid]
            metadata: EmptyMetadata
            buffer: buffer_0 host 29 B (align=1) (1.75%)
            buffer: buffer_1 host 28 B (align=1) (1.69%)
            buffer: views host 1.60 kB (align=16) (96.56%)
        ");
    }

    #[test]
    fn test_varbinview_zip() {
        let if_true = {
            let mut builder = VarBinViewBuilder::new(
                DType::Utf8(Nullability::NonNullable),
                10,
                Default::default(),
                BufferGrowthStrategy::fixed(64 * 1024),
                0.0,
            );
            for _ in 0..100 {
                builder.append_value("Hello");
                builder.append_value("Hello this is a long string that won't be inlined.");
            }
            builder.finish()
        };

        let if_false = {
            let mut builder = VarBinViewBuilder::new(
                DType::Utf8(Nullability::NonNullable),
                10,
                Default::default(),
                BufferGrowthStrategy::fixed(64 * 1024),
                0.0,
            );
            for _ in 0..100 {
                builder.append_value("Hello2");
                builder.append_value("Hello2 this is a long string that won't be inlined.");
            }
            builder.finish()
        };

        // [1,2,4,5,7,8,..]
        let mask = Mask::from_indices(200, (0..100).filter(|i| i % 3 != 0).collect());
        let mask_array = mask.clone().into_array();

        let zipped = if_true.zip(if_false.clone(), mask_array).unwrap();
        let zipped = zipped.as_opt::<VarBinViewVTable>().unwrap();
        assert_eq!(zipped.nbuffers(), 2);

        // assert the result is the same as arrow
        let expected = arrow_zip(
            mask.into_array()
                .into_arrow_preferred()
                .unwrap()
                .as_boolean(),
            &if_true.into_arrow_preferred().unwrap(),
            &if_false.into_arrow_preferred().unwrap(),
        )
        .unwrap();

        let actual = zipped.clone().into_array().into_arrow_preferred().unwrap();
        assert_eq!(actual.as_ref(), expected.as_ref());
    }
}
