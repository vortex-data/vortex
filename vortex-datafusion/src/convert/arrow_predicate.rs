// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;
use std::hash::Hash;
use std::hash::Hasher;
use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_array::cast::AsArray;
use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::FieldRef;
use arrow_schema::SchemaRef;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_expr::utils::reassign_expr_columns;
use vortex::array::ArrayRef;
use vortex::array::ExecutionCtx;
use vortex::array::IntoArray;
use vortex::array::arrays::ConstantArray;
use vortex::array::builtins::ArrayBuiltins;
use vortex::dtype::DType;
use vortex::dtype::Nullability;
use vortex::error::VortexError;
use vortex::error::VortexResult;
use vortex::error::vortex_ensure;
use vortex::expr::Expression;
use vortex::expr::display::ExprDisplay;
use vortex::scalar_fn::Arity;
use vortex::scalar_fn::ChildName;
use vortex::scalar_fn::ExecutionArgs;
use vortex::scalar_fn::ScalarFnId;
use vortex::scalar_fn::ScalarFnVTable;
use vortex::scalar_fn::ScalarFnVTableExt;
use vortex::session::registry::CachedId;
use vortex_arrow::ArrowSessionExt;

pub(super) fn dynamic_predicate(
    expression: Arc<dyn PhysicalExpr>,
    schema: SchemaRef,
    inputs: Expression,
) -> Option<Expression> {
    let expression = reassign_expr_columns(expression, &schema).ok()?;
    let field = Arc::new(Field::new_struct("", schema.fields().clone(), false));
    Some(ArrowPredicate.new_expr(PredicateOptions { expression, field }, [inputs]))
}

#[derive(Clone)]
struct ArrowPredicate;

#[derive(Clone)]
struct PredicateOptions {
    expression: Arc<dyn PhysicalExpr>,
    field: FieldRef,
}

impl fmt::Debug for PredicateOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.expression, f)
    }
}
impl fmt::Display for PredicateOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.expression, f)
    }
}
impl PartialEq for PredicateOptions {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.expression, &other.expression) && self.field == other.field
    }
}
impl Eq for PredicateOptions {}
impl Hash for PredicateOptions {
    fn hash<H: Hasher>(&self, state: &mut H) {
        Arc::as_ptr(&self.expression).cast::<()>().hash(state);
        self.field.hash(state);
    }
}

impl ScalarFnVTable for ArrowPredicate {
    type Options = PredicateOptions;
    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.datafusion.dynamic_predicate");
        *ID
    }
    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Exact(1)
    }
    fn child_name(&self, _options: &Self::Options, _index: usize) -> ChildName {
        ChildName::from("keys")
    }
    fn return_dtype(&self, _options: &Self::Options, inputs: &[DType]) -> VortexResult<DType> {
        vortex_ensure!(
            matches!(inputs, [DType::Struct(..)]),
            "Dynamic predicate requires a key struct"
        );
        Ok(DType::Bool(Nullability::Nullable))
    }
    fn is_infallible(&self, _options: &Self::Options) -> bool {
        true
    }
    fn fmt_sql(
        &self,
        options: &Self::Options,
        expr: &dyn ExprDisplay,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        write!(
            f,
            "dynamic_predicate({}, {})",
            options.expression,
            expr.display_child(0)
        )
    }
    fn execute(
        &self,
        options: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let keys = args.get(0)?;
        let session = ctx.session().clone();
        let predicate = (|| {
            let arrow = session
                .arrow()
                .execute_arrow(keys, Some(options.field.as_ref()), ctx)?;
            let batch = RecordBatch::from(arrow.as_struct().clone());
            let result = options
                .expression
                .evaluate(&batch)
                .map_err(VortexError::external)?
                .into_array(batch.num_rows())
                .map_err(VortexError::external)?;
            vortex_ensure!(
                result.data_type() == &DataType::Boolean,
                "Dynamic predicate must return Boolean"
            );
            session
                .arrow()
                .from_arrow_array(result, Nullability::Nullable)
        })();
        // These predicates are optional pruning hints; the producer retains exact enforcement.
        // An unavailable hint must retain rows rather than introduce an execution error.
        predicate.or_else(|_| {
            ConstantArray::new(true, args.row_count())
                .into_array()
                .cast(DType::Bool(Nullability::Nullable))?
                .execute(ctx)
        })
    }
}
