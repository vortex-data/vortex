// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;
use std::sync::Arc;

use parking_lot::RwLock;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_session::registry::CachedId;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::StructArray;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::FieldName;
use crate::dtype::FieldNames;
use crate::dtype::Nullability;
use crate::dtype::StructFields;
use crate::expr::BoundExpression;
use crate::expr::display::ExprDisplay;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::fns::literal::Literal;
use crate::validity::Validity;

/// A source of a predicate that can change while a query runs, such as the build-side bounds of
/// a hash join or the threshold of a top-N operator in a compute engine.
pub trait DynamicExpressionSource: Debug + Display + Send + Sync + 'static {
    /// A counter that changes whenever [`Self::current`] can return a different predicate.
    fn generation(&self) -> u64;

    /// The current predicate, bound against `scope`.
    ///
    /// The scope is a struct whose field `i`, named by [`input_field_name`], holds the child `i`
    /// of the dynamic expression. The predicate must return a boolean.
    fn current(&self, scope: &DType) -> VortexResult<BoundExpression>;
}

/// The name of the scope field that holds child `idx` of a dynamic expression.
pub fn input_field_name(idx: usize) -> FieldName {
    FieldName::from(idx.to_string())
}

/// A predicate that a [`DynamicExpressionSource`] can replace while a query runs.
///
/// The children of the expression are the inputs that every predicate of the source can read, so
/// the columns a scan must read are known before the source has its first predicate. Each
/// evaluation reads the current predicate of the source.
#[derive(Clone)]
pub struct DynamicExpression;

impl ScalarFnVTable for DynamicExpression {
    type Options = DynamicExpressionOptions;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.dynamic_expression");
        *ID
    }

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Variadic { min: 0, max: None }
    }

    fn child_name(&self, _options: &Self::Options, child_idx: usize) -> ChildName {
        ChildName::from(Arc::from(input_field_name(child_idx).as_ref()))
    }

    fn fmt_sql(
        &self,
        options: &Self::Options,
        expr: &dyn ExprDisplay,
        f: &mut Formatter<'_>,
    ) -> std::fmt::Result {
        write!(f, "dynamic({}", options.source)?;
        for idx in 0..expr.display_children_count() {
            write!(f, ", ")?;
            Display::fmt(expr.display_child(idx), f)?;
        }
        write!(f, ")")
    }

    fn return_dtype(&self, _options: &Self::Options, _arg_dtypes: &[DType]) -> VortexResult<DType> {
        // The predicate can change, so the dtype must cover every predicate of the source.
        Ok(DType::Bool(Nullability::Nullable))
    }

    fn execute(
        &self,
        options: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let inputs = (0..args.num_inputs())
            .map(|idx| args.get(idx))
            .collect::<VortexResult<Vec<_>>>()?;
        let names = (0..inputs.len())
            .map(input_field_name)
            .collect::<FieldNames>();
        let scope = StructArray::try_new(names, inputs, args.row_count(), Validity::NonNullable)?
            .into_array();

        let predicate = options.current(scope.dtype())?;
        scope
            .apply_bound(&predicate)?
            .cast(DType::Bool(Nullability::Nullable))?
            .execute(ctx)
    }
}

/// The options of a [`DynamicExpression`]: its source, and the predicate last bound from it.
#[derive(Clone)]
pub struct DynamicExpressionOptions {
    source: Arc<dyn DynamicExpressionSource>,
    bound: Arc<RwLock<BoundPredicates>>,
}

/// The predicate of the source as of a generation, bound against each scope it was read with.
///
/// A scan reads the predicate with the dtypes of the bound inputs and executes it with the dtypes
/// of the input arrays, which can differ, so each scope keeps its own binding.
#[derive(Default)]
struct BoundPredicates {
    generation: u64,
    by_scope: Vec<(DType, BoundExpression)>,
}

impl DynamicExpressionOptions {
    /// Create options that read predicates from `source`.
    pub fn new(source: Arc<dyn DynamicExpressionSource>) -> Self {
        Self {
            source,
            bound: Arc::new(RwLock::new(BoundPredicates::default())),
        }
    }

    /// The source of the predicates.
    pub fn source(&self) -> &Arc<dyn DynamicExpressionSource> {
        &self.source
    }

    /// The current predicate of the source, bound against `scope`.
    ///
    /// A predicate is bound once for each generation of the source and scope.
    pub fn current(&self, scope: &DType) -> VortexResult<BoundExpression> {
        let generation = self.source.generation();
        {
            let bound = self.bound.read();
            if bound.generation == generation
                && let Some((_, predicate)) = bound
                    .by_scope
                    .iter()
                    .find(|(bound_scope, _)| bound_scope == scope)
            {
                return Ok(predicate.clone());
            }
        }

        let mut bound = self.bound.write();
        if bound.generation != generation {
            bound.generation = generation;
            bound.by_scope.clear();
        }
        if let Some((_, predicate)) = bound
            .by_scope
            .iter()
            .find(|(bound_scope, _)| bound_scope == scope)
        {
            return Ok(predicate.clone());
        }

        let predicate = self.source.current(scope)?;
        vortex_ensure!(
            matches!(predicate.dtype(), DType::Bool(_)),
            "Dynamic expression {} must return a boolean, got {}",
            self.source,
            predicate.dtype()
        );
        bound.by_scope.push((scope.clone(), predicate.clone()));
        Ok(predicate)
    }
}

impl Debug for DynamicExpressionOptions {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DynamicExpressionOptions")
            .field("source", &self.source)
            .finish()
    }
}

impl Display for DynamicExpressionOptions {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&self.source, f)
    }
}

/// Equality and hashing use the identity of the source, so a new predicate of the source does not
/// change the identity of the expression tree.
impl PartialEq for DynamicExpressionOptions {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.source, &other.source)
    }
}

impl Eq for DynamicExpressionOptions {}

impl Hash for DynamicExpressionOptions {
    fn hash<H: Hasher>(&self, state: &mut H) {
        Arc::as_ptr(&self.source).cast::<()>().hash(state);
    }
}

/// Whether `expr` is a dynamic expression whose current predicate is the literal `true`, such as
/// a filter of an engine that has no value yet. A scan can skip such a predicate for free.
pub fn is_currently_true(expr: &BoundExpression) -> bool {
    let Some(options) = expr.as_opt::<DynamicExpression>() else {
        return false;
    };
    let scope = DType::Struct(
        StructFields::new(
            (0..expr.children().len())
                .map(input_field_name)
                .collect::<FieldNames>(),
            expr.children()
                .iter()
                .map(|child| child.dtype().clone())
                .collect(),
        ),
        Nullability::NonNullable,
    );
    options.current(&scope).is_ok_and(|predicate| {
        predicate
            .as_opt::<Literal>()
            .is_some_and(|value| value.as_bool_opt().and_then(|value| value.value()) == Some(true))
    })
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;
    use std::sync::atomic::AtomicI32;
    use std::sync::atomic::AtomicU64;
    use std::sync::atomic::Ordering;

    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use super::*;
    use crate::VortexSessionExecute;
    use crate::arrays::BoolArray;
    use crate::assert_arrays_eq;
    use crate::dtype::PType;
    use crate::expr::col;
    use crate::expr::dynamic_expression;
    use crate::expr::get_item as unbound_get_item;
    use crate::expr::lit as unbound_lit;
    use crate::expr::lt;
    use crate::expr::root;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    /// `input 0 < threshold`, where a threshold of zero means there is no bound yet.
    #[derive(Debug, Default)]
    struct Threshold {
        value: AtomicI32,
        generation: AtomicU64,
    }

    impl Threshold {
        fn set(&self, value: i32) {
            self.value.store(value, Ordering::SeqCst);
            self.generation.fetch_add(1, Ordering::SeqCst);
        }
    }

    impl Display for Threshold {
        fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
            write!(f, "threshold({})", self.value.load(Ordering::SeqCst))
        }
    }

    impl DynamicExpressionSource for Threshold {
        fn generation(&self) -> u64 {
            self.generation.load(Ordering::SeqCst)
        }

        fn current(&self, scope: &DType) -> VortexResult<BoundExpression> {
            let value = self.value.load(Ordering::SeqCst);
            if value == 0 {
                return unbound_lit(true).bind(scope);
            }
            lt(
                unbound_get_item(input_field_name(0), root()),
                unbound_lit(value),
            )
            .bind(scope)
        }
    }

    #[test]
    fn execute_reads_the_current_predicate() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let threshold = Arc::new(Threshold::default());
        let expr = dynamic_expression(threshold.clone(), [col("a")]);
        let input =
            StructArray::from_fields(&[("a", buffer![1i32, 5, 10].into_array())])?.into_array();

        let result = input.clone().apply(&expr)?;
        assert_arrays_eq!(
            result,
            BoolArray::from_iter([Some(true), Some(true), Some(true)]),
            &mut ctx
        );

        threshold.set(6);
        let result = input.apply(&expr)?;
        assert_arrays_eq!(
            result,
            BoolArray::from_iter([Some(true), Some(true), Some(false)]),
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn is_currently_true_follows_the_source() -> VortexResult<()> {
        let threshold = Arc::new(Threshold::default());
        let scope = DType::Struct(
            StructFields::new(
                FieldNames::from(vec!["a"]),
                vec![DType::Primitive(PType::I32, Nullability::NonNullable)],
            ),
            Nullability::NonNullable,
        );
        let expr = dynamic_expression(threshold.clone(), [col("a")]).bind(&scope)?;
        assert!(is_currently_true(&expr));

        threshold.set(6);
        assert!(!is_currently_true(&expr));
        Ok(())
    }

}
