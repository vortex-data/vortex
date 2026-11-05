// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A Vortex scalar function that evaluates a DataFusion predicate over Arrow exports of its
//! inputs.
//!
//! A hash join's dynamic filter tests the probe keys against the build side's hash table, which
//! has no Vortex equivalent. This function lets a scan run that test itself, so it decodes the
//! rest of a row only when the join would keep it. Zone pruning cannot prove anything about the
//! predicate, so it treats every zone as possibly matching.

use std::fmt;
use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;
use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_array::RecordBatchOptions;
use arrow_array::cast::AsArray;
use arrow_schema::Field;
use arrow_schema::Schema;
use arrow_schema::SchemaRef;
use datafusion_physical_expr::PhysicalExpr;
use vortex::array::ArrayRef;
use vortex::array::ExecutionCtx;
use vortex::array::IntoArray;
use vortex::array::arrays::ConstantArray;
use vortex::dtype::DType;
use vortex::dtype::Nullability;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_ensure_eq;
use vortex::error::vortex_err;
use vortex::expr::display::ExprDisplay;
use vortex::scalar::Scalar;
use vortex::scalar_fn::Arity;
use vortex::scalar_fn::ChildName;
use vortex::scalar_fn::ExecutionArgs;
use vortex::scalar_fn::ScalarFnId;
use vortex::scalar_fn::ScalarFnVTable;
use vortex::session::registry::CachedId;
use vortex_arrow::ArrowSessionExt;
use vortex_arrow::convert::from_arrow_boolean;

/// Evaluates a DataFusion boolean predicate, whose column `i` is child `i` of the expression.
///
/// The function is process-local and not serializable.
#[derive(Clone)]
pub(crate) struct DataFusionPredicate;

/// The predicate a [`DataFusionPredicate`] evaluates, and the Arrow field of each child.
#[derive(Clone)]
pub(crate) struct DataFusionPredicateOptions {
    predicate: Arc<dyn PhysicalExpr>,
    /// The Arrow field each child is exported to. The predicate reads column `i` from field `i`.
    export: SchemaRef,
    /// The schema of the batch the predicate evaluates: `export`, with every field nullable.
    batch: SchemaRef,
}

impl DataFusionPredicateOptions {
    /// Options to evaluate `predicate`, whose column `i` has the Arrow field `fields[i]`.
    pub(crate) fn new(predicate: Arc<dyn PhysicalExpr>, fields: Vec<Field>) -> Self {
        // The predicate never relies on a field being non-nullable, while a batch rejects nulls
        // in a non-nullable field, and Vortex may hand over a nullable array without any nulls.
        let batch = fields
            .iter()
            .map(|field| field.clone().with_nullable(true))
            .collect::<Vec<_>>();
        Self {
            predicate,
            export: Arc::new(Schema::new(fields)),
            batch: Arc::new(Schema::new(batch)),
        }
    }
}

impl fmt::Debug for DataFusionPredicateOptions {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("DataFusionPredicateOptions")
            .field("predicate", &self.predicate.to_string())
            .finish()
    }
}

impl Display for DataFusionPredicateOptions {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Display::fmt(&self.predicate, f)
    }
}

/// Equality and hashing use the identity of the predicate, which DataFusion does not compare
/// cheaply.
impl PartialEq for DataFusionPredicateOptions {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.predicate, &other.predicate) && Arc::ptr_eq(&self.export, &other.export)
    }
}

impl Eq for DataFusionPredicateOptions {}

impl Hash for DataFusionPredicateOptions {
    fn hash<H: Hasher>(&self, state: &mut H) {
        Arc::as_ptr(&self.predicate).cast::<()>().hash(state);
    }
}

impl ScalarFnVTable for DataFusionPredicate {
    type Options = DataFusionPredicateOptions;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.datafusion.predicate");
        *ID
    }

    fn arity(&self, options: &Self::Options) -> Arity {
        Arity::Exact(options.export.fields().len())
    }

    fn child_name(&self, options: &Self::Options, child_idx: usize) -> ChildName {
        ChildName::from(Arc::from(options.export.field(child_idx).name().as_str()))
    }

    fn fmt_sql(
        &self,
        options: &Self::Options,
        _expr: &dyn ExprDisplay,
        f: &mut Formatter<'_>,
    ) -> fmt::Result {
        write!(f, "datafusion({})", options.predicate)
    }

    fn return_dtype(&self, _options: &Self::Options, _args: &[DType]) -> VortexResult<DType> {
        Ok(DType::Bool(Nullability::Nullable))
    }

    fn execute(
        &self,
        options: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let row_count = args.row_count();
        let session = ctx.session().clone();
        let columns = options
            .export
            .fields()
            .iter()
            .enumerate()
            .map(|(idx, field)| {
                session
                    .arrow()
                    .execute_arrow(args.get(idx)?, Some(field), ctx)
            })
            .collect::<VortexResult<Vec<_>>>()?;
        let batch = RecordBatch::try_new_with_options(
            Arc::clone(&options.batch),
            columns,
            &RecordBatchOptions::new().with_row_count(Some(row_count)),
        )
        .map_err(|e| vortex_err!("Invalid batch for DataFusion predicate: {e}"))?;

        let result = match options
            .predicate
            .evaluate(&batch)
            .and_then(|value| value.into_array(row_count))
        {
            Ok(result) => result,
            Err(error) => {
                // The predicate is a dynamic filter that the operator producing it enforces
                // exactly, so keeping every row loses nothing but this evaluation. The scan can
                // evaluate it on rows that another predicate would have rejected first.
                tracing::debug!(
                    predicate = %options.predicate,
                    %error,
                    "DataFusion predicate failed, keeping every row"
                );
                return Ok(ConstantArray::new(
                    Scalar::bool(true, Nullability::Nullable),
                    row_count,
                )
                .into_array());
            }
        };
        let Some(result) = result.as_boolean_opt() else {
            vortex_bail!(
                "DataFusion predicate {} returned {}, not a boolean",
                options.predicate,
                result.data_type()
            );
        };
        vortex_ensure_eq!(result.len(), row_count);
        from_arrow_boolean(result, true)
    }
}
