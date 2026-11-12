// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use anyhow::Result;
use arrow_array::ArrayRef;
use arrow_array::BooleanArray;
use arrow_array::Float32Array;
use arrow_array::Float64Array;
use arrow_array::Int64Array;
use arrow_array::ListArray;
use arrow_array::RecordBatch;
use arrow_array::StringArray;
use arrow_array::types::Int64Type;
use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::Schema;
use datafusion_common::Result as DFResult;
use datafusion_common::ScalarValue;
use datafusion_common::config::ConfigOptions;
use datafusion_common::hash_utils::create_hashes;
use datafusion_expr::ColumnarValue;
use datafusion_expr::Operator as DFOperator;
use datafusion_expr::ScalarFunctionArgs;
use datafusion_expr::ScalarUDF;
use datafusion_expr::ScalarUDFImpl;
use datafusion_expr::Signature;
use datafusion_expr::Volatility;
use datafusion_functions::core::named_struct::NamedStructFunc;
use datafusion_functions::string::lower::LowerFunc;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_expr::ScalarFunctionExpr;
use datafusion_physical_plan::expressions::BinaryExpr;
use datafusion_physical_plan::expressions::CaseExpr;
use datafusion_physical_plan::expressions::CastExpr;
use datafusion_physical_plan::expressions::Column;
use datafusion_physical_plan::expressions::IsNotNullExpr;
use datafusion_physical_plan::expressions::NotExpr;
use datafusion_physical_plan::expressions::lit;
use datafusion_physical_plan::joins::HashExpr;
use datafusion_physical_plan::joins::HashTableLookupExpr;
use datafusion_physical_plan::joins::Map;
use datafusion_physical_plan::joins::SeededRandomState;
use datafusion_physical_plan::joins::join_hash_map::JoinHashMapU32;
use datafusion_physical_plan::joins::utils::JoinHashMapType;
use vortex::VortexSessionDefault;
use vortex::array::VortexSessionExecute;
use vortex::expr::Expression;
use vortex::scalar_fn::fns::binary::Binary;
use vortex::scalar_fn::fns::operators::Operator;
use vortex::session::VortexSession;
use vortex_arrow::ArrowSessionExt;
use vortex_arrow::ExecuteArrowExt;

use super::DefaultExpressionConverter;
use super::ExpressionConverter;
use super::is_infallible;
use super::tests::array_transform_expr;
use crate::df_expression::df_expression;

pub(crate) fn lookup(
    values: ArrayRef,
    key: Arc<dyn PhysicalExpr>,
) -> Result<Arc<dyn PhysicalExpr>> {
    let state = SeededRandomState::with_seed(17);
    let mut hashes = vec![0; values.len()];
    create_hashes(&[values], state.random_state(), &mut hashes)?;
    let mut map = JoinHashMapU32::with_capacity(hashes.len());
    map.update_from_iter(Box::new(hashes.iter().enumerate()), 0);
    Ok(Arc::new(HashTableLookupExpr::new(
        vec![key],
        state,
        Arc::new(Map::HashMap(Box::new(map))),
        "membership".into(),
    )))
}
fn evaluate(
    expression: &Expression,
    batch: &RecordBatch,
    output: &Field,
    session: &VortexSession,
) -> Result<ArrayRef> {
    let input = session
        .arrow()
        .from_arrow_record_batch(batch.clone(), &batch.schema())?;
    Ok(input
        .apply(expression)?
        .execute_arrow(Some(output), &mut session.create_execution_ctx())?)
}

#[test]
fn df_expression_keeps_native_outer_and_inner_sums() -> Result<()> {
    let batch = RecordBatch::try_from_iter(vec![
        (
            "x",
            Arc::new(Float64Array::from(vec![1., 2., 4.])) as ArrayRef,
        ),
        (
            "y",
            Arc::new(Float64Array::from(vec![2., 2., 1.])) as ArrayRef,
        ),
        (
            "z",
            Arc::new(Float64Array::from(vec![10., 20., 30.])) as ArrayRef,
        ),
    ])?;
    let inner: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
        Arc::new(Column::new("x", 0)),
        DFOperator::Plus,
        Arc::new(Column::new("y", 1)),
    ));
    let membership = lookup(Arc::new(Float64Array::from(vec![3., 5.])), inner)?;
    let outer: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
        Arc::new(CastExpr::new(membership, DataType::Float64, None)),
        DFOperator::Plus,
        Arc::new(Column::new("z", 2)),
    ));
    let session = VortexSession::default();
    let converter = DefaultExpressionConverter::new(session.clone());
    let converted = converter.try_convert(&outer, &batch.schema())?.unwrap();
    assert_eq!(converted.as_opt::<Binary>(), Some(&Operator::Add));
    let display = converted.to_string();
    assert_eq!(
        display.matches("datafusion.df_expression").count(),
        1,
        "{display}"
    );
    // The single DF fragment receives the native x+y result, not x and y.
    assert!(display.contains("opts=membership"), "{display}");
    let expected = outer
        .evaluate(&batch)?
        .into_array_of_size(batch.num_rows())?;
    let actual = evaluate(
        &converted,
        &batch,
        outer.return_field(&batch.schema())?.as_ref(),
        &session,
    )?;
    assert_eq!(&actual, &expected);
    Ok(())
}

#[test]
fn df_expression_groups_wrapping_operators_and_preserves_lazy_errors() -> Result<()> {
    let batch = RecordBatch::try_from_iter(vec![(
        "id",
        Arc::new(Int64Array::from(vec![0, 1])) as ArrayRef,
    )])?;
    let column = Arc::new(Column::new("id", 0)) as Arc<dyn PhysicalExpr>;
    let wrapped: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
        Arc::new(BinaryExpr::new(
            Arc::clone(&column),
            DFOperator::Plus,
            lit(i64::MAX),
        )),
        DFOperator::Plus,
        lit(1_i64),
    ));
    let key: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
        lit(1_i64),
        DFOperator::Divide,
        Arc::clone(&column),
    ));
    let membership = lookup(Arc::new(Int64Array::from(vec![1])), key)?;
    let conditional: Arc<dyn PhysicalExpr> = Arc::new(CaseExpr::try_new(
        None,
        vec![(lit(false), Arc::clone(&membership))],
        Some(lit(true)),
    )?);
    let session = VortexSession::default();
    let converter = DefaultExpressionConverter::new(session.clone());
    for expression in [wrapped, conditional] {
        let converted = converter
            .try_convert(&expression, &batch.schema())?
            .unwrap();
        assert_eq!(
            converted
                .to_string()
                .matches("datafusion.df_expression")
                .count(),
            1
        );
        let expected = expression
            .evaluate(&batch)?
            .into_array_of_size(batch.num_rows())?;
        let actual = evaluate(
            &converted,
            &batch,
            expression.return_field(&batch.schema())?.as_ref(),
            &session,
        )?;
        assert_eq!(&actual, &expected);
    }
    let negated: Arc<dyn PhysicalExpr> = Arc::new(NotExpr::new(membership));
    let hint = converter
        .try_convert_dynamic(&negated, &batch.schema())?
        .unwrap();
    assert!(is_infallible(&hint));
    let actual = evaluate(
        &hint,
        &batch,
        &Field::new("hint", DataType::Boolean, false),
        &session,
    )?;
    assert_eq!(
        &actual,
        &(Arc::new(BooleanArray::from(vec![true, true])) as ArrayRef)
    );
    Ok(())
}

pub(crate) fn routed(key: Arc<dyn PhysicalExpr>) -> Result<Arc<dyn PhysicalExpr>> {
    let route: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
        Arc::new(HashExpr::new(
            vec![Arc::clone(&key)],
            SeededRandomState::with_seed(0),
            "route".into(),
        )),
        DFOperator::Modulo,
        lit(2_u64),
    ));
    let first = lookup(Arc::new(Int64Array::from(vec![1, 3])), Arc::clone(&key))?;
    let second = lookup(Arc::new(Int64Array::from(vec![2, 4])), key)?;
    Ok(Arc::new(CaseExpr::try_new(
        Some(route),
        vec![(lit(0_u64), first), (lit(1_u64), second)],
        Some(lit(false)),
    )?))
}
#[test]
fn df_expression_accepts_partition_routing_case_as_a_whole_hint() -> Result<()> {
    let batch = RecordBatch::try_from_iter(vec![(
        "id",
        Arc::new(Int64Array::from(vec![
            Some(1),
            Some(2),
            None,
            Some(3),
            Some(4),
        ])) as ArrayRef,
    )])?;
    let expression = routed(Arc::new(Column::new("id", 0)))?;
    let session = VortexSession::default();
    let converter = DefaultExpressionConverter::new(session.clone());
    let converted = converter
        .try_convert_dynamic(&expression, &batch.schema())?
        .unwrap();
    assert!(is_infallible(&converted));
    let expected = expression
        .evaluate(&batch)?
        .into_array_of_size(batch.num_rows())?;
    let actual = evaluate(
        &converted,
        &batch,
        &Field::new("hint", DataType::Boolean, true),
        &session,
    )?;
    assert_eq!(&actual, &expected);
    Ok(())
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct ObserveArgument {
    signature: Signature,
    name: String,
    nullable: bool,
    scalar: bool,
}
impl ScalarUDFImpl for ObserveArgument {
    fn name(&self) -> &str {
        "observe_argument"
    }
    fn signature(&self) -> &Signature {
        &self.signature
    }
    fn return_type(&self, _: &[DataType]) -> DFResult<DataType> {
        Ok(DataType::Boolean)
    }
    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> DFResult<ColumnarValue> {
        Ok(ColumnarValue::Scalar(ScalarValue::Boolean(Some(
            args.arg_fields[0].name() == &self.name
                && args.arg_fields[0].is_nullable() == self.nullable
                && matches!(args.args[0], ColumnarValue::Scalar(_)) == self.scalar,
        ))))
    }
}
#[test]
fn df_expression_preserves_udf_argument_fields_and_scalar_shape() -> Result<()> {
    let batch = RecordBatch::try_from_iter(vec![
        ("unused", Arc::new(Int64Array::from(vec![0, 0])) as ArrayRef),
        (
            "x",
            Arc::new(Float32Array::from(vec![Some(1.), None])) as ArrayRef,
        ),
    ])?;
    let cast: Arc<dyn PhysicalExpr> = Arc::new(CastExpr::new_with_target_field(
        Arc::new(Column::new("x", 1)),
        Arc::new(Field::new("original", DataType::Float64, false)),
        None,
    ));
    let lower: Arc<dyn PhysicalExpr> = Arc::new(ScalarFunctionExpr::try_new(
        Arc::new(ScalarUDF::from(LowerFunc::new())),
        vec![lit("ABC")],
        &batch.schema(),
        Arc::new(ConfigOptions::new()),
    )?);
    let session = VortexSession::default();
    let converter = DefaultExpressionConverter::new(session.clone());
    let short_circuit: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
        lit(false),
        DFOperator::And,
        Arc::new(IsNotNullExpr::new(Arc::new(Column::new("x", 1)))),
    ));
    for (child, scalar) in [(cast, false), (lower, true), (short_circuit, true)] {
        let field = child.return_field(&batch.schema())?;
        let fun = ScalarUDF::from(ObserveArgument {
            signature: Signature::any(1, Volatility::Immutable),
            name: field.name().clone(),
            nullable: field.is_nullable(),
            scalar,
        });
        let expression: Arc<dyn PhysicalExpr> = Arc::new(ScalarFunctionExpr::try_new(
            Arc::new(fun),
            vec![child],
            &batch.schema(),
            Arc::new(ConfigOptions::new()),
        )?);
        let mixed = converter
            .try_convert(&expression, &batch.schema())?
            .unwrap();
        let raw = df_expression(Arc::clone(&expression), &batch.schema(), &session)?.unwrap();
        let expected = expression
            .evaluate(&batch)?
            .into_array_of_size(batch.num_rows())?;
        for converted in [mixed, raw] {
            let actual = evaluate(
                &converted,
                &batch,
                expression.return_field(&batch.schema())?.as_ref(),
                &session,
            )?;
            assert_eq!(&actual, &expected);
            assert_eq!(
                &actual,
                &(Arc::new(BooleanArray::from(vec![true, true])) as ArrayRef)
            );
        }
    }
    Ok(())
}

#[test]
fn df_expression_advisory_rejects_invalid_original_column_bindings() -> Result<()> {
    let converter = DefaultExpressionConverter::default();
    let duplicate = Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("id", DataType::Int64, false),
    ]);
    let predicate: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
        Arc::new(Column::new("id", 1)),
        DFOperator::Eq,
        lit(2_i64),
    ));
    assert!(
        converter
            .try_convert_dynamic(&predicate, &duplicate)?
            .is_none()
    );
    let distinct = Schema::new(vec![
        Field::new("a", DataType::Int64, false),
        Field::new("id", DataType::Int64, false),
    ]);
    for index in [0, 2] {
        let invalid: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("id", index)),
            DFOperator::Eq,
            lit(2_i64),
        ));
        assert!(converter.try_convert_dynamic(&invalid, &distinct).is_err());
    }
    Ok(())
}

#[test]
fn df_expression_preserves_lambda_input_positions() -> Result<()> {
    let batch = RecordBatch::try_from_iter(vec![
        (
            "x",
            Arc::new(Float64Array::from(vec![10., 20.])) as ArrayRef,
        ),
        (
            "list",
            Arc::new(ListArray::from_iter_primitive::<Int64Type, _, _>([
                Some(vec![Some(1), Some(2)]),
                Some(vec![Some(3), None]),
            ])) as ArrayRef,
        ),
    ])?;
    let transform = array_transform_expr(
        Arc::new(Column::new("list", 1)),
        "u",
        2,
        |u| Ok(Arc::new(BinaryExpr::new(u, DFOperator::Modulo, lit(2_i64)))),
        &batch.schema(),
    )?;
    let sum: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
        Arc::new(Column::new("x", 0)),
        DFOperator::Plus,
        lit(1_f64),
    ));
    let product: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
        Arc::new(Column::new("x", 0)),
        DFOperator::Multiply,
        lit(2_f64),
    ));
    let named: Arc<dyn PhysicalExpr> = Arc::new(ScalarFunctionExpr::try_new(
        Arc::new(ScalarUDF::from(NamedStructFunc::new())),
        vec![
            lit("sum"),
            sum,
            lit("product"),
            product,
            lit("list"),
            Arc::clone(&transform),
        ],
        &batch.schema(),
        Arc::new(ConfigOptions::new()),
    )?);
    let session = VortexSession::default();
    let converter = DefaultExpressionConverter::new(session.clone());
    for expression in [transform, named] {
        let mixed = converter
            .try_convert(&expression, &batch.schema())?
            .unwrap();
        let raw = df_expression(Arc::clone(&expression), &batch.schema(), &session)?.unwrap();
        let expected = expression
            .evaluate(&batch)?
            .into_array_of_size(batch.num_rows())?;
        for converted in [mixed, raw] {
            let actual = evaluate(
                &converted,
                &batch,
                expression.return_field(&batch.schema())?.as_ref(),
                &session,
            )?;
            assert_eq!(&actual, &expected);
        }
    }
    Ok(())
}

#[test]
fn df_expression_preserves_fields_through_datafusion_case_projection() -> Result<()> {
    let batch = RecordBatch::try_from_iter(vec![
        ("s", Arc::new(StringArray::from(vec!["a", "b"])) as ArrayRef),
        (
            "flag",
            Arc::new(BooleanArray::from(vec![true, false])) as ArrayRef,
        ),
        ("n", Arc::new(Int64Array::from(vec![10, 20])) as ArrayRef),
    ])?;
    let case: Arc<dyn PhysicalExpr> = Arc::new(CaseExpr::try_new(
        None,
        vec![(
            Arc::new(Column::new("flag", 1)) as Arc<dyn PhysicalExpr>,
            Arc::new(Column::new("n", 2)) as Arc<dyn PhysicalExpr>,
        )],
        Some(lit(0_i64)),
    )?);
    let expression: Arc<dyn PhysicalExpr> = Arc::new(ScalarFunctionExpr::try_new(
        Arc::new(ScalarUDF::from(NamedStructFunc::new())),
        vec![lit("s"), Arc::new(Column::new("s", 0)), lit("v"), case],
        &batch.schema(),
        Arc::new(ConfigOptions::new()),
    )?);
    let session = VortexSession::default();
    let mixed = DefaultExpressionConverter::new(session.clone())
        .try_convert(&expression, &batch.schema())?
        .unwrap();
    let raw = df_expression(Arc::clone(&expression), &batch.schema(), &session)?.unwrap();
    let expected = expression
        .evaluate(&batch)?
        .into_array_of_size(batch.num_rows())?;
    for converted in [mixed, raw] {
        let actual = evaluate(
            &converted,
            &batch,
            expression.return_field(&batch.schema())?.as_ref(),
            &session,
        )?;
        assert_eq!(&actual, &expected);
    }
    Ok(())
}
