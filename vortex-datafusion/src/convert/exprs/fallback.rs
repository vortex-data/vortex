// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Lower one DF region, extracting native subtrees without changing evaluation domains.

use std::sync::Arc;

use arrow_schema::DataType;
use arrow_schema::FieldRef;
use arrow_schema::Schema;
use datafusion_common::HashMap;
use datafusion_expr::Operator;
use datafusion_functions::core::getfield::GetFieldFunc;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_expr::ScalarFunctionExpr;
use datafusion_physical_plan::expressions::BinaryExpr;
use datafusion_physical_plan::expressions::CastExpr;
use datafusion_physical_plan::expressions::Column;
use datafusion_physical_plan::expressions::IsNotNullExpr;
use datafusion_physical_plan::expressions::IsNullExpr;
use datafusion_physical_plan::expressions::Literal;
use datafusion_physical_plan::expressions::NegativeExpr;
use datafusion_physical_plan::expressions::NotExpr;
use datafusion_physical_plan::expressions::TryCastExpr;
use datafusion_physical_plan::joins::HashExpr;
use datafusion_physical_plan::joins::HashTableLookupExpr;
use vortex::expr::Expression;
use vortex::expr::get_item;
use vortex::expr::root;

use super::Conversion;
use super::DefaultExpressionConverter;
use super::Unconverted;
use super::is_infallible;

pub(super) fn eager(expression: &Arc<dyn PhysicalExpr>) -> bool {
    expression.is::<Column>() || expression.is::<Literal>() || expression.is::<CastExpr>()
        || expression.is::<TryCastExpr>() || expression.is::<NegativeExpr>()
        || expression.is::<NotExpr>() || expression.is::<IsNullExpr>() || expression.is::<IsNotNullExpr>()
        // These concrete DF implementations evaluate all children over the input batch.
        || expression.is::<ScalarFunctionExpr>() || expression.is::<HashExpr>() || expression.is::<HashTableLookupExpr>()
        || expression.downcast_ref::<BinaryExpr>().is_some_and(|binary| !matches!(binary.op(), Operator::And | Operator::Or))
}

pub(super) fn compatible(expression: &Arc<dyn PhysicalExpr>, schema: &Schema) -> Conversion<bool> {
    if let Some(binary) = expression.downcast_ref::<BinaryExpr>() {
        let left = binary.left().data_type(schema)?;
        let right = binary.right().data_type(schema)?;
        let floating = |dtype: &DataType| match dtype {
            DataType::Dictionary(_, values) => values.is_floating(),
            other => other.is_floating(),
        };
        if matches!(
            binary.op(),
            Operator::Eq
                | Operator::NotEq
                | Operator::Lt
                | Operator::LtEq
                | Operator::Gt
                | Operator::GtEq
        ) && (floating(&left) || floating(&right))
        {
            return Ok(false);
        }
        // DF may wrap integer arithmetic; the Vortex operation is checked. DF does
        // not expose that expression option, so preserve either behavior in DF.
        if matches!(
            binary.op(),
            Operator::Plus | Operator::Minus | Operator::Multiply
        ) && (left.is_integer() || right.is_integer())
        {
            return Ok(false);
        }
    }
    // Native extraction is equivalent only when every parent struct is non-null.
    if let Some(function) = expression.downcast_ref::<ScalarFunctionExpr>()
        && ScalarFunctionExpr::try_downcast_func::<GetFieldFunc>(function).is_some()
    {
        let [source, paths @ ..] = function.args() else {
            return Ok(true);
        };
        let mut dtype = source.data_type(schema)?;
        let mut nullable = source.nullable(schema)?;
        for path in paths {
            let Some(name) = path
                .downcast_ref::<Literal>()
                .and_then(|literal| literal.value().try_as_str())
                .flatten()
            else {
                return Ok(true);
            };
            let DataType::Struct(fields) = &dtype else {
                return Ok(true);
            };
            let Some((_, field)) = fields.find(name) else {
                return Ok(true);
            };
            if nullable {
                return Ok(false);
            }
            nullable = field.is_nullable();
            dtype = field.data_type().clone();
        }
    }
    Ok(true)
}

pub(super) fn lower(
    converter: &DefaultExpressionConverter,
    expression: &Arc<dyn PhysicalExpr>,
    schema: &Schema,
    advisory: bool,
) -> Conversion<Expression> {
    super::validate_columns(expression, schema)?;
    if !crate::df_expression::eligible(expression) {
        return Err(Unconverted::Unsupported);
    }
    if crate::df_expression::contains_higher_order_function(expression) {
        return crate::df_expression::raw_expression(
            Arc::clone(expression),
            schema,
            &converter.session,
            advisory,
        )?
        .ok_or(Unconverted::Unsupported);
    }
    let native = converter.native_only();
    let mut extraction = Extraction {
        native: &native,
        schema,
        advisory,
        inputs: vec![],
        fields: vec![],
        originals: vec![],
        columns: HashMap::default(),
        arrays: HashMap::default(),
    };
    let expression = extraction.rewrite(expression, true, false)?;
    crate::df_expression::from_inputs(
        expression,
        Arc::new(Schema::new(extraction.fields)),
        extraction.inputs,
        &converter.session,
        advisory,
    )?
    .ok_or(Unconverted::Unsupported)
}

struct Extraction<'a> {
    native: &'a DefaultExpressionConverter,
    schema: &'a Schema,
    advisory: bool,
    inputs: Vec<Expression>,
    fields: Vec<FieldRef>,
    originals: Vec<Arc<dyn PhysicalExpr>>,
    columns: HashMap<usize, bool>,
    arrays: HashMap<usize, bool>,
}
impl Extraction<'_> {
    fn has_columns(&mut self, expression: &Arc<dyn PhysicalExpr>) -> bool {
        let key = Arc::as_ptr(expression) as *const () as usize;
        if let Some(value) = self.columns.get(&key) {
            return *value;
        }
        let value = expression.is::<Column>()
            || expression
                .children()
                .iter()
                .any(|child| self.has_columns(child));
        self.columns.insert(key, value);
        value
    }
    // DF may return Scalar from short-circuit expressions, conditional regions,
    // or arbitrary UDFs even when they reference a column. Extract only concrete
    // operators whose evaluator preserves Array shape from an Array operand.
    fn array_valued(&mut self, expression: &Arc<dyn PhysicalExpr>) -> bool {
        let key = Arc::as_ptr(expression) as *const () as usize;
        if let Some(value) = self.arrays.get(&key) {
            return *value;
        }
        let preserves_array = expression.is::<CastExpr>()
            || expression.is::<TryCastExpr>()
            || expression.is::<NegativeExpr>()
            || expression.is::<NotExpr>()
            || expression.is::<IsNullExpr>()
            || expression.is::<IsNotNullExpr>()
            || expression
                .downcast_ref::<BinaryExpr>()
                .is_some_and(|binary| !matches!(binary.op(), Operator::And | Operator::Or));
        let value = expression.is::<Column>()
            || preserves_array
                && expression
                    .children()
                    .iter()
                    .any(|child| self.array_valued(child));
        self.arrays.insert(key, value);
        value
    }
    fn input(
        &mut self,
        original: &Arc<dyn PhysicalExpr>,
        native: Expression,
    ) -> Conversion<Arc<dyn PhysicalExpr>> {
        let index = if let Some(index) = self
            .originals
            .iter()
            .position(|existing| existing.eq(original))
        {
            index
        } else {
            let index = self.inputs.len();
            self.fields.push(original.return_field(self.schema)?);
            self.originals.push(Arc::clone(original));
            self.inputs.push(native);
            index
        };
        Ok(Arc::new(Column::new(self.fields[index].name(), index)))
    }
    fn rewrite(
        &mut self,
        expression: &Arc<dyn PhysicalExpr>,
        eager_region: bool,
        child: bool,
    ) -> Conversion<Arc<dyn PhysicalExpr>> {
        if let Some(column) = expression.downcast_ref::<Column>() {
            return self.input(expression, get_item(column.name(), root()));
        }
        // Keep scalar-producing constant regions in DF: turning them into
        // input columns changes ColumnarValue::Scalar into Array for a UDF.
        if expression.is::<Literal>() || !self.has_columns(expression) {
            return Ok(Arc::clone(expression));
        }
        if child && self.array_valued(expression) {
            match self.native.convert_checked(expression, self.schema) {
                Ok(native)
                    if (eager_region || is_infallible(&native))
                        && (!self.advisory || is_infallible(&native)) =>
                {
                    // DF UDFs observe the original Field, including nullability.
                    let field = expression.return_field(self.schema)?;
                    if field.is_nullable() == expression.nullable(self.schema)? {
                        return self.input(expression, native);
                    }
                }
                Ok(_) | Err(Unconverted::Unsupported) => {}
                Err(error) => return Err(error),
            }
        }
        let allow = eager_region && eager(expression);
        let children = expression
            .children()
            .into_iter()
            .map(|child| self.rewrite(child, allow, true))
            .collect::<Conversion<Vec<_>>>()?;
        let rewritten = Arc::clone(expression).with_new_children(children)?;
        crate::df_expression::preserve_field(
            expression,
            rewritten,
            self.schema,
            &Schema::new(self.fields.clone()),
        )
        .map_err(Unconverted::Failed)
    }
}
