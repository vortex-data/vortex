// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use async_trait::async_trait;
use futures::future::try_join_all;
use futures::try_join;
use itertools::Itertools as _;
use std::any::Any;
use std::hash::Hash;
use std::sync::Arc;
use vortex_array::arrays::StructArray;
use vortex_array::operator::getitem::GetItemOperator;
use vortex_array::operator::{
    BatchBindCtx, BatchExecution, BatchExecutionRef, BatchOperator, MaskExecution, Operator,
    OperatorEq, OperatorHash, OperatorId, OperatorRef,
};
use vortex_array::validity::Validity;
use vortex_array::{ArrayRef, Canonical, DeserializeMetadata, IntoArray, ProstMetadata};
use vortex_dtype::{DType, FieldName, FieldNames, Nullability, StructFields};
use vortex_error::{vortex_bail, vortex_err, VortexExpect as _, VortexResult};
use vortex_proto::expr as pb;

use crate::display::{DisplayAs, DisplayFormat};
use crate::{vtable, AnalysisExpr, ExprEncodingRef, ExprId, ExprRef, IntoExpr, Scope, VTable};

vtable!(Pack);

/// Pack zero or more expressions into a structure with named fields.
///
/// # Examples
///
/// ```
/// use vortex_array::{IntoArray, ToCanonical};
/// use vortex_buffer::buffer;
/// use vortex_expr::{root, PackExpr, Scope, VortexExpr};
/// use vortex_scalar::Scalar;
/// use vortex_dtype::Nullability;
///
/// let example = PackExpr::try_new(
///     ["x", "x copy", "second x copy"].into(),
///     vec![root(), root(), root()],
///     Nullability::NonNullable,
/// ).unwrap();
/// let packed = example.evaluate(&Scope::new(buffer![100, 110, 200].into_array())).unwrap();
/// let x_copy = packed
///     .to_struct()
///     .field_by_name("x copy")
///     .unwrap()
///     .clone();
/// assert_eq!(x_copy.scalar_at(0), Scalar::from(100));
/// assert_eq!(x_copy.scalar_at(1), Scalar::from(110));
/// assert_eq!(x_copy.scalar_at(2), Scalar::from(200));
/// ```
///
#[allow(clippy::derived_hash_with_manual_eq)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PackExpr {
    names: FieldNames,
    values: Vec<ExprRef>,
    nullability: Nullability,
}

pub struct PackExprEncoding;

impl VTable for PackVTable {
    type Expr = PackExpr;
    type Encoding = PackExprEncoding;
    type Metadata = ProstMetadata<pb::PackOpts>;

    fn id(_encoding: &Self::Encoding) -> ExprId {
        ExprId::new_ref("pack")
    }

    fn encoding(_expr: &Self::Expr) -> ExprEncodingRef {
        ExprEncodingRef::new_ref(PackExprEncoding.as_ref())
    }

    fn metadata(expr: &Self::Expr) -> Option<Self::Metadata> {
        Some(ProstMetadata(pb::PackOpts {
            paths: expr.names.iter().map(|n| n.to_string()).collect(),
            nullable: expr.nullability.into(),
        }))
    }

    fn children(expr: &Self::Expr) -> Vec<&ExprRef> {
        expr.values.iter().collect()
    }

    fn with_children(expr: &Self::Expr, children: Vec<ExprRef>) -> VortexResult<Self::Expr> {
        PackExpr::try_new(expr.names.clone(), children, expr.nullability)
    }

    fn build(
        _encoding: &Self::Encoding,
        metadata: &<Self::Metadata as DeserializeMetadata>::Output,
        children: Vec<ExprRef>,
    ) -> VortexResult<Self::Expr> {
        if children.len() != metadata.paths.len() {
            vortex_bail!(
                "Pack expression expects {} children, got {}",
                metadata.paths.len(),
                children.len()
            );
        }
        let names: FieldNames = metadata
            .paths
            .iter()
            .map(|name| FieldName::from(name.as_str()))
            .collect();
        PackExpr::try_new(names, children, metadata.nullable.into())
    }

    fn evaluate(expr: &Self::Expr, scope: &Scope) -> VortexResult<ArrayRef> {
        let len = scope.len();
        let value_arrays = expr
            .values
            .iter()
            .zip_eq(expr.names.iter())
            .map(|(value_expr, name)| {
                value_expr
                    .unchecked_evaluate(scope)
                    .map_err(|e| e.with_context(format!("Can't evaluate '{name}'")))
            })
            .process_results(|it| it.collect::<Vec<_>>())?;
        let validity = match expr.nullability {
            Nullability::NonNullable => Validity::NonNullable,
            Nullability::Nullable => Validity::AllValid,
        };
        Ok(StructArray::try_new(expr.names.clone(), value_arrays, len, validity)?.into_array())
    }

    fn return_dtype(expr: &Self::Expr, scope: &DType) -> VortexResult<DType> {
        let value_dtypes = expr
            .values
            .iter()
            .map(|value_expr| value_expr.return_dtype(scope))
            .collect::<VortexResult<Vec<_>>>()?;
        Ok(DType::Struct(
            StructFields::new(expr.names.clone(), value_dtypes),
            expr.nullability,
        ))
    }

    fn operator(expr: &Self::Expr, scope: &OperatorRef) -> VortexResult<Option<OperatorRef>> {
        let mut values = Vec::with_capacity(expr.values.len());
        for value in &expr.values {
            if let Some(op) = value.operator(scope)? {
                values.push(op);
            } else {
                // One of the children cannot be converted to an operator, so the whole pack cannot
                return Ok(None);
            }
        }

        Ok(Some(Arc::new(PackOperator::try_new(
            expr.names.clone(),
            values,
            expr.nullability,
        )?)))
    }
}

impl PackExpr {
    pub fn try_new(
        names: FieldNames,
        values: Vec<ExprRef>,
        nullability: Nullability,
    ) -> VortexResult<Self> {
        if names.len() != values.len() {
            vortex_bail!("length mismatch {} {}", names.len(), values.len());
        }
        Ok(PackExpr {
            names,
            values,
            nullability,
        })
    }

    pub fn try_new_expr(
        names: FieldNames,
        values: Vec<ExprRef>,
        nullability: Nullability,
    ) -> VortexResult<ExprRef> {
        Self::try_new(names, values, nullability).map(|v| v.into_expr())
    }

    pub fn names(&self) -> &FieldNames {
        &self.names
    }

    pub fn field(&self, field_name: &FieldName) -> VortexResult<ExprRef> {
        let idx = self
            .names
            .iter()
            .position(|name| name == field_name)
            .ok_or_else(|| {
                vortex_err!(
                    "Cannot find field {} in pack fields {:?}",
                    field_name,
                    self.names
                )
            })?;

        self.values
            .get(idx)
            .cloned()
            .ok_or_else(|| vortex_err!("field index out of bounds: {}", idx))
    }

    pub fn nullability(&self) -> Nullability {
        self.nullability
    }
}

/// Creates an expression that packs values into a struct with named fields.
///
/// ```rust
/// # use vortex_dtype::Nullability;
/// # use vortex_expr::{pack, col, lit};
/// let expr = pack([("id", col("user_id")), ("constant", lit(42))], Nullability::NonNullable);
/// ```
pub fn pack(
    elements: impl IntoIterator<Item = (impl Into<FieldName>, ExprRef)>,
    nullability: Nullability,
) -> ExprRef {
    let (names, values): (Vec<_>, Vec<_>) = elements
        .into_iter()
        .map(|(name, value)| (name.into(), value))
        .unzip();
    PackExpr::try_new(names.into(), values, nullability)
        .vortex_expect("pack names and values have the same length")
        .into_expr()
}

impl DisplayAs for PackExpr {
    fn fmt_as(&self, df: DisplayFormat, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match df {
            DisplayFormat::Compact => {
                write!(
                    f,
                    "pack({}){}",
                    self.names
                        .iter()
                        .zip(&self.values)
                        .format_with(", ", |(name, expr), f| f(&format_args!("{name}: {expr}"))),
                    self.nullability
                )
            }
            DisplayFormat::Tree => {
                write!(f, "Pack")
            }
        }
    }

    fn child_names(&self) -> Option<Vec<String>> {
        Some(self.names.iter().map(|n| n.to_string()).collect())
    }
}

impl AnalysisExpr for PackExpr {}

#[derive(Debug)]
pub struct PackOperator {
    names: FieldNames,
    values: Vec<OperatorRef>,
    nullability: Nullability,

    dtype: DType,
    len: usize,
}

impl PackOperator {
    pub fn try_new(
        names: FieldNames,
        values: Vec<OperatorRef>,
        nullability: Nullability,
    ) -> VortexResult<Self> {
        if names.len() != values.len() {
            vortex_bail!("length mismatch {} {}", names.len(), values.len());
        }
        let len = values
            .iter()
            .map(|v| v.len())
            .all_equal_value()
            .map_err(|_| vortex_err!("length mismatch among values"))?;
        let value_dtypes = values.iter().map(|v| v.dtype().clone()).collect();
        let dtype = DType::Struct(StructFields::new(names.clone(), value_dtypes), nullability);
        Ok(PackOperator {
            names,
            values,
            nullability,
            dtype,
            len,
        })
    }
}

impl OperatorHash for PackOperator {
    fn operator_hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.names.hash(state);
        self.nullability.hash(state);
        for value in &self.values {
            value.operator_hash(state);
        }
    }
}

impl OperatorEq for PackOperator {
    fn operator_eq(&self, other: &Self) -> bool {
        self.names == other.names
            && self.nullability == other.nullability
            && self.values.len() == other.values.len()
            && self
                .values
                .iter()
                .zip(&other.values)
                .all(|(a, b)| a.operator_eq(b))
    }
}

impl Operator for PackOperator {
    fn id(&self) -> OperatorId {
        OperatorId::from("vortex.pack")
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn len(&self) -> usize {
        self.len
    }

    fn children(&self) -> &[OperatorRef] {
        &self.values
    }

    fn with_children(self: Arc<Self>, children: Vec<OperatorRef>) -> VortexResult<OperatorRef> {
        Ok(Arc::new(PackOperator::try_new(
            self.names.clone(),
            children,
            self.nullability,
        )?))
    }

    fn reduce_parent(
        &self,
        parent: OperatorRef,
        _child_idx: usize,
    ) -> VortexResult<Option<OperatorRef>> {
        if parent.as_any().downcast_ref::<GetItemOperator>().is_some() {
            vortex_bail!("TODO: PackOperator should reduce a GetItemOperator into its children");
        }

        Ok(None)
    }

    fn as_batch(&self) -> Option<&dyn BatchOperator> {
        Some(self)
    }
}

impl BatchOperator for PackOperator {
    fn project(
        &self,
        mask: &OperatorRef,
        ctx: &mut dyn BatchBindCtx,
    ) -> VortexResult<BatchExecutionRef> {
        let values: Vec<_> = self
            .values
            .iter()
            .map(|value| ctx.bind_project(value, Some(mask)))
            .try_collect()?;

        let DType::Struct(fields, nullability) = &self.dtype else {
            vortex_bail!("PackOperator must have Struct dtype");
        };

        let validity = match nullability {
            Nullability::NonNullable => Validity::NonNullable,
            Nullability::Nullable => Validity::AllValid,
        };

        Ok(Box::new(PackExecution {
            values,
            mask: ctx.bind_mask(mask)?,
            fields: fields.clone(),
            validity,
        }))
    }
}

struct PackExecution {
    values: Vec<BatchExecutionRef>,
    mask: MaskExecution,
    fields: StructFields,
    validity: Validity,
}

#[async_trait]
impl BatchExecution for PackExecution {
    async fn execute(self: Box<Self>) -> VortexResult<Canonical> {
        let values = try_join_all(self.values.into_iter().map(|exec| exec.execute()));
        let (values, mask) = try_join!(values, self.mask)?;
        let values = values.into_iter().map(|c| c.into_array()).collect();

        Ok(Canonical::Struct(StructArray::try_new_with_dtype(
            values,
            self.fields.clone(),
            mask.true_count(),
            self.validity,
        )?))
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::arrays::{PrimitiveArray, StructArray};
    use vortex_array::validity::Validity;
    use vortex_array::vtable::ValidityHelper;
    use vortex_array::{Array, ArrayRef, IntoArray, ToCanonical};
    use vortex_buffer::buffer;
    use vortex_dtype::{FieldNames, Nullability};
    use vortex_error::{vortex_bail, VortexResult};

    use crate::{col, pack, IntoExpr, PackExpr, Scope};

    fn test_array() -> ArrayRef {
        StructArray::from_fields(&[
            ("a", buffer![0, 1, 2].into_array()),
            ("b", buffer![4, 5, 6].into_array()),
        ])
        .unwrap()
        .into_array()
    }

    fn primitive_field(array: &dyn Array, field_path: &[&str]) -> VortexResult<PrimitiveArray> {
        let mut field_path = field_path.iter();

        let Some(field) = field_path.next() else {
            vortex_bail!("empty field path");
        };

        let mut array = array.to_struct().field_by_name(field)?.clone();
        for field in field_path {
            array = array.to_struct().field_by_name(field)?.clone();
        }
        Ok(array.to_primitive())
    }

    #[test]
    pub fn test_empty_pack() {
        let expr =
            PackExpr::try_new(FieldNames::default(), Vec::new(), Nullability::NonNullable).unwrap();

        let test_array = test_array();
        let actual_array = expr.evaluate(&Scope::new(test_array.clone())).unwrap();
        assert_eq!(actual_array.len(), test_array.len());
        assert_eq!(actual_array.to_struct().struct_fields().nfields(), 0);
    }

    #[test]
    pub fn test_simple_pack() {
        let expr = PackExpr::try_new(
            ["one", "two", "three"].into(),
            vec![col("a"), col("b"), col("a")],
            Nullability::NonNullable,
        )
        .unwrap();

        let actual_array = expr
            .evaluate(&Scope::new(test_array()))
            .unwrap()
            .to_struct();

        assert_eq!(actual_array.names(), ["one", "two", "three"]);
        assert_eq!(actual_array.validity(), &Validity::NonNullable);

        assert_eq!(
            primitive_field(actual_array.as_ref(), &["one"])
                .unwrap()
                .as_slice::<i32>(),
            [0, 1, 2]
        );
        assert_eq!(
            primitive_field(actual_array.as_ref(), &["two"])
                .unwrap()
                .as_slice::<i32>(),
            [4, 5, 6]
        );
        assert_eq!(
            primitive_field(actual_array.as_ref(), &["three"])
                .unwrap()
                .as_slice::<i32>(),
            [0, 1, 2]
        );
    }

    #[test]
    pub fn test_nested_pack() {
        let expr = PackExpr::try_new(
            ["one", "two", "three"].into(),
            vec![
                col("a"),
                PackExpr::try_new(
                    ["two_one", "two_two"].into(),
                    vec![col("b"), col("b")],
                    Nullability::NonNullable,
                )
                .unwrap()
                .into_expr(),
                col("a"),
            ],
            Nullability::NonNullable,
        )
        .unwrap();

        let actual_array = expr
            .evaluate(&Scope::new(test_array()))
            .unwrap()
            .to_struct();

        assert_eq!(actual_array.names(), ["one", "two", "three"]);

        assert_eq!(
            primitive_field(actual_array.as_ref(), &["one"])
                .unwrap()
                .as_slice::<i32>(),
            [0, 1, 2]
        );
        assert_eq!(
            primitive_field(actual_array.as_ref(), &["two", "two_one"])
                .unwrap()
                .as_slice::<i32>(),
            [4, 5, 6]
        );
        assert_eq!(
            primitive_field(actual_array.as_ref(), &["two", "two_two"])
                .unwrap()
                .as_slice::<i32>(),
            [4, 5, 6]
        );
        assert_eq!(
            primitive_field(actual_array.as_ref(), &["three"])
                .unwrap()
                .as_slice::<i32>(),
            [0, 1, 2]
        );
    }

    #[test]
    pub fn test_pack_nullable() {
        let expr = PackExpr::try_new(
            ["one", "two", "three"].into(),
            vec![col("a"), col("b"), col("a")],
            Nullability::Nullable,
        )
        .unwrap();

        let actual_array = expr
            .evaluate(&Scope::new(test_array()))
            .unwrap()
            .to_struct();

        assert_eq!(actual_array.names(), ["one", "two", "three"]);
        assert_eq!(actual_array.validity(), &Validity::AllValid);
    }

    #[test]
    pub fn test_display() {
        let expr = pack(
            [("id", col("user_id")), ("name", col("username"))],
            Nullability::NonNullable,
        );
        assert_eq!(expr.to_string(), "pack(id: $.user_id, name: $.username)");

        let expr2 = PackExpr::try_new(
            ["x", "y"].into(),
            vec![col("a"), col("b")],
            Nullability::Nullable,
        )
        .unwrap();
        assert_eq!(expr2.to_string(), "pack(x: $.a, y: $.b)?");
    }
}
