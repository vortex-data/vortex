// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::hash::Hash;

use itertools::Itertools as _;
use vortex_array::arrays::StructArray;
use vortex_array::validity::Validity;
use vortex_array::{ArrayRef, DeserializeMetadata, IntoArray, ProstMetadata};
use vortex_dtype::{DType, FieldName, FieldNames, Nullability, StructFields};
use vortex_error::{VortexExpect as _, VortexResult, vortex_bail, vortex_err};
use vortex_proto::expr as pb;

use crate::{AnalysisExpr, ExprEncodingRef, ExprId, ExprRef, IntoExpr, Scope, VTable, vtable};

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
///     .unwrap()
///     .field_by_name("x copy")
///     .unwrap()
///     .clone();
/// assert_eq!(x_copy.scalar_at(0).unwrap(), Scalar::from(100));
/// assert_eq!(x_copy.scalar_at(1).unwrap(), Scalar::from(110));
/// assert_eq!(x_copy.scalar_at(2).unwrap(), Scalar::from(200));
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
            .map(|value_expr| value_expr.unchecked_evaluate(scope))
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
            .process_results(|it| it.collect())?;
        Ok(DType::Struct(
            StructFields::new(expr.names.clone(), value_dtypes),
            expr.nullability,
        ))
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

impl Display for PackExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
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
}

impl AnalysisExpr for PackExpr {}

#[cfg(test)]
mod tests {

    use vortex_array::arrays::{PrimitiveArray, StructArray};
    use vortex_array::validity::Validity;
    use vortex_array::vtable::ValidityHelper;
    use vortex_array::{Array, ArrayRef, IntoArray, ToCanonical};
    use vortex_buffer::buffer;
    use vortex_dtype::{FieldNames, Nullability};
    use vortex_error::{VortexResult, vortex_bail};

    use crate::{IntoExpr, PackExpr, Scope, col};

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

        let mut array = array.to_struct()?.field_by_name(field)?.clone();
        for field in field_path {
            array = array.to_struct()?.field_by_name(field)?.clone();
        }
        Ok(array.to_primitive().unwrap())
    }

    #[test]
    pub fn test_empty_pack() {
        let expr =
            PackExpr::try_new(FieldNames::default(), Vec::new(), Nullability::NonNullable).unwrap();

        let test_array = test_array();
        let actual_array = expr.evaluate(&Scope::new(test_array.clone())).unwrap();
        assert_eq!(actual_array.len(), test_array.len());
        assert_eq!(
            actual_array.to_struct().unwrap().struct_fields().nfields(),
            0
        );
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
            .to_struct()
            .unwrap();

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
            .to_struct()
            .unwrap();

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
            .to_struct()
            .unwrap();

        assert_eq!(actual_array.names(), ["one", "two", "three"]);
        assert_eq!(actual_array.validity(), &Validity::AllValid);
    }
}
