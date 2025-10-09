// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools as _;
use vortex_array::arrays::ChunkedArray;
use vortex_array::{ArrayRef, DeserializeMetadata, EmptyMetadata, IntoArray};
use vortex_dtype::DType;
use vortex_error::{VortexResult, vortex_bail};

use crate::display::{DisplayAs, DisplayFormat};
use crate::{AnalysisExpr, ExprEncodingRef, ExprId, ExprRef, IntoExpr, Scope, VTable, vtable};

vtable!(Concat);

/// Concatenate zero or more expressions into a single array.
///
/// All child expressions must evaluate to arrays of the same dtype.
///
/// # Examples
///
/// ```
/// use vortex_array::{IntoArray, ToCanonical};
/// use vortex_buffer::buffer;
/// use vortex_expr::{ConcatExpr, Scope, lit};
/// use vortex_scalar::Scalar;
///
/// let example = ConcatExpr::new(vec![
///     lit(Scalar::from(100)),
///     lit(Scalar::from(200)),
///     lit(Scalar::from(300)),
/// ]);
/// let concatenated = example.evaluate(&Scope::empty(1)).unwrap();
/// assert_eq!(concatenated.len(), 3);
/// ```
#[allow(clippy::derived_hash_with_manual_eq)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ConcatExpr {
    values: Vec<ExprRef>,
}

pub struct ConcatExprEncoding;

impl VTable for ConcatVTable {
    type Expr = ConcatExpr;
    type Encoding = ConcatExprEncoding;
    type Metadata = EmptyMetadata;

    fn id(_encoding: &Self::Encoding) -> ExprId {
        ExprId::new_ref("concat")
    }

    fn encoding(_expr: &Self::Expr) -> ExprEncodingRef {
        ExprEncodingRef::new_ref(ConcatExprEncoding.as_ref())
    }

    fn metadata(_expr: &Self::Expr) -> Option<Self::Metadata> {
        Some(EmptyMetadata)
    }

    fn children(expr: &Self::Expr) -> Vec<&ExprRef> {
        expr.values.iter().collect()
    }

    fn with_children(_expr: &Self::Expr, children: Vec<ExprRef>) -> VortexResult<Self::Expr> {
        Ok(ConcatExpr { values: children })
    }

    fn build(
        _encoding: &Self::Encoding,
        _metadata: &<Self::Metadata as DeserializeMetadata>::Output,
        children: Vec<ExprRef>,
    ) -> VortexResult<Self::Expr> {
        Ok(ConcatExpr { values: children })
    }

    fn evaluate(expr: &Self::Expr, scope: &Scope) -> VortexResult<ArrayRef> {
        if expr.values.is_empty() {
            vortex_bail!("Concat expression must have at least one child");
        }

        let value_arrays = expr
            .values
            .iter()
            .map(|value_expr| value_expr.unchecked_evaluate(scope))
            .process_results(|it| it.collect::<Vec<_>>())?;

        // Get the common dtype from the first array
        let dtype = value_arrays[0].dtype().clone();

        // Validate all arrays have the same dtype
        for array in &value_arrays[1..] {
            if array.dtype() != &dtype {
                vortex_bail!(
                    "All arrays in concat must have the same dtype, expected {:?} but got {:?}",
                    dtype,
                    array.dtype()
                );
            }
        }

        Ok(ChunkedArray::try_new(value_arrays, dtype)?.into_array())
    }

    fn return_dtype(expr: &Self::Expr, scope: &DType) -> VortexResult<DType> {
        if expr.values.is_empty() {
            vortex_bail!("Concat expression must have at least one child");
        }

        // Return the dtype of the first child - all children must have the same dtype
        let dtype = expr.values[0].return_dtype(scope)?;

        // Validate all children have the same dtype
        for value_expr in &expr.values[1..] {
            let child_dtype = value_expr.return_dtype(scope)?;
            if child_dtype != dtype {
                vortex_bail!(
                    "All expressions in concat must return the same dtype, expected {:?} but got {:?}",
                    dtype,
                    child_dtype
                );
            }
        }

        Ok(dtype)
    }
}

impl ConcatExpr {
    pub fn new(values: Vec<ExprRef>) -> Self {
        ConcatExpr { values }
    }

    pub fn new_expr(values: Vec<ExprRef>) -> ExprRef {
        Self::new(values).into_expr()
    }

    pub fn values(&self) -> &[ExprRef] {
        &self.values
    }
}

/// Creates an expression that concatenates multiple expressions into a single array.
///
/// All input expressions must evaluate to arrays of the same dtype.
///
/// ```rust
/// # use vortex_expr::{concat, col, lit};
/// # use vortex_scalar::Scalar;
/// let expr = concat([col("chunk1"), col("chunk2"), lit(Scalar::from(42))]);
/// ```
pub fn concat(elements: impl IntoIterator<Item = impl Into<ExprRef>>) -> ExprRef {
    let values = elements.into_iter().map(|value| value.into()).collect_vec();
    ConcatExpr::new(values).into_expr()
}

impl DisplayAs for ConcatExpr {
    fn fmt_as(&self, df: DisplayFormat, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match df {
            DisplayFormat::Compact => {
                write!(f, "concat({})", self.values.iter().format(", "))
            }
            DisplayFormat::Tree => {
                write!(f, "Concat")
            }
        }
    }
}

impl AnalysisExpr for ConcatExpr {}

#[cfg(test)]
mod tests {
    use vortex_array::arrays::ChunkedVTable;
    use vortex_array::{IntoArray, ToCanonical};
    use vortex_buffer::buffer;
    use vortex_dtype::{DType, Nullability, PType};

    use crate::{ConcatExpr, Scope, col, concat, lit, root};

    fn test_array() -> vortex_array::ArrayRef {
        vortex_array::arrays::StructArray::from_fields(&[
            ("a", buffer![1, 2, 3].into_array()),
            ("b", buffer![4, 5, 6].into_array()),
        ])
        .unwrap()
        .into_array()
    }

    #[test]
    pub fn test_concat_literals() {
        let expr = ConcatExpr::new(vec![
            lit(vortex_scalar::Scalar::from(1i32)),
            lit(vortex_scalar::Scalar::from(2i32)),
            lit(vortex_scalar::Scalar::from(3i32)),
        ]);

        // Literals expand to scope.len(), so use a scope of len 1
        let scope_array = buffer![0i32].into_array();
        let actual_array = expr.evaluate(&Scope::new(scope_array)).unwrap();

        let chunked = actual_array.as_::<ChunkedVTable>();
        assert_eq!(chunked.nchunks(), 3);
        assert_eq!(chunked.len(), 3);

        let canonical = chunked.to_canonical().into_array();
        let primitive = canonical.to_primitive();
        assert_eq!(primitive.as_slice::<i32>(), &[1, 2, 3]);
    }

    #[test]
    pub fn test_concat_columns() {
        let expr = ConcatExpr::new(vec![col("a"), col("b"), col("a")]);

        let actual_array = expr.evaluate(&Scope::new(test_array())).unwrap();

        let chunked = actual_array.as_::<ChunkedVTable>();
        assert_eq!(chunked.nchunks(), 3);
        assert_eq!(chunked.len(), 9);

        let canonical = chunked.to_canonical().into_array();
        let primitive = canonical.to_primitive();
        assert_eq!(primitive.as_slice::<i32>(), &[1, 2, 3, 4, 5, 6, 1, 2, 3]);
    }

    #[test]
    pub fn test_concat_mixed() {
        let expr = ConcatExpr::new(vec![
            col("a"),
            lit(vortex_scalar::Scalar::from(99i32)),
            col("b"),
        ]);

        let actual_array = expr.evaluate(&Scope::new(test_array())).unwrap();

        let chunked = actual_array.as_::<ChunkedVTable>();
        assert_eq!(chunked.nchunks(), 3);
        // len = 3 (col a) + 3 (lit 99 expanded to scope.len()) + 3 (col b) = 9
        assert_eq!(chunked.len(), 9);

        let canonical = chunked.to_canonical().into_array();
        let primitive = canonical.to_primitive();
        assert_eq!(primitive.as_slice::<i32>(), &[1, 2, 3, 99, 99, 99, 4, 5, 6]);
    }

    #[test]
    pub fn test_concat_dtype_mismatch() {
        let expr = ConcatExpr::new(vec![
            lit(vortex_scalar::Scalar::from(1i32)),
            lit(vortex_scalar::Scalar::from(2i64)),
        ]);

        let result = expr.evaluate(&Scope::new(test_array()));
        assert!(result.is_err());
    }

    #[test]
    pub fn test_return_dtype() {
        let expr = ConcatExpr::new(vec![
            lit(vortex_scalar::Scalar::from(1i32)),
            lit(vortex_scalar::Scalar::from(2i32)),
        ]);

        let dtype = expr
            .return_dtype(&DType::Primitive(PType::I32, Nullability::NonNullable))
            .unwrap();

        assert_eq!(
            dtype,
            DType::Primitive(PType::I32, Nullability::NonNullable)
        );
    }

    #[test]
    pub fn test_display() {
        let expr = concat([col("a"), col("b"), col("c")]);
        assert_eq!(expr.to_string(), "concat($.a, $.b, $.c)");
    }

    #[test]
    pub fn test_concat_with_root() {
        let expr = concat([root(), root()]);

        let test_array = buffer![1, 2, 3].into_array();
        let actual_array = expr.evaluate(&Scope::new(test_array)).unwrap();

        let chunked = actual_array.as_::<ChunkedVTable>();
        assert_eq!(chunked.nchunks(), 2);
        assert_eq!(chunked.len(), 6);

        let canonical = chunked.to_canonical().into_array();
        let primitive = canonical.to_primitive();
        assert_eq!(primitive.as_slice::<i32>(), &[1, 2, 3, 1, 2, 3]);
    }
}
