use std::ops::Not;
use vortex_array::{
    ArrayRef, DeserializeMetadata, EmptyMetadata, IntoArray, ToCanonical, arrays::StructArray,
    compute::mask, validity::Validity,
};
use vortex_dtype::{DType, Nullability, StructFields};
use vortex_error::{VortexExpect as _, VortexResult, vortex_bail};

use crate::{
    AnalysisExpr, ExprEncodingRef, ExprId, ExprRef, IntoExpr as _, Scope, VTable,
    display::{DisplayAs, DisplayFormat},
    vtable,
};

/// Push top-level nulls of a struct into its fields yielding a non-null struct.
///
/// All arrays other than nullable struct arrays are unchanged.
///
/// See Also
/// --------
///
/// [`push_nulls`](push_nulls).
#[allow(clippy::derived_hash_with_manual_eq)]
#[derive(Debug, Clone, Eq, Hash)]
pub struct PushNullsExpr {
    child: ExprRef,
}

impl PartialEq for PushNullsExpr {
    fn eq(&self, other: &Self) -> bool {
        self.child.eq(&other.child)
    }
}

impl PushNullsExpr {
    pub fn new(child: ExprRef) -> Self {
        Self { child }
    }

    pub fn child(&self) -> &ExprRef {
        &self.child
    }
}

pub struct PushNullsExprEncoding;

impl AnalysisExpr for PushNullsExpr {}

impl DisplayAs for PushNullsExpr {
    fn fmt_as(&self, df: DisplayFormat, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match df {
            DisplayFormat::Compact => write!(f, "push_nulls({})", self.child),
            DisplayFormat::Tree => write!(f, "PushNulls"),
        }
    }
}

vtable!(PushNulls);

impl VTable for PushNullsVTable {
    type Expr = PushNullsExpr;

    type Encoding = PushNullsExprEncoding;

    type Metadata = EmptyMetadata;

    fn id(_encoding: &Self::Encoding) -> ExprId {
        ExprId::new_ref("push_nulls")
    }

    fn encoding(_expr: &Self::Expr) -> ExprEncodingRef {
        ExprEncodingRef::new_ref(PushNullsExprEncoding.as_ref())
    }

    fn metadata(_expr: &Self::Expr) -> Option<Self::Metadata> {
        Some(EmptyMetadata)
    }

    fn children(expr: &Self::Expr) -> Vec<&ExprRef> {
        vec![&expr.child]
    }

    fn with_children(_expr: &Self::Expr, children: Vec<ExprRef>) -> VortexResult<Self::Expr> {
        Self::build(&PushNullsExprEncoding, &EmptyMetadata, children)
    }

    fn build(
        _encoding: &Self::Encoding,
        _metadata: &<Self::Metadata as DeserializeMetadata>::Output,
        mut children: Vec<ExprRef>,
    ) -> VortexResult<Self::Expr> {
        if children.len() != 1 {
            vortex_bail!(
                "PushNulls should have exactly one child, found: {}",
                children.len()
            )
        }
        let child = children.pop().vortex_expect("verified length one above");
        Ok(PushNullsExpr { child })
    }

    fn evaluate(expr: &Self::Expr, scope: &Scope) -> VortexResult<ArrayRef> {
        let child = expr.child.unchecked_evaluate(scope)?;
        if !child.dtype().is_struct() {
            return Ok(child);
        }

        let child = child.to_struct();

        let top_level_invalidity = child.validity_mask().not();
        let new_fields = child
            .fields()
            .iter()
            .map(|field| mask(field, &top_level_invalidity))
            .collect::<Result<Vec<_>, _>>()?;
        StructArray::try_new(
            child.names().clone(),
            new_fields,
            child.len(),
            Validity::NonNullable,
        )
        .map(IntoArray::into_array)
    }

    fn return_dtype(expr: &Self::Expr, scope: &DType) -> VortexResult<DType> {
        let child = expr.child.return_dtype(scope)?;
        let Some(struct_fields) = child.as_struct_fields_opt() else {
            return Ok(child);
        };
        let top_level_nullability = child.nullability();
        let new_field_dtypes = struct_fields
            .fields()
            .map(|f| f.union_nullability(top_level_nullability))
            .collect();
        Ok(DType::Struct(
            StructFields::new(struct_fields.names().clone(), new_field_dtypes),
            Nullability::NonNullable,
        ))
    }
}

/// Push top-level nulls of a struct into its fields yielding a non-null struct.
///
/// All arrays other than nullable struct arrays are unchanged.
///
/// Examples
/// --------
///
/// Push top-level nulls of a struct with one integral field named a:
///
/// ```
/// use vortex_array::arrays::{StructArray};
/// use vortex_array::validity::Validity;
/// use vortex_array::{IntoArray};
/// use vortex_buffer::buffer;
/// use vortex_expr::push_nulls;
/// use vortex_expr::{Scope, root};
///
/// let array = StructArray::try_from_iter_with_validity(
///     [("a", buffer![0, 1, 2])],
///     Validity::from_iter([true, false, true]),
/// )
/// .unwrap();
///
/// let result = push_nulls(root())
///     .evaluate(&Scope::new(array.into_array()))
///     .unwrap();
/// assert_eq!(
///     result.display_values().to_string(),
///     "[{a: 0i32}, {a: null}, {a: 2i32}]",
/// );
/// ```
///
/// Push top-level nulls of a struct with one struct field named a:
///
/// ```
/// use vortex_array::arrays::{StructArray};
/// use vortex_array::validity::Validity;
/// use vortex_array::{IntoArray};
/// use vortex_buffer::buffer;
/// use vortex_expr::push_nulls;
/// use vortex_expr::{Scope, root};
///
/// let array = StructArray::try_from_iter_with_validity(
///     [(
///         "a",
///         StructArray::try_from_iter([("inner", buffer![0, 1, 2])]).unwrap(),
///     )],
///     Validity::from_iter([true, false, true]),
/// )
/// .unwrap();
///
/// let result = push_nulls(root())
///     .evaluate(&Scope::new(array.into_array()))
///     .unwrap();
/// assert_eq!(
///     result.display_values().to_string(),
///     "[{a: {inner: 0i32}}, {a: null}, {a: {inner: 2i32}}]",
/// );
/// ```
pub fn push_nulls(operand: ExprRef) -> ExprRef {
    PushNullsExpr::new(operand).into_expr()
}

#[cfg(test)]
mod tests {
    use vortex_array::IntoArray;
    use vortex_array::arrays::{PrimitiveArray, StructArray, VarBinArray};
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;
    use vortex_dtype::FieldNames;

    use crate::push_nulls;
    use crate::{Scope, root};

    #[test]
    pub fn test_push_nulls_nullable_empty_struct() {
        let actual = push_nulls(root())
            .evaluate(&Scope::new(
                StructArray::new(
                    FieldNames::default(),
                    Vec::new(),
                    2,
                    Validity::from_iter([true, false]),
                )
                .into_array(),
            ))
            .unwrap();
        assert_eq!(actual.display_values().to_string(), "[{}, {}]");
    }

    #[test]
    pub fn test_push_nulls_into_non_nullable_field() {
        let actual = push_nulls(root())
            .evaluate(&Scope::new(
                StructArray::try_from_iter_with_validity(
                    [("a", buffer![0, 1])],
                    Validity::from_iter([true, false]),
                )
                .unwrap()
                .into_array(),
            ))
            .unwrap();
        assert_eq!(
            actual.display_values().to_string(),
            "[{a: 0i32}, {a: null}]"
        );
    }

    #[test]
    pub fn test_push_nulls_into_nullable_field() {
        let actual = push_nulls(root())
            .evaluate(&Scope::new(
                StructArray::try_from_iter_with_validity(
                    [(
                        "a",
                        PrimitiveArray::from_option_iter([None, Some(1), Some(2)]),
                    )],
                    Validity::from_iter([true, false, true]),
                )
                .unwrap()
                .into_array(),
            ))
            .unwrap();
        assert_eq!(
            actual.display_values().to_string(),
            "[{a: null}, {a: null}, {a: 2i32}]"
        );
    }

    #[test]
    pub fn test_push_nulls_into_non_nullable_struct_with_nullable_field() {
        let actual = push_nulls(root())
            .evaluate(&Scope::new(
                StructArray::try_from_iter_with_validity(
                    [(
                        "a",
                        StructArray::try_from_iter([(
                            "inner",
                            PrimitiveArray::from_option_iter([None, Some(1), Some(2)]),
                        )])
                        .unwrap(),
                    )],
                    Validity::from_iter([true, false, true]),
                )
                .unwrap()
                .into_array(),
            ))
            .unwrap();
        assert_eq!(
            actual.display_values().to_string(),
            "[{a: {inner: null}}, {a: null}, {a: {inner: 2i32}}]"
        );
    }

    #[test]
    pub fn test_push_nulls_into_non_nullable_struct_with_non_nullable_field() {
        let actual = push_nulls(root())
            .evaluate(&Scope::new(
                StructArray::try_from_iter_with_validity(
                    [(
                        "a",
                        StructArray::try_from_iter([("inner", buffer![0, 1, 2])]).unwrap(),
                    )],
                    Validity::from_iter([true, false, true]),
                )
                .unwrap()
                .into_array(),
            ))
            .unwrap();
        assert_eq!(
            actual.display_values().to_string(),
            "[{a: {inner: 0i32}}, {a: null}, {a: {inner: 2i32}}]"
        );
    }

    #[test]
    pub fn test_push_nulls_into_many_fields() {
        let actual = push_nulls(root())
            .evaluate(&Scope::new(
                StructArray::try_from_iter_with_validity(
                    [
                        (
                            "a",
                            StructArray::try_from_iter([("inner", buffer![0, 1, 2])])
                                .unwrap()
                                .into_array(),
                        ),
                        ("b", buffer![0, 1, 2].into_array()),
                        (
                            "c",
                            <VarBinArray as FromIterator<_>>::from_iter([
                                Some("zero"),
                                Some("one"),
                                None,
                            ])
                            .into_array(),
                        ),
                    ],
                    Validity::from_iter([true, false, true]),
                )
                .unwrap()
                .into_array(),
            ))
            .unwrap();
        assert_eq!(
            actual.display_values().to_string(),
            "[{a: {inner: 0i32}, b: 0i32, c: \"zero\"}, {a: null, b: null, c: null}, {a: {inner: 2i32}, b: 2i32, c: null}]"
        );
    }
}
