// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;

use itertools::Itertools;
use vortex_array::{ArrayRef, DeserializeMetadata, EmptyMetadata, IntoArray, ToCanonical};
use vortex_dtype::{DType, FieldNames};
use vortex_error::{VortexResult, vortex_bail, vortex_err};

use crate::field::DisplayFieldNames;
use crate::{AnalysisExpr, ExprEncodingRef, ExprId, ExprRef, IntoExpr, Scope, VTable, vtable};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SelectField {
    Include(FieldNames),
    Exclude(FieldNames),
}

vtable!(Select);

#[derive(Debug, Clone, Hash)]
#[allow(clippy::derived_hash_with_manual_eq)]
pub struct SelectExpr {
    fields: SelectField,
    child: ExprRef,
}

impl PartialEq for SelectExpr {
    fn eq(&self, other: &Self) -> bool {
        self.fields == other.fields && self.child.eq(&other.child)
    }
}

pub struct SelectExprEncoding;

impl VTable for SelectVTable {
    type Expr = SelectExpr;
    type Encoding = SelectExprEncoding;
    type Metadata = EmptyMetadata;

    fn id(_encoding: &Self::Encoding) -> ExprId {
        ExprId::new_ref("select")
    }

    fn encoding(_expr: &Self::Expr) -> ExprEncodingRef {
        ExprEncodingRef::new_ref(SelectExprEncoding.as_ref())
    }

    fn metadata(_expr: &Self::Expr) -> Option<Self::Metadata> {
        // Select does not support serialization
        None
    }

    fn children(expr: &Self::Expr) -> Vec<&ExprRef> {
        vec![&expr.child]
    }

    fn with_children(expr: &Self::Expr, children: Vec<ExprRef>) -> VortexResult<Self::Expr> {
        Ok(SelectExpr {
            fields: expr.fields.clone(),
            child: children[0].clone(),
        })
    }

    fn build(
        _encoding: &Self::Encoding,
        _metadata: &<Self::Metadata as DeserializeMetadata>::Output,
        _children: Vec<ExprRef>,
    ) -> VortexResult<Self::Expr> {
        vortex_bail!("Select does not support deserialization")
    }

    fn evaluate(expr: &Self::Expr, scope: &Scope) -> VortexResult<ArrayRef> {
        let batch = expr.child.unchecked_evaluate(scope)?.to_struct()?;
        Ok(match &expr.fields {
            SelectField::Include(f) => batch.project(f.as_ref()),
            SelectField::Exclude(names) => {
                let included_names = batch
                    .names()
                    .iter()
                    .filter(|&f| !names.as_ref().contains(f))
                    .cloned()
                    .collect::<Vec<_>>();
                batch.project(included_names.as_slice())
            }
        }?
        .into_array())
    }

    fn return_dtype(expr: &Self::Expr, scope: &DType) -> VortexResult<DType> {
        let child_dtype = expr.child.return_dtype(scope)?;
        let child_struct_dtype = child_dtype
            .as_struct()
            .ok_or_else(|| vortex_err!("Select child not a struct dtype"))?;

        let projected = match &expr.fields {
            SelectField::Include(fields) => child_struct_dtype.project(fields.as_ref())?,
            SelectField::Exclude(fields) => child_struct_dtype
                .names()
                .iter()
                .cloned()
                .zip_eq(child_struct_dtype.fields())
                .filter(|(name, _)| !fields.as_ref().contains(name))
                .collect(),
        };

        Ok(DType::Struct(projected, child_dtype.nullability()))
    }
}

pub fn select(fields: impl Into<FieldNames>, child: ExprRef) -> ExprRef {
    SelectExpr::include_expr(fields.into(), child)
}

pub fn select_exclude(fields: impl Into<FieldNames>, child: ExprRef) -> ExprRef {
    SelectExpr::exclude_expr(fields.into(), child)
}

impl SelectExpr {
    pub fn new(fields: SelectField, child: ExprRef) -> Self {
        Self { fields, child }
    }

    pub fn include_expr(columns: FieldNames, child: ExprRef) -> ExprRef {
        Self::new(SelectField::Include(columns), child).into_expr()
    }

    pub fn exclude_expr(columns: FieldNames, child: ExprRef) -> ExprRef {
        Self::new(SelectField::Exclude(columns), child).into_expr()
    }

    pub fn fields(&self) -> &SelectField {
        &self.fields
    }

    pub fn child(&self) -> &ExprRef {
        &self.child
    }

    pub fn as_include(&self, field_names: &FieldNames) -> VortexResult<ExprRef> {
        Ok(Self::new(
            SelectField::Include(self.fields.as_include_names(field_names)?),
            self.child.clone(),
        )
        .into_expr())
    }
}

impl SelectField {
    pub fn include(columns: FieldNames) -> Self {
        assert_eq!(columns.iter().unique().collect_vec().len(), columns.len());
        Self::Include(columns)
    }

    pub fn exclude(columns: FieldNames) -> Self {
        assert_eq!(columns.iter().unique().collect_vec().len(), columns.len());
        Self::Exclude(columns)
    }

    pub fn is_include(&self) -> bool {
        matches!(self, Self::Include(_))
    }

    pub fn is_exclude(&self) -> bool {
        matches!(self, Self::Exclude(_))
    }

    pub fn fields(&self) -> &FieldNames {
        match self {
            SelectField::Include(fields) => fields,
            SelectField::Exclude(fields) => fields,
        }
    }

    pub fn as_include_names(&self, field_names: &FieldNames) -> VortexResult<FieldNames> {
        if self
            .fields()
            .iter()
            .any(|f| !field_names.iter().contains(f))
        {
            vortex_bail!(
                "Field {:?} in select not in field names {:?}",
                self,
                field_names
            );
        }
        match self {
            SelectField::Include(fields) => Ok(fields.clone()),
            SelectField::Exclude(exc_fields) => Ok(field_names
                .iter()
                .filter(|f| exc_fields.iter().contains(f))
                .cloned()
                .collect()),
        }
    }
}

impl Display for SelectField {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SelectField::Include(fields) => write!(f, "{{{}}}", DisplayFieldNames(fields)),
            SelectField::Exclude(fields) => write!(f, "~{{{}}}", DisplayFieldNames(fields)),
        }
    }
}

impl Display for SelectExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}{}", self.child, self.fields)
    }
}

impl AnalysisExpr for SelectExpr {}

#[cfg(test)]
mod tests {

    use vortex_array::arrays::StructArray;
    use vortex_array::{IntoArray, ToCanonical};
    use vortex_buffer::buffer;
    use vortex_dtype::{DType, FieldName, Nullability};

    use crate::{Scope, root, select, select_exclude, test_harness};

    fn test_array() -> StructArray {
        StructArray::from_fields(&[
            ("a", buffer![0, 1, 2].into_array()),
            ("b", buffer![4, 5, 6].into_array()),
        ])
        .unwrap()
    }

    #[test]
    pub fn include_columns() {
        let st = test_array();
        let select = select(vec![FieldName::from("a")], root());
        let selected = select
            .evaluate(&Scope::new(st.to_array()))
            .unwrap()
            .to_struct()
            .unwrap();
        let selected_names = selected.names().clone();
        assert_eq!(selected_names.as_ref(), &["a".into()]);
    }

    #[test]
    pub fn exclude_columns() {
        let st = test_array();
        let select = select_exclude(vec![FieldName::from("a")], root());
        let selected = select
            .evaluate(&Scope::new(st.to_array()))
            .unwrap()
            .to_struct()
            .unwrap();
        let selected_names = selected.names().clone();
        assert_eq!(selected_names.as_ref(), &["b".into()]);
    }

    #[test]
    fn dtype() {
        let dtype = test_harness::struct_dtype();

        let select_expr = select(vec![FieldName::from("a")], root());
        let expected_dtype = DType::Struct(
            dtype.as_struct().unwrap().project(&["a".into()]).unwrap(),
            Nullability::NonNullable,
        );
        assert_eq!(select_expr.return_dtype(&dtype).unwrap(), expected_dtype);

        let select_expr_exclude = select_exclude(
            vec![
                FieldName::from("col1"),
                FieldName::from("col2"),
                FieldName::from("bool1"),
                FieldName::from("bool2"),
            ],
            root(),
        );
        assert_eq!(
            select_expr_exclude.return_dtype(&dtype).unwrap(),
            expected_dtype
        );

        let select_expr_exclude = select_exclude(
            vec![FieldName::from("col1"), FieldName::from("col2")],
            root(),
        );
        assert_eq!(
            select_expr_exclude.return_dtype(&dtype).unwrap(),
            DType::Struct(
                dtype
                    .as_struct()
                    .unwrap()
                    .project(&["a".into(), "bool1".into(), "bool2".into()])
                    .unwrap(),
                Nullability::NonNullable
            )
        );
    }
}
