use std::any::Any;
use std::fmt::Display;
use std::sync::Arc;

use itertools::Itertools;
use vortex_array::Array;
use vortex_dtype::FieldNames;
use vortex_error::{vortex_bail, vortex_err, VortexResult};

use crate::field::DisplayFieldNames;
use crate::{ExprRef, VortexExpr};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SelectField {
    Include(FieldNames),
    Exclude(FieldNames),
}

#[derive(Debug, Clone, Eq, Hash)]
#[allow(clippy::derived_hash_with_manual_eq)]
pub struct Select {
    fields: SelectField,
    child: ExprRef,
}

pub fn select(fields: impl Into<FieldNames>, child: ExprRef) -> ExprRef {
    Select::include_expr(fields.into(), child)
}

pub fn select_exclude(fields: impl Into<FieldNames>, child: ExprRef) -> ExprRef {
    Select::exclude_expr(fields.into(), child)
}

impl Select {
    pub fn new_expr(fields: SelectField, child: ExprRef) -> ExprRef {
        Arc::new(Self { fields, child })
    }

    pub fn include_expr(columns: FieldNames, child: ExprRef) -> ExprRef {
        Self::new_expr(SelectField::Include(columns), child)
    }

    pub fn exclude_expr(columns: FieldNames, child: ExprRef) -> ExprRef {
        Self::new_expr(SelectField::Exclude(columns), child)
    }

    pub fn fields(&self) -> &SelectField {
        &self.fields
    }

    pub fn child(&self) -> &ExprRef {
        &self.child
    }

    pub fn as_include(&self, field_names: &FieldNames) -> VortexResult<ExprRef> {
        Ok(Self::new_expr(
            SelectField::Include(self.fields.as_include_names(field_names)?),
            self.child.clone(),
        ))
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
        if self.fields().iter().any(|f| !field_names.contains(f)) {
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
                .filter(|f| exc_fields.contains(f))
                .cloned()
                .collect()),
        }
    }
}

impl Display for SelectField {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SelectField::Include(fields) => write!(f, "+({})", DisplayFieldNames(fields)),
            SelectField::Exclude(fields) => write!(f, "-({})", DisplayFieldNames(fields)),
        }
    }
}

impl Display for Select {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "select {} {}", self.fields, self.child)
    }
}

impl VortexExpr for Select {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn unchecked_evaluate(&self, batch: &Array) -> VortexResult<Array> {
        let batch = self.child.evaluate(batch)?;
        let st = batch
            .as_struct_array()
            .ok_or_else(|| vortex_err!("Not a struct array"))?;
        match &self.fields {
            SelectField::Include(f) => st.project(f),
            SelectField::Exclude(names) => {
                let included_names = st
                    .names()
                    .iter()
                    .filter(|&f| !names.contains(f))
                    .cloned()
                    .collect::<Vec<_>>();
                st.project(included_names.as_slice())
            }
        }
    }

    fn children(&self) -> Vec<&ExprRef> {
        vec![&self.child]
    }

    fn replacing_children(self: Arc<Self>, children: Vec<ExprRef>) -> ExprRef {
        assert_eq!(children.len(), 1);
        Self::new_expr(self.fields.clone(), children[0].clone())
    }
}

impl PartialEq for Select {
    fn eq(&self, other: &Select) -> bool {
        self.fields == other.fields && self.child.eq(&other.child)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_array::array::StructArray;
    use vortex_array::IntoArray;
    use vortex_buffer::buffer;
    use vortex_dtype::{DType, Field, FieldName, Nullability};

    use crate::{ident, select, select_exclude, test_harness};

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
        let select = select(vec![FieldName::from("a")], ident());
        let selected = select.evaluate(st.as_ref()).unwrap();
        let selected_names = selected.as_struct_array().unwrap().names().clone();
        assert_eq!(selected_names.as_ref(), &["a".into()]);
    }

    #[test]
    pub fn exclude_columns() {
        let st = test_array();
        let select = select_exclude(vec![FieldName::from("a")], ident());
        let selected = select.evaluate(st.as_ref()).unwrap();
        let selected_names = selected.as_struct_array().unwrap().names().clone();
        assert_eq!(selected_names.as_ref(), &["b".into()]);
    }

    #[test]
    fn dtype() {
        let dtype = test_harness::struct_dtype();

        let select_expr = select(vec![FieldName::from("a")], ident());
        let expected_dtype = DType::Struct(
            Arc::new(
                dtype
                    .as_struct()
                    .unwrap()
                    .project(&[Field::from("a")])
                    .unwrap(),
            ),
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
            ident(),
        );
        assert_eq!(
            select_expr_exclude.return_dtype(&dtype).unwrap(),
            expected_dtype
        );

        let select_expr_exclude = select_exclude(
            vec![FieldName::from("col1"), FieldName::from("col2")],
            ident(),
        );
        assert_eq!(
            select_expr_exclude.return_dtype(&dtype).unwrap(),
            DType::Struct(
                Arc::new(
                    dtype
                        .as_struct()
                        .unwrap()
                        .project(&[Field::from("a"), Field::from("bool1"), Field::from("bool2")])
                        .unwrap()
                ),
                Nullability::NonNullable
            )
        );
    }
}
