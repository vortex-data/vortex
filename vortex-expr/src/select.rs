use std::any::Any;
use std::fmt::Display;
use std::sync::Arc;

use itertools::Itertools;
use vortex_array::aliases::hash_set::HashSet;
use vortex_array::ArrayData;
use vortex_dtype::Field;
use vortex_error::{vortex_err, VortexResult};

use crate::{ExprRef, VortexExpr};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SelectField {
    Include(Vec<Field>),
    Exclude(Vec<Field>),
}

#[derive(Debug, Clone, Eq, Hash)]
#[allow(clippy::derived_hash_with_manual_eq)]
pub struct Select {
    fields: SelectField,
    child: ExprRef,
}

impl Select {
    pub fn new_expr(fields: SelectField, child: ExprRef) -> ExprRef {
        Arc::new(Self { fields, child })
    }

    pub fn include_expr(columns: Vec<Field>, child: ExprRef) -> ExprRef {
        Self::new_expr(SelectField::Include(columns), child)
    }

    pub fn exclude_expr(columns: Vec<Field>, child: ExprRef) -> ExprRef {
        Self::new_expr(SelectField::Exclude(columns), child)
    }

    pub fn fields(&self) -> &SelectField {
        &self.fields
    }

    pub fn child(&self) -> &ExprRef {
        &self.child
    }
}

impl SelectField {
    pub fn include(columns: Vec<Field>) -> Self {
        Self::Include(columns)
    }

    pub fn exclude(columns: Vec<Field>) -> Self {
        Self::Exclude(columns)
    }

    pub fn fields(&self) -> &[Field] {
        match self {
            SelectField::Include(fields) => fields,
            SelectField::Exclude(fields) => fields,
        }
    }
}

impl Display for SelectField {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SelectField::Include(fields) => write!(f, "+({})", fields.iter().format(",")),
            SelectField::Exclude(fields) => write!(f, "-({})", fields.iter().format(",")),
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

    fn unchecked_evaluate(&self, batch: &ArrayData) -> VortexResult<ArrayData> {
        let batch = self.child.evaluate(batch)?;
        let st = batch
            .as_struct_array()
            .ok_or_else(|| vortex_err!("Not a struct array"))?;
        match &self.fields {
            SelectField::Include(f) => st.project(f),
            SelectField::Exclude(e) => {
                let normalized_exclusion = e
                    .iter()
                    .map(|ef| match ef {
                        Field::Name(n) => Ok(&**n),
                        Field::Index(i) => st
                            .names()
                            .get(*i)
                            .map(|s| &**s)
                            .ok_or_else(|| vortex_err!("Column doesn't exist")),
                    })
                    .collect::<VortexResult<HashSet<_>>>()?;
                let included_names = st
                    .names()
                    .iter()
                    .filter(|f| !normalized_exclusion.contains(&&***f))
                    .map(|f| Field::from(&**f))
                    .collect::<Vec<_>>();
                st.project(&included_names)
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
    use vortex_array::array::StructArray;
    use vortex_array::IntoArrayData;
    use vortex_buffer::buffer;
    use vortex_dtype::{DType, Field, Nullability};

    use crate::{ident, test_harness, Select};

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
        let select = Select::include_expr(vec![Field::from("a")], ident());
        let selected = select.evaluate(st.as_ref()).unwrap();
        let selected_names = selected.as_struct_array().unwrap().names().clone();
        assert_eq!(selected_names.as_ref(), &["a".into()]);
    }

    #[test]
    pub fn exclude_columns() {
        let st = test_array();
        let select = Select::exclude_expr(vec![Field::from("a")], ident());
        let selected = select.evaluate(st.as_ref()).unwrap();
        let selected_names = selected.as_struct_array().unwrap().names().clone();
        assert_eq!(selected_names.as_ref(), &["b".into()]);
    }

    #[test]
    fn dtype() {
        let dtype = test_harness::struct_dtype();

        let select_expr = Select::include_expr(vec![Field::from("a")], ident());
        let expected_dtype = DType::Struct(
            dtype
                .as_struct()
                .unwrap()
                .project(&[Field::from("a")])
                .unwrap(),
            Nullability::NonNullable,
        );
        assert_eq!(select_expr.return_dtype(&dtype).unwrap(), expected_dtype);

        let select_expr_exclude = Select::exclude_expr(
            vec![
                Field::from("col1"),
                Field::from("col2"),
                Field::from("bool1"),
                Field::from("bool2"),
            ],
            ident(),
        );
        assert_eq!(
            select_expr_exclude.return_dtype(&dtype).unwrap(),
            expected_dtype
        );

        let select_expr_exclude = Select::exclude_expr(
            vec![Field::from("col1"), Field::from("col2"), Field::Index(1)],
            ident(),
        );
        assert_eq!(
            select_expr_exclude.return_dtype(&dtype).unwrap(),
            DType::Struct(
                dtype
                    .as_struct()
                    .unwrap()
                    .project(&[Field::from("a"), Field::from("bool1"), Field::from("bool2")])
                    .unwrap(),
                Nullability::NonNullable
            )
        );
    }
}
