use std::any::Any;
use std::fmt::Display;
use std::sync::Arc;

use itertools::Itertools;
use vortex_array::aliases::hash_set::HashSet;
use vortex_array::ArrayData;
use vortex_dtype::Field;
use vortex_error::{vortex_err, VortexResult};

use crate::{ExprRef, VortexExpr};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Select {
    Include(Vec<Field>),
    Exclude(Vec<Field>),
}

impl Select {
    pub fn include(columns: Vec<Field>) -> Self {
        Self::Include(columns)
    }

    pub fn include_expr(columns: Vec<Field>) -> Arc<Self> {
        Arc::new(Self::include(columns))
    }

    pub fn exclude(columns: Vec<Field>) -> Self {
        Self::Exclude(columns)
    }

    pub fn exclude_expr(columns: Vec<Field>) -> Arc<Self> {
        Arc::new(Self::exclude(columns))
    }

    pub fn fields(&self) -> &[Field] {
        match self {
            Select::Include(fields) => fields,
            Select::Exclude(fields) => fields,
        }
    }
}

impl Display for Select {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Select::Include(fields) => write!(f, "Include({})", fields.iter().format(",")),
            Select::Exclude(fields) => write!(f, "Exclude({})", fields.iter().format(",")),
        }
    }
}

impl VortexExpr for Select {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn evaluate(&self, batch: &ArrayData) -> VortexResult<ArrayData> {
        let st = batch
            .as_struct_array()
            .ok_or_else(|| vortex_err!("Not a struct array"))?;
        match self {
            Select::Include(f) => st.project(f),
            Select::Exclude(e) => {
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
        vec![]
    }

    fn replacing_children(self: Arc<Self>, children: Vec<ExprRef>) -> ExprRef {
        assert_eq!(children.len(), 0);
        self
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::array::StructArray;
    use vortex_array::IntoArrayData;
    use vortex_buffer::buffer;
    use vortex_dtype::Field;

    use crate::{Select, VortexExpr};

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
        let select = Select::include(vec![Field::from("a")]);
        let selected = select.evaluate(st.as_ref()).unwrap();
        let selected_names = selected.as_struct_array().unwrap().names().clone();
        assert_eq!(selected_names.as_ref(), &["a".into()]);
    }

    #[test]
    pub fn exclude_columns() {
        let st = test_array();
        let select = Select::exclude(vec![Field::from("a")]);
        let selected = select.evaluate(st.as_ref()).unwrap();
        let selected_names = selected.as_struct_array().unwrap().names().clone();
        assert_eq!(selected_names.as_ref(), &["b".into()]);
    }
}
