use std::any::Any;
use std::fmt::Display;

use itertools::Itertools;
use vortex_array::aliases::hash_set::HashSet;
use vortex_array::ArrayData;
use vortex_dtype::field::Field;
use vortex_error::{vortex_err, VortexResult};

use crate::{unbox_any, VortexExpr};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Select {
    Include(Vec<Field>),
    Exclude(Vec<Field>),
}

impl Select {
    pub fn include(columns: Vec<Field>) -> Self {
        Self::Include(columns)
    }

    pub fn exclude(columns: Vec<Field>) -> Self {
        Self::Exclude(columns)
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
        batch.with_dyn(|s| {
            let st = s
                .as_struct_array()
                .ok_or_else(|| vortex_err!("Not a struct array"))?;
            match self {
                Select::Include(f) => st.project(f),
                Select::Exclude(e) => {
                    let normalized_exclusion = e
                        .iter()
                        .map(|ef| match ef {
                            Field::Name(n) => Ok(n.as_str()),
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
        })
    }

    fn collect_references<'a>(&'a self, references: &mut HashSet<&'a Field>) {
        match self {
            Select::Include(f) => references.extend(f.iter()),
            // It's weird that we treat the references of exclusions and inclusions the same, we need to have a wrapper around Field in the return
            Select::Exclude(e) => references.extend(e.iter()),
        }
    }
}

impl PartialEq<dyn Any> for Select {
    fn eq(&self, other: &dyn Any) -> bool {
        unbox_any(other)
            .downcast_ref::<Self>()
            .map(|x| self == x)
            .unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::array::{PrimitiveArray, StructArray};
    use vortex_array::IntoArrayData;
    use vortex_dtype::field::Field;

    use crate::{Select, VortexExpr};

    fn test_array() -> StructArray {
        StructArray::from_fields(&[
            ("a", PrimitiveArray::from(vec![0, 1, 2]).into_array()),
            ("b", PrimitiveArray::from(vec![4, 5, 6]).into_array()),
        ])
        .unwrap()
    }

    #[test]
    pub fn include_columns() {
        let st = test_array();
        let select = Select::include(vec![Field::from("a")]);
        let selected = select.evaluate(st.as_ref()).unwrap();
        let selected_names = selected.with_dyn(|a| a.as_struct_array_unchecked().names().clone());
        assert_eq!(selected_names.as_ref(), &["a".into()]);
    }

    #[test]
    pub fn exclude_columns() {
        let st = test_array();
        let select = Select::exclude(vec![Field::from("a")]);
        let selected = select.evaluate(st.as_ref()).unwrap();
        let selected_names = selected.with_dyn(|a| a.as_struct_array_unchecked().names().clone());
        assert_eq!(selected_names.as_ref(), &["b".into()]);
    }
}
