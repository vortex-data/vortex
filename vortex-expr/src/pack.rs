use std::any::Any;
use std::fmt::Display;
use std::sync::Arc;
use itertools::Itertools;
use vortex_array::aliases::hash_set::HashSet;
use vortex_array::array::StructArray;
use vortex_array::{ArrayData, IntoArrayData};
use vortex_dtype::field::Field;
use vortex_error::VortexResult;

use crate::{unbox_any, ExprRef, VortexExpr};

#[derive(Debug, Clone)]
pub struct Pack {
    children: Vec<ExprRef>,
    names: Vec<String>,
}

impl Pack {
    pub fn new_expr(children: Vec<ExprRef>, names: Vec<String>) -> ExprRef {
        Arc::new(Self { children, names })
    }
}

impl Display for Pack {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "pack({})", self.names.join(", "))
    }
}

impl VortexExpr for Pack {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn evaluate(&self, batch: &ArrayData) -> VortexResult<ArrayData> {
        let mut arrays = Vec::with_capacity(self.children.len());
        for child in &self.children {
            arrays.push(child.evaluate(batch)?);
        }
        let fields = self.names.iter().zip(arrays).map(|(n, a)| (n.as_str(), a)).collect_vec();
        let struct_array = StructArray::from_fields(fields.as_slice())?;
        Ok(struct_array.into_array())
    }

    fn collect_references<'a>(&'a self, references: &mut HashSet<&'a Field>) {
        for child in &self.children {
            child.collect_references(references)
        }
    }
}

impl PartialEq<dyn Any> for Pack {
    fn eq(&self, other: &dyn Any) -> bool {
        unbox_any(other)
            .downcast_ref::<Self>()
            .map(|x| {
                x.children
                    .iter()
                    .zip(&self.children)
                    .all(|(a, b)| a.eq(b))
                    && x.names.iter().zip(&self.names).all(|(a, b)| a == b)
            })
            .unwrap_or(false)
    }
}
