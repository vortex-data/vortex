// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::hash::{Hash, Hasher};
use std::slice;
use std::sync::Arc;

use vortex_dtype::{DType, FieldName};
use vortex_error::{VortexExpect, VortexResult};

use crate::operator::{LengthBounds, Operator, OperatorEq, OperatorHash, OperatorId, OperatorRef};

/// An operator that extracts a field from a struct array.
#[derive(Debug)]
pub struct GetItemOperator {
    // The struct-like child operator.
    child: OperatorRef,
    field: FieldName,
    // The dtype of the extracted field.
    dtype: DType,
}

impl OperatorHash for GetItemOperator {
    fn operator_hash<H: Hasher>(&self, state: &mut H) {
        self.child.operator_hash(state);
        self.field.hash(state);
        self.dtype.hash(state);
    }
}
impl OperatorEq for GetItemOperator {
    fn operator_eq(&self, other: &Self) -> bool {
        self.child.operator_eq(&other.child)
            && self.field == other.field
            && self.dtype == other.dtype
    }
}

impl GetItemOperator {
    pub fn field_name(&self) -> &FieldName {
        &self.field
    }
}

impl Operator for GetItemOperator {
    fn id(&self) -> OperatorId {
        OperatorId::from("vortex.getitem")
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn bounds(&self) -> LengthBounds {
        self.child.bounds()
    }

    fn children(&self) -> &[OperatorRef] {
        slice::from_ref(&self.child)
    }

    fn with_children(self: Arc<Self>, children: Vec<OperatorRef>) -> VortexResult<OperatorRef> {
        Ok(Arc::new(GetItemOperator {
            child: children.into_iter().next().vortex_expect("missing child"),
            field: self.field.clone(),
            dtype: self.dtype.clone(),
        }))
    }
}
