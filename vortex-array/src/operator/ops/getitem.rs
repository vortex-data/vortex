// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::slice;
use std::sync::Arc;

use vortex_dtype::{DType, FieldName};
use vortex_error::{VortexExpect, VortexResult};

use crate::operator::{Operator, OperatorId, OperatorRef};

/// An operator that extracts a field from a struct array.
#[derive(Debug, Hash)]
pub struct GetItemOperator {
    child: OperatorRef,
    field: FieldName,
    dtype: DType,
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

    fn len(&self) -> usize {
        self.child.len()
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
