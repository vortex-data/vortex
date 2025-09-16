// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::operator::ArrayOperator;
use std::any::Any;
use std::slice;
use std::sync::Arc;
use vortex_dtype::{DType, FieldName};
use vortex_error::{VortexExpect, VortexResult};

/// An operator that extracts a field from a struct array.
pub struct GetItemOperator {
    child: Arc<dyn ArrayOperator>,
    field: FieldName,
    dtype: DType,
}

impl GetItemOperator {
    pub fn field_name(&self) -> &FieldName {
        &self.field
    }
}

impl ArrayOperator for GetItemOperator {
    fn id(&self) -> Arc<str> {
        Arc::from("vortex.getitem")
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

    fn children(&self) -> &[Arc<dyn ArrayOperator>] {
        slice::from_ref(&self.child)
    }

    fn with_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ArrayOperator>>,
    ) -> VortexResult<Arc<dyn ArrayOperator>> {
        Ok(Arc::new(GetItemOperator {
            child: children.into_iter().next().vortex_expect("missing child"),
            field: self.field.clone(),
            dtype: self.dtype.clone(),
        }))
    }
}
