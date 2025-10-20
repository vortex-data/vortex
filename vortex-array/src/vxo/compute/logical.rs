// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::stats::ArrayStats;
use crate::vxo::{Array, ArrayEq, ArrayHash, ArrayRef, BatchKernel, BindCtx};
use crate::EncodingId;
use itertools::Itertools;
use std::any::Any;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use vortex_dtype::DType;
use vortex_error::{VortexExpect, VortexResult};

/// Logical operators for boolean arrays.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LogicalOperator {
    And,
    AndKleene,
    Or,
    OrKleene,
    AndNot,
}

#[derive(Clone, Debug)]
pub struct LogicalArray {
    children: [ArrayRef; 2],
    operator: LogicalOperator,
    dtype: DType,
    stats: ArrayStats,
}

impl LogicalArray {
    pub fn new(lhs: ArrayRef, rhs: ArrayRef, operator: LogicalOperator) -> Self {
        assert_eq!(
            lhs.len(),
            rhs.len(),
            "Logical arrays must have the same length"
        );
        assert!(
            lhs.dtype().eq_ignore_nullability(rhs.dtype()),
            "Logical arrays must have the same dtype (excluding nullability), got {} and {}",
            lhs.dtype(),
            rhs.dtype()
        );

        let dtype = DType::Bool(lhs.dtype().nullability() | rhs.dtype().nullability());

        Self {
            children: [lhs, rhs],
            operator,
            dtype,
            stats: ArrayStats::default(),
        }
    }

    /// Returns the left-hand side array.
    pub fn lhs(&self) -> &ArrayRef {
        &self.children[0]
    }

    /// Returns the right-hand side array.
    pub fn rhs(&self) -> &ArrayRef {
        &self.children[1]
    }
}

impl ArrayHash for LogicalArray {
    fn array_hash<H: Hasher>(&self, state: &mut H) {
        for child in &self.children {
            child.array_hash(state);
        }
        self.operator.hash(state);
    }
}

impl ArrayEq for LogicalArray {
    fn array_eq(&self, other: &Self) -> bool {
        self.operator == other.operator
            && self.lhs().array_eq(&other.lhs())
            && self.rhs().array_eq(&other.rhs())
    }
}

impl Array for LogicalArray {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn encoding_id(&self) -> EncodingId {
        match self.operator {
            LogicalOperator::And => EncodingId::from("vortex.and"),
            LogicalOperator::AndKleene => EncodingId::from("vortex.and_kleene"),
            LogicalOperator::Or => EncodingId::from("vortex.or"),
            LogicalOperator::OrKleene => EncodingId::from("vortex.or_kleene"),
            LogicalOperator::AndNot => EncodingId::from("vortex.and_not"),
        }
    }

    fn len(&self) -> usize {
        self.lhs().len()
    }

    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn children(&self) -> &[ArrayRef] {
        &self.children
    }

    fn with_children(&self, children: Vec<ArrayRef>) -> ArrayRef {
        // TODO(ngates): what's the contract for what we have to assert here?
        Arc::new(LogicalArray {
            children: children
                .into_iter()
                .collect_array()
                .vortex_expect("LogicalArray requires exactly 2 children"),
            operator: self.operator,
            dtype: self.dtype.clone(),
            stats: self.stats.clone(),
        })
    }

    fn bind(
        &self,
        selection: Option<&ArrayRef>,
        ctx: &mut dyn BindCtx,
    ) -> VortexResult<Box<dyn BatchKernel>> {
        todo!()
    }
}
