// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::operator::{OperatorEq, OperatorRef};

/// Each operator has a row selection over the domain of input rows.
#[derive(Debug, Clone)]
pub enum RowSelection {
    /// Defines a new domain of N rows.
    Domain(usize),
    /// Returns all rows from the domain.
    All,
    /// Selects rows from the range where the boolean operator resolves to a true bit.
    MaskOperator(OperatorRef),
}

impl PartialEq for RowSelection {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (RowSelection::Domain(n1), RowSelection::Domain(n2)) => n1 == n2,
            (RowSelection::All, RowSelection::All) => true,
            (RowSelection::MaskOperator(o1), RowSelection::MaskOperator(o2)) => o1.operator_eq(o2),
            _ => false,
        }
    }
}
impl Eq for RowSelection {}
