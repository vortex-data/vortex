// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::ArrayRef;
use crate::scalar::Scalar;
use crate::stats::ArrayStats;

pub(super) const NUM_SLOTS: usize = 0;

#[derive(Clone, Debug)]
pub struct ConstantArray {
    pub(super) scalar: Scalar,
    pub(super) len: usize,
    pub(super) slots: Vec<Option<ArrayRef>>,
    pub(super) stats_set: ArrayStats,
}

impl ConstantArray {
    pub fn new<S>(scalar: S, len: usize) -> Self
    where
        S: Into<Scalar>,
    {
        let scalar = scalar.into();
        Self {
            scalar,
            len,
            slots: vec![],
            stats_set: Default::default(),
        }
    }

    /// Returns the [`Scalar`] value of this constant array.
    pub fn scalar(&self) -> &Scalar {
        &self.scalar
    }

    pub fn into_parts(self) -> Scalar {
        self.scalar
    }
}
