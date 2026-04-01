// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect;

use crate::ArrayRef;
use crate::array::Array;
use crate::arrays::Constant;
use crate::dtype::DType;
use crate::scalar::Scalar;
use crate::stats::ArrayStats;

pub(super) const NUM_SLOTS: usize = 0;

#[derive(Clone, Debug)]
pub struct ConstantData {
    pub(super) scalar: Scalar,
    pub(super) len: usize,
    pub(super) slots: Vec<Option<ArrayRef>>,
    pub(super) stats_set: ArrayStats,
}

impl ConstantData {
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

    /// Returns the length of this array.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns the [`DType`] of this array.
    pub fn dtype(&self) -> &DType {
        self.scalar.dtype()
    }

    /// Returns `true` if this array is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the [`Scalar`] value of this constant array.
    pub fn scalar(&self) -> &Scalar {
        &self.scalar
    }

    pub fn into_parts(self) -> Scalar {
        self.scalar
    }
}

impl Array<Constant> {
    pub fn new<S>(scalar: S, len: usize) -> Self
    where
        S: Into<Scalar>,
    {
        Array::try_from_data(ConstantData::new(scalar, len))
            .vortex_expect("ConstantData is always valid")
    }
}
