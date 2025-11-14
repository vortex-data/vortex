// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Definition and implementation of [`NullVector`] and [`NullVector`].

mod scalar;
pub use scalar::NullScalar;

mod vector_mut;
pub use vector_mut::NullVector;

use crate::Vector;

impl From<NullVector> for Vector {
    fn from(v: NullVector) -> Self {
        Self::Null(v)
    }
}
