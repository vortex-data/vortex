// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::experiment::N;

pub struct PrimitiveVector<T> {
    elements: Box<[T; N]>,
}
