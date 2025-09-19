// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod operator;

use crate::operator::Operator;

pub trait WebGpuOperator: Operator {}
