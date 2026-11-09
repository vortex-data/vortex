// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;

use itertools::Itertools;

use crate::dtype::FieldNames;

pub struct DisplayFieldNames<'a>(pub &'a FieldNames);

impl Display for DisplayFieldNames<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0.iter().format(", "))
    }
}
