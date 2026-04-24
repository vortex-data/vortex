// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod datasets;
mod synthetic;

use super::DatasetFixture;
use super::FlatLayoutFixture;

/// All synthetic (flat-layout) fixtures.
pub fn synthetic_fixtures() -> Vec<Box<dyn FlatLayoutFixture>> {
    synthetic::fixtures()
}

/// All dataset fixtures.
pub fn dataset_fixtures() -> Vec<Box<dyn DatasetFixture>> {
    datasets::fixtures()
}
