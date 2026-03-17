// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod clickbench;
#[allow(clippy::cast_possible_truncation)]
mod tpch;

use crate::fixtures::DatasetFixture;

/// All dataset-derived fixtures.
pub fn fixtures() -> Vec<Box<dyn DatasetFixture>> {
    let mut fixtures = Vec::new();
    fixtures.extend(tpch::fixtures());
    fixtures.extend(clickbench::fixtures());
    fixtures
}

#[cfg(test)]
mod tests {
    use vortex::file::WriteStrategyBuilder;

    use super::fixtures;
    use crate::adapter;

    fn is_clickbench_fixture(name: &str) -> bool {
        name.contains("clickbench")
    }

    #[test]
    fn roundtrip_non_clickbench_fixtures_to_bytes() {
        for dataset in fixtures()
            .into_iter()
            .filter(|fixture| !is_clickbench_fixture(fixture.name()))
        {
            let array = dataset.build().unwrap();
            let regular_bytes = adapter::write_compressed_to_bytes(
                array.clone(),
                WriteStrategyBuilder::default().build(),
            )
            .unwrap();
            let _regular = adapter::read_file(regular_bytes).unwrap();

            let compact_bytes = adapter::write_compressed_to_bytes(
                array,
                WriteStrategyBuilder::default()
                    .with_compact_encodings()
                    .build(),
            )
            .unwrap();
            let _compact = adapter::read_file(compact_bytes).unwrap();
        }
    }
}
