// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod clickbench;
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
    use vortex::VortexSessionDefault;
    use vortex::compressor::BtrBlocksCompressorBuilder;
    use vortex::editions::CORE_2026_08_3;
    use vortex::editions::EditionSessionExt;
    use vortex::file::WriteStrategyBuilder;
    use vortex::session::VortexSession;
    use vortex_arrow::ArrowSessionExt;
    use vortex_error::VortexResult;
    use vortex_error::vortex_err;

    use super::fixtures;
    use crate::adapter;

    fn is_clickbench_fixture(name: &str) -> bool {
        name.contains("clickbench")
    }

    #[test]
    fn roundtrip_non_clickbench_fixtures_to_bytes() -> VortexResult<()> {
        let session = VortexSession::default();
        session
            .enable_edition(CORE_2026_08_3)
            .map_err(|error| vortex_err!("{error}"))?;
        for dataset in fixtures()
            .into_iter()
            .filter(|fixture| !is_clickbench_fixture(fixture.name()))
        {
            let array = dataset.build(&session.arrow())?;
            let regular_bytes = adapter::write_compressed_to_bytes_with_session(
                &session,
                array.clone(),
                WriteStrategyBuilder::default().build(),
            )?;
            let _regular = adapter::read_file(regular_bytes)?;

            let compact_bytes = adapter::write_compressed_to_bytes_with_session(
                &session,
                array,
                WriteStrategyBuilder::default()
                    .with_btrblocks_builder(BtrBlocksCompressorBuilder::default().with_compact())
                    .build(),
            )?;
            let _compact = adapter::read_file(compact_bytes)?;
        }
        Ok(())
    }
}
