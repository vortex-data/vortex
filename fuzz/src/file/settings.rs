// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::num::NonZeroU16;
use std::sync::Arc;

use arbitrary::Arbitrary;
use arbitrary::Unstructured;
use vortex::layout::layouts::flat::writer::FlatLayoutStrategy;
use vortex_btrblocks::BtrBlocksCompressorBuilder;
use vortex_btrblocks::CompressionSessionExt;
use vortex_btrblocks::SchemeExt;
use vortex_file::VortexWriteOptions;
use vortex_file::WriteOptionsSessionExt;
use vortex_file::WriteStrategyBuilder;

use crate::SESSION;

#[derive(Debug, Clone, Copy)]
pub struct WriteSettings {
    pub scheme_mask: u32,
    pub row_block_size: u16,
    pub dict_max_bytes: u16,
    pub file_statistics: bool,
    pub include_padding: bool,
    pub disable_editions: bool,
    pub data_block_target_bytes: Option<NonZeroU16>,
    pub max_variable_length_statistics_size: u16,
    pub unrestricted: bool,
}

impl<'a> Arbitrary<'a> for WriteSettings {
    #[expect(clippy::cast_possible_truncation)]
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        // tight packing gives roughly same distribution but less entropy consumption
        let first = u64::arbitrary(u)?;
        let second = u64::arbitrary(u)?;
        Ok(Self {
            scheme_mask: first as u32,
            row_block_size: ((first >> 32) as u16 & 0x1fff) + 1,
            dict_max_bytes: (first >> 45) as u16,
            file_statistics: first & 1 << 61 != 0,
            include_padding: first & 1 << 62 != 0,
            disable_editions: first & 1 << 63 != 0,
            data_block_target_bytes: NonZeroU16::new(second as u16),
            max_variable_length_statistics_size: (second >> 16) as u16 & 0x3ff,
            unrestricted: second & 1 << 26 != 0,
        })
    }
}

impl WriteSettings {
    pub fn write_options(&self) -> VortexWriteOptions {
        let excluded = SESSION
            .compression()
            .schemes()
            .iter()
            .enumerate()
            .filter(|(index, _)| self.scheme_mask >> (index % 32) & 1 != 0)
            .map(|(_, scheme)| scheme.id())
            .collect::<Vec<_>>();
        let mut compressor =
            BtrBlocksCompressorBuilder::from_session(&SESSION).exclude_schemes(excluded);
        if self.disable_editions {
            compressor = compressor.disable_editions();
        }
        if self.unrestricted {
            compressor = compressor.unrestricted();
        }
        let flat = FlatLayoutStrategy::default()
            .with_include_padding(self.include_padding)
            .with_max_variable_length_statistics_size(usize::from(
                self.max_variable_length_statistics_size,
            ));
        let strategy = WriteStrategyBuilder::from_session(&SESSION)
            .with_row_block_size(usize::from(self.row_block_size))
            .with_data_block_target_bytes(
                self.data_block_target_bytes
                    .map(|bytes| u64::from(bytes.get())),
            )
            .with_dict_max_bytes(usize::from(self.dict_max_bytes))
            .with_flat_strategy(Arc::new(flat))
            .with_btrblocks_builder(compressor)
            .build();
        let mut options = SESSION.write_options().with_strategy(strategy);
        if self.disable_editions || self.unrestricted {
            options = options.disable_editions();
        }
        if !self.file_statistics {
            options = options.with_file_statistics(Vec::new());
        }
        options
    }
}
