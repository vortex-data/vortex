// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Shared sizing policy for array benchmarks.

use std::mem::size_of;

const DEFAULT_DATA_BLOCK_BYTES: usize = 1 << 20;
const DEFAULT_SCAN_SPLIT_ROWS: usize = 100_000;

/// Returns a production-sized row count for one fixed-width input array.
///
/// The sizing policy mirrors Vortex's default [1 MiB writer block] and [100,000-row scan split].
/// The smaller limit models the amount of one column processed by a scan task.
///
/// [1 MiB writer block]: https://github.com/vortex-data/vortex/blob/aaed723dffe1d2c54fcc7f1bbcf760726d4e8056/vortex-file/src/strategy.rs#L67-L75
/// [100,000-row scan split]: https://github.com/vortex-data/vortex/blob/aaed723dffe1d2c54fcc7f1bbcf760726d4e8056/vortex-layout/src/scan/mod.rs#L16-L19
pub const fn fixed_width_array_len<T>() -> usize {
    let data_block_rows = DEFAULT_DATA_BLOCK_BYTES / size_of::<T>();
    if data_block_rows < DEFAULT_SCAN_SPLIT_ROWS {
        data_block_rows
    } else {
        DEFAULT_SCAN_SPLIT_ROWS
    }
}
