// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Name ID to string mapping for benchmark data.

use phf::Map;
use phf::phf_map;

// Name ID constants.
pub const NULL: u32 = 0;
pub const INVALID: u32 = 1;
pub const RANDOM_ACCESS: u32 = 2;
pub const VORTEX_NVME: u32 = 3;
pub const PARQUET_NVME: u32 = 4;
pub const LANCE_NVME: u32 = 5;

// TODO(connor): This should probably be generated smarter.
/// Maps name IDs to their string representations.
pub static NAMES: Map<u32, &'static str> = phf_map! {
    0u32 => "null",
    1u32 => "invalid",
    2u32 => "random-access",
    3u32 => "vortex-nvme",
    4u32 => "parquet-nvme",
    5u32 => "lance-nvme",
};
