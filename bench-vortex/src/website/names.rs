// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use phf::Map;
use phf::phf_map;

// TODO(connor): This should probably be generated smarter.
pub static NAMES: Map<u32, &'static str> = phf_map! {
    0 => "null",
    1 => "invalid",
    2 => "random-access",
    3 => "vortex-nvme",
    4 => "parquet-nvme",
    5 => "lance-nvme",
};
