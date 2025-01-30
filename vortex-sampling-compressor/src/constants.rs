pub use cost::*;
pub use decompression::*;

mod cost {
    pub const DEFAULT_MAX_COST: u8 = 3;

    // structural pass-throughs have no cost
    pub const CHUNKED_COST: u8 = 0;
    pub const SPARSE_COST: u8 = 0;
    pub const STRUCT_COST: u8 = 0;
    pub const LIST_COST: u8 = 0;
    pub const VARBIN_COST: u8 = 0;

    // so fast that we can ignore the cost
    pub const BITPACKED_NO_PATCHES_COST: u8 = 0;
    pub const BITPACKED_WITH_PATCHES_COST: u8 = 0;
    pub const CONSTANT_COST: u8 = 0;
    pub const ZIGZAG_COST: u8 = 0;

    // "normal" encodings
    pub const ALP_COST: u8 = 1;
    pub const ALP_RD_COST: u8 = 1;
    pub const DATE_TIME_PARTS_COST: u8 = 1;
    pub const DICT_COST: u8 = 1;
    pub const FOR_COST: u8 = 1;
    pub const FSST_COST: u8 = 1;
    pub const RUN_END_COST: u8 = 1;

    // "expensive" encodings
    pub const DELTA_COST: u8 = 2;
}

mod decompression {
    // Macbook Pro with M4 Max CPU has 546 GB/s of memory bandwidth
    pub const MAX_GIB_PER_S: f64 = 546.0;

    // structural pass-throughs
    pub const SPARSE_GIB_PER_S: f64 = MAX_GIB_PER_S;
    pub const STRUCT_GIB_PER_S: f64 = MAX_GIB_PER_S;
    pub const CHUNKED_GIB_PER_S: f64 = MAX_GIB_PER_S;
    pub const LIST_GIB_PER_S: f64 = MAX_GIB_PER_S;
    pub const VARBIN_GIB_PER_S: f64 = MAX_GIB_PER_S;

    // benchmarked decompression throughput
    pub const ALP_GIB_PER_S: f64 = 10.8;
    pub const ALP_RD_GIB_PER_S: f64 = 4.4;
    pub const BITPACKED_NO_PATCHES_GIB_PER_S: f64 = 48.2;
    pub const BITPACKED_WITH_PATCHES_GIB_PER_S: f64 = 46.6;
    pub const CONSTANT_GIB_PER_S: f64 = 200.0;
    pub const DATE_TIME_PARTS_GIB_PER_S: f64 = 50.0; // this is a guess
    pub const DELTA_GIB_PER_S: f64 = 12.8;
    pub const DICT_GIB_PER_S: f64 = 30.0; // ranges from 15-45 depending on data, picked the midpoint
    pub const FOR_GIB_PER_S: f64 = 11.3;
    pub const FSST_GIB_PER_S: f64 = 6.7;
    pub const RUN_END_GIB_PER_S: f64 = 10.0;
    pub const ZIGZAG_GIB_PER_S: f64 = 30.0;
}
