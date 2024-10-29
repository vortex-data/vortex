use vortex::stats::Stat;

mod read;
mod write;

mod pruning;
#[cfg(test)]
mod tests;

pub const VERSION: u16 = 1;
pub const MAGIC_BYTES: [u8; 4] = *b"VRTX";
// Size of serialized Postscript Flatbuffer
pub const FOOTER_POSTSCRIPT_SIZE: usize = 32;
pub const EOF_SIZE: usize = 8;
pub const FLAT_LAYOUT_ID: LayoutId = LayoutId(1);
pub const CHUNKED_LAYOUT_ID: LayoutId = LayoutId(2);
pub const COLUMN_LAYOUT_ID: LayoutId = LayoutId(3);
pub const INLINE_SCHEMA_LAYOUT_ID: LayoutId = LayoutId(4);

pub const PRUNING_STATS: [Stat; 4] = [Stat::Min, Stat::Max, Stat::NullCount, Stat::TrueCount];
pub const METADATA_FIELD_NAMES: [&str; 5] =
    ["row_offset", "min", "max", "null_count", "true_count"];

pub use read::*;
pub use write::*;
