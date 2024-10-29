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

pub use read::*;
pub use write::*;
