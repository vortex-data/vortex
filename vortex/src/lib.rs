// https://github.com/rust-lang/cargo/pull/11645#issuecomment-1536905941
#![doc = include_str!(concat!("../", env!("CARGO_PKG_README")))]

pub use vortex_array::*;
#[cfg(feature = "files")]
pub use vortex_file as file;
#[cfg(feature = "files")]
pub use vortex_io as io;
pub use {
    vortex_btrblocks as compressor, vortex_buffer as buffer, vortex_dtype as dtype,
    vortex_error as error, vortex_expr as expr, vortex_flatbuffers as flatbuffers,
    vortex_ipc as ipc, vortex_layout as layout, vortex_mask as mask, vortex_proto as proto,
    vortex_scalar as scalar,
};

pub mod encodings {
    pub use {
        vortex_alp as alp, vortex_bytebool as bytebool, vortex_datetime_parts as datetime_parts,
        vortex_dict as dict, vortex_fastlanes as fastlanes, vortex_fsst as fsst,
        vortex_runend as runend, vortex_sparse as sparse, vortex_zigzag as zigzag,
    };
}
