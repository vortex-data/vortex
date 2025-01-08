use std::fs::File;
use std::os::unix::prelude::MetadataExt;
use std::path::PathBuf;

use vortex::array::{ChunkedArray, StructArray};
use vortex::dtype::DType;
use vortex::error::VortexResult;
use vortex::variants::StructArrayTrait;
use vortex::ArrayDType;

use crate::data_downloads::FileType;
use crate::reader::open_vortex;
use crate::CompressionRunStats;

pub async fn vortex_chunk_sizes(path: PathBuf) -> VortexResult<CompressionRunStats> {
    let file = File::open(path.as_path())?;
    let total_compressed_size = file.metadata()?.size();
    let vortex = open_vortex(path.as_path()).await?;
    let DType::Struct(st, _) = vortex.dtype() else {
        unreachable!()
    };

    let mut compressed_sizes = vec![0; st.names().len()];
    let chunked_array = ChunkedArray::try_from(vortex).unwrap();
    for chunk in chunked_array.chunks() {
        let struct_arr = StructArray::try_from(chunk).unwrap();
        for (i, f) in
            (0..struct_arr.nfields()).map(|i| (i, struct_arr.maybe_null_field_by_idx(i).unwrap()))
        {
            compressed_sizes[i] += f.nbytes() as u64;
        }
    }

    let stats = CompressionRunStats {
        schema: chunked_array.dtype().clone(),
        file_type: FileType::Vortex,
        total_compressed_size: Some(total_compressed_size),
        compressed_sizes,
        file_name: path
            .with_extension("")
            .file_name()
            .unwrap()
            .to_string_lossy()
            .to_string(),
    };
    Ok(stats)
}
