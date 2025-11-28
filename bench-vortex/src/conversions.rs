// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fs::File;
use std::path::PathBuf;

use arrow_array::RecordBatchReader;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use vortex::array::ArrayRef;
use vortex::array::arrow::FromArrowArray;
use vortex::array::iter::ArrayIteratorAdapter;
use vortex::array::iter::ArrayIteratorExt;
use vortex::array::stream::ArrayStream;
use vortex::dtype::DType;
use vortex::dtype::arrow::FromArrowType;
use vortex::error::VortexError;

pub fn parquet_to_vortex(parquet_path: PathBuf) -> anyhow::Result<impl ArrayStream> {
    let reader = ParquetRecordBatchReaderBuilder::try_new(File::open(parquet_path)?)?.build()?;

    let array_iter = ArrayIteratorAdapter::new(
        DType::from_arrow(reader.schema()),
        reader.map(|br| {
            br.map_err(VortexError::from)
                .map(|b| ArrayRef::from_arrow(b, false))
        }),
    );

    Ok(array_iter.into_array_stream())
}
