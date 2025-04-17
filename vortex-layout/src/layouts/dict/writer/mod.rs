use bytes::Bytes;
use vortex_array::arcref::ArcRef;
use vortex_array::compute::slice;
use vortex_array::{Array, ArrayContext, ArrayRef, RkyvMetadata, SerializeMetadata};
use vortex_dict::builders::{DictConstraints, DictEncoder, dict_encoder};
use vortex_dtype::{DType, PType};
use vortex_error::{VortexResult, vortex_err};

mod repeating;
mod single;

use crate::layouts::dict::DictLayout;
use crate::{Layout, LayoutStrategy, LayoutVTableRef, LayoutWriter, LayoutWriterExt};

#[derive(Clone)]
pub struct DictLayoutOptions {
    pub repeat: bool,
    pub constraints: DictConstraints,
}

impl Default for DictLayoutOptions {
    fn default() -> Self {
        Self {
            repeat: true,
            constraints: DictConstraints {
                max_bytes: 1024 * 1024,
                max_len: u16::MAX as usize,
            },
        }
    }
}

pub struct DictStrategy {
    pub options: DictLayoutOptions,
    pub child: ArcRef<dyn LayoutStrategy>,
    pub values: ArcRef<dyn LayoutStrategy>,
}

impl LayoutStrategy for DictStrategy {
    fn new_writer(&self, ctx: &ArrayContext, dtype: &DType) -> VortexResult<Box<dyn LayoutWriter>> {
        if !dict_layout_supported(dtype) {
            return self.child.new_writer(ctx, dtype);
        }
        if self.options.repeat {
            Ok(repeating::RepeatingDictLayoutWriter::new(
                ctx.clone(),
                dtype,
                self.child.clone(),
                self.values.clone(),
                self.options.constraints.clone(),
            )
            .boxed())
        } else {
            Ok(single::DictLayoutWriter::new(
                ctx.clone(),
                dtype,
                self.child.clone(),
                self.values.clone(),
                self.options.constraints.clone(),
            )
            .boxed())
        }
    }
}

pub fn dict_layout_supported(dtype: &DType) -> bool {
    matches!(
        dtype,
        DType::Primitive(..) | DType::Utf8(_) | DType::Binary(_)
    )
}

#[derive(Debug, Clone, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct DictLayoutMetadata {
    pub codes_ptype: PType,
}

fn dict_layout(values: Layout, codes: Layout) -> VortexResult<Layout> {
    let codes_ptype = codes.dtype().try_into()?;
    let metadata = Bytes::copy_from_slice(
        &RkyvMetadata(DictLayoutMetadata { codes_ptype })
            .serialize()
            .ok_or_else(|| vortex_err!("could not serialize dict layout metadata"))?,
    );
    Ok(Layout::new_owned(
        "dict".into(),
        LayoutVTableRef::new_ref(&DictLayout),
        values.dtype().clone(),
        codes.row_count(),
        vec![],
        vec![values, codes],
        Some(metadata),
    ))
}

enum EncodingState {
    Continue((Box<dyn DictEncoder>, ArrayRef)),
    // (values, encoded, unencoded)
    Done((ArrayRef, ArrayRef, ArrayRef)),
}

fn start_encoding(constraints: &DictConstraints, chunk: &dyn Array) -> VortexResult<EncodingState> {
    let encoder = dict_encoder(chunk, constraints)?;
    encode_chunk(encoder, chunk)
}

fn encode_chunk(
    mut encoder: Box<dyn DictEncoder>,
    chunk: &dyn Array,
) -> VortexResult<EncodingState> {
    let encoded = encoder.encode(chunk)?;
    Ok(match remainder(chunk, encoded.len())? {
        None => EncodingState::Continue((encoder, encoded)),
        Some(unencoded) => EncodingState::Done((encoder.values()?, encoded, unencoded)),
    })
}

fn remainder(array: &dyn Array, encoded_len: usize) -> VortexResult<Option<ArrayRef>> {
    (encoded_len < array.len())
        .then(|| slice(array, encoded_len, array.len()))
        .transpose()
}
