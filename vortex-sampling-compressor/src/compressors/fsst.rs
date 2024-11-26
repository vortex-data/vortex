use std::any::Any;
use std::fmt::Debug;
use std::sync::Arc;

use fsst::Compressor;
use vortex_array::aliases::hash_set::HashSet;
use vortex_array::array::{VarBin, VarBinView};
use vortex_array::encoding::EncodingRef;
use vortex_array::stats::ArrayStatistics;
use vortex_array::{ArrayDType, ArrayDef, IntoArrayData};
use vortex_dtype::DType;
use vortex_error::{vortex_bail, VortexResult};
use vortex_fsst::{fsst_compress, fsst_train_compressor, FSSTArray, FSSTEncoding, FSST};

use super::bitpacked::BITPACK_WITH_PATCHES;
use super::delta::DeltaCompressor;
use super::r#for::FoRCompressor;
use super::varbin::VarBinCompressor;
use super::{CompressedArray, CompressionTree, EncoderMetadata, EncodingCompressor};
use crate::{constants, SamplingCompressor};

#[derive(Debug)]
pub struct FSSTCompressor;

/// Maximum size in bytes of the FSST symbol table
const FSST_SYMTAB_MAX_SIZE: usize = 8 * 255 + 255;

impl EncoderMetadata for Compressor {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl EncodingCompressor for FSSTCompressor {
    fn id(&self) -> &str {
        FSST::ID.as_ref()
    }

    fn cost(&self) -> u8 {
        constants::FSST_COST
    }

    fn can_compress(&self, array: &vortex_array::ArrayData) -> Option<&dyn EncodingCompressor> {
        // FSST arrays must have DType::Utf8.
        //
        // Note that while it can accept binary data, it is unlikely to perform well.
        if !matches!(array.dtype(), &DType::Utf8(_)) {
            return None;
        }

        // FSST can be applied on top of VarBin and VarBinView
        if array.is_encoding(VarBin::ID) || array.is_encoding(VarBinView::ID) {
            return Some(self);
        }

        None
    }

    fn compress<'a>(
        &'a self,
        array: &vortex_array::ArrayData,
        // TODO(aduffy): reuse compressor from sample run if we have saved it off.
        like: Option<CompressionTree<'a>>,
        ctx: SamplingCompressor<'a>,
    ) -> VortexResult<CompressedArray<'a>> {
        // Size-check: FSST has a builtin 2KB overhead due to the symbol table, and usually compresses
        // between 2-3x depending on the text quality.
        //
        // It's not worth running a full compression step unless the array is large enough.
        if array.nbytes() < 5 * FSST_SYMTAB_MAX_SIZE {
            return Ok(CompressedArray::uncompressed(array.clone()));
        }

        let compressor = like
            .clone()
            .and_then(|mut tree| tree.metadata())
            .map(VortexResult::Ok)
            .unwrap_or_else(|| Ok(Arc::new(fsst_train_compressor(array)?)))?;

        let Some(fsst_compressor) = compressor.as_any().downcast_ref::<Compressor>() else {
            vortex_bail!("Could not downcast metadata as FSST Compressor")
        };

        let fsst_array = if array.is_encoding(VarBin::ID) || array.is_encoding(VarBinView::ID) {
            // For a VarBinArray or VarBinViewArray, compress directly.
            fsst_compress(array, fsst_compressor)?
        } else {
            vortex_bail!(
                "Unsupported encoding for FSSTCompressor: {}",
                array.encoding().id()
            )
        };

        let codes = fsst_array.codes();
        let compressed_codes = ctx
            .auxiliary("fsst_codes")
            .excluding(self)
            .including_only(&[
                &VarBinCompressor,
                &DeltaCompressor,
                &FoRCompressor,
                &BITPACK_WITH_PATCHES,
            ])
            .compress(&codes, like.as_ref().and_then(|l| l.child(2)))?;

        // Compress the uncompressed_lengths array.
        let uncompressed_lengths = ctx
            .auxiliary("uncompressed_lengths")
            .excluding(self)
            .compress(
                &fsst_array.uncompressed_lengths(),
                like.as_ref().and_then(|l| l.child(3)),
            )?;

        Ok(CompressedArray::compressed(
            FSSTArray::try_new(
                fsst_array.dtype().clone(),
                fsst_array.symbols(),
                fsst_array.symbol_lengths(),
                compressed_codes.array,
                uncompressed_lengths.array,
            )?
            .into_array(),
            Some(CompressionTree::new_with_metadata(
                self,
                vec![None, None, compressed_codes.path, uncompressed_lengths.path],
                compressor,
            )),
            Some(array.statistics()),
        ))
    }

    fn used_encodings(&self) -> HashSet<EncodingRef> {
        HashSet::from([&FSSTEncoding as EncodingRef])
    }
}
