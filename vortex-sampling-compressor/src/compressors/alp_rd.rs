use std::any::Any;
use std::sync::Arc;

use vortex_alp::{match_each_alp_float_ptype, ALPRDEncoding, RDEncoder as ALPRDEncoder, ALPRD};
use vortex_array::aliases::hash_set::HashSet;
use vortex_array::array::PrimitiveArray;
use vortex_array::encoding::EncodingRef;
use vortex_array::stats::ArrayStatistics;
use vortex_array::variants::PrimitiveArrayTrait;
use vortex_array::{ArrayData, ArrayDef, IntoArrayData, IntoArrayVariant};
use vortex_dtype::PType;
use vortex_error::{vortex_bail, VortexResult};
use vortex_fastlanes::BitPackedEncoding;

use crate::compressors::{CompressedArray, CompressionTree, EncoderMetadata, EncodingCompressor};
use crate::{constants, SamplingCompressor};

#[derive(Debug)]
pub struct ALPRDCompressor;

impl EncoderMetadata for ALPRDEncoder {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl EncodingCompressor for ALPRDCompressor {
    fn id(&self) -> &str {
        ALPRD::ID.as_ref()
    }

    fn cost(&self) -> u8 {
        constants::ALP_RD_COST
    }

    fn can_compress(&self, array: &ArrayData) -> Option<&dyn EncodingCompressor> {
        // Only support primitive arrays
        let parray = PrimitiveArray::try_from(array.clone()).ok()?;

        // Only supports f32 and f64
        if !matches!(parray.ptype(), PType::F32 | PType::F64) {
            return None;
        }

        Some(self)
    }

    fn compress<'a>(
        &'a self,
        array: &ArrayData,
        like: Option<CompressionTree<'a>>,
        _ctx: SamplingCompressor<'a>,
    ) -> VortexResult<CompressedArray<'a>> {
        let primitive = array.clone().into_primitive()?;

        // Train a new compressor or reuse an existing compressor.
        let encoder = like
            .clone()
            .and_then(|mut tree| tree.metadata())
            .map(VortexResult::Ok)
            .unwrap_or_else(|| Ok(Arc::new(alp_rd_new_encoder(&primitive))))?;

        let Some(alp_rd_encoder) = encoder.as_any().downcast_ref::<ALPRDEncoder>() else {
            vortex_bail!("Could not downcast metadata as ALPRDEncoder");
        };

        let encoded = alp_rd_encoder.encode(&primitive).into_array();
        Ok(CompressedArray::compressed(
            encoded,
            Some(CompressionTree::new_with_metadata(self, vec![], encoder)),
            Some(array.statistics()),
        ))
    }

    fn used_encodings(&self) -> HashSet<EncodingRef> {
        HashSet::from([&ALPRDEncoding as EncodingRef, &BitPackedEncoding])
    }
}

/// Create a new `ALPRDEncoder` from the given array of samples.
fn alp_rd_new_encoder(array: &PrimitiveArray) -> ALPRDEncoder {
    match_each_alp_float_ptype!(array.ptype(), |$P| {
        ALPRDEncoder::new(array.maybe_null_slice::<$P>())
    })
}
