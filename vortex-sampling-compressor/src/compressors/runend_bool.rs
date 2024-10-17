use vortex_array::aliases::hash_set::HashSet;
use vortex_array::array::Bool;
use vortex_array::encoding::EncodingRef;
use vortex_array::stats::ArrayStatistics as _;
use vortex_array::{ArrayDType, ArrayData, ArrayDef, IntoArrayData, IntoArrayVariant};
use vortex_dtype::DType;
use vortex_dtype::Nullability::NonNullable;
use vortex_error::VortexResult;
use vortex_runend_bool::compress::runend_bool_encode;
use vortex_runend_bool::{RunEndBool, RunEndBoolEncoding};

use crate::compressors::{CompressedArray, CompressionTree, EncodingCompressor};
use crate::{constants, SamplingCompressor};

#[derive(Debug)]
pub struct RunEndBoolCompressor;

impl EncodingCompressor for RunEndBoolCompressor {
    fn id(&self) -> &str {
        RunEndBool::ID.as_ref()
    }

    fn cost(&self) -> u8 {
        constants::RUN_END_BOOL_COST
    }

    fn can_compress(&self, array: &ArrayData) -> Option<&dyn EncodingCompressor> {
        // Only support bool arrays
        if array.encoding().id() != Bool::ID {
            return None;
        }

        if array.len() > u32::MAX as usize {
            return None;
        }

        Some(self)
    }

    fn compress<'a>(
        &'a self,
        array: &ArrayData,
        _like: Option<CompressionTree<'a>>,
        _ctx: SamplingCompressor<'a>,
    ) -> VortexResult<CompressedArray<'a>> {
        Ok(CompressedArray::compressed(
            runend_bool_encode(array.clone().into_bool()?)?.into_array(),
            Some(CompressionTree::flat(self)),
            Some(array.statistics()),
        ))
    }

    fn used_encodings(&self) -> HashSet<EncodingRef> {
        HashSet::from([&RunEndBoolEncoding as EncodingRef])
    }
}
