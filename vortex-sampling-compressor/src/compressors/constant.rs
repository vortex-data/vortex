use vortex_array::aliases::hash_set::HashSet;
use vortex_array::array::{Constant, ConstantArray, ConstantEncoding};
use vortex_array::encoding::EncodingRef;
use vortex_array::stats::ArrayStatistics;
use vortex_array::{ArrayData, ArrayDef, IntoArrayData};
use vortex_error::{VortexExpect, VortexResult};

use crate::compressors::{CompressedArray, CompressionTree, EncodingCompressor};
use crate::{constants, SamplingCompressor};

#[derive(Debug)]
pub struct ConstantCompressor;

impl EncodingCompressor for ConstantCompressor {
    fn id(&self) -> &str {
        Constant::ID.as_ref()
    }

    fn cost(&self) -> u8 {
        constants::CONSTANT_COST
    }

    fn can_compress(&self, array: &ArrayData) -> Option<&dyn EncodingCompressor> {
        array
            .statistics()
            .compute_is_constant()?
            .then_some(self as &dyn EncodingCompressor)
    }

    fn compress<'a>(
        &'a self,
        array: &ArrayData,
        _like: Option<CompressionTree<'a>>,
        _ctx: SamplingCompressor<'a>,
    ) -> VortexResult<CompressedArray<'a>> {
        Ok(CompressedArray::compressed(
            ConstantArray::new(
                array
                    .as_constant()
                    .vortex_expect("ConstantCompressor expects constant array"),
                array.len(),
            )
            .into_array(),
            Some(CompressionTree::flat(self)),
            Some(array.statistics()),
        ))
    }

    fn used_encodings(&self) -> HashSet<EncodingRef> {
        HashSet::from([&ConstantEncoding as EncodingRef])
    }
}
