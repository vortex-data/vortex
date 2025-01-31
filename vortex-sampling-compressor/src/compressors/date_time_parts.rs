use vortex_array::aliases::hash_set::HashSet;
use vortex_array::array::TemporalArray;
use vortex_array::{Array, Encoding, EncodingId, IntoArray};
use vortex_datetime_dtype::TemporalMetadata;
use vortex_datetime_parts::{
    split_temporal, DateTimePartsArray, DateTimePartsEncoding, TemporalParts,
};
use vortex_error::VortexResult;

use crate::compressors::{CompressedArray, CompressionTree, EncodingCompressor};
use crate::downscale::downscale_integer_array;
use crate::{constants, SamplingCompressor};

#[derive(Debug)]
pub struct DateTimePartsCompressor;

impl EncodingCompressor for DateTimePartsCompressor {
    fn id(&self) -> &str {
        DateTimePartsEncoding::ID.as_ref()
    }

    fn cost(&self) -> u8 {
        constants::DATE_TIME_PARTS_COST
    }

    fn can_compress(&self, array: &Array) -> Option<&dyn EncodingCompressor> {
        if let Ok(temporal_array) = TemporalArray::try_from(array.clone()) {
            match temporal_array.temporal_metadata() {
                // We only attempt to compress Timestamp arrays.
                TemporalMetadata::Timestamp(..) => Some(self),
                _ => None,
            }
        } else {
            None
        }
    }

    fn compress<'a>(
        &'a self,
        array: &Array,
        like: Option<CompressionTree<'a>>,
        ctx: SamplingCompressor<'a>,
    ) -> VortexResult<CompressedArray<'a>> {
        let TemporalParts {
            days,
            seconds,
            subseconds,
        } = split_temporal(TemporalArray::try_from(array.clone())?)?;

        let days = ctx.named("days").compress(
            &downscale_integer_array(days)?,
            like.as_ref().and_then(|l| l.child(0)),
        )?;
        let seconds = ctx.named("seconds").compress(
            &downscale_integer_array(seconds)?,
            like.as_ref().and_then(|l| l.child(1)),
        )?;
        let subsecond = ctx.named("subsecond").compress(
            &downscale_integer_array(subseconds)?,
            like.as_ref().and_then(|l| l.child(2)),
        )?;
        Ok(CompressedArray::compressed(
            DateTimePartsArray::try_new(
                array.dtype().clone(),
                days.array,
                seconds.array,
                subsecond.array,
            )?
            .into_array(),
            Some(CompressionTree::new(
                self,
                vec![days.path, seconds.path, subsecond.path],
            )),
            array,
        ))
    }

    fn used_encodings(&self) -> HashSet<EncodingId> {
        HashSet::from([DateTimePartsEncoding::ID])
    }
}
