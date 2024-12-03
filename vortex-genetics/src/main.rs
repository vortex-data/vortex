use std::sync::{Arc, LazyLock};

use vortex::alp::{ALPEncoding, ALPRDEncoding};
use vortex::array::{
    PrimitiveEncoding, SparseEncoding, StructEncoding, VarBinEncoding, VarBinViewEncoding,
};
use vortex::bytebool::ByteBoolEncoding;
use vortex::compute::list_mean;
use vortex::datetime_parts::DateTimePartsEncoding;
use vortex::dict::DictEncoding;
use vortex::dtype::field::Field;
use vortex::encoding::EncodingRef;
use vortex::error::VortexResult;
use vortex::fastlanes::{BitPackedEncoding, DeltaEncoding, FoREncoding};
use vortex::file::{LayoutContext, LayoutDeserializer, Projection, VortexReadBuilder};
use vortex::fsst::FSSTEncoding;
use vortex::io::TokioFile;
use vortex::roaring::{RoaringBoolEncoding, RoaringIntEncoding};
use vortex::runend::RunEndEncoding;
use vortex::runend_bool::RunEndBoolEncoding;
use vortex::zigzag::ZigZagEncoding;
use vortex::Context;

pub static ALL_ENCODINGS_CONTEXT: LazyLock<Arc<Context>> = LazyLock::new(|| {
    Arc::new(Context::default().with_encodings([
        &ALPEncoding as EncodingRef,
        &ALPRDEncoding,
        &ByteBoolEncoding,
        &DateTimePartsEncoding,
        &DictEncoding,
        &BitPackedEncoding,
        &DeltaEncoding,
        &FoREncoding,
        &FSSTEncoding,
        &PrimitiveEncoding,
        &RoaringBoolEncoding,
        &RoaringIntEncoding,
        &RunEndEncoding,
        &RunEndBoolEncoding,
        &SparseEncoding,
        &StructEncoding,
        &VarBinEncoding,
        &VarBinViewEncoding,
        &ZigZagEncoding,
    ]))
});

#[tokio::main]
async fn main() -> VortexResult<()> {
    let file_path = "100_000-no-lists-of-lists.vcf.vortex";

    let builder = VortexReadBuilder::new(
        TokioFile::open(file_path)?,
        LayoutDeserializer::new(
            ALL_ENCODINGS_CONTEXT.clone(),
            LayoutContext::default().into(),
        ),
    )
    .with_projection(Projection::Flat(vec![Field::from("GT")]));

    let reader = builder.build().await?;

    let array = reader.read_all().await?;

    let means = list_mean(
        array
            .as_struct_array()
            .unwrap()
            .field_by_name("GT")
            .unwrap(),
    )?;

    println!("the means {}", means);

    Ok(())
}
