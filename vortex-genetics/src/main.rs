use std::sync::{Arc, LazyLock};

use tokio::fs::File;
// use itertools::Itertools as _;
use vortex::alp::{ALPEncoding, ALPRDEncoding};
use vortex::array::{
    PrimitiveEncoding, SparseEncoding, StructArray, StructEncoding, VarBinEncoding,
    VarBinViewEncoding,
};
use vortex::bytebool::ByteBoolEncoding;
use vortex::compute::{
    list_mean, // scalar_at
};
use vortex::datetime_parts::DateTimePartsEncoding;
use vortex::dict::DictEncoding;
use vortex::dtype::field::Field;
use vortex::encoding::EncodingRef;
use vortex::error::VortexResult;
use vortex::fastlanes::{BitPackedEncoding, DeltaEncoding, FoREncoding};
use vortex::file::{
    LayoutContext, LayoutDeserializer, Projection, VortexFileWriter, VortexReadBuilder,
};
use vortex::fsst::FSSTEncoding;
use vortex::io::{TokioAdapter, TokioFile};
use vortex::roaring::{RoaringBoolEncoding, RoaringIntEncoding};
use vortex::runend::RunEndEncoding;
use vortex::runend_bool::RunEndBoolEncoding;
use vortex::zigzag::ZigZagEncoding;
use vortex::{Context, IntoArrayData as _};

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
    let input = "100_000-no-lists-of-lists.vcf.vortex";
    let output = "100_000-GT_mean.vortex";

    let builder = VortexReadBuilder::new(
        TokioFile::open(input)?,
        LayoutDeserializer::new(
            ALL_ENCODINGS_CONTEXT.clone(),
            LayoutContext::default().into(),
        ),
    )
    .with_projection(Projection::Flat(vec![Field::from("GT")]));

    let reader = builder.build().await?;

    let array = reader.read_all().await?;

    let gt_mean = list_mean(
        array
            .as_struct_array()
            .unwrap()
            .field_by_name("GT")
            .unwrap(),
    )?;

    let mut writer = VortexFileWriter::new(TokioAdapter(File::create(output).await?));
    writer = writer
        .write_array_columns(StructArray::from_fields(&[("GT_mean", gt_mean)])?.into_array())
        .await?;
    writer.finalize().await?;

    Ok(())
}
