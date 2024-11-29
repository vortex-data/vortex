use std::path::Path;
use std::sync::{Arc, LazyLock};

use arrow_schema::{DataType, Field, Schema, TimeUnit};
use datafusion::datasource::listing::{
    ListingOptions, ListingTable, ListingTableConfig, ListingTableUrl,
};
use datafusion::prelude::{ParquetReadOptions, SessionContext};
use tokio::fs::{create_dir_all, OpenOptions};
use vortex::aliases::hash_map::HashMap;
use vortex::array::{ChunkedArray, StructArray};
use vortex::dtype::DType;
use vortex::error::vortex_err;
use vortex::file::{VortexFileWriter, VORTEX_FILE_EXTENSION};
use vortex::sampling_compressor::SamplingCompressor;
use vortex::variants::StructArrayTrait;
use vortex::{ArrayDType, ArrayData, IntoArrayData, IntoArrayVariant};
use vortex_datafusion::persistent::format::VortexFormat;

use crate::{idempotent_async, CTX};

pub static HITS_SCHEMA: LazyLock<Schema> = LazyLock::new(|| {
    use DataType::*;
    Schema::new(vec![
        Field::new("watchid", Int64, false),
        Field::new("javaenable", Int16, false),
        Field::new("title", Utf8View, false),
        Field::new("goodevent", Int16, false),
        Field::new("eventtime", Timestamp(TimeUnit::Microsecond, None), false),
        Field::new("eventdate", Timestamp(TimeUnit::Microsecond, None), false),
        Field::new("counterid", Int32, false),
        Field::new("clientip", Int32, false),
        Field::new("regionid", Int32, false),
        Field::new("userid", Int64, false),
        Field::new("counterclass", Int16, false),
        Field::new("os", Int16, false),
        Field::new("useragent", Int16, false),
        Field::new("url", Utf8View, false),
        Field::new("referer", Utf8View, false),
        Field::new("isrefresh", Int16, false),
        Field::new("referercategoryid", Int16, false),
        Field::new("refererregionid", Int32, false),
        Field::new("urlcategoryid", Int16, false),
        Field::new("urlregionid", Int32, false),
        Field::new("resolutionwidth", Int16, false),
        Field::new("resolutionheight", Int16, false),
        Field::new("resolutiondepth", Int16, false),
        Field::new("flashmajor", Int16, false),
        Field::new("flashminor", Int16, false),
        Field::new("flashminor2", Utf8View, false),
        Field::new("netmajor", Int16, false),
        Field::new("netminor", Int16, false),
        Field::new("useragentmajor", Int16, false),
        Field::new("useragentminor", Utf8View, false),
        Field::new("cookieenable", Int16, false),
        Field::new("javascriptenable", Int16, false),
        Field::new("ismobile", Int16, false),
        Field::new("mobilephone", Int16, false),
        Field::new("mobilephonemodel", Utf8View, false),
        Field::new("params", Utf8View, false),
        Field::new("ipnetworkid", Int32, false),
        Field::new("traficsourceid", Int16, false),
        Field::new("searchengineid", Int16, false),
        Field::new("searchphrase", Utf8View, false),
        Field::new("advengineid", Int16, false),
        Field::new("isartifical", Int16, false),
        Field::new("windowclientwidth", Int16, false),
        Field::new("windowclientheight", Int16, false),
        Field::new("clienttimezone", Int16, false),
        Field::new(
            "clienteventtime",
            Timestamp(TimeUnit::Microsecond, None),
            false,
        ),
        Field::new("silverlightversion1", Int16, false),
        Field::new("silverlightversion2", Int16, false),
        Field::new("silverlightversion3", Int32, false),
        Field::new("silverlightversion4", Int16, false),
        Field::new("pagecharset", Utf8View, false),
        Field::new("codeversion", Int32, false),
        Field::new("islink", Int16, false),
        Field::new("isdownload", Int16, false),
        Field::new("isnotbounce", Int16, false),
        Field::new("funiqid", Int64, false),
        Field::new("originalurl", Utf8View, false),
        Field::new("hid", Int32, false),
        Field::new("isoldcounter", Int16, false),
        Field::new("isevent", Int16, false),
        Field::new("isparameter", Int16, false),
        Field::new("dontcounthits", Int16, false),
        Field::new("withhash", Int16, false),
        Field::new("hitcolor", Utf8View, false),
        Field::new(
            "localeventtime",
            Timestamp(TimeUnit::Microsecond, None),
            false,
        ),
        Field::new("age", Int16, false),
        Field::new("sex", Int16, false),
        Field::new("income", Int16, false),
        Field::new("interests", Int16, false),
        Field::new("robotness", Int16, false),
        Field::new("remoteip", Int32, false),
        Field::new("windowname", Int32, false),
        Field::new("openername", Int32, false),
        Field::new("historylength", Int16, false),
        Field::new("browserlanguage", Utf8View, false),
        Field::new("browsercountry", Utf8View, false),
        Field::new("socialnetwork", Utf8View, false),
        Field::new("socialaction", Utf8View, false),
        Field::new("httperror", Int16, false),
        Field::new("sendtiming", Int32, false),
        Field::new("dnstiming", Int32, false),
        Field::new("connecttiming", Int32, false),
        Field::new("responsestarttiming", Int32, false),
        Field::new("responseendtiming", Int32, false),
        Field::new("fetchtiming", Int32, false),
        Field::new("socialsourcenetworkid", Int16, false),
        Field::new("socialsourcepage", Utf8View, false),
        Field::new("paramprice", Int64, false),
        Field::new("paramorderid", Utf8View, false),
        Field::new("paramcurrency", Utf8View, false),
        Field::new("paramcurrencyid", Int16, false),
        Field::new("openstatservicename", Utf8View, false),
        Field::new("openstatcampaignid", Utf8View, false),
        Field::new("openstatadid", Utf8View, false),
        Field::new("openstatsourceid", Utf8View, false),
        Field::new("utmsource", Utf8View, false),
        Field::new("utmmedium", Utf8View, false),
        Field::new("utmcampaign", Utf8View, false),
        Field::new("utmcontent", Utf8View, false),
        Field::new("utmterm", Utf8View, false),
        Field::new("fromtag", Utf8View, false),
        Field::new("hasgclid", Int16, false),
        Field::new("refererhash", Int64, false),
        Field::new("urlhash", Int64, false),
        Field::new("clid", Int32, false),
    ])
});

pub async fn register_vortex_file(
    session: &SessionContext,
    table_name: &str,
    input_path: &Path,
    schema: &Schema,
) -> anyhow::Result<()> {
    let vortex_dir = input_path.parent().unwrap().join("vortex_compressed");
    create_dir_all(&vortex_dir).await?;

    for idx in 0..100 {
        let parquet_file_path = input_path.join(format!("hits_{idx}.parquet"));
        let output_path = vortex_dir.join(format!("hits_{idx}.{VORTEX_FILE_EXTENSION}"));
        idempotent_async(&output_path, |vtx_file| async move {
            eprintln!("Processing file {idx}");
            let record_batches = session
                .read_parquet(
                    parquet_file_path.to_str().unwrap(),
                    ParquetReadOptions::default(),
                )
                .await?
                .collect()
                .await?;

            // Create a ChunkedArray from the set of chunks.
            let sts = record_batches
                .into_iter()
                .map(ArrayData::try_from)
                .map(|a| a.unwrap().into_struct().unwrap())
                .collect::<Vec<_>>();

            let mut arrays_map: HashMap<Arc<str>, Vec<ArrayData>> = HashMap::default();
            let mut types_map: HashMap<Arc<str>, DType> = HashMap::default();

            for st in sts.into_iter() {
                let struct_dtype = st.dtype().as_struct().unwrap();
                let names = struct_dtype.names().iter();
                let types = struct_dtype.dtypes().iter();

                for (field_name, field_type) in names.zip(types) {
                    let lower_case: Arc<str> = field_name.to_lowercase().into();
                    let val = arrays_map.entry(lower_case.clone()).or_default();
                    val.push(st.field_by_name(field_name.as_ref()).unwrap());

                    types_map.insert(lower_case, field_type.clone());
                }
            }

            let fields = schema
                .fields()
                .iter()
                .map(|field| {
                    let name: Arc<str> = field.name().to_ascii_lowercase().as_str().into();
                    let dtype = types_map[&name].clone();
                    let chunks = arrays_map.remove(&name).unwrap();
                    let chunked_child = ChunkedArray::try_new(chunks, dtype).unwrap();

                    (name, chunked_child.into_array())
                })
                .collect::<Vec<_>>();

            let data = StructArray::from_fields(&fields)?.into_array();

            let compressor = SamplingCompressor::default();
            let data = compressor.compress(&data, None)?.into_array();

            let f = OpenOptions::new()
                .write(true)
                .truncate(true)
                .create(true)
                .open(&vtx_file)
                .await?;

            let mut writer = VortexFileWriter::new(f);
            writer = writer.write_array_columns(data).await?;
            writer.finalize().await?;

            anyhow::Ok(())
        })
        .await?;
    }

    let format = Arc::new(VortexFormat::new(&CTX));
    let table_path = vortex_dir
        .to_str()
        .ok_or_else(|| vortex_err!("Path is not valid UTF-8"))?;
    let table_path = format!("file://{table_path}/");
    let table_url = ListingTableUrl::parse(table_path)?;

    let config = ListingTableConfig::new(table_url)
        .with_listing_options(ListingOptions::new(format as _))
        .with_schema(schema.clone().into());

    let listing_table = Arc::new(ListingTable::try_new(config)?);

    session.register_table(table_name, listing_table as _)?;

    Ok(())
}

pub fn clickbench_queries() -> Vec<(usize, String)> {
    let queries_file = Path::new(env!("CARGO_MANIFEST_DIR")).join("clickbench_queries.sql");

    std::fs::read_to_string(queries_file)
        .unwrap()
        .split(';')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .enumerate()
        .collect()
}
