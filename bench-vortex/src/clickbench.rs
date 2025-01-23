use std::path::Path;
use std::sync::{Arc, LazyLock};

use arrow_schema::{DataType, Field, Schema, TimeUnit};
use datafusion::datasource::file_format::parquet::ParquetFormat;
use datafusion::datasource::listing::{
    ListingOptions, ListingTable, ListingTableConfig, ListingTableUrl,
};
use datafusion::prelude::{ParquetReadOptions, SessionContext};
use futures::{stream, StreamExt, TryStreamExt};
use tokio::fs::{create_dir_all, OpenOptions};
use vortex::aliases::hash_map::HashMap;
use vortex::array::{ChunkedArray, StructArray};
use vortex::dtype::DType;
use vortex::error::vortex_err;
use vortex::file::{VortexWriteOptions, VORTEX_FILE_EXTENSION};
use vortex::sampling_compressor::SamplingCompressor;
use vortex::variants::StructArrayTrait;
use vortex::{ArrayDType, ArrayData, IntoArrayData, IntoArrayVariant};
use vortex_datafusion::persistent::VortexFormat;

use crate::{idempotent_async, CTX};

pub static HITS_SCHEMA: LazyLock<Schema> = LazyLock::new(|| {
    use DataType::*;
    Schema::new(vec![
        Field::new("WatchID", Int64, false),
        Field::new("JavaEnable", Int16, false),
        Field::new("Title", Utf8View, false),
        Field::new("GoodEvent", Int16, false),
        Field::new("EventTime", Timestamp(TimeUnit::Microsecond, None), false),
        Field::new("EventDate", Timestamp(TimeUnit::Microsecond, None), false),
        Field::new("CounterID", Int32, false),
        Field::new("ClientIP", Int32, false),
        Field::new("RegionID", Int32, false),
        Field::new("UserID", Int64, false),
        Field::new("CounterClass", Int16, false),
        Field::new("OS", Int16, false),
        Field::new("UserAgent", Int16, false),
        Field::new("URL", Utf8View, false),
        Field::new("Referer", Utf8View, false),
        Field::new("IsRefresh", Int16, false),
        Field::new("RefererCategoryID", Int16, false),
        Field::new("RefererRegionID", Int32, false),
        Field::new("URLCategoryID", Int16, false),
        Field::new("URLRegionID", Int32, false),
        Field::new("ResolutionWidth", Int16, false),
        Field::new("ResolutionHeight", Int16, false),
        Field::new("ResolutionDepth", Int16, false),
        Field::new("FlashMajor", Int16, false),
        Field::new("FlashMinor", Int16, false),
        Field::new("FlashMinor2", Utf8View, false),
        Field::new("NetMajor", Int16, false),
        Field::new("NetMinor", Int16, false),
        Field::new("UserAgentMajor", Int16, false),
        Field::new("UserAgentMinor", Utf8View, false),
        Field::new("CookieEnable", Int16, false),
        Field::new("JavascriptEnable", Int16, false),
        Field::new("IsMobile", Int16, false),
        Field::new("MobilePhone", Int16, false),
        Field::new("MobilePhoneModel", Utf8View, false),
        Field::new("Params", Utf8View, false),
        Field::new("IPNetworkID", Int32, false),
        Field::new("TraficSourceID", Int16, false),
        Field::new("SearchEngineID", Int16, false),
        Field::new("SearchPhrase", Utf8View, false),
        Field::new("AdvEngineID", Int16, false),
        Field::new("IsArtifical", Int16, false),
        Field::new("WindowClientWidth", Int16, false),
        Field::new("WindowClientHeight", Int16, false),
        Field::new("ClientTimeZone", Int16, false),
        Field::new(
            "ClientEventTime",
            Timestamp(TimeUnit::Microsecond, None),
            false,
        ),
        Field::new("SilverlightVersion1", Int16, false),
        Field::new("SilverlightVersion2", Int16, false),
        Field::new("SilverlightVersion3", Int32, false),
        Field::new("SilverlightVersion4", Int16, false),
        Field::new("PageCharset", Utf8View, false),
        Field::new("CodeVersion", Int32, false),
        Field::new("IsLink", Int16, false),
        Field::new("IsDownload", Int16, false),
        Field::new("IsNotBounce", Int16, false),
        Field::new("FUniqID", Int64, false),
        Field::new("OriginalURL", Utf8View, false),
        Field::new("HID", Int32, false),
        Field::new("IsOldCounter", Int16, false),
        Field::new("IsEvent", Int16, false),
        Field::new("IsParameter", Int16, false),
        Field::new("DontCountHits", Int16, false),
        Field::new("WithHash", Int16, false),
        Field::new("HitColor", Utf8View, false),
        Field::new(
            "LocalEventTime",
            Timestamp(TimeUnit::Microsecond, None),
            false,
        ),
        Field::new("Age", Int16, false),
        Field::new("Sex", Int16, false),
        Field::new("Income", Int16, false),
        Field::new("Interests", Int16, false),
        Field::new("Robotness", Int16, false),
        Field::new("RemoteIP", Int32, false),
        Field::new("WindowName", Int32, false),
        Field::new("OpenerName", Int32, false),
        Field::new("HistoryLength", Int16, false),
        Field::new("BrowserLanguage", Utf8View, false),
        Field::new("BrowserCountry", Utf8View, false),
        Field::new("SocialNetwork", Utf8View, false),
        Field::new("SocialAction", Utf8View, false),
        Field::new("HTTPError", Int16, false),
        Field::new("SendTiming", Int32, false),
        Field::new("DNSTiming", Int32, false),
        Field::new("ConnectTiming", Int32, false),
        Field::new("ResponseStartTiming", Int32, false),
        Field::new("ResponseEndTiming", Int32, false),
        Field::new("FetchTiming", Int32, false),
        Field::new("SocialSourceNetworkID", Int16, false),
        Field::new("SocialSourcePage", Utf8View, false),
        Field::new("ParamPrice", Int64, false),
        Field::new("ParamOrderID", Utf8View, false),
        Field::new("ParamCurrency", Utf8View, false),
        Field::new("ParamCurrencyID", Int16, false),
        Field::new("OpenstatServiceName", Utf8View, false),
        Field::new("OpenstatCampaignID", Utf8View, false),
        Field::new("OpenstatAdID", Utf8View, false),
        Field::new("OpenstatSourceID", Utf8View, false),
        Field::new("UTMSource", Utf8View, false),
        Field::new("UTMMedium", Utf8View, false),
        Field::new("UTMCampaign", Utf8View, false),
        Field::new("UTMContent", Utf8View, false),
        Field::new("UTMTerm", Utf8View, false),
        Field::new("FromTag", Utf8View, false),
        Field::new("HasGCLID", Int16, false),
        Field::new("RefererHash", Int64, false),
        Field::new("URLHash", Int64, false),
        Field::new("CLID", Int32, false),
    ])
});

pub async fn register_vortex_files(
    session: SessionContext,
    table_name: &str,
    input_path: &Path,
    schema: &Schema,
) -> anyhow::Result<()> {
    let vortex_dir = input_path.join("vortex");
    create_dir_all(&vortex_dir).await?;

    stream::iter(0..100)
        .map(|idx| {
            let parquet_file_path = input_path
                .join("parquet")
                .join(format!("hits_{idx}.parquet"));
            let output_path = vortex_dir.join(format!("hits_{idx}.{VORTEX_FILE_EXTENSION}"));
            let session = session.clone();
            let schema = schema.clone();

            tokio::spawn(async move {
                let output_path = output_path.clone();
                idempotent_async(&output_path, move |vtx_file| async move {
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
                        let types = struct_dtype.dtypes();

                        for (field_name, field_type) in names.zip(types) {
                            let val = arrays_map.entry(field_name.clone()).or_default();
                            val.push(st.maybe_null_field_by_name(field_name.as_ref()).unwrap());

                            types_map.insert(field_name.clone(), field_type.clone());
                        }
                    }

                    let fields = schema
                        .fields()
                        .iter()
                        .map(|field| {
                            let name: Arc<str> = field.name().as_str().into();
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

                    VortexWriteOptions::default()
                        .write(f, data.into_array_stream())
                        .await?;

                    anyhow::Ok(())
                })
                .await
                .expect("Failed to write Vortex file")
            })
        })
        .buffer_unordered(16)
        .try_collect::<Vec<_>>()
        .await?;

    let format = Arc::new(VortexFormat::new(CTX.clone()));
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

pub async fn register_parquet_files(
    session: &SessionContext,
    table_name: &str,
    input_path: &Path,
    schema: &Schema,
) -> anyhow::Result<()> {
    let format = Arc::new(ParquetFormat::new());
    let table_path = input_path.join("parquet");
    let table_path = format!(
        "file://{}/",
        table_path
            .to_str()
            .ok_or_else(|| vortex_err!("Path is not valid UTF-8"))?
    );
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
