// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use arrow_array::builder::{Int32Builder, ListBuilder, StringBuilder};
use futures::StreamExt;
use indicatif::ProgressBar;
use noodles_vcf::variant::record::info::field::value::{Array, Value};
use noodles_vcf::variant::record::samples::Series;
use noodles_vcf::variant::record::samples::series::value::Array as EntryArray;
use noodles_vcf::variant::record::samples::series::value::Value as EntryValue;
use noodles_vcf::{Header, Record};
use parquet::arrow::AsyncArrowWriter;
use parquet::arrow::ParquetRecordBatchStreamBuilder;
use reqwest::Client;
use std::borrow::Cow;
use std::io;
use std::sync::Arc;
use tokio::fs::{File, create_dir_all};
use tokio::io::BufReader;
use tokio::runtime::Handle;
use tokio_util::io::StreamReader;
use tracing::info;
use vortex::ArrayRef;
use vortex::arrow::FromArrowArray;
use vortex::compressor::CompactCompressor;
use vortex::dtype::DType;
use vortex::dtype::arrow::FromArrowType;
use vortex::error::{VortexError, VortexExpect, VortexResult};
use vortex::error::{vortex_bail, vortex_err};
use vortex::file::{VortexLayoutStrategy, VortexWriteOptions};
use vortex::stream::ArrayStreamAdapter;
use vortex::utils::aliases::hash_map::HashMap;

use crate::Format;
use crate::idempotent_async;
use crate::statpopgen::builder::GnomADBuilder;
use crate::statpopgen::schema::SCHEMA;
use crate::vortex_panic;

use super::StatPopGenBenchmark;

impl StatPopGenBenchmark {
    pub async fn download_parquet(&self) -> VortexResult<()> {
        let url = "https://gnomad-public-us-east-1.s3.amazonaws.com/release/3.1.2/vcf/genomes/gnomad.genomes.v3.1.2.hgdp_tgp.chr21.vcf.bgz";
        let parquet_output_path = self.parquet_path()?;
        idempotent_async(&parquet_output_path, async |parquet_output_path| {
            info!(
                "Downloading first {} lines of gnomAD v3.1.2 HGDP-1kG chr21.",
                self.n_rows
            );

            // Fetch the remote stream
            let client = Client::new();
            let response = client
                .get(url)
                .send()
                .await
                .map_err(|err| vortex_err!("reqwest failed: {err}"))?
                .error_for_status()
                .map_err(|err| vortex_err!("reqwest bad status: {err}"))?;
            let byte_stream = response.bytes_stream().map(|x| x.map_err(io::Error::other));
            let stream_reader = StreamReader::new(byte_stream);

            // Wrap in BGZF reader
            let buf_reader = BufReader::new(stream_reader);
            let mut bgzf_reader = noodles_bgzf::r#async::io::Reader::new(buf_reader);

            // Read and parse VCF header
            let mut vcf_reader = noodles_vcf::AsyncReader::new(&mut bgzf_reader);

            let mut builder = GnomADBuilder::new();

            // Read and print the first 100,000 records
            let mut record = Record::default();
            let header = vcf_reader.read_header().await?;
            let progress = ProgressBar::new(self.n_rows);
            for i in progress.wrap_iter(0..self.n_rows) {
                let bytes_read = vcf_reader.read_record(&mut record).await?;
                if bytes_read == 0 {
                    vortex_bail!("Reached end of stream after only {} records.", i)
                }

                builder
                    .CHROM_builder
                    .append_value(record.reference_sequence_name());
                builder.POS_builder.append_value(
                    record
                        .variant_start()
                        .expect("pos must not be null")
                        .expect("pos must not err")
                        .get() as u64,
                );
                builder.ID_builder.append_value(record.ids().as_ref());
                builder.REF_builder.append_value(record.reference_bases());
                builder.ALT_builder.append_value(
                    record
                        .alternate_bases()
                        .as_ref()
                        .split(",")
                        .map(|x| (x != ".").then_some(x)),
                );
                builder
                    .QUAL_builder
                    .append_option(record.quality_score().transpose()?);
                builder.FILTER_builder.append_value(
                    record
                        .filters()
                        .as_ref()
                        .split(";")
                        .map(|x| (x != ".").then_some(x)),
                );

                {
                    let info = record.info();
                    let info_fields: HashMap<&str, Option<Value>> = info
                        .iter(&header)
                        .map(|x| x.expect("no errors allowed"))
                        .collect();

                    fn value_int32(v: Option<&Value>) -> VortexResult<Option<i32>> {
                        Ok(match v {
                            None => None,
                            Some(Value::Integer(x)) => Some(*x),
                            _ => vortex_bail!("expected int32 {:?}", v),
                        })
                    }
                    fn value_float64(v: Option<&Value>) -> VortexResult<Option<f64>> {
                        Ok(match v {
                            None => None,
                            Some(Value::Float(x)) => Some(*x as f64),
                            _ => vortex_bail!("expected f64 {:?}", v),
                        })
                    }
                    fn value_string<'a>(v: Option<&'a Value>) -> VortexResult<Option<&'a str>> {
                        Ok(match v {
                            None => None,
                            Some(Value::String(x)) => Some(x),
                            _ => vortex_bail!("expected string {:?}", v),
                        })
                    }
                    fn value_boolean(v: Option<&Value>) -> VortexResult<bool> {
                        Ok(match v {
                            None => false,
                            Some(Value::Flag) => true,
                            _ => vortex_bail!("expected bool {:?}", v),
                        })
                    }
                    fn value_list_int32(
                        v: Option<&Value>,
                    ) -> VortexResult<Option<impl Iterator<Item = Option<i32>>>>
                    {
                        Ok(match v {
                            None => None,
                            Some(Value::Array(a)) => match a {
                                Array::Integer(values) => {
                                    Some(values.iter().map(|x| x.expect("no errors")))
                                }
                                _ => vortex_bail!("expected int32 {:?}", v),
                            },
                            _ => vortex_bail!("expected int32 {:?}", v),
                        })
                    }
                    fn value_list_float64(
                        v: Option<&Value>,
                    ) -> VortexResult<Option<impl Iterator<Item = Option<f64>>>>
                    {
                        Ok(match v {
                            None => None,
                            Some(Value::Array(a)) => match a {
                                Array::Float(values) => Some(
                                    values
                                        .iter()
                                        .map(|x| x.expect("no errors").map(|x| x as f64)),
                                ),
                                _ => vortex_bail!("expected int32 {:?}", v),
                            },
                            _ => vortex_bail!("expected f64 {:?}", v),
                        })
                    }
                    fn value_list_string<'a>(
                        v: Option<&'a Value>,
                    ) -> VortexResult<Option<impl Iterator<Item = Option<Cow<'a, str>>>>>
                    {
                        Ok(match v {
                            None => None,
                            Some(Value::Array(a)) => match a {
                                Array::String(values) => {
                                    Some(values.iter().map(|x| x.expect("no errors")))
                                }
                                _ => vortex_bail!("expected int32 {:?}", v),
                            },
                            _ => vortex_bail!("expected string {:?}", v),
                        })
                    }

                    builder.AC_builder.append_option(value_list_int32(
                        info_fields.get("AC").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_builder.append_option(value_int32(
                        info_fields.get("AN").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_builder.append_option(value_list_float64(
                        info_fields.get("AF").and_then(|x| x.as_ref()),
                    )?);
                    builder.AC_raw_builder.append_option(value_list_int32(
                        info_fields.get("AC_raw").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_raw_builder.append_option(value_int32(
                        info_fields.get("AN_raw").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_raw_builder.append_option(value_list_float64(
                        info_fields.get("AF_raw").and_then(|x| x.as_ref()),
                    )?);
                    builder.gnomad_AC_builder.append_option(value_list_int32(
                        info_fields.get("gnomad_AC").and_then(|x| x.as_ref()),
                    )?);
                    builder.gnomad_AN_builder.append_option(value_int32(
                        info_fields.get("gnomad_AN").and_then(|x| x.as_ref()),
                    )?);
                    builder.gnomad_AF_builder.append_option(value_list_float64(
                        info_fields.get("gnomad_AF").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .gnomad_popmax_builder
                        .append_option(value_list_string(
                            info_fields.get("gnomad_popmax").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .gnomad_faf95_popmax_builder
                        .append_option(value_list_float64(
                            info_fields
                                .get("gnomad_faf95_popmax")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .gnomad_AC_raw_builder
                        .append_option(value_list_int32(
                            info_fields.get("gnomad_AC_raw").and_then(|x| x.as_ref()),
                        )?);
                    builder.gnomad_AN_raw_builder.append_option(value_int32(
                        info_fields.get("gnomad_AN_raw").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .gnomad_AF_raw_builder
                        .append_option(value_list_float64(
                            info_fields.get("gnomad_AF_raw").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .AC_italian_XY_builder
                        .append_option(value_list_int32(
                            info_fields.get("AC_italian_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder.AN_italian_XY_builder.append_option(value_int32(
                        info_fields.get("AN_italian_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .AF_italian_XY_builder
                        .append_option(value_list_float64(
                            info_fields.get("AF_italian_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .nhomalt_italian_XY_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("nhomalt_italian_XY")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_gwd_XX_builder.append_option(value_list_int32(
                        info_fields.get("AC_gwd_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_gwd_XX_builder.append_option(value_int32(
                        info_fields.get("AN_gwd_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_gwd_XX_builder.append_option(value_list_float64(
                        info_fields.get("AF_gwd_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .nhomalt_gwd_XX_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_gwd_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_she_XY_builder.append_option(value_list_int32(
                        info_fields.get("AC_she_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_she_XY_builder.append_option(value_int32(
                        info_fields.get("AN_she_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_she_XY_builder.append_option(value_list_float64(
                        info_fields.get("AF_she_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .nhomalt_she_XY_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_she_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .AC_biakapygmy_builder
                        .append_option(value_list_int32(
                            info_fields.get("AC_biakapygmy").and_then(|x| x.as_ref()),
                        )?);
                    builder.AN_biakapygmy_builder.append_option(value_int32(
                        info_fields.get("AN_biakapygmy").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .AF_biakapygmy_builder
                        .append_option(value_list_float64(
                            info_fields.get("AF_biakapygmy").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .nhomalt_biakapygmy_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("nhomalt_biakapygmy")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_tsi_XY_builder.append_option(value_list_int32(
                        info_fields.get("AC_tsi_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_tsi_XY_builder.append_option(value_int32(
                        info_fields.get("AN_tsi_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_tsi_XY_builder.append_option(value_list_float64(
                        info_fields.get("AF_tsi_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .nhomalt_tsi_XY_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_tsi_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_surui_builder.append_option(value_list_int32(
                        info_fields.get("AC_surui").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_surui_builder.append_option(value_int32(
                        info_fields.get("AN_surui").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_surui_builder.append_option(value_list_float64(
                        info_fields.get("AF_surui").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .nhomalt_surui_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_surui").and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_esn_XX_builder.append_option(value_list_int32(
                        info_fields.get("AC_esn_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_esn_XX_builder.append_option(value_int32(
                        info_fields.get("AN_esn_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_esn_XX_builder.append_option(value_list_float64(
                        info_fields.get("AF_esn_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .nhomalt_esn_XX_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_esn_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_ceu_builder.append_option(value_list_int32(
                        info_fields.get("AC_ceu").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_ceu_builder.append_option(value_int32(
                        info_fields.get("AN_ceu").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_ceu_builder.append_option(value_list_float64(
                        info_fields.get("AF_ceu").and_then(|x| x.as_ref()),
                    )?);
                    builder.nhomalt_ceu_builder.append_option(value_list_int32(
                        info_fields.get("nhomalt_ceu").and_then(|x| x.as_ref()),
                    )?);
                    builder.AC_pjl_XX_builder.append_option(value_list_int32(
                        info_fields.get("AC_pjl_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_pjl_XX_builder.append_option(value_int32(
                        info_fields.get("AN_pjl_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_pjl_XX_builder.append_option(value_list_float64(
                        info_fields.get("AF_pjl_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .nhomalt_pjl_XX_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_pjl_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_gbr_XX_builder.append_option(value_list_int32(
                        info_fields.get("AC_gbr_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_gbr_XX_builder.append_option(value_int32(
                        info_fields.get("AN_gbr_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_gbr_XX_builder.append_option(value_list_float64(
                        info_fields.get("AF_gbr_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .nhomalt_gbr_XX_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_gbr_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_druze_builder.append_option(value_list_int32(
                        info_fields.get("AC_druze").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_druze_builder.append_option(value_int32(
                        info_fields.get("AN_druze").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_druze_builder.append_option(value_list_float64(
                        info_fields.get("AF_druze").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .nhomalt_druze_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_druze").and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_khv_XY_builder.append_option(value_list_int32(
                        info_fields.get("AC_khv_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_khv_XY_builder.append_option(value_int32(
                        info_fields.get("AN_khv_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_khv_XY_builder.append_option(value_list_float64(
                        info_fields.get("AF_khv_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .nhomalt_khv_XY_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_khv_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_chs_XX_builder.append_option(value_list_int32(
                        info_fields.get("AC_chs_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_chs_XX_builder.append_option(value_int32(
                        info_fields.get("AN_chs_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_chs_XX_builder.append_option(value_list_float64(
                        info_fields.get("AF_chs_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .nhomalt_chs_XX_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_chs_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_french_builder.append_option(value_list_int32(
                        info_fields.get("AC_french").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_french_builder.append_option(value_int32(
                        info_fields.get("AN_french").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_french_builder.append_option(value_list_float64(
                        info_fields.get("AF_french").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .nhomalt_french_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_french").and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_daur_XX_builder.append_option(value_list_int32(
                        info_fields.get("AC_daur_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_daur_XX_builder.append_option(value_int32(
                        info_fields.get("AN_daur_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_daur_XX_builder.append_option(value_list_float64(
                        info_fields.get("AF_daur_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .nhomalt_daur_XX_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_daur_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_itu_builder.append_option(value_list_int32(
                        info_fields.get("AC_itu").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_itu_builder.append_option(value_int32(
                        info_fields.get("AN_itu").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_itu_builder.append_option(value_list_float64(
                        info_fields.get("AF_itu").and_then(|x| x.as_ref()),
                    )?);
                    builder.nhomalt_itu_builder.append_option(value_list_int32(
                        info_fields.get("nhomalt_itu").and_then(|x| x.as_ref()),
                    )?);
                    builder.AC_yizu_XY_builder.append_option(value_list_int32(
                        info_fields.get("AC_yizu_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_yizu_XY_builder.append_option(value_int32(
                        info_fields.get("AN_yizu_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_yizu_XY_builder.append_option(value_list_float64(
                        info_fields.get("AF_yizu_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .nhomalt_yizu_XY_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_yizu_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_yri_XX_builder.append_option(value_list_int32(
                        info_fields.get("AC_yri_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_yri_XX_builder.append_option(value_int32(
                        info_fields.get("AN_yri_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_yri_XX_builder.append_option(value_list_float64(
                        info_fields.get("AF_yri_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .nhomalt_yri_XX_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_yri_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_oroqen_XY_builder.append_option(value_list_int32(
                        info_fields.get("AC_oroqen_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_oroqen_XY_builder.append_option(value_int32(
                        info_fields.get("AN_oroqen_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .AF_oroqen_XY_builder
                        .append_option(value_list_float64(
                            info_fields.get("AF_oroqen_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .nhomalt_oroqen_XY_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("nhomalt_oroqen_XY")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_clm_XY_builder.append_option(value_list_int32(
                        info_fields.get("AC_clm_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_clm_XY_builder.append_option(value_int32(
                        info_fields.get("AN_clm_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_clm_XY_builder.append_option(value_list_float64(
                        info_fields.get("AF_clm_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .nhomalt_clm_XY_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_clm_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .AC_makrani_XX_builder
                        .append_option(value_list_int32(
                            info_fields.get("AC_makrani_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder.AN_makrani_XX_builder.append_option(value_int32(
                        info_fields.get("AN_makrani_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .AF_makrani_XX_builder
                        .append_option(value_list_float64(
                            info_fields.get("AF_makrani_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .nhomalt_makrani_XX_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("nhomalt_makrani_XX")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_fin_XX_builder.append_option(value_list_int32(
                        info_fields.get("AC_fin_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_fin_XX_builder.append_option(value_int32(
                        info_fields.get("AN_fin_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_fin_XX_builder.append_option(value_list_float64(
                        info_fields.get("AF_fin_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .nhomalt_fin_XX_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_fin_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .AC_karitiana_XY_builder
                        .append_option(value_list_int32(
                            info_fields.get("AC_karitiana_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder.AN_karitiana_XY_builder.append_option(value_int32(
                        info_fields.get("AN_karitiana_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .AF_karitiana_XY_builder
                        .append_option(value_list_float64(
                            info_fields.get("AF_karitiana_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .nhomalt_karitiana_XY_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("nhomalt_karitiana_XY")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_adygei_builder.append_option(value_list_int32(
                        info_fields.get("AC_adygei").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_adygei_builder.append_option(value_int32(
                        info_fields.get("AN_adygei").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_adygei_builder.append_option(value_list_float64(
                        info_fields.get("AF_adygei").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .nhomalt_adygei_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_adygei").and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_sindhi_XY_builder.append_option(value_list_int32(
                        info_fields.get("AC_sindhi_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_sindhi_XY_builder.append_option(value_int32(
                        info_fields.get("AN_sindhi_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .AF_sindhi_XY_builder
                        .append_option(value_list_float64(
                            info_fields.get("AF_sindhi_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .nhomalt_sindhi_XY_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("nhomalt_sindhi_XY")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_acb_XX_builder.append_option(value_list_int32(
                        info_fields.get("AC_acb_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_acb_XX_builder.append_option(value_int32(
                        info_fields.get("AN_acb_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_acb_XX_builder.append_option(value_list_float64(
                        info_fields.get("AF_acb_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .nhomalt_acb_XX_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_acb_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_papuan_XY_builder.append_option(value_list_int32(
                        info_fields.get("AC_papuan_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_papuan_XY_builder.append_option(value_int32(
                        info_fields.get("AN_papuan_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .AF_papuan_XY_builder
                        .append_option(value_list_float64(
                            info_fields.get("AF_papuan_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .nhomalt_papuan_XY_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("nhomalt_papuan_XY")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_pel_XX_builder.append_option(value_list_int32(
                        info_fields.get("AC_pel_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_pel_XX_builder.append_option(value_int32(
                        info_fields.get("AN_pel_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_pel_XX_builder.append_option(value_list_float64(
                        info_fields.get("AF_pel_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .nhomalt_pel_XX_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_pel_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_daur_builder.append_option(value_list_int32(
                        info_fields.get("AC_daur").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_daur_builder.append_option(value_int32(
                        info_fields.get("AN_daur").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_daur_builder.append_option(value_list_float64(
                        info_fields.get("AF_daur").and_then(|x| x.as_ref()),
                    )?);
                    builder.nhomalt_daur_builder.append_option(value_list_int32(
                        info_fields.get("nhomalt_daur").and_then(|x| x.as_ref()),
                    )?);
                    builder.AC_pel_XY_builder.append_option(value_list_int32(
                        info_fields.get("AC_pel_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_pel_XY_builder.append_option(value_int32(
                        info_fields.get("AN_pel_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_pel_XY_builder.append_option(value_list_float64(
                        info_fields.get("AF_pel_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .nhomalt_pel_XY_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_pel_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_colombian_builder.append_option(value_list_int32(
                        info_fields.get("AC_colombian").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_colombian_builder.append_option(value_int32(
                        info_fields.get("AN_colombian").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .AF_colombian_builder
                        .append_option(value_list_float64(
                            info_fields.get("AF_colombian").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .nhomalt_colombian_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("nhomalt_colombian")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_surui_XY_builder.append_option(value_list_int32(
                        info_fields.get("AC_surui_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_surui_XY_builder.append_option(value_int32(
                        info_fields.get("AN_surui_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .AF_surui_XY_builder
                        .append_option(value_list_float64(
                            info_fields.get("AF_surui_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .nhomalt_surui_XY_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_surui_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_gih_builder.append_option(value_list_int32(
                        info_fields.get("AC_gih").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_gih_builder.append_option(value_int32(
                        info_fields.get("AN_gih").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_gih_builder.append_option(value_list_float64(
                        info_fields.get("AF_gih").and_then(|x| x.as_ref()),
                    )?);
                    builder.nhomalt_gih_builder.append_option(value_list_int32(
                        info_fields.get("nhomalt_gih").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .AC_russian_XY_builder
                        .append_option(value_list_int32(
                            info_fields.get("AC_russian_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder.AN_russian_XY_builder.append_option(value_int32(
                        info_fields.get("AN_russian_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .AF_russian_XY_builder
                        .append_option(value_list_float64(
                            info_fields.get("AF_russian_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .nhomalt_russian_XY_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("nhomalt_russian_XY")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .AC_karitiana_XX_builder
                        .append_option(value_list_int32(
                            info_fields.get("AC_karitiana_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder.AN_karitiana_XX_builder.append_option(value_int32(
                        info_fields.get("AN_karitiana_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .AF_karitiana_XX_builder
                        .append_option(value_list_float64(
                            info_fields.get("AF_karitiana_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .nhomalt_karitiana_XX_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("nhomalt_karitiana_XX")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_pima_builder.append_option(value_list_int32(
                        info_fields.get("AC_pima").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_pima_builder.append_option(value_int32(
                        info_fields.get("AN_pima").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_pima_builder.append_option(value_list_float64(
                        info_fields.get("AF_pima").and_then(|x| x.as_ref()),
                    )?);
                    builder.nhomalt_pima_builder.append_option(value_list_int32(
                        info_fields.get("nhomalt_pima").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .AC_japanese_XX_builder
                        .append_option(value_list_int32(
                            info_fields.get("AC_japanese_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder.AN_japanese_XX_builder.append_option(value_int32(
                        info_fields.get("AN_japanese_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .AF_japanese_XX_builder
                        .append_option(value_list_float64(
                            info_fields.get("AF_japanese_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .nhomalt_japanese_XX_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("nhomalt_japanese_XX")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_beb_XY_builder.append_option(value_list_int32(
                        info_fields.get("AC_beb_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_beb_XY_builder.append_option(value_int32(
                        info_fields.get("AN_beb_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_beb_XY_builder.append_option(value_list_float64(
                        info_fields.get("AF_beb_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .nhomalt_beb_XY_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_beb_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .AC_bedouin_XY_builder
                        .append_option(value_list_int32(
                            info_fields.get("AC_bedouin_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder.AN_bedouin_XY_builder.append_option(value_int32(
                        info_fields.get("AN_bedouin_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .AF_bedouin_XY_builder
                        .append_option(value_list_float64(
                            info_fields.get("AF_bedouin_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .nhomalt_bedouin_XY_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("nhomalt_bedouin_XY")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_hazara_XX_builder.append_option(value_list_int32(
                        info_fields.get("AC_hazara_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_hazara_XX_builder.append_option(value_int32(
                        info_fields.get("AN_hazara_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .AF_hazara_XX_builder
                        .append_option(value_list_float64(
                            info_fields.get("AF_hazara_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .nhomalt_hazara_XX_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("nhomalt_hazara_XX")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_han_builder.append_option(value_list_int32(
                        info_fields.get("AC_han").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_han_builder.append_option(value_int32(
                        info_fields.get("AN_han").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_han_builder.append_option(value_list_float64(
                        info_fields.get("AF_han").and_then(|x| x.as_ref()),
                    )?);
                    builder.nhomalt_han_builder.append_option(value_list_int32(
                        info_fields.get("nhomalt_han").and_then(|x| x.as_ref()),
                    )?);
                    builder.AC_tujia_XY_builder.append_option(value_list_int32(
                        info_fields.get("AC_tujia_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_tujia_XY_builder.append_option(value_int32(
                        info_fields.get("AN_tujia_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .AF_tujia_XY_builder
                        .append_option(value_list_float64(
                            info_fields.get("AF_tujia_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .nhomalt_tujia_XY_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_tujia_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_druze_XY_builder.append_option(value_list_int32(
                        info_fields.get("AC_druze_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_druze_XY_builder.append_option(value_int32(
                        info_fields.get("AN_druze_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .AF_druze_XY_builder
                        .append_option(value_list_float64(
                            info_fields.get("AF_druze_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .nhomalt_druze_XY_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_druze_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .AC_melanesian_XX_builder
                        .append_option(value_list_int32(
                            info_fields.get("AC_melanesian_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder.AN_melanesian_XX_builder.append_option(value_int32(
                        info_fields.get("AN_melanesian_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .AF_melanesian_XX_builder
                        .append_option(value_list_float64(
                            info_fields.get("AF_melanesian_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .nhomalt_melanesian_XX_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("nhomalt_melanesian_XX")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_surui_XX_builder.append_option(value_list_int32(
                        info_fields.get("AC_surui_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_surui_XX_builder.append_option(value_int32(
                        info_fields.get("AN_surui_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .AF_surui_XX_builder
                        .append_option(value_list_float64(
                            info_fields.get("AF_surui_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .nhomalt_surui_XX_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_surui_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_sindhi_XX_builder.append_option(value_list_int32(
                        info_fields.get("AC_sindhi_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_sindhi_XX_builder.append_option(value_int32(
                        info_fields.get("AN_sindhi_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .AF_sindhi_XX_builder
                        .append_option(value_list_float64(
                            info_fields.get("AF_sindhi_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .nhomalt_sindhi_XX_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("nhomalt_sindhi_XX")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_oroqen_builder.append_option(value_list_int32(
                        info_fields.get("AC_oroqen").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_oroqen_builder.append_option(value_int32(
                        info_fields.get("AN_oroqen").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_oroqen_builder.append_option(value_list_float64(
                        info_fields.get("AF_oroqen").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .nhomalt_oroqen_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_oroqen").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .AC_cambodian_XY_builder
                        .append_option(value_list_int32(
                            info_fields.get("AC_cambodian_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder.AN_cambodian_XY_builder.append_option(value_int32(
                        info_fields.get("AN_cambodian_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .AF_cambodian_XY_builder
                        .append_option(value_list_float64(
                            info_fields.get("AF_cambodian_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .nhomalt_cambodian_XY_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("nhomalt_cambodian_XY")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .AC_mandenka_XX_builder
                        .append_option(value_list_int32(
                            info_fields.get("AC_mandenka_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder.AN_mandenka_XX_builder.append_option(value_int32(
                        info_fields.get("AN_mandenka_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .AF_mandenka_XX_builder
                        .append_option(value_list_float64(
                            info_fields.get("AF_mandenka_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .nhomalt_mandenka_XX_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("nhomalt_mandenka_XX")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_stu_XY_builder.append_option(value_list_int32(
                        info_fields.get("AC_stu_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_stu_XY_builder.append_option(value_int32(
                        info_fields.get("AN_stu_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_stu_XY_builder.append_option(value_list_float64(
                        info_fields.get("AF_stu_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .nhomalt_stu_XY_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_stu_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .AC_balochi_XY_builder
                        .append_option(value_list_int32(
                            info_fields.get("AC_balochi_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder.AN_balochi_XY_builder.append_option(value_int32(
                        info_fields.get("AN_balochi_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .AF_balochi_XY_builder
                        .append_option(value_list_float64(
                            info_fields.get("AF_balochi_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .nhomalt_balochi_XY_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("nhomalt_balochi_XY")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_tuscan_XX_builder.append_option(value_list_int32(
                        info_fields.get("AC_tuscan_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_tuscan_XX_builder.append_option(value_int32(
                        info_fields.get("AN_tuscan_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .AF_tuscan_XX_builder
                        .append_option(value_list_float64(
                            info_fields.get("AF_tuscan_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .nhomalt_tuscan_XX_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("nhomalt_tuscan_XX")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_clm_builder.append_option(value_list_int32(
                        info_fields.get("AC_clm").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_clm_builder.append_option(value_int32(
                        info_fields.get("AN_clm").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_clm_builder.append_option(value_list_float64(
                        info_fields.get("AF_clm").and_then(|x| x.as_ref()),
                    )?);
                    builder.nhomalt_clm_builder.append_option(value_list_int32(
                        info_fields.get("nhomalt_clm").and_then(|x| x.as_ref()),
                    )?);
                    builder.AC_pur_builder.append_option(value_list_int32(
                        info_fields.get("AC_pur").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_pur_builder.append_option(value_int32(
                        info_fields.get("AN_pur").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_pur_builder.append_option(value_list_float64(
                        info_fields.get("AF_pur").and_then(|x| x.as_ref()),
                    )?);
                    builder.nhomalt_pur_builder.append_option(value_list_int32(
                        info_fields.get("nhomalt_pur").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .AC_mandenka_XY_builder
                        .append_option(value_list_int32(
                            info_fields.get("AC_mandenka_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder.AN_mandenka_XY_builder.append_option(value_int32(
                        info_fields.get("AN_mandenka_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .AF_mandenka_XY_builder
                        .append_option(value_list_float64(
                            info_fields.get("AF_mandenka_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .nhomalt_mandenka_XY_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("nhomalt_mandenka_XY")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_xibo_XX_builder.append_option(value_list_int32(
                        info_fields.get("AC_xibo_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_xibo_XX_builder.append_option(value_int32(
                        info_fields.get("AN_xibo_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_xibo_XX_builder.append_option(value_list_float64(
                        info_fields.get("AF_xibo_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .nhomalt_xibo_XX_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_xibo_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_acb_XY_builder.append_option(value_list_int32(
                        info_fields.get("AC_acb_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_acb_XY_builder.append_option(value_int32(
                        info_fields.get("AN_acb_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_acb_XY_builder.append_option(value_list_float64(
                        info_fields.get("AF_acb_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .nhomalt_acb_XY_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_acb_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_dai_builder.append_option(value_list_int32(
                        info_fields.get("AC_dai").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_dai_builder.append_option(value_int32(
                        info_fields.get("AN_dai").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_dai_builder.append_option(value_list_float64(
                        info_fields.get("AF_dai").and_then(|x| x.as_ref()),
                    )?);
                    builder.nhomalt_dai_builder.append_option(value_list_int32(
                        info_fields.get("nhomalt_dai").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .AC_bantukenya_builder
                        .append_option(value_list_int32(
                            info_fields.get("AC_bantukenya").and_then(|x| x.as_ref()),
                        )?);
                    builder.AN_bantukenya_builder.append_option(value_int32(
                        info_fields.get("AN_bantukenya").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .AF_bantukenya_builder
                        .append_option(value_list_float64(
                            info_fields.get("AF_bantukenya").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .nhomalt_bantukenya_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("nhomalt_bantukenya")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_lahu_XX_builder.append_option(value_list_int32(
                        info_fields.get("AC_lahu_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_lahu_XX_builder.append_option(value_int32(
                        info_fields.get("AN_lahu_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_lahu_XX_builder.append_option(value_list_float64(
                        info_fields.get("AF_lahu_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .nhomalt_lahu_XX_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_lahu_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_tsi_builder.append_option(value_list_int32(
                        info_fields.get("AC_tsi").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_tsi_builder.append_option(value_int32(
                        info_fields.get("AN_tsi").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_tsi_builder.append_option(value_list_float64(
                        info_fields.get("AF_tsi").and_then(|x| x.as_ref()),
                    )?);
                    builder.nhomalt_tsi_builder.append_option(value_list_int32(
                        info_fields.get("nhomalt_tsi").and_then(|x| x.as_ref()),
                    )?);
                    builder.AC_mozabite_builder.append_option(value_list_int32(
                        info_fields.get("AC_mozabite").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_mozabite_builder.append_option(value_int32(
                        info_fields.get("AN_mozabite").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .AF_mozabite_builder
                        .append_option(value_list_float64(
                            info_fields.get("AF_mozabite").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .nhomalt_mozabite_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_mozabite").and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_tu_builder.append_option(value_list_int32(
                        info_fields.get("AC_tu").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_tu_builder.append_option(value_int32(
                        info_fields.get("AN_tu").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_tu_builder.append_option(value_list_float64(
                        info_fields.get("AF_tu").and_then(|x| x.as_ref()),
                    )?);
                    builder.nhomalt_tu_builder.append_option(value_list_int32(
                        info_fields.get("nhomalt_tu").and_then(|x| x.as_ref()),
                    )?);
                    builder.AC_jpt_builder.append_option(value_list_int32(
                        info_fields.get("AC_jpt").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_jpt_builder.append_option(value_int32(
                        info_fields.get("AN_jpt").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_jpt_builder.append_option(value_list_float64(
                        info_fields.get("AF_jpt").and_then(|x| x.as_ref()),
                    )?);
                    builder.nhomalt_jpt_builder.append_option(value_list_int32(
                        info_fields.get("nhomalt_jpt").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .AC_mozabite_XX_builder
                        .append_option(value_list_int32(
                            info_fields.get("AC_mozabite_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder.AN_mozabite_XX_builder.append_option(value_int32(
                        info_fields.get("AN_mozabite_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .AF_mozabite_XX_builder
                        .append_option(value_list_float64(
                            info_fields.get("AF_mozabite_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .nhomalt_mozabite_XX_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("nhomalt_mozabite_XX")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .AC_biakapygmy_XY_builder
                        .append_option(value_list_int32(
                            info_fields.get("AC_biakapygmy_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder.AN_biakapygmy_XY_builder.append_option(value_int32(
                        info_fields.get("AN_biakapygmy_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .AF_biakapygmy_XY_builder
                        .append_option(value_list_float64(
                            info_fields.get("AF_biakapygmy_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .nhomalt_biakapygmy_XY_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("nhomalt_biakapygmy_XY")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .AC_burusho_XY_builder
                        .append_option(value_list_int32(
                            info_fields.get("AC_burusho_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder.AN_burusho_XY_builder.append_option(value_int32(
                        info_fields.get("AN_burusho_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .AF_burusho_XY_builder
                        .append_option(value_list_float64(
                            info_fields.get("AF_burusho_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .nhomalt_burusho_XY_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("nhomalt_burusho_XY")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_itu_XX_builder.append_option(value_list_int32(
                        info_fields.get("AC_itu_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_itu_XX_builder.append_option(value_int32(
                        info_fields.get("AN_itu_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_itu_XX_builder.append_option(value_list_float64(
                        info_fields.get("AF_itu_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .nhomalt_itu_XX_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_itu_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_gwd_XY_builder.append_option(value_list_int32(
                        info_fields.get("AC_gwd_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_gwd_XY_builder.append_option(value_int32(
                        info_fields.get("AN_gwd_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_gwd_XY_builder.append_option(value_list_float64(
                        info_fields.get("AF_gwd_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .nhomalt_gwd_XY_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_gwd_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_druze_XX_builder.append_option(value_list_int32(
                        info_fields.get("AC_druze_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_druze_XX_builder.append_option(value_int32(
                        info_fields.get("AN_druze_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .AF_druze_XX_builder
                        .append_option(value_list_float64(
                            info_fields.get("AF_druze_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .nhomalt_druze_XX_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_druze_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .AC_melanesian_XY_builder
                        .append_option(value_list_int32(
                            info_fields.get("AC_melanesian_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder.AN_melanesian_XY_builder.append_option(value_int32(
                        info_fields.get("AN_melanesian_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .AF_melanesian_XY_builder
                        .append_option(value_list_float64(
                            info_fields.get("AF_melanesian_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .nhomalt_melanesian_XY_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("nhomalt_melanesian_XY")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .AC_mongola_XX_builder
                        .append_option(value_list_int32(
                            info_fields.get("AC_mongola_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder.AN_mongola_XX_builder.append_option(value_int32(
                        info_fields.get("AN_mongola_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .AF_mongola_XX_builder
                        .append_option(value_list_float64(
                            info_fields.get("AF_mongola_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .nhomalt_mongola_XX_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("nhomalt_mongola_XX")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_XX_builder.append_option(value_list_int32(
                        info_fields.get("AC_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_XX_builder.append_option(value_int32(
                        info_fields.get("AN_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_XX_builder.append_option(value_list_float64(
                        info_fields.get("AF_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder.nhomalt_XX_builder.append_option(value_list_int32(
                        info_fields.get("nhomalt_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .AC_bantukenya_XX_builder
                        .append_option(value_list_int32(
                            info_fields.get("AC_bantukenya_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder.AN_bantukenya_XX_builder.append_option(value_int32(
                        info_fields.get("AN_bantukenya_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .AF_bantukenya_XX_builder
                        .append_option(value_list_float64(
                            info_fields.get("AF_bantukenya_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .nhomalt_bantukenya_XX_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("nhomalt_bantukenya_XX")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_hezhen_XX_builder.append_option(value_list_int32(
                        info_fields.get("AC_hezhen_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_hezhen_XX_builder.append_option(value_int32(
                        info_fields.get("AN_hezhen_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .AF_hezhen_XX_builder
                        .append_option(value_list_float64(
                            info_fields.get("AF_hezhen_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .nhomalt_hezhen_XX_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("nhomalt_hezhen_XX")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_itu_XY_builder.append_option(value_list_int32(
                        info_fields.get("AC_itu_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_itu_XY_builder.append_option(value_int32(
                        info_fields.get("AN_itu_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_itu_XY_builder.append_option(value_list_float64(
                        info_fields.get("AF_itu_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .nhomalt_itu_XY_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_itu_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .AC_bantusafrica_builder
                        .append_option(value_list_int32(
                            info_fields.get("AC_bantusafrica").and_then(|x| x.as_ref()),
                        )?);
                    builder.AN_bantusafrica_builder.append_option(value_int32(
                        info_fields.get("AN_bantusafrica").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .AF_bantusafrica_builder
                        .append_option(value_list_float64(
                            info_fields.get("AF_bantusafrica").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .nhomalt_bantusafrica_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("nhomalt_bantusafrica")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_ceu_XY_builder.append_option(value_list_int32(
                        info_fields.get("AC_ceu_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_ceu_XY_builder.append_option(value_int32(
                        info_fields.get("AN_ceu_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_ceu_XY_builder.append_option(value_list_float64(
                        info_fields.get("AF_ceu_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .nhomalt_ceu_XY_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_ceu_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_maya_XX_builder.append_option(value_list_int32(
                        info_fields.get("AC_maya_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_maya_XX_builder.append_option(value_int32(
                        info_fields.get("AN_maya_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_maya_XX_builder.append_option(value_list_float64(
                        info_fields.get("AF_maya_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .nhomalt_maya_XX_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_maya_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_gbr_builder.append_option(value_list_int32(
                        info_fields.get("AC_gbr").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_gbr_builder.append_option(value_int32(
                        info_fields.get("AN_gbr").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_gbr_builder.append_option(value_list_float64(
                        info_fields.get("AF_gbr").and_then(|x| x.as_ref()),
                    )?);
                    builder.nhomalt_gbr_builder.append_option(value_list_int32(
                        info_fields.get("nhomalt_gbr").and_then(|x| x.as_ref()),
                    )?);
                    builder.AC_xibo_XY_builder.append_option(value_list_int32(
                        info_fields.get("AC_xibo_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_xibo_XY_builder.append_option(value_int32(
                        info_fields.get("AN_xibo_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_xibo_XY_builder.append_option(value_list_float64(
                        info_fields.get("AF_xibo_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .nhomalt_xibo_XY_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_xibo_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_fin_builder.append_option(value_list_int32(
                        info_fields.get("AC_fin").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_fin_builder.append_option(value_int32(
                        info_fields.get("AN_fin").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_fin_builder.append_option(value_list_float64(
                        info_fields.get("AF_fin").and_then(|x| x.as_ref()),
                    )?);
                    builder.nhomalt_fin_builder.append_option(value_list_int32(
                        info_fields.get("nhomalt_fin").and_then(|x| x.as_ref()),
                    )?);
                    builder.AC_tujia_builder.append_option(value_list_int32(
                        info_fields.get("AC_tujia").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_tujia_builder.append_option(value_int32(
                        info_fields.get("AN_tujia").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_tujia_builder.append_option(value_list_float64(
                        info_fields.get("AF_tujia").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .nhomalt_tujia_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_tujia").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .AC_mbutipygmy_XX_builder
                        .append_option(value_list_int32(
                            info_fields.get("AC_mbutipygmy_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder.AN_mbutipygmy_XX_builder.append_option(value_int32(
                        info_fields.get("AN_mbutipygmy_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .AF_mbutipygmy_XX_builder
                        .append_option(value_list_float64(
                            info_fields.get("AF_mbutipygmy_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .nhomalt_mbutipygmy_XX_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("nhomalt_mbutipygmy_XX")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_hazara_XY_builder.append_option(value_list_int32(
                        info_fields.get("AC_hazara_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_hazara_XY_builder.append_option(value_int32(
                        info_fields.get("AN_hazara_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .AF_hazara_XY_builder
                        .append_option(value_list_float64(
                            info_fields.get("AF_hazara_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .nhomalt_hazara_XY_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("nhomalt_hazara_XY")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_papuan_XX_builder.append_option(value_list_int32(
                        info_fields.get("AC_papuan_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_papuan_XX_builder.append_option(value_int32(
                        info_fields.get("AN_papuan_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .AF_papuan_XX_builder
                        .append_option(value_list_float64(
                            info_fields.get("AF_papuan_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .nhomalt_papuan_XX_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("nhomalt_papuan_XX")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_japanese_builder.append_option(value_list_int32(
                        info_fields.get("AC_japanese").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_japanese_builder.append_option(value_int32(
                        info_fields.get("AN_japanese").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .AF_japanese_builder
                        .append_option(value_list_float64(
                            info_fields.get("AF_japanese").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .nhomalt_japanese_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_japanese").and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_xibo_builder.append_option(value_list_int32(
                        info_fields.get("AC_xibo").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_xibo_builder.append_option(value_int32(
                        info_fields.get("AN_xibo").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_xibo_builder.append_option(value_list_float64(
                        info_fields.get("AF_xibo").and_then(|x| x.as_ref()),
                    )?);
                    builder.nhomalt_xibo_builder.append_option(value_list_int32(
                        info_fields.get("nhomalt_xibo").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .AC_sardinian_XY_builder
                        .append_option(value_list_int32(
                            info_fields.get("AC_sardinian_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder.AN_sardinian_XY_builder.append_option(value_int32(
                        info_fields.get("AN_sardinian_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .AF_sardinian_XY_builder
                        .append_option(value_list_float64(
                            info_fields.get("AF_sardinian_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .nhomalt_sardinian_XY_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("nhomalt_sardinian_XY")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .AC_colombian_XY_builder
                        .append_option(value_list_int32(
                            info_fields.get("AC_colombian_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder.AN_colombian_XY_builder.append_option(value_int32(
                        info_fields.get("AN_colombian_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .AF_colombian_XY_builder
                        .append_option(value_list_float64(
                            info_fields.get("AF_colombian_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .nhomalt_colombian_XY_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("nhomalt_colombian_XY")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_balochi_builder.append_option(value_list_int32(
                        info_fields.get("AC_balochi").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_balochi_builder.append_option(value_int32(
                        info_fields.get("AN_balochi").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_balochi_builder.append_option(value_list_float64(
                        info_fields.get("AF_balochi").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .nhomalt_balochi_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_balochi").and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_gih_XX_builder.append_option(value_list_int32(
                        info_fields.get("AC_gih_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_gih_XX_builder.append_option(value_int32(
                        info_fields.get("AN_gih_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_gih_XX_builder.append_option(value_list_float64(
                        info_fields.get("AF_gih_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .nhomalt_gih_XX_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_gih_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_esn_XY_builder.append_option(value_list_int32(
                        info_fields.get("AC_esn_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_esn_XY_builder.append_option(value_int32(
                        info_fields.get("AN_esn_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_esn_XY_builder.append_option(value_list_float64(
                        info_fields.get("AF_esn_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .nhomalt_esn_XY_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_esn_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_msl_XY_builder.append_option(value_list_int32(
                        info_fields.get("AC_msl_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_msl_XY_builder.append_option(value_int32(
                        info_fields.get("AN_msl_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_msl_XY_builder.append_option(value_list_float64(
                        info_fields.get("AF_msl_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .nhomalt_msl_XY_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_msl_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_pjl_XY_builder.append_option(value_list_int32(
                        info_fields.get("AC_pjl_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_pjl_XY_builder.append_option(value_int32(
                        info_fields.get("AN_pjl_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_pjl_XY_builder.append_option(value_list_float64(
                        info_fields.get("AF_pjl_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .nhomalt_pjl_XY_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_pjl_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_makrani_builder.append_option(value_list_int32(
                        info_fields.get("AC_makrani").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_makrani_builder.append_option(value_int32(
                        info_fields.get("AN_makrani").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_makrani_builder.append_option(value_list_float64(
                        info_fields.get("AF_makrani").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .nhomalt_makrani_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_makrani").and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_ceu_XX_builder.append_option(value_list_int32(
                        info_fields.get("AC_ceu_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_ceu_XX_builder.append_option(value_int32(
                        info_fields.get("AN_ceu_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_ceu_XX_builder.append_option(value_list_float64(
                        info_fields.get("AF_ceu_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .nhomalt_ceu_XX_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_ceu_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_miaozu_XX_builder.append_option(value_list_int32(
                        info_fields.get("AC_miaozu_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_miaozu_XX_builder.append_option(value_int32(
                        info_fields.get("AN_miaozu_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .AF_miaozu_XX_builder
                        .append_option(value_list_float64(
                            info_fields.get("AF_miaozu_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .nhomalt_miaozu_XX_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("nhomalt_miaozu_XX")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_naxi_XY_builder.append_option(value_list_int32(
                        info_fields.get("AC_naxi_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_naxi_XY_builder.append_option(value_int32(
                        info_fields.get("AN_naxi_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_naxi_XY_builder.append_option(value_list_float64(
                        info_fields.get("AF_naxi_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .nhomalt_naxi_XY_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_naxi_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .AC_sardinian_XX_builder
                        .append_option(value_list_int32(
                            info_fields.get("AC_sardinian_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder.AN_sardinian_XX_builder.append_option(value_int32(
                        info_fields.get("AN_sardinian_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .AF_sardinian_XX_builder
                        .append_option(value_list_float64(
                            info_fields.get("AF_sardinian_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .nhomalt_sardinian_XX_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("nhomalt_sardinian_XX")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_mongola_builder.append_option(value_list_int32(
                        info_fields.get("AC_mongola").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_mongola_builder.append_option(value_int32(
                        info_fields.get("AN_mongola").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_mongola_builder.append_option(value_list_float64(
                        info_fields.get("AF_mongola").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .nhomalt_mongola_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_mongola").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .AC_orcadian_XY_builder
                        .append_option(value_list_int32(
                            info_fields.get("AC_orcadian_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder.AN_orcadian_XY_builder.append_option(value_int32(
                        info_fields.get("AN_orcadian_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .AF_orcadian_XY_builder
                        .append_option(value_list_float64(
                            info_fields.get("AF_orcadian_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .nhomalt_orcadian_XY_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("nhomalt_orcadian_XY")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_hazara_builder.append_option(value_list_int32(
                        info_fields.get("AC_hazara").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_hazara_builder.append_option(value_int32(
                        info_fields.get("AN_hazara").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_hazara_builder.append_option(value_list_float64(
                        info_fields.get("AF_hazara").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .nhomalt_hazara_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_hazara").and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_tsi_XX_builder.append_option(value_list_int32(
                        info_fields.get("AC_tsi_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_tsi_XX_builder.append_option(value_int32(
                        info_fields.get("AN_tsi_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_tsi_XX_builder.append_option(value_list_float64(
                        info_fields.get("AF_tsi_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .nhomalt_tsi_XX_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_tsi_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_msl_XX_builder.append_option(value_list_int32(
                        info_fields.get("AC_msl_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_msl_XX_builder.append_option(value_int32(
                        info_fields.get("AN_msl_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_msl_XX_builder.append_option(value_list_float64(
                        info_fields.get("AF_msl_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .nhomalt_msl_XX_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_msl_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_pur_XY_builder.append_option(value_list_int32(
                        info_fields.get("AC_pur_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_pur_XY_builder.append_option(value_int32(
                        info_fields.get("AN_pur_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_pur_XY_builder.append_option(value_list_float64(
                        info_fields.get("AF_pur_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .nhomalt_pur_XY_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_pur_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_clm_XX_builder.append_option(value_list_int32(
                        info_fields.get("AC_clm_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_clm_XX_builder.append_option(value_int32(
                        info_fields.get("AN_clm_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_clm_XX_builder.append_option(value_list_float64(
                        info_fields.get("AF_clm_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .nhomalt_clm_XX_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_clm_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .AC_palestinian_builder
                        .append_option(value_list_int32(
                            info_fields.get("AC_palestinian").and_then(|x| x.as_ref()),
                        )?);
                    builder.AN_palestinian_builder.append_option(value_int32(
                        info_fields.get("AN_palestinian").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .AF_palestinian_builder
                        .append_option(value_list_float64(
                            info_fields.get("AF_palestinian").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .nhomalt_palestinian_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("nhomalt_palestinian")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_han_XY_builder.append_option(value_list_int32(
                        info_fields.get("AC_han_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_han_XY_builder.append_option(value_int32(
                        info_fields.get("AN_han_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_han_XY_builder.append_option(value_list_float64(
                        info_fields.get("AF_han_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .nhomalt_han_XY_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_han_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .AC_bedouin_XX_builder
                        .append_option(value_list_int32(
                            info_fields.get("AC_bedouin_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder.AN_bedouin_XX_builder.append_option(value_int32(
                        info_fields.get("AN_bedouin_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .AF_bedouin_XX_builder
                        .append_option(value_list_float64(
                            info_fields.get("AF_bedouin_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .nhomalt_bedouin_XX_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("nhomalt_bedouin_XX")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_yizu_builder.append_option(value_list_int32(
                        info_fields.get("AC_yizu").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_yizu_builder.append_option(value_int32(
                        info_fields.get("AN_yizu").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_yizu_builder.append_option(value_list_float64(
                        info_fields.get("AF_yizu").and_then(|x| x.as_ref()),
                    )?);
                    builder.nhomalt_yizu_builder.append_option(value_list_int32(
                        info_fields.get("nhomalt_yizu").and_then(|x| x.as_ref()),
                    )?);
                    builder.AC_XY_builder.append_option(value_list_int32(
                        info_fields.get("AC_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_XY_builder.append_option(value_int32(
                        info_fields.get("AN_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_XY_builder.append_option(value_list_float64(
                        info_fields.get("AF_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder.nhomalt_XY_builder.append_option(value_list_int32(
                        info_fields.get("nhomalt_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder.AC_ibs_XX_builder.append_option(value_list_int32(
                        info_fields.get("AC_ibs_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_ibs_XX_builder.append_option(value_int32(
                        info_fields.get("AN_ibs_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_ibs_XX_builder.append_option(value_list_float64(
                        info_fields.get("AF_ibs_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .nhomalt_ibs_XX_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_ibs_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_brahui_XX_builder.append_option(value_list_int32(
                        info_fields.get("AC_brahui_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_brahui_XX_builder.append_option(value_int32(
                        info_fields.get("AN_brahui_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .AF_brahui_XX_builder
                        .append_option(value_list_float64(
                            info_fields.get("AF_brahui_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .nhomalt_brahui_XX_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("nhomalt_brahui_XX")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_yakut_builder.append_option(value_list_int32(
                        info_fields.get("AC_yakut").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_yakut_builder.append_option(value_int32(
                        info_fields.get("AN_yakut").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_yakut_builder.append_option(value_list_float64(
                        info_fields.get("AF_yakut").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .nhomalt_yakut_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_yakut").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .AC_russian_XX_builder
                        .append_option(value_list_int32(
                            info_fields.get("AC_russian_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder.AN_russian_XX_builder.append_option(value_int32(
                        info_fields.get("AN_russian_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .AF_russian_XX_builder
                        .append_option(value_list_float64(
                            info_fields.get("AF_russian_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .nhomalt_russian_XX_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("nhomalt_russian_XX")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .AC_mozabite_XY_builder
                        .append_option(value_list_int32(
                            info_fields.get("AC_mozabite_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder.AN_mozabite_XY_builder.append_option(value_int32(
                        info_fields.get("AN_mozabite_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .AF_mozabite_XY_builder
                        .append_option(value_list_float64(
                            info_fields.get("AF_mozabite_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .nhomalt_mozabite_XY_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("nhomalt_mozabite_XY")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_lahu_builder.append_option(value_list_int32(
                        info_fields.get("AC_lahu").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_lahu_builder.append_option(value_int32(
                        info_fields.get("AN_lahu").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_lahu_builder.append_option(value_list_float64(
                        info_fields.get("AF_lahu").and_then(|x| x.as_ref()),
                    )?);
                    builder.nhomalt_lahu_builder.append_option(value_list_int32(
                        info_fields.get("nhomalt_lahu").and_then(|x| x.as_ref()),
                    )?);
                    builder.AC_lwk_builder.append_option(value_list_int32(
                        info_fields.get("AC_lwk").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_lwk_builder.append_option(value_int32(
                        info_fields.get("AN_lwk").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_lwk_builder.append_option(value_list_float64(
                        info_fields.get("AF_lwk").and_then(|x| x.as_ref()),
                    )?);
                    builder.nhomalt_lwk_builder.append_option(value_list_int32(
                        info_fields.get("nhomalt_lwk").and_then(|x| x.as_ref()),
                    )?);
                    builder.AC_basque_builder.append_option(value_list_int32(
                        info_fields.get("AC_basque").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_basque_builder.append_option(value_int32(
                        info_fields.get("AN_basque").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_basque_builder.append_option(value_list_float64(
                        info_fields.get("AF_basque").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .nhomalt_basque_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_basque").and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_fin_XY_builder.append_option(value_list_int32(
                        info_fields.get("AC_fin_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_fin_XY_builder.append_option(value_int32(
                        info_fields.get("AN_fin_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_fin_XY_builder.append_option(value_list_float64(
                        info_fields.get("AF_fin_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .nhomalt_fin_XY_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_fin_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_uygur_builder.append_option(value_list_int32(
                        info_fields.get("AC_uygur").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_uygur_builder.append_option(value_int32(
                        info_fields.get("AN_uygur").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_uygur_builder.append_option(value_list_float64(
                        info_fields.get("AF_uygur").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .nhomalt_uygur_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_uygur").and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_yoruba_XX_builder.append_option(value_list_int32(
                        info_fields.get("AC_yoruba_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_yoruba_XX_builder.append_option(value_int32(
                        info_fields.get("AN_yoruba_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .AF_yoruba_XX_builder
                        .append_option(value_list_float64(
                            info_fields.get("AF_yoruba_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .nhomalt_yoruba_XX_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("nhomalt_yoruba_XX")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_orcadian_builder.append_option(value_list_int32(
                        info_fields.get("AC_orcadian").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_orcadian_builder.append_option(value_int32(
                        info_fields.get("AN_orcadian").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .AF_orcadian_builder
                        .append_option(value_list_float64(
                            info_fields.get("AF_orcadian").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .nhomalt_orcadian_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_orcadian").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .AC_bantusafrica_XX_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("AC_bantusafrica_XX")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .AN_bantusafrica_XX_builder
                        .append_option(value_int32(
                            info_fields
                                .get("AN_bantusafrica_XX")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .AF_bantusafrica_XX_builder
                        .append_option(value_list_float64(
                            info_fields
                                .get("AF_bantusafrica_XX")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .nhomalt_bantusafrica_XX_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("nhomalt_bantusafrica_XX")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_french_XY_builder.append_option(value_list_int32(
                        info_fields.get("AC_french_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_french_XY_builder.append_option(value_int32(
                        info_fields.get("AN_french_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .AF_french_XY_builder
                        .append_option(value_list_float64(
                            info_fields.get("AF_french_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .nhomalt_french_XY_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("nhomalt_french_XY")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_pur_XX_builder.append_option(value_list_int32(
                        info_fields.get("AC_pur_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_pur_XX_builder.append_option(value_int32(
                        info_fields.get("AN_pur_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_pur_XX_builder.append_option(value_list_float64(
                        info_fields.get("AF_pur_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .nhomalt_pur_XX_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_pur_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_khv_builder.append_option(value_list_int32(
                        info_fields.get("AC_khv").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_khv_builder.append_option(value_int32(
                        info_fields.get("AN_khv").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_khv_builder.append_option(value_list_float64(
                        info_fields.get("AF_khv").and_then(|x| x.as_ref()),
                    )?);
                    builder.nhomalt_khv_builder.append_option(value_list_int32(
                        info_fields.get("nhomalt_khv").and_then(|x| x.as_ref()),
                    )?);
                    builder.AC_asw_XY_builder.append_option(value_list_int32(
                        info_fields.get("AC_asw_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_asw_XY_builder.append_option(value_int32(
                        info_fields.get("AN_asw_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_asw_XY_builder.append_option(value_list_float64(
                        info_fields.get("AF_asw_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .nhomalt_asw_XY_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_asw_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_she_builder.append_option(value_list_int32(
                        info_fields.get("AC_she").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_she_builder.append_option(value_int32(
                        info_fields.get("AN_she").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_she_builder.append_option(value_list_float64(
                        info_fields.get("AF_she").and_then(|x| x.as_ref()),
                    )?);
                    builder.nhomalt_she_builder.append_option(value_list_int32(
                        info_fields.get("nhomalt_she").and_then(|x| x.as_ref()),
                    )?);
                    builder.AC_dai_XX_builder.append_option(value_list_int32(
                        info_fields.get("AC_dai_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_dai_XX_builder.append_option(value_int32(
                        info_fields.get("AN_dai_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_dai_XX_builder.append_option(value_list_float64(
                        info_fields.get("AF_dai_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .nhomalt_dai_XX_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_dai_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_she_XX_builder.append_option(value_list_int32(
                        info_fields.get("AC_she_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_she_XX_builder.append_option(value_int32(
                        info_fields.get("AN_she_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_she_XX_builder.append_option(value_list_float64(
                        info_fields.get("AF_she_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .nhomalt_she_XX_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_she_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_ibs_XY_builder.append_option(value_list_int32(
                        info_fields.get("AC_ibs_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_ibs_XY_builder.append_option(value_int32(
                        info_fields.get("AN_ibs_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_ibs_XY_builder.append_option(value_list_float64(
                        info_fields.get("AF_ibs_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .nhomalt_ibs_XY_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_ibs_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_uygur_XY_builder.append_option(value_list_int32(
                        info_fields.get("AC_uygur_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_uygur_XY_builder.append_option(value_int32(
                        info_fields.get("AN_uygur_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .AF_uygur_XY_builder
                        .append_option(value_list_float64(
                            info_fields.get("AF_uygur_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .nhomalt_uygur_XY_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_uygur_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .AC_cambodian_XX_builder
                        .append_option(value_list_int32(
                            info_fields.get("AC_cambodian_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder.AN_cambodian_XX_builder.append_option(value_int32(
                        info_fields.get("AN_cambodian_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .AF_cambodian_XX_builder
                        .append_option(value_list_float64(
                            info_fields.get("AF_cambodian_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .nhomalt_cambodian_XX_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("nhomalt_cambodian_XX")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_pima_XY_builder.append_option(value_list_int32(
                        info_fields.get("AC_pima_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_pima_XY_builder.append_option(value_int32(
                        info_fields.get("AN_pima_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_pima_XY_builder.append_option(value_list_float64(
                        info_fields.get("AF_pima_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .nhomalt_pima_XY_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_pima_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_cambodian_builder.append_option(value_list_int32(
                        info_fields.get("AC_cambodian").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_cambodian_builder.append_option(value_int32(
                        info_fields.get("AN_cambodian").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .AF_cambodian_builder
                        .append_option(value_list_float64(
                            info_fields.get("AF_cambodian").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .nhomalt_cambodian_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("nhomalt_cambodian")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_san_XX_builder.append_option(value_list_int32(
                        info_fields.get("AC_san_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_san_XX_builder.append_option(value_int32(
                        info_fields.get("AN_san_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_san_XX_builder.append_option(value_list_float64(
                        info_fields.get("AF_san_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .nhomalt_san_XX_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_san_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .AC_bantusafrica_XY_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("AC_bantusafrica_XY")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .AN_bantusafrica_XY_builder
                        .append_option(value_int32(
                            info_fields
                                .get("AN_bantusafrica_XY")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .AF_bantusafrica_XY_builder
                        .append_option(value_list_float64(
                            info_fields
                                .get("AF_bantusafrica_XY")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .nhomalt_bantusafrica_XY_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("nhomalt_bantusafrica_XY")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_yri_builder.append_option(value_list_int32(
                        info_fields.get("AC_yri").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_yri_builder.append_option(value_int32(
                        info_fields.get("AN_yri").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_yri_builder.append_option(value_list_float64(
                        info_fields.get("AF_yri").and_then(|x| x.as_ref()),
                    )?);
                    builder.nhomalt_yri_builder.append_option(value_list_int32(
                        info_fields.get("nhomalt_yri").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .AC_makrani_XY_builder
                        .append_option(value_list_int32(
                            info_fields.get("AC_makrani_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder.AN_makrani_XY_builder.append_option(value_int32(
                        info_fields.get("AN_makrani_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .AF_makrani_XY_builder
                        .append_option(value_list_float64(
                            info_fields.get("AF_makrani_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .nhomalt_makrani_XY_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("nhomalt_makrani_XY")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .AC_balochi_XX_builder
                        .append_option(value_list_int32(
                            info_fields.get("AC_balochi_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder.AN_balochi_XX_builder.append_option(value_int32(
                        info_fields.get("AN_balochi_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .AF_balochi_XX_builder
                        .append_option(value_list_float64(
                            info_fields.get("AF_balochi_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .nhomalt_balochi_XX_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("nhomalt_balochi_XX")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_tuscan_builder.append_option(value_list_int32(
                        info_fields.get("AC_tuscan").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_tuscan_builder.append_option(value_int32(
                        info_fields.get("AN_tuscan").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_tuscan_builder.append_option(value_list_float64(
                        info_fields.get("AF_tuscan").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .nhomalt_tuscan_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_tuscan").and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_stu_builder.append_option(value_list_int32(
                        info_fields.get("AC_stu").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_stu_builder.append_option(value_int32(
                        info_fields.get("AN_stu").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_stu_builder.append_option(value_list_float64(
                        info_fields.get("AF_stu").and_then(|x| x.as_ref()),
                    )?);
                    builder.nhomalt_stu_builder.append_option(value_list_int32(
                        info_fields.get("nhomalt_stu").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .AC_bantukenya_XY_builder
                        .append_option(value_list_int32(
                            info_fields.get("AC_bantukenya_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder.AN_bantukenya_XY_builder.append_option(value_int32(
                        info_fields.get("AN_bantukenya_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .AF_bantukenya_XY_builder
                        .append_option(value_list_float64(
                            info_fields.get("AF_bantukenya_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .nhomalt_bantukenya_XY_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("nhomalt_bantukenya_XY")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_italian_builder.append_option(value_list_int32(
                        info_fields.get("AC_italian").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_italian_builder.append_option(value_int32(
                        info_fields.get("AN_italian").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_italian_builder.append_option(value_list_float64(
                        info_fields.get("AF_italian").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .nhomalt_italian_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_italian").and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_msl_builder.append_option(value_list_int32(
                        info_fields.get("AC_msl").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_msl_builder.append_option(value_int32(
                        info_fields.get("AN_msl").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_msl_builder.append_option(value_list_float64(
                        info_fields.get("AF_msl").and_then(|x| x.as_ref()),
                    )?);
                    builder.nhomalt_msl_builder.append_option(value_list_int32(
                        info_fields.get("nhomalt_msl").and_then(|x| x.as_ref()),
                    )?);
                    builder.nhomalt_raw_builder.append_option(value_list_int32(
                        info_fields.get("nhomalt_raw").and_then(|x| x.as_ref()),
                    )?);
                    builder.AC_french_XX_builder.append_option(value_list_int32(
                        info_fields.get("AC_french_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_french_XX_builder.append_option(value_int32(
                        info_fields.get("AN_french_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .AF_french_XX_builder
                        .append_option(value_list_float64(
                            info_fields.get("AF_french_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .nhomalt_french_XX_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("nhomalt_french_XX")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .AC_colombian_XX_builder
                        .append_option(value_list_int32(
                            info_fields.get("AC_colombian_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder.AN_colombian_XX_builder.append_option(value_int32(
                        info_fields.get("AN_colombian_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .AF_colombian_XX_builder
                        .append_option(value_list_float64(
                            info_fields.get("AF_colombian_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .nhomalt_colombian_XX_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("nhomalt_colombian_XX")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_gbr_XY_builder.append_option(value_list_int32(
                        info_fields.get("AC_gbr_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_gbr_XY_builder.append_option(value_int32(
                        info_fields.get("AN_gbr_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_gbr_XY_builder.append_option(value_list_float64(
                        info_fields.get("AF_gbr_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .nhomalt_gbr_XY_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_gbr_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_chs_builder.append_option(value_list_int32(
                        info_fields.get("AC_chs").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_chs_builder.append_option(value_int32(
                        info_fields.get("AN_chs").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_chs_builder.append_option(value_list_float64(
                        info_fields.get("AF_chs").and_then(|x| x.as_ref()),
                    )?);
                    builder.nhomalt_chs_builder.append_option(value_list_int32(
                        info_fields.get("nhomalt_chs").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .AC_palestinian_XX_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("AC_palestinian_XX")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder.AN_palestinian_XX_builder.append_option(value_int32(
                        info_fields
                            .get("AN_palestinian_XX")
                            .and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .AF_palestinian_XX_builder
                        .append_option(value_list_float64(
                            info_fields
                                .get("AF_palestinian_XX")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .nhomalt_palestinian_XX_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("nhomalt_palestinian_XX")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_maya_builder.append_option(value_list_int32(
                        info_fields.get("AC_maya").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_maya_builder.append_option(value_int32(
                        info_fields.get("AN_maya").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_maya_builder.append_option(value_list_float64(
                        info_fields.get("AF_maya").and_then(|x| x.as_ref()),
                    )?);
                    builder.nhomalt_maya_builder.append_option(value_list_int32(
                        info_fields.get("nhomalt_maya").and_then(|x| x.as_ref()),
                    )?);
                    builder.AC_brahui_XY_builder.append_option(value_list_int32(
                        info_fields.get("AC_brahui_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_brahui_XY_builder.append_option(value_int32(
                        info_fields.get("AN_brahui_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .AF_brahui_XY_builder
                        .append_option(value_list_float64(
                            info_fields.get("AF_brahui_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .nhomalt_brahui_XY_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("nhomalt_brahui_XY")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .AC_italian_XX_builder
                        .append_option(value_list_int32(
                            info_fields.get("AC_italian_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder.AN_italian_XX_builder.append_option(value_int32(
                        info_fields.get("AN_italian_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .AF_italian_XX_builder
                        .append_option(value_list_float64(
                            info_fields.get("AF_italian_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .nhomalt_italian_XX_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("nhomalt_italian_XX")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_miaozu_builder.append_option(value_list_int32(
                        info_fields.get("AC_miaozu").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_miaozu_builder.append_option(value_int32(
                        info_fields.get("AN_miaozu").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_miaozu_builder.append_option(value_list_float64(
                        info_fields.get("AF_miaozu").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .nhomalt_miaozu_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_miaozu").and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_pjl_builder.append_option(value_list_int32(
                        info_fields.get("AC_pjl").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_pjl_builder.append_option(value_int32(
                        info_fields.get("AN_pjl").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_pjl_builder.append_option(value_list_float64(
                        info_fields.get("AF_pjl").and_then(|x| x.as_ref()),
                    )?);
                    builder.nhomalt_pjl_builder.append_option(value_list_int32(
                        info_fields.get("nhomalt_pjl").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .AC_burusho_XX_builder
                        .append_option(value_list_int32(
                            info_fields.get("AC_burusho_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder.AN_burusho_XX_builder.append_option(value_int32(
                        info_fields.get("AN_burusho_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .AF_burusho_XX_builder
                        .append_option(value_list_float64(
                            info_fields.get("AF_burusho_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .nhomalt_burusho_XX_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("nhomalt_burusho_XX")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_khv_XX_builder.append_option(value_list_int32(
                        info_fields.get("AC_khv_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_khv_XX_builder.append_option(value_int32(
                        info_fields.get("AN_khv_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_khv_XX_builder.append_option(value_list_float64(
                        info_fields.get("AF_khv_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .nhomalt_khv_XX_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_khv_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_mxl_XX_builder.append_option(value_list_int32(
                        info_fields.get("AC_mxl_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_mxl_XX_builder.append_option(value_int32(
                        info_fields.get("AN_mxl_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_mxl_XX_builder.append_option(value_list_float64(
                        info_fields.get("AF_mxl_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .nhomalt_mxl_XX_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_mxl_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_dai_XY_builder.append_option(value_list_int32(
                        info_fields.get("AC_dai_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_dai_XY_builder.append_option(value_int32(
                        info_fields.get("AN_dai_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_dai_XY_builder.append_option(value_list_float64(
                        info_fields.get("AF_dai_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .nhomalt_dai_XY_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_dai_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_hezhen_XY_builder.append_option(value_list_int32(
                        info_fields.get("AC_hezhen_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_hezhen_XY_builder.append_option(value_int32(
                        info_fields.get("AN_hezhen_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .AF_hezhen_XY_builder
                        .append_option(value_list_float64(
                            info_fields.get("AF_hezhen_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .nhomalt_hezhen_XY_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("nhomalt_hezhen_XY")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_sindhi_builder.append_option(value_list_int32(
                        info_fields.get("AC_sindhi").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_sindhi_builder.append_option(value_int32(
                        info_fields.get("AN_sindhi").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_sindhi_builder.append_option(value_list_float64(
                        info_fields.get("AF_sindhi").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .nhomalt_sindhi_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_sindhi").and_then(|x| x.as_ref()),
                        )?);
                    builder.nhomalt_builder.append_option(value_list_int32(
                        info_fields.get("nhomalt").and_then(|x| x.as_ref()),
                    )?);
                    builder.AC_pel_builder.append_option(value_list_int32(
                        info_fields.get("AC_pel").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_pel_builder.append_option(value_int32(
                        info_fields.get("AN_pel").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_pel_builder.append_option(value_list_float64(
                        info_fields.get("AF_pel").and_then(|x| x.as_ref()),
                    )?);
                    builder.nhomalt_pel_builder.append_option(value_list_int32(
                        info_fields.get("nhomalt_pel").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .AC_mongola_XY_builder
                        .append_option(value_list_int32(
                            info_fields.get("AC_mongola_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder.AN_mongola_XY_builder.append_option(value_int32(
                        info_fields.get("AN_mongola_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .AF_mongola_XY_builder
                        .append_option(value_list_float64(
                            info_fields.get("AF_mongola_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .nhomalt_mongola_XY_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("nhomalt_mongola_XY")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_kalash_XX_builder.append_option(value_list_int32(
                        info_fields.get("AC_kalash_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_kalash_XX_builder.append_option(value_int32(
                        info_fields.get("AN_kalash_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .AF_kalash_XX_builder
                        .append_option(value_list_float64(
                            info_fields.get("AF_kalash_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .nhomalt_kalash_XX_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("nhomalt_kalash_XX")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_burusho_builder.append_option(value_list_int32(
                        info_fields.get("AC_burusho").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_burusho_builder.append_option(value_int32(
                        info_fields.get("AN_burusho").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_burusho_builder.append_option(value_list_float64(
                        info_fields.get("AF_burusho").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .nhomalt_burusho_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_burusho").and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_hezhen_builder.append_option(value_list_int32(
                        info_fields.get("AC_hezhen").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_hezhen_builder.append_option(value_int32(
                        info_fields.get("AN_hezhen").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_hezhen_builder.append_option(value_list_float64(
                        info_fields.get("AF_hezhen").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .nhomalt_hezhen_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_hezhen").and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_beb_XX_builder.append_option(value_list_int32(
                        info_fields.get("AC_beb_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_beb_XX_builder.append_option(value_int32(
                        info_fields.get("AN_beb_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_beb_XX_builder.append_option(value_list_float64(
                        info_fields.get("AF_beb_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .nhomalt_beb_XX_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_beb_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_asw_XX_builder.append_option(value_list_int32(
                        info_fields.get("AC_asw_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_asw_XX_builder.append_option(value_int32(
                        info_fields.get("AN_asw_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_asw_XX_builder.append_option(value_list_float64(
                        info_fields.get("AF_asw_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .nhomalt_asw_XX_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_asw_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_cdx_XY_builder.append_option(value_list_int32(
                        info_fields.get("AC_cdx_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_cdx_XY_builder.append_option(value_int32(
                        info_fields.get("AN_cdx_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_cdx_XY_builder.append_option(value_list_float64(
                        info_fields.get("AF_cdx_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .nhomalt_cdx_XY_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_cdx_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_mxl_XY_builder.append_option(value_list_int32(
                        info_fields.get("AC_mxl_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_mxl_XY_builder.append_option(value_int32(
                        info_fields.get("AN_mxl_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_mxl_XY_builder.append_option(value_list_float64(
                        info_fields.get("AF_mxl_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .nhomalt_mxl_XY_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_mxl_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .AC_orcadian_XX_builder
                        .append_option(value_list_int32(
                            info_fields.get("AC_orcadian_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder.AN_orcadian_XX_builder.append_option(value_int32(
                        info_fields.get("AN_orcadian_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .AF_orcadian_XX_builder
                        .append_option(value_list_float64(
                            info_fields.get("AF_orcadian_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .nhomalt_orcadian_XX_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("nhomalt_orcadian_XX")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_san_builder.append_option(value_list_int32(
                        info_fields.get("AC_san").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_san_builder.append_option(value_int32(
                        info_fields.get("AN_san").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_san_builder.append_option(value_list_float64(
                        info_fields.get("AF_san").and_then(|x| x.as_ref()),
                    )?);
                    builder.nhomalt_san_builder.append_option(value_list_int32(
                        info_fields.get("nhomalt_san").and_then(|x| x.as_ref()),
                    )?);
                    builder.AC_bedouin_builder.append_option(value_list_int32(
                        info_fields.get("AC_bedouin").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_bedouin_builder.append_option(value_int32(
                        info_fields.get("AN_bedouin").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_bedouin_builder.append_option(value_list_float64(
                        info_fields.get("AF_bedouin").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .nhomalt_bedouin_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_bedouin").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .AC_palestinian_XY_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("AC_palestinian_XY")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder.AN_palestinian_XY_builder.append_option(value_int32(
                        info_fields
                            .get("AN_palestinian_XY")
                            .and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .AF_palestinian_XY_builder
                        .append_option(value_list_float64(
                            info_fields
                                .get("AF_palestinian_XY")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .nhomalt_palestinian_XY_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("nhomalt_palestinian_XY")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_naxi_XX_builder.append_option(value_list_int32(
                        info_fields.get("AC_naxi_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_naxi_XX_builder.append_option(value_int32(
                        info_fields.get("AN_naxi_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_naxi_XX_builder.append_option(value_list_float64(
                        info_fields.get("AF_naxi_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .nhomalt_naxi_XX_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_naxi_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_ibs_builder.append_option(value_list_int32(
                        info_fields.get("AC_ibs").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_ibs_builder.append_option(value_int32(
                        info_fields.get("AN_ibs").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_ibs_builder.append_option(value_list_float64(
                        info_fields.get("AF_ibs").and_then(|x| x.as_ref()),
                    )?);
                    builder.nhomalt_ibs_builder.append_option(value_list_int32(
                        info_fields.get("nhomalt_ibs").and_then(|x| x.as_ref()),
                    )?);
                    builder.AC_asw_builder.append_option(value_list_int32(
                        info_fields.get("AC_asw").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_asw_builder.append_option(value_int32(
                        info_fields.get("AN_asw").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_asw_builder.append_option(value_list_float64(
                        info_fields.get("AF_asw").and_then(|x| x.as_ref()),
                    )?);
                    builder.nhomalt_asw_builder.append_option(value_list_int32(
                        info_fields.get("nhomalt_asw").and_then(|x| x.as_ref()),
                    )?);
                    builder.AC_yizu_XX_builder.append_option(value_list_int32(
                        info_fields.get("AC_yizu_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_yizu_XX_builder.append_option(value_int32(
                        info_fields.get("AN_yizu_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_yizu_XX_builder.append_option(value_list_float64(
                        info_fields.get("AF_yizu_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .nhomalt_yizu_XX_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_yizu_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_chb_XY_builder.append_option(value_list_int32(
                        info_fields.get("AC_chb_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_chb_XY_builder.append_option(value_int32(
                        info_fields.get("AN_chb_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_chb_XY_builder.append_option(value_list_float64(
                        info_fields.get("AF_chb_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .nhomalt_chb_XY_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_chb_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_sardinian_builder.append_option(value_list_int32(
                        info_fields.get("AC_sardinian").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_sardinian_builder.append_option(value_int32(
                        info_fields.get("AN_sardinian").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .AF_sardinian_builder
                        .append_option(value_list_float64(
                            info_fields.get("AF_sardinian").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .nhomalt_sardinian_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("nhomalt_sardinian")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_tujia_XX_builder.append_option(value_list_int32(
                        info_fields.get("AC_tujia_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_tujia_XX_builder.append_option(value_int32(
                        info_fields.get("AN_tujia_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .AF_tujia_XX_builder
                        .append_option(value_list_float64(
                            info_fields.get("AF_tujia_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .nhomalt_tujia_XX_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_tujia_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_mandenka_builder.append_option(value_list_int32(
                        info_fields.get("AC_mandenka").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_mandenka_builder.append_option(value_int32(
                        info_fields.get("AN_mandenka").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .AF_mandenka_builder
                        .append_option(value_list_float64(
                            info_fields.get("AF_mandenka").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .nhomalt_mandenka_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_mandenka").and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_naxi_builder.append_option(value_list_int32(
                        info_fields.get("AC_naxi").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_naxi_builder.append_option(value_int32(
                        info_fields.get("AN_naxi").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_naxi_builder.append_option(value_list_float64(
                        info_fields.get("AF_naxi").and_then(|x| x.as_ref()),
                    )?);
                    builder.nhomalt_naxi_builder.append_option(value_list_int32(
                        info_fields.get("nhomalt_naxi").and_then(|x| x.as_ref()),
                    )?);
                    builder.AC_yri_XY_builder.append_option(value_list_int32(
                        info_fields.get("AC_yri_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_yri_XY_builder.append_option(value_int32(
                        info_fields.get("AN_yri_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_yri_XY_builder.append_option(value_list_float64(
                        info_fields.get("AF_yri_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .nhomalt_yri_XY_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_yri_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_jpt_XY_builder.append_option(value_list_int32(
                        info_fields.get("AC_jpt_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_jpt_XY_builder.append_option(value_int32(
                        info_fields.get("AN_jpt_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_jpt_XY_builder.append_option(value_list_float64(
                        info_fields.get("AF_jpt_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .nhomalt_jpt_XY_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_jpt_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_pathan_XX_builder.append_option(value_list_int32(
                        info_fields.get("AC_pathan_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_pathan_XX_builder.append_option(value_int32(
                        info_fields.get("AN_pathan_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .AF_pathan_XX_builder
                        .append_option(value_list_float64(
                            info_fields.get("AF_pathan_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .nhomalt_pathan_XX_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("nhomalt_pathan_XX")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_mxl_builder.append_option(value_list_int32(
                        info_fields.get("AC_mxl").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_mxl_builder.append_option(value_int32(
                        info_fields.get("AN_mxl").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_mxl_builder.append_option(value_list_float64(
                        info_fields.get("AF_mxl").and_then(|x| x.as_ref()),
                    )?);
                    builder.nhomalt_mxl_builder.append_option(value_list_int32(
                        info_fields.get("nhomalt_mxl").and_then(|x| x.as_ref()),
                    )?);
                    builder.AC_uygur_XX_builder.append_option(value_list_int32(
                        info_fields.get("AC_uygur_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_uygur_XX_builder.append_option(value_int32(
                        info_fields.get("AN_uygur_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .AF_uygur_XX_builder
                        .append_option(value_list_float64(
                            info_fields.get("AF_uygur_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .nhomalt_uygur_XX_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_uygur_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_adygei_XY_builder.append_option(value_list_int32(
                        info_fields.get("AC_adygei_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_adygei_XY_builder.append_option(value_int32(
                        info_fields.get("AN_adygei_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .AF_adygei_XY_builder
                        .append_option(value_list_float64(
                            info_fields.get("AF_adygei_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .nhomalt_adygei_XY_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("nhomalt_adygei_XY")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_lwk_XY_builder.append_option(value_list_int32(
                        info_fields.get("AC_lwk_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_lwk_XY_builder.append_option(value_int32(
                        info_fields.get("AN_lwk_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_lwk_XY_builder.append_option(value_list_float64(
                        info_fields.get("AF_lwk_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .nhomalt_lwk_XY_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_lwk_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_han_XX_builder.append_option(value_list_int32(
                        info_fields.get("AC_han_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_han_XX_builder.append_option(value_int32(
                        info_fields.get("AN_han_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_han_XX_builder.append_option(value_list_float64(
                        info_fields.get("AF_han_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .nhomalt_han_XX_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_han_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_basque_XX_builder.append_option(value_list_int32(
                        info_fields.get("AC_basque_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_basque_XX_builder.append_option(value_int32(
                        info_fields.get("AN_basque_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .AF_basque_XX_builder
                        .append_option(value_list_float64(
                            info_fields.get("AF_basque_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .nhomalt_basque_XX_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("nhomalt_basque_XX")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_beb_builder.append_option(value_list_int32(
                        info_fields.get("AC_beb").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_beb_builder.append_option(value_int32(
                        info_fields.get("AN_beb").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_beb_builder.append_option(value_list_float64(
                        info_fields.get("AF_beb").and_then(|x| x.as_ref()),
                    )?);
                    builder.nhomalt_beb_builder.append_option(value_list_int32(
                        info_fields.get("nhomalt_beb").and_then(|x| x.as_ref()),
                    )?);
                    builder.AC_daur_XY_builder.append_option(value_list_int32(
                        info_fields.get("AC_daur_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_daur_XY_builder.append_option(value_int32(
                        info_fields.get("AN_daur_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_daur_XY_builder.append_option(value_list_float64(
                        info_fields.get("AF_daur_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .nhomalt_daur_XY_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_daur_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_russian_builder.append_option(value_list_int32(
                        info_fields.get("AC_russian").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_russian_builder.append_option(value_int32(
                        info_fields.get("AN_russian").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_russian_builder.append_option(value_list_float64(
                        info_fields.get("AF_russian").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .nhomalt_russian_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_russian").and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_pima_XX_builder.append_option(value_list_int32(
                        info_fields.get("AC_pima_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_pima_XX_builder.append_option(value_int32(
                        info_fields.get("AN_pima_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_pima_XX_builder.append_option(value_list_float64(
                        info_fields.get("AF_pima_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .nhomalt_pima_XX_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_pima_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .AC_mbutipygmy_builder
                        .append_option(value_list_int32(
                            info_fields.get("AC_mbutipygmy").and_then(|x| x.as_ref()),
                        )?);
                    builder.AN_mbutipygmy_builder.append_option(value_int32(
                        info_fields.get("AN_mbutipygmy").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .AF_mbutipygmy_builder
                        .append_option(value_list_float64(
                            info_fields.get("AF_mbutipygmy").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .nhomalt_mbutipygmy_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("nhomalt_mbutipygmy")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_san_XY_builder.append_option(value_list_int32(
                        info_fields.get("AC_san_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_san_XY_builder.append_option(value_int32(
                        info_fields.get("AN_san_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_san_XY_builder.append_option(value_list_float64(
                        info_fields.get("AF_san_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .nhomalt_san_XY_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_san_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_chs_XY_builder.append_option(value_list_int32(
                        info_fields.get("AC_chs_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_chs_XY_builder.append_option(value_int32(
                        info_fields.get("AN_chs_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_chs_XY_builder.append_option(value_list_float64(
                        info_fields.get("AF_chs_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .nhomalt_chs_XY_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_chs_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_tu_XY_builder.append_option(value_list_int32(
                        info_fields.get("AC_tu_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_tu_XY_builder.append_option(value_int32(
                        info_fields.get("AN_tu_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_tu_XY_builder.append_option(value_list_float64(
                        info_fields.get("AF_tu_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .nhomalt_tu_XY_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_tu_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_jpt_XX_builder.append_option(value_list_int32(
                        info_fields.get("AC_jpt_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_jpt_XX_builder.append_option(value_int32(
                        info_fields.get("AN_jpt_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_jpt_XX_builder.append_option(value_list_float64(
                        info_fields.get("AF_jpt_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .nhomalt_jpt_XX_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_jpt_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_gwd_builder.append_option(value_list_int32(
                        info_fields.get("AC_gwd").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_gwd_builder.append_option(value_int32(
                        info_fields.get("AN_gwd").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_gwd_builder.append_option(value_list_float64(
                        info_fields.get("AF_gwd").and_then(|x| x.as_ref()),
                    )?);
                    builder.nhomalt_gwd_builder.append_option(value_list_int32(
                        info_fields.get("nhomalt_gwd").and_then(|x| x.as_ref()),
                    )?);
                    builder.AC_cdx_XX_builder.append_option(value_list_int32(
                        info_fields.get("AC_cdx_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_cdx_XX_builder.append_option(value_int32(
                        info_fields.get("AN_cdx_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_cdx_XX_builder.append_option(value_list_float64(
                        info_fields.get("AF_cdx_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .nhomalt_cdx_XX_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_cdx_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_gih_XY_builder.append_option(value_list_int32(
                        info_fields.get("AC_gih_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_gih_XY_builder.append_option(value_int32(
                        info_fields.get("AN_gih_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_gih_XY_builder.append_option(value_list_float64(
                        info_fields.get("AF_gih_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .nhomalt_gih_XY_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_gih_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_kalash_builder.append_option(value_list_int32(
                        info_fields.get("AC_kalash").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_kalash_builder.append_option(value_int32(
                        info_fields.get("AN_kalash").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_kalash_builder.append_option(value_list_float64(
                        info_fields.get("AF_kalash").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .nhomalt_kalash_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_kalash").and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_brahui_builder.append_option(value_list_int32(
                        info_fields.get("AC_brahui").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_brahui_builder.append_option(value_int32(
                        info_fields.get("AN_brahui").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_brahui_builder.append_option(value_list_float64(
                        info_fields.get("AF_brahui").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .nhomalt_brahui_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_brahui").and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_chb_builder.append_option(value_list_int32(
                        info_fields.get("AC_chb").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_chb_builder.append_option(value_int32(
                        info_fields.get("AN_chb").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_chb_builder.append_option(value_list_float64(
                        info_fields.get("AF_chb").and_then(|x| x.as_ref()),
                    )?);
                    builder.nhomalt_chb_builder.append_option(value_list_int32(
                        info_fields.get("nhomalt_chb").and_then(|x| x.as_ref()),
                    )?);
                    builder.AC_maya_XY_builder.append_option(value_list_int32(
                        info_fields.get("AC_maya_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_maya_XY_builder.append_option(value_int32(
                        info_fields.get("AN_maya_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_maya_XY_builder.append_option(value_list_float64(
                        info_fields.get("AF_maya_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .nhomalt_maya_XY_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_maya_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_papuan_builder.append_option(value_list_int32(
                        info_fields.get("AC_papuan").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_papuan_builder.append_option(value_int32(
                        info_fields.get("AN_papuan").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_papuan_builder.append_option(value_list_float64(
                        info_fields.get("AF_papuan").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .nhomalt_papuan_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_papuan").and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_tuscan_XY_builder.append_option(value_list_int32(
                        info_fields.get("AC_tuscan_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_tuscan_XY_builder.append_option(value_int32(
                        info_fields.get("AN_tuscan_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .AF_tuscan_XY_builder
                        .append_option(value_list_float64(
                            info_fields.get("AF_tuscan_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .nhomalt_tuscan_XY_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("nhomalt_tuscan_XY")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_yakut_XY_builder.append_option(value_list_int32(
                        info_fields.get("AC_yakut_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_yakut_XY_builder.append_option(value_int32(
                        info_fields.get("AN_yakut_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .AF_yakut_XY_builder
                        .append_option(value_list_float64(
                            info_fields.get("AF_yakut_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .nhomalt_yakut_XY_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_yakut_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .AC_biakapygmy_XX_builder
                        .append_option(value_list_int32(
                            info_fields.get("AC_biakapygmy_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder.AN_biakapygmy_XX_builder.append_option(value_int32(
                        info_fields.get("AN_biakapygmy_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .AF_biakapygmy_XX_builder
                        .append_option(value_list_float64(
                            info_fields.get("AF_biakapygmy_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .nhomalt_biakapygmy_XX_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("nhomalt_biakapygmy_XX")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_yakut_XX_builder.append_option(value_list_int32(
                        info_fields.get("AC_yakut_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_yakut_XX_builder.append_option(value_int32(
                        info_fields.get("AN_yakut_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .AF_yakut_XX_builder
                        .append_option(value_list_float64(
                            info_fields.get("AF_yakut_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .nhomalt_yakut_XX_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_yakut_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_chb_XX_builder.append_option(value_list_int32(
                        info_fields.get("AC_chb_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_chb_XX_builder.append_option(value_int32(
                        info_fields.get("AN_chb_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_chb_XX_builder.append_option(value_list_float64(
                        info_fields.get("AF_chb_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .nhomalt_chb_XX_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_chb_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_lwk_XX_builder.append_option(value_list_int32(
                        info_fields.get("AC_lwk_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_lwk_XX_builder.append_option(value_int32(
                        info_fields.get("AN_lwk_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_lwk_XX_builder.append_option(value_list_float64(
                        info_fields.get("AF_lwk_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .nhomalt_lwk_XX_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_lwk_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_basque_XY_builder.append_option(value_list_int32(
                        info_fields.get("AC_basque_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_basque_XY_builder.append_option(value_int32(
                        info_fields.get("AN_basque_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .AF_basque_XY_builder
                        .append_option(value_list_float64(
                            info_fields.get("AF_basque_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .nhomalt_basque_XY_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("nhomalt_basque_XY")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .AC_melanesian_builder
                        .append_option(value_list_int32(
                            info_fields.get("AC_melanesian").and_then(|x| x.as_ref()),
                        )?);
                    builder.AN_melanesian_builder.append_option(value_int32(
                        info_fields.get("AN_melanesian").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .AF_melanesian_builder
                        .append_option(value_list_float64(
                            info_fields.get("AF_melanesian").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .nhomalt_melanesian_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("nhomalt_melanesian")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_karitiana_builder.append_option(value_list_int32(
                        info_fields.get("AC_karitiana").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_karitiana_builder.append_option(value_int32(
                        info_fields.get("AN_karitiana").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .AF_karitiana_builder
                        .append_option(value_list_float64(
                            info_fields.get("AF_karitiana").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .nhomalt_karitiana_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("nhomalt_karitiana")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_yoruba_XY_builder.append_option(value_list_int32(
                        info_fields.get("AC_yoruba_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_yoruba_XY_builder.append_option(value_int32(
                        info_fields.get("AN_yoruba_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .AF_yoruba_XY_builder
                        .append_option(value_list_float64(
                            info_fields.get("AF_yoruba_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .nhomalt_yoruba_XY_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("nhomalt_yoruba_XY")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_kalash_XY_builder.append_option(value_list_int32(
                        info_fields.get("AC_kalash_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_kalash_XY_builder.append_option(value_int32(
                        info_fields.get("AN_kalash_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .AF_kalash_XY_builder
                        .append_option(value_list_float64(
                            info_fields.get("AF_kalash_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .nhomalt_kalash_XY_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("nhomalt_kalash_XY")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_stu_XX_builder.append_option(value_list_int32(
                        info_fields.get("AC_stu_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_stu_XX_builder.append_option(value_int32(
                        info_fields.get("AN_stu_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_stu_XX_builder.append_option(value_list_float64(
                        info_fields.get("AF_stu_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .nhomalt_stu_XX_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_stu_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .AC_mbutipygmy_XY_builder
                        .append_option(value_list_int32(
                            info_fields.get("AC_mbutipygmy_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder.AN_mbutipygmy_XY_builder.append_option(value_int32(
                        info_fields.get("AN_mbutipygmy_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .AF_mbutipygmy_XY_builder
                        .append_option(value_list_float64(
                            info_fields.get("AF_mbutipygmy_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .nhomalt_mbutipygmy_XY_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("nhomalt_mbutipygmy_XY")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_yoruba_builder.append_option(value_list_int32(
                        info_fields.get("AC_yoruba").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_yoruba_builder.append_option(value_int32(
                        info_fields.get("AN_yoruba").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_yoruba_builder.append_option(value_list_float64(
                        info_fields.get("AF_yoruba").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .nhomalt_yoruba_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_yoruba").and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_oroqen_XX_builder.append_option(value_list_int32(
                        info_fields.get("AC_oroqen_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_oroqen_XX_builder.append_option(value_int32(
                        info_fields.get("AN_oroqen_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .AF_oroqen_XX_builder
                        .append_option(value_list_float64(
                            info_fields.get("AF_oroqen_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .nhomalt_oroqen_XX_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("nhomalt_oroqen_XX")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_acb_builder.append_option(value_list_int32(
                        info_fields.get("AC_acb").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_acb_builder.append_option(value_int32(
                        info_fields.get("AN_acb").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_acb_builder.append_option(value_list_float64(
                        info_fields.get("AF_acb").and_then(|x| x.as_ref()),
                    )?);
                    builder.nhomalt_acb_builder.append_option(value_list_int32(
                        info_fields.get("nhomalt_acb").and_then(|x| x.as_ref()),
                    )?);
                    builder.AC_miaozu_XY_builder.append_option(value_list_int32(
                        info_fields.get("AC_miaozu_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_miaozu_XY_builder.append_option(value_int32(
                        info_fields.get("AN_miaozu_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .AF_miaozu_XY_builder
                        .append_option(value_list_float64(
                            info_fields.get("AF_miaozu_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .nhomalt_miaozu_XY_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("nhomalt_miaozu_XY")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_lahu_XY_builder.append_option(value_list_int32(
                        info_fields.get("AC_lahu_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_lahu_XY_builder.append_option(value_int32(
                        info_fields.get("AN_lahu_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_lahu_XY_builder.append_option(value_list_float64(
                        info_fields.get("AF_lahu_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .nhomalt_lahu_XY_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_lahu_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_esn_builder.append_option(value_list_int32(
                        info_fields.get("AC_esn").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_esn_builder.append_option(value_int32(
                        info_fields.get("AN_esn").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_esn_builder.append_option(value_list_float64(
                        info_fields.get("AF_esn").and_then(|x| x.as_ref()),
                    )?);
                    builder.nhomalt_esn_builder.append_option(value_list_int32(
                        info_fields.get("nhomalt_esn").and_then(|x| x.as_ref()),
                    )?);
                    builder.AC_adygei_XX_builder.append_option(value_list_int32(
                        info_fields.get("AC_adygei_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_adygei_XX_builder.append_option(value_int32(
                        info_fields.get("AN_adygei_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .AF_adygei_XX_builder
                        .append_option(value_list_float64(
                            info_fields.get("AF_adygei_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .nhomalt_adygei_XX_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("nhomalt_adygei_XX")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_tu_XX_builder.append_option(value_list_int32(
                        info_fields.get("AC_tu_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_tu_XX_builder.append_option(value_int32(
                        info_fields.get("AN_tu_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_tu_XX_builder.append_option(value_list_float64(
                        info_fields.get("AF_tu_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .nhomalt_tu_XX_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_tu_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_pathan_builder.append_option(value_list_int32(
                        info_fields.get("AC_pathan").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_pathan_builder.append_option(value_int32(
                        info_fields.get("AN_pathan").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_pathan_builder.append_option(value_list_float64(
                        info_fields.get("AF_pathan").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .nhomalt_pathan_builder
                        .append_option(value_list_int32(
                            info_fields.get("nhomalt_pathan").and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_pathan_XY_builder.append_option(value_list_int32(
                        info_fields.get("AC_pathan_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_pathan_XY_builder.append_option(value_int32(
                        info_fields.get("AN_pathan_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .AF_pathan_XY_builder
                        .append_option(value_list_float64(
                            info_fields.get("AF_pathan_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .nhomalt_pathan_XY_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("nhomalt_pathan_XY")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .AC_japanese_XY_builder
                        .append_option(value_list_int32(
                            info_fields.get("AC_japanese_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder.AN_japanese_XY_builder.append_option(value_int32(
                        info_fields.get("AN_japanese_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .AF_japanese_XY_builder
                        .append_option(value_list_float64(
                            info_fields.get("AF_japanese_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .nhomalt_japanese_XY_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("nhomalt_japanese_XY")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder.AC_cdx_builder.append_option(value_list_int32(
                        info_fields.get("AC_cdx").and_then(|x| x.as_ref()),
                    )?);
                    builder.AN_cdx_builder.append_option(value_int32(
                        info_fields.get("AN_cdx").and_then(|x| x.as_ref()),
                    )?);
                    builder.AF_cdx_builder.append_option(value_list_float64(
                        info_fields.get("AF_cdx").and_then(|x| x.as_ref()),
                    )?);
                    builder.nhomalt_cdx_builder.append_option(value_list_int32(
                        info_fields.get("nhomalt_cdx").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .gnomad_AC_amr_XY_builder
                        .append_option(value_list_int32(
                            info_fields.get("gnomad_AC_amr_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder.gnomad_AN_amr_XY_builder.append_option(value_int32(
                        info_fields.get("gnomad_AN_amr_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .gnomad_AF_amr_XY_builder
                        .append_option(value_list_float64(
                            info_fields.get("gnomad_AF_amr_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .gnomad_nhomalt_amr_XY_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("gnomad_nhomalt_amr_XY")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .gnomad_AC_oth_builder
                        .append_option(value_list_int32(
                            info_fields.get("gnomad_AC_oth").and_then(|x| x.as_ref()),
                        )?);
                    builder.gnomad_AN_oth_builder.append_option(value_int32(
                        info_fields.get("gnomad_AN_oth").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .gnomad_AF_oth_builder
                        .append_option(value_list_float64(
                            info_fields.get("gnomad_AF_oth").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .gnomad_nhomalt_oth_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("gnomad_nhomalt_oth")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .gnomad_AC_sas_XY_builder
                        .append_option(value_list_int32(
                            info_fields.get("gnomad_AC_sas_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder.gnomad_AN_sas_XY_builder.append_option(value_int32(
                        info_fields.get("gnomad_AN_sas_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .gnomad_AF_sas_XY_builder
                        .append_option(value_list_float64(
                            info_fields.get("gnomad_AF_sas_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .gnomad_nhomalt_sas_XY_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("gnomad_nhomalt_sas_XY")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .gnomad_AC_fin_XX_builder
                        .append_option(value_list_int32(
                            info_fields.get("gnomad_AC_fin_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder.gnomad_AN_fin_XX_builder.append_option(value_int32(
                        info_fields.get("gnomad_AN_fin_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .gnomad_AF_fin_XX_builder
                        .append_option(value_list_float64(
                            info_fields.get("gnomad_AF_fin_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .gnomad_nhomalt_fin_XX_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("gnomad_nhomalt_fin_XX")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .gnomad_AC_nfe_XX_builder
                        .append_option(value_list_int32(
                            info_fields.get("gnomad_AC_nfe_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder.gnomad_AN_nfe_XX_builder.append_option(value_int32(
                        info_fields.get("gnomad_AN_nfe_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .gnomad_AF_nfe_XX_builder
                        .append_option(value_list_float64(
                            info_fields.get("gnomad_AF_nfe_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .gnomad_nhomalt_nfe_XX_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("gnomad_nhomalt_nfe_XX")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .gnomad_AC_ami_builder
                        .append_option(value_list_int32(
                            info_fields.get("gnomad_AC_ami").and_then(|x| x.as_ref()),
                        )?);
                    builder.gnomad_AN_ami_builder.append_option(value_int32(
                        info_fields.get("gnomad_AN_ami").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .gnomad_AF_ami_builder
                        .append_option(value_list_float64(
                            info_fields.get("gnomad_AF_ami").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .gnomad_nhomalt_ami_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("gnomad_nhomalt_ami")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .gnomad_AC_sas_builder
                        .append_option(value_list_int32(
                            info_fields.get("gnomad_AC_sas").and_then(|x| x.as_ref()),
                        )?);
                    builder.gnomad_AN_sas_builder.append_option(value_int32(
                        info_fields.get("gnomad_AN_sas").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .gnomad_AF_sas_builder
                        .append_option(value_list_float64(
                            info_fields.get("gnomad_AF_sas").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .gnomad_nhomalt_sas_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("gnomad_nhomalt_sas")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .gnomad_AC_ami_XY_builder
                        .append_option(value_list_int32(
                            info_fields.get("gnomad_AC_ami_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder.gnomad_AN_ami_XY_builder.append_option(value_int32(
                        info_fields.get("gnomad_AN_ami_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .gnomad_AF_ami_XY_builder
                        .append_option(value_list_float64(
                            info_fields.get("gnomad_AF_ami_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .gnomad_nhomalt_ami_XY_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("gnomad_nhomalt_ami_XY")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .gnomad_AC_oth_XX_builder
                        .append_option(value_list_int32(
                            info_fields.get("gnomad_AC_oth_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder.gnomad_AN_oth_XX_builder.append_option(value_int32(
                        info_fields.get("gnomad_AN_oth_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .gnomad_AF_oth_XX_builder
                        .append_option(value_list_float64(
                            info_fields.get("gnomad_AF_oth_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .gnomad_nhomalt_oth_XX_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("gnomad_nhomalt_oth_XX")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .gnomad_AC_amr_XX_builder
                        .append_option(value_list_int32(
                            info_fields.get("gnomad_AC_amr_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder.gnomad_AN_amr_XX_builder.append_option(value_int32(
                        info_fields.get("gnomad_AN_amr_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .gnomad_AF_amr_XX_builder
                        .append_option(value_list_float64(
                            info_fields.get("gnomad_AF_amr_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .gnomad_nhomalt_amr_XX_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("gnomad_nhomalt_amr_XX")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder.gnomad_AC_XX_builder.append_option(value_list_int32(
                        info_fields.get("gnomad_AC_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder.gnomad_AN_XX_builder.append_option(value_int32(
                        info_fields.get("gnomad_AN_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .gnomad_AF_XX_builder
                        .append_option(value_list_float64(
                            info_fields.get("gnomad_AF_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .gnomad_nhomalt_XX_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("gnomad_nhomalt_XX")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .gnomad_AC_fin_builder
                        .append_option(value_list_int32(
                            info_fields.get("gnomad_AC_fin").and_then(|x| x.as_ref()),
                        )?);
                    builder.gnomad_AN_fin_builder.append_option(value_int32(
                        info_fields.get("gnomad_AN_fin").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .gnomad_AF_fin_builder
                        .append_option(value_list_float64(
                            info_fields.get("gnomad_AF_fin").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .gnomad_nhomalt_fin_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("gnomad_nhomalt_fin")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .gnomad_AC_asj_XX_builder
                        .append_option(value_list_int32(
                            info_fields.get("gnomad_AC_asj_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder.gnomad_AN_asj_XX_builder.append_option(value_int32(
                        info_fields.get("gnomad_AN_asj_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .gnomad_AF_asj_XX_builder
                        .append_option(value_list_float64(
                            info_fields.get("gnomad_AF_asj_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .gnomad_nhomalt_asj_XX_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("gnomad_nhomalt_asj_XX")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .gnomad_AC_sas_XX_builder
                        .append_option(value_list_int32(
                            info_fields.get("gnomad_AC_sas_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder.gnomad_AN_sas_XX_builder.append_option(value_int32(
                        info_fields.get("gnomad_AN_sas_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .gnomad_AF_sas_XX_builder
                        .append_option(value_list_float64(
                            info_fields.get("gnomad_AF_sas_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .gnomad_nhomalt_sas_XX_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("gnomad_nhomalt_sas_XX")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .gnomad_AC_mid_XY_builder
                        .append_option(value_list_int32(
                            info_fields.get("gnomad_AC_mid_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder.gnomad_AN_mid_XY_builder.append_option(value_int32(
                        info_fields.get("gnomad_AN_mid_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .gnomad_AF_mid_XY_builder
                        .append_option(value_list_float64(
                            info_fields.get("gnomad_AF_mid_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .gnomad_nhomalt_mid_XY_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("gnomad_nhomalt_mid_XY")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder.gnomad_AC_XY_builder.append_option(value_list_int32(
                        info_fields.get("gnomad_AC_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder.gnomad_AN_XY_builder.append_option(value_int32(
                        info_fields.get("gnomad_AN_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .gnomad_AF_XY_builder
                        .append_option(value_list_float64(
                            info_fields.get("gnomad_AF_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .gnomad_nhomalt_XY_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("gnomad_nhomalt_XY")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .gnomad_AC_eas_builder
                        .append_option(value_list_int32(
                            info_fields.get("gnomad_AC_eas").and_then(|x| x.as_ref()),
                        )?);
                    builder.gnomad_AN_eas_builder.append_option(value_int32(
                        info_fields.get("gnomad_AN_eas").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .gnomad_AF_eas_builder
                        .append_option(value_list_float64(
                            info_fields.get("gnomad_AF_eas").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .gnomad_nhomalt_eas_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("gnomad_nhomalt_eas")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .gnomad_AC_asj_XY_builder
                        .append_option(value_list_int32(
                            info_fields.get("gnomad_AC_asj_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder.gnomad_AN_asj_XY_builder.append_option(value_int32(
                        info_fields.get("gnomad_AN_asj_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .gnomad_AF_asj_XY_builder
                        .append_option(value_list_float64(
                            info_fields.get("gnomad_AF_asj_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .gnomad_nhomalt_asj_XY_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("gnomad_nhomalt_asj_XY")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .gnomad_AC_fin_XY_builder
                        .append_option(value_list_int32(
                            info_fields.get("gnomad_AC_fin_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder.gnomad_AN_fin_XY_builder.append_option(value_int32(
                        info_fields.get("gnomad_AN_fin_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .gnomad_AF_fin_XY_builder
                        .append_option(value_list_float64(
                            info_fields.get("gnomad_AF_fin_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .gnomad_nhomalt_fin_XY_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("gnomad_nhomalt_fin_XY")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .gnomad_AC_amr_builder
                        .append_option(value_list_int32(
                            info_fields.get("gnomad_AC_amr").and_then(|x| x.as_ref()),
                        )?);
                    builder.gnomad_AN_amr_builder.append_option(value_int32(
                        info_fields.get("gnomad_AN_amr").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .gnomad_AF_amr_builder
                        .append_option(value_list_float64(
                            info_fields.get("gnomad_AF_amr").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .gnomad_nhomalt_amr_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("gnomad_nhomalt_amr")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .gnomad_AC_afr_builder
                        .append_option(value_list_int32(
                            info_fields.get("gnomad_AC_afr").and_then(|x| x.as_ref()),
                        )?);
                    builder.gnomad_AN_afr_builder.append_option(value_int32(
                        info_fields.get("gnomad_AN_afr").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .gnomad_AF_afr_builder
                        .append_option(value_list_float64(
                            info_fields.get("gnomad_AF_afr").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .gnomad_nhomalt_afr_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("gnomad_nhomalt_afr")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .gnomad_nhomalt_raw_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("gnomad_nhomalt_raw")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .gnomad_AC_ami_XX_builder
                        .append_option(value_list_int32(
                            info_fields.get("gnomad_AC_ami_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder.gnomad_AN_ami_XX_builder.append_option(value_int32(
                        info_fields.get("gnomad_AN_ami_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .gnomad_AF_ami_XX_builder
                        .append_option(value_list_float64(
                            info_fields.get("gnomad_AF_ami_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .gnomad_nhomalt_ami_XX_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("gnomad_nhomalt_ami_XX")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .gnomad_AC_eas_XY_builder
                        .append_option(value_list_int32(
                            info_fields.get("gnomad_AC_eas_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder.gnomad_AN_eas_XY_builder.append_option(value_int32(
                        info_fields.get("gnomad_AN_eas_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .gnomad_AF_eas_XY_builder
                        .append_option(value_list_float64(
                            info_fields.get("gnomad_AF_eas_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .gnomad_nhomalt_eas_XY_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("gnomad_nhomalt_eas_XY")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .gnomad_AC_mid_builder
                        .append_option(value_list_int32(
                            info_fields.get("gnomad_AC_mid").and_then(|x| x.as_ref()),
                        )?);
                    builder.gnomad_AN_mid_builder.append_option(value_int32(
                        info_fields.get("gnomad_AN_mid").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .gnomad_AF_mid_builder
                        .append_option(value_list_float64(
                            info_fields.get("gnomad_AF_mid").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .gnomad_nhomalt_mid_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("gnomad_nhomalt_mid")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .gnomad_AC_oth_XY_builder
                        .append_option(value_list_int32(
                            info_fields.get("gnomad_AC_oth_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder.gnomad_AN_oth_XY_builder.append_option(value_int32(
                        info_fields.get("gnomad_AN_oth_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .gnomad_AF_oth_XY_builder
                        .append_option(value_list_float64(
                            info_fields.get("gnomad_AF_oth_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .gnomad_nhomalt_oth_XY_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("gnomad_nhomalt_oth_XY")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .gnomad_AC_mid_XX_builder
                        .append_option(value_list_int32(
                            info_fields.get("gnomad_AC_mid_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder.gnomad_AN_mid_XX_builder.append_option(value_int32(
                        info_fields.get("gnomad_AN_mid_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .gnomad_AF_mid_XX_builder
                        .append_option(value_list_float64(
                            info_fields.get("gnomad_AF_mid_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .gnomad_nhomalt_mid_XX_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("gnomad_nhomalt_mid_XX")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .gnomad_nhomalt_builder
                        .append_option(value_list_int32(
                            info_fields.get("gnomad_nhomalt").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .gnomad_AC_asj_builder
                        .append_option(value_list_int32(
                            info_fields.get("gnomad_AC_asj").and_then(|x| x.as_ref()),
                        )?);
                    builder.gnomad_AN_asj_builder.append_option(value_int32(
                        info_fields.get("gnomad_AN_asj").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .gnomad_AF_asj_builder
                        .append_option(value_list_float64(
                            info_fields.get("gnomad_AF_asj").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .gnomad_nhomalt_asj_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("gnomad_nhomalt_asj")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .gnomad_AC_afr_XX_builder
                        .append_option(value_list_int32(
                            info_fields.get("gnomad_AC_afr_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder.gnomad_AN_afr_XX_builder.append_option(value_int32(
                        info_fields.get("gnomad_AN_afr_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .gnomad_AF_afr_XX_builder
                        .append_option(value_list_float64(
                            info_fields.get("gnomad_AF_afr_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .gnomad_nhomalt_afr_XX_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("gnomad_nhomalt_afr_XX")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .gnomad_AC_afr_XY_builder
                        .append_option(value_list_int32(
                            info_fields.get("gnomad_AC_afr_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder.gnomad_AN_afr_XY_builder.append_option(value_int32(
                        info_fields.get("gnomad_AN_afr_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .gnomad_AF_afr_XY_builder
                        .append_option(value_list_float64(
                            info_fields.get("gnomad_AF_afr_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .gnomad_nhomalt_afr_XY_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("gnomad_nhomalt_afr_XY")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .gnomad_AC_eas_XX_builder
                        .append_option(value_list_int32(
                            info_fields.get("gnomad_AC_eas_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder.gnomad_AN_eas_XX_builder.append_option(value_int32(
                        info_fields.get("gnomad_AN_eas_XX").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .gnomad_AF_eas_XX_builder
                        .append_option(value_list_float64(
                            info_fields.get("gnomad_AF_eas_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .gnomad_nhomalt_eas_XX_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("gnomad_nhomalt_eas_XX")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .gnomad_AC_nfe_XY_builder
                        .append_option(value_list_int32(
                            info_fields.get("gnomad_AC_nfe_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder.gnomad_AN_nfe_XY_builder.append_option(value_int32(
                        info_fields.get("gnomad_AN_nfe_XY").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .gnomad_AF_nfe_XY_builder
                        .append_option(value_list_float64(
                            info_fields.get("gnomad_AF_nfe_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .gnomad_nhomalt_nfe_XY_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("gnomad_nhomalt_nfe_XY")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .gnomad_AC_nfe_builder
                        .append_option(value_list_int32(
                            info_fields.get("gnomad_AC_nfe").and_then(|x| x.as_ref()),
                        )?);
                    builder.gnomad_AN_nfe_builder.append_option(value_int32(
                        info_fields.get("gnomad_AN_nfe").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .gnomad_AF_nfe_builder
                        .append_option(value_list_float64(
                            info_fields.get("gnomad_AF_nfe").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .gnomad_nhomalt_nfe_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("gnomad_nhomalt_nfe")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .gnomad_AC_popmax_builder
                        .append_option(value_list_int32(
                            info_fields.get("gnomad_AC_popmax").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .gnomad_AN_popmax_builder
                        .append_option(value_list_int32(
                            info_fields.get("gnomad_AN_popmax").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .gnomad_AF_popmax_builder
                        .append_option(value_list_float64(
                            info_fields.get("gnomad_AF_popmax").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .gnomad_nhomalt_popmax_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("gnomad_nhomalt_popmax")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .gnomad_faf95_amr_XY_builder
                        .append_option(value_list_float64(
                            info_fields
                                .get("gnomad_faf95_amr_XY")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .gnomad_faf99_amr_XY_builder
                        .append_option(value_list_float64(
                            info_fields
                                .get("gnomad_faf99_amr_XY")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .gnomad_faf95_sas_XY_builder
                        .append_option(value_list_float64(
                            info_fields
                                .get("gnomad_faf95_sas_XY")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .gnomad_faf99_sas_XY_builder
                        .append_option(value_list_float64(
                            info_fields
                                .get("gnomad_faf99_sas_XY")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .gnomad_faf95_nfe_XX_builder
                        .append_option(value_list_float64(
                            info_fields
                                .get("gnomad_faf95_nfe_XX")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .gnomad_faf99_nfe_XX_builder
                        .append_option(value_list_float64(
                            info_fields
                                .get("gnomad_faf99_nfe_XX")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .gnomad_faf95_sas_builder
                        .append_option(value_list_float64(
                            info_fields.get("gnomad_faf95_sas").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .gnomad_faf99_sas_builder
                        .append_option(value_list_float64(
                            info_fields.get("gnomad_faf99_sas").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .gnomad_faf95_amr_XX_builder
                        .append_option(value_list_float64(
                            info_fields
                                .get("gnomad_faf95_amr_XX")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .gnomad_faf99_amr_XX_builder
                        .append_option(value_list_float64(
                            info_fields
                                .get("gnomad_faf99_amr_XX")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .gnomad_faf95_XX_builder
                        .append_option(value_list_float64(
                            info_fields.get("gnomad_faf95_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .gnomad_faf99_XX_builder
                        .append_option(value_list_float64(
                            info_fields.get("gnomad_faf99_XX").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .gnomad_faf95_sas_XX_builder
                        .append_option(value_list_float64(
                            info_fields
                                .get("gnomad_faf95_sas_XX")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .gnomad_faf99_sas_XX_builder
                        .append_option(value_list_float64(
                            info_fields
                                .get("gnomad_faf99_sas_XX")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .gnomad_faf95_XY_builder
                        .append_option(value_list_float64(
                            info_fields.get("gnomad_faf95_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .gnomad_faf99_XY_builder
                        .append_option(value_list_float64(
                            info_fields.get("gnomad_faf99_XY").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .gnomad_faf95_eas_builder
                        .append_option(value_list_float64(
                            info_fields.get("gnomad_faf95_eas").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .gnomad_faf99_eas_builder
                        .append_option(value_list_float64(
                            info_fields.get("gnomad_faf99_eas").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .gnomad_faf95_amr_builder
                        .append_option(value_list_float64(
                            info_fields.get("gnomad_faf95_amr").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .gnomad_faf99_amr_builder
                        .append_option(value_list_float64(
                            info_fields.get("gnomad_faf99_amr").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .gnomad_faf95_afr_builder
                        .append_option(value_list_float64(
                            info_fields.get("gnomad_faf95_afr").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .gnomad_faf99_afr_builder
                        .append_option(value_list_float64(
                            info_fields.get("gnomad_faf99_afr").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .gnomad_faf95_eas_XY_builder
                        .append_option(value_list_float64(
                            info_fields
                                .get("gnomad_faf95_eas_XY")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .gnomad_faf99_eas_XY_builder
                        .append_option(value_list_float64(
                            info_fields
                                .get("gnomad_faf99_eas_XY")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .gnomad_faf95_builder
                        .append_option(value_list_float64(
                            info_fields.get("gnomad_faf95").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .gnomad_faf99_builder
                        .append_option(value_list_float64(
                            info_fields.get("gnomad_faf99").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .gnomad_faf95_afr_XX_builder
                        .append_option(value_list_float64(
                            info_fields
                                .get("gnomad_faf95_afr_XX")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .gnomad_faf99_afr_XX_builder
                        .append_option(value_list_float64(
                            info_fields
                                .get("gnomad_faf99_afr_XX")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .gnomad_faf95_afr_XY_builder
                        .append_option(value_list_float64(
                            info_fields
                                .get("gnomad_faf95_afr_XY")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .gnomad_faf99_afr_XY_builder
                        .append_option(value_list_float64(
                            info_fields
                                .get("gnomad_faf99_afr_XY")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .gnomad_faf95_eas_XX_builder
                        .append_option(value_list_float64(
                            info_fields
                                .get("gnomad_faf95_eas_XX")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .gnomad_faf99_eas_XX_builder
                        .append_option(value_list_float64(
                            info_fields
                                .get("gnomad_faf99_eas_XX")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .gnomad_faf95_nfe_XY_builder
                        .append_option(value_list_float64(
                            info_fields
                                .get("gnomad_faf95_nfe_XY")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .gnomad_faf99_nfe_XY_builder
                        .append_option(value_list_float64(
                            info_fields
                                .get("gnomad_faf99_nfe_XY")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .gnomad_faf95_nfe_builder
                        .append_option(value_list_float64(
                            info_fields.get("gnomad_faf95_nfe").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .gnomad_faf99_nfe_builder
                        .append_option(value_list_float64(
                            info_fields.get("gnomad_faf99_nfe").and_then(|x| x.as_ref()),
                        )?);
                    builder.FS_builder.append_option(value_float64(
                        info_fields.get("FS").and_then(|x| x.as_ref()),
                    )?);
                    builder.MQ_builder.append_option(value_float64(
                        info_fields.get("MQ").and_then(|x| x.as_ref()),
                    )?);
                    builder.MQRankSum_builder.append_option(value_float64(
                        info_fields.get("MQRankSum").and_then(|x| x.as_ref()),
                    )?);
                    builder.QUALapprox_builder.append_option(value_int32(
                        info_fields.get("QUALapprox").and_then(|x| x.as_ref()),
                    )?);
                    builder.QD_builder.append_option(value_float64(
                        info_fields.get("QD").and_then(|x| x.as_ref()),
                    )?);
                    builder.ReadPosRankSum_builder.append_option(value_float64(
                        info_fields.get("ReadPosRankSum").and_then(|x| x.as_ref()),
                    )?);
                    builder.VarDP_builder.append_option(value_int32(
                        info_fields.get("VarDP").and_then(|x| x.as_ref()),
                    )?);
                    builder.monoallelic_builder.append_value(value_boolean(
                        info_fields.get("monoallelic").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .transmitted_singleton_builder
                        .append_value(value_boolean(
                            info_fields
                                .get("transmitted_singleton")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder.AS_FS_builder.append_option(value_list_float64(
                        info_fields.get("AS_FS").and_then(|x| x.as_ref()),
                    )?);
                    builder.AS_MQ_builder.append_option(value_list_float64(
                        info_fields.get("AS_MQ").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .AS_MQRankSum_builder
                        .append_option(value_list_float64(
                            info_fields.get("AS_MQRankSum").and_then(|x| x.as_ref()),
                        )?);
                    builder.AS_pab_max_builder.append_option(value_list_float64(
                        info_fields.get("AS_pab_max").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .AS_QUALapprox_builder
                        .append_option(value_list_int32(
                            info_fields.get("AS_QUALapprox").and_then(|x| x.as_ref()),
                        )?);
                    builder.AS_QD_builder.append_option(value_list_float64(
                        info_fields.get("AS_QD").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .AS_ReadPosRankSum_builder
                        .append_option(value_list_float64(
                            info_fields
                                .get("AS_ReadPosRankSum")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder.AS_SB_TABLE_builder.append_option(value_list_string(
                        info_fields.get("AS_SB_TABLE").and_then(|x| x.as_ref()),
                    )?);
                    builder.AS_SOR_builder.append_option(value_list_float64(
                        info_fields.get("AS_SOR").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .InbreedingCoeff_builder
                        .append_option(value_list_float64(
                            info_fields.get("InbreedingCoeff").and_then(|x| x.as_ref()),
                        )?);
                    builder.AS_culprit_builder.append_option(value_list_string(
                        info_fields.get("AS_culprit").and_then(|x| x.as_ref()),
                    )?);
                    builder.AS_VQSLOD_builder.append_option(value_list_float64(
                        info_fields.get("AS_VQSLOD").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .NEGATIVE_TRAIN_SITE_builder
                        .append_value(value_boolean(
                            info_fields
                                .get("NEGATIVE_TRAIN_SITE")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .POSITIVE_TRAIN_SITE_builder
                        .append_value(value_boolean(
                            info_fields
                                .get("POSITIVE_TRAIN_SITE")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder.allele_type_builder.append_option(value_string(
                        info_fields.get("allele_type").and_then(|x| x.as_ref()),
                    )?);
                    builder.n_alt_alleles_builder.append_option(value_int32(
                        info_fields.get("n_alt_alleles").and_then(|x| x.as_ref()),
                    )?);
                    builder.variant_type_builder.append_option(value_string(
                        info_fields.get("variant_type").and_then(|x| x.as_ref()),
                    )?);
                    builder.was_mixed_builder.append_value(value_boolean(
                        info_fields.get("was_mixed").and_then(|x| x.as_ref()),
                    )?);
                    builder.lcr_builder.append_value(value_boolean(
                        info_fields.get("lcr").and_then(|x| x.as_ref()),
                    )?);
                    builder.nonpar_builder.append_value(value_boolean(
                        info_fields.get("nonpar").and_then(|x| x.as_ref()),
                    )?);
                    builder.segdup_builder.append_value(value_boolean(
                        info_fields.get("segdup").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .gq_hist_alt_bin_freq_builder
                        .append_option(value_list_string(
                            info_fields
                                .get("gq_hist_alt_bin_freq")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .gq_hist_all_bin_freq_builder
                        .append_option(value_list_string(
                            info_fields
                                .get("gq_hist_all_bin_freq")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .dp_hist_alt_bin_freq_builder
                        .append_option(value_list_string(
                            info_fields
                                .get("dp_hist_alt_bin_freq")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .dp_hist_alt_n_larger_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("dp_hist_alt_n_larger")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .dp_hist_all_bin_freq_builder
                        .append_option(value_list_string(
                            info_fields
                                .get("dp_hist_all_bin_freq")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .dp_hist_all_n_larger_builder
                        .append_option(value_list_int32(
                            info_fields
                                .get("dp_hist_all_n_larger")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .ab_hist_alt_bin_freq_builder
                        .append_option(value_list_string(
                            info_fields
                                .get("ab_hist_alt_bin_freq")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder.cadd_raw_score_builder.append_option(value_float64(
                        info_fields.get("cadd_raw_score").and_then(|x| x.as_ref()),
                    )?);
                    builder.cadd_phred_builder.append_option(value_float64(
                        info_fields.get("cadd_phred").and_then(|x| x.as_ref()),
                    )?);
                    builder.revel_score_builder.append_option(value_float64(
                        info_fields.get("revel_score").and_then(|x| x.as_ref()),
                    )?);
                    builder
                        .splice_ai_max_ds_builder
                        .append_option(value_float64(
                            info_fields.get("splice_ai_max_ds").and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .splice_ai_consequence_builder
                        .append_option(value_string(
                            info_fields
                                .get("splice_ai_consequence")
                                .and_then(|x| x.as_ref()),
                        )?);
                    builder
                        .primate_ai_score_builder
                        .append_option(value_float64(
                            info_fields.get("primate_ai_score").and_then(|x| x.as_ref()),
                        )?);
                    builder.vep_builder.append_option(value_list_string(
                        info_fields.get("vep").and_then(|x| x.as_ref()),
                    )?);
                }

                if let Some(gts) = record.samples().select("GT") {
                    builder.GT_builder.append_value(gts.iter(&header).map(|x| {
                        x.expect("no io errors").and_then(|x| match x {
                            EntryValue::Genotype(x) => {
                                match x
                                    .iter()
                                    .map(|x| x.expect("no io errors").0)
                                    .collect::<Vec<_>>()[..]
                                {
                                    [None, None] => None,
                                    [Some(l), Some(r)] => Some(l as u64 + r as u64),
                                    _ => vortex_panic!("wtf {:?}", x),
                                }
                            }
                            _ => vortex_panic!("expected genotype {:?}", x),
                        })
                    }));
                } else {
                    builder.GT_builder.append_null()
                }

                fn parse_int32_format_field(
                    record: &mut Record,
                    header: &Header,
                    builder: &mut ListBuilder<Int32Builder>,
                    name: &str,
                ) {
                    if let Some(entries) = record.samples().select(name) {
                        builder.append_value(entries.iter(header).map(|x| {
                            x.expect("no io errors").map(|x| match x {
                                EntryValue::Integer(x) => x,
                                _ => vortex_panic!("expected int32 {:?}", x),
                            })
                        }));
                    } else {
                        builder.append_null()
                    }
                }

                fn parse_pgt_format_field(
                    record: &mut Record,
                    header: &Header,
                    builder: &mut ListBuilder<Int32Builder>,
                    name: &str,
                ) {
                    // DK: bioinfomatics is a dumpster fire
                    if let Some(entries) = record.samples().select(name) {
                        builder.append_value(entries.iter(header).map(|x| {
                            x.expect("no io errors").and_then(|x| match x {
                                EntryValue::String(x) if x == "./." || x == "." => None,
                                EntryValue::String(x) if x == "0|0" => Some(0),
                                EntryValue::String(x) if x == "0|1" => Some(1),
                                EntryValue::String(x) if x == "1|0" => Some(2),
                                EntryValue::String(x) if x == "1|1" => Some(3),
                                _ => vortex_panic!("expected biallelic phased genotype {:?}", x),
                            })
                        }));
                    } else {
                        builder.append_null()
                    }
                }

                fn parse_string_format_field(
                    record: &mut Record,
                    header: &Header,
                    builder: &mut ListBuilder<StringBuilder>,
                    name: &str,
                ) {
                    if let Some(entries) = record.samples().select(name) {
                        builder.append_value(entries.iter(header).map(|x| {
                            x.expect("no io errors").map(|x| match x {
                                EntryValue::String(x) => x,
                                _ => vortex_panic!("expected string {:?}", x),
                            })
                        }));
                    } else {
                        builder.append_null()
                    }
                }

                fn parse_list_int32_format_field(
                    record: &mut Record,
                    header: &Header,
                    builder: &mut ListBuilder<ListBuilder<Int32Builder>>,
                    name: &str,
                ) {
                    if let Some(entries) = record.samples().select(name) {
                        builder.append_value(entries.iter(header).map(|x| {
                            x.expect("no io errors").map(|x| match x {
                                EntryValue::Array(x) => match x {
                                    EntryArray::Integer(values) => values
                                        .iter()
                                        .map(|x| x.expect("no io errors"))
                                        .collect::<Vec<_>>(),
                                    _ => vortex_panic!("expected list int32 {:?}", x),
                                },
                                _ => vortex_panic!("expected list list int32 {:?}", x),
                            })
                        }));
                    } else {
                        builder.append_null()
                    }
                }

                parse_int32_format_field(&mut record, &header, &mut builder.GQ_builder, "GQ");
                parse_int32_format_field(&mut record, &header, &mut builder.DP_builder, "DP");
                parse_list_int32_format_field(&mut record, &header, &mut builder.AD_builder, "AD");
                parse_int32_format_field(
                    &mut record,
                    &header,
                    &mut builder.MIN_DP_builder,
                    "MIN_DP",
                );
                parse_pgt_format_field(&mut record, &header, &mut builder.PGT_builder, "PGT");
                parse_string_format_field(&mut record, &header, &mut builder.PID_builder, "PID");
                parse_list_int32_format_field(&mut record, &header, &mut builder.PL_builder, "PL");
                parse_list_int32_format_field(&mut record, &header, &mut builder.SB_builder, "SB");
            }

            let rb = builder.finish()?;
            let file = File::create(parquet_output_path).await?;
            let mut writer = AsyncArrowWriter::try_new(file, SCHEMA.clone(), None)?;
            writer.write(&rb).await?;
            writer.close().await?;

            Ok(())
        })
        .await?;
        Ok(())
    }

    const BATCH_SIZE: usize = 8192;

    pub async fn parquet_to_vortex(&self, format: Format) -> VortexResult<()> {
        let parquet_path = self.parquet_path()?;
        let (output_path, strategy) = match format {
            Format::OnDiskVortex => {
                info!("Converting StatPopGen dataset from Parquet to Vortex.");
                (
                    self.vortex_path()?,
                    VortexLayoutStrategy::with_executor(Arc::new(Handle::current())),
                )
            }
            Format::VortexCompact => {
                info!("Converting StatPopGen dataset from Parquet to Vortex-compact.");
                (
                    self.vortex_compact_path()?,
                    VortexLayoutStrategy::compact_with_executor(
                        Arc::new(Handle::current()),
                        CompactCompressor::default(),
                    ),
                )
            }
            otherwise => {
                vortex_bail!("you asked for vortex but gave me {}", otherwise)
            }
        };

        create_dir_all(
            &output_path
                .parent()
                .ok_or_else(|| vortex_err!("vortex path must be a file in a directory"))?,
        )
        .await?;
        let file = File::open(parquet_path).await?;

        let parquet = ParquetRecordBatchStreamBuilder::new(file)
            .await?
            .with_batch_size(Self::BATCH_SIZE);
        let num_rows = parquet.metadata().file_metadata().num_rows();

        let dtype = DType::from_arrow(parquet.schema().as_ref());
        let mut vortex_stream = parquet
            .build()?
            .map(|record_batch| {
                record_batch
                    .map_err(VortexError::from)
                    .map(|rb| ArrayRef::from_arrow(rb, false))
            })
            .boxed();

        // Parquet reader returns batches, rather than row groups. So make sure we correctly
        // configure the progress bar.
        let nbatches = u64::try_from(num_rows)
            .vortex_expect("negative row count?")
            .div_ceil(Self::BATCH_SIZE as u64);
        vortex_stream = ProgressBar::new(nbatches)
            .wrap_stream(vortex_stream)
            .boxed();

        VortexWriteOptions::default()
            .with_strategy(strategy)
            .write(
                File::create(output_path).await?,
                ArrayStreamAdapter::new(dtype, vortex_stream),
            )
            .await?;

        Ok(())
    }

    pub fn parquet_to_duckdb(&self) -> VortexResult<()> {
        todo!()
    }
}
