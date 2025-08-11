use crate::statpopgen::schema::SCHEMA;

use super::vcf_value_conversions::*;
use arrow_array::ArrayRef;
use arrow_array::RecordBatch;
use arrow_array::builder::ArrayBuilder;
use arrow_array::builder::BooleanBuilder;
use arrow_array::builder::Float32Builder;
use arrow_array::builder::Int32Builder;
use arrow_array::builder::ListBuilder;
use arrow_array::builder::StringBuilder;
use arrow_array::builder::UInt64Builder;
use arrow_schema::ArrowError;
use itertools::Itertools as _;
use noodles_vcf::header::record::value::map::info::{Number, Type};
use noodles_vcf::record::Info;
use noodles_vcf::variant::record::info::field::Value;
use noodles_vcf::{Header, Record};
use std::sync::Arc;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex::utils::aliases::hash_map::HashMap;

#[allow(dead_code)]
#[allow(non_snake_case)]
#[derive(Default)]
pub struct GnomADBuilder<'a> {
    pub CHROM_builder: StringBuilder,
    pub POS_builder: UInt64Builder,
    pub ID_builder: StringBuilder,
    pub REF_builder: StringBuilder,
    pub ALT_builder: ListBuilder<StringBuilder>,
    pub QUAL_builder: Float32Builder,
    pub FILTER_builder: ListBuilder<StringBuilder>,

    pub info_builder: HashMap<&'a str, InfoArrayBuilder>,

    pub GT_builder: ListBuilder<UInt64Builder>,
    pub GQ_builder: ListBuilder<Int32Builder>,
    pub DP_builder: ListBuilder<Int32Builder>,
    pub AD_builder: ListBuilder<ListBuilder<Int32Builder>>,
    pub MIN_DP_builder: ListBuilder<Int32Builder>,
    pub PGT_builder: ListBuilder<Int32Builder>,
    pub PID_builder: ListBuilder<StringBuilder>,
    pub PL_builder: ListBuilder<ListBuilder<Int32Builder>>,
    pub SB_builder: ListBuilder<ListBuilder<Int32Builder>>,
}

pub enum InfoArrayBuilder {
    Integer(Int32Builder),
    Float(Float32Builder),
    Flag(BooleanBuilder),
    String(StringBuilder),
    ListInteger(ListBuilder<Int32Builder>),
    ListFloat(ListBuilder<Float32Builder>),
    ListString(ListBuilder<StringBuilder>),
}

impl InfoArrayBuilder {
    pub fn push(&mut self, v: Option<Value>) -> VortexResult<()> {
        match self {
            InfoArrayBuilder::Integer(b) => b.append_option(value_int32(v)?),
            InfoArrayBuilder::Float(b) => b.append_option(value_float32(v)?),
            InfoArrayBuilder::Flag(b) => b.append_value(value_boolean(v)?),
            InfoArrayBuilder::String(b) => b.append_option(value_string(v)?),
            InfoArrayBuilder::ListInteger(b) => {
                if let Some(v) = value_list_int32(v)? {
                    v.iter().process_results(|iter| b.append_value(iter))?;
                } else {
                    b.append_null();
                }
            }
            InfoArrayBuilder::ListFloat(b) => {
                if let Some(v) = value_list_float32(v)? {
                    v.iter().process_results(|iter| b.append_value(iter))?;
                } else {
                    b.append_null();
                }
            }
            InfoArrayBuilder::ListString(b) => {
                if let Some(v) = value_list_string(v)? {
                    v.iter().process_results(|iter| b.append_value(iter))?;
                } else {
                    b.append_null();
                }
            }
        }

        Ok(())
    }

    pub fn len(&self) -> usize {
        match self {
            InfoArrayBuilder::Integer(b) => b.len(),
            InfoArrayBuilder::Float(b) => b.len(),
            InfoArrayBuilder::Flag(b) => b.len(),
            InfoArrayBuilder::String(b) => b.len(),
            InfoArrayBuilder::ListInteger(b) => b.len(),
            InfoArrayBuilder::ListFloat(b) => b.len(),
            InfoArrayBuilder::ListString(b) => b.len(),
        }
    }

    pub fn finish(self) -> ArrayRef {
        match self {
            InfoArrayBuilder::Integer(mut b) => Arc::new(b.finish()) as ArrayRef,
            InfoArrayBuilder::Float(mut b) => Arc::new(b.finish()),
            InfoArrayBuilder::Flag(mut b) => Arc::new(b.finish()),
            InfoArrayBuilder::String(mut b) => Arc::new(b.finish()),
            InfoArrayBuilder::ListInteger(mut b) => Arc::new(b.finish()),
            InfoArrayBuilder::ListFloat(mut b) => Arc::new(b.finish()),
            InfoArrayBuilder::ListString(mut b) => Arc::new(b.finish()),
        }
    }
}

impl<'a> GnomADBuilder<'a> {
    #[allow(non_snake_case)]
    pub fn new(header: &'a Header) -> Self {
        let info_builder: HashMap<&'a str, InfoArrayBuilder> = header
            .infos()
            .iter()
            .map(|(name, info)| {
                let builder = match (info.number(), info.ty()) {
                    (Number::Count(1), Type::Integer) => {
                        InfoArrayBuilder::Integer(Default::default())
                    }
                    (Number::Count(1), Type::Float) => InfoArrayBuilder::Float(Default::default()),
                    (Number::Count(0), Type::Flag) => InfoArrayBuilder::Flag(Default::default()),
                    (Number::Count(1), Type::Character) => todo!(),
                    (Number::Count(1), Type::String) => {
                        InfoArrayBuilder::String(Default::default())
                    }
                    (_, Type::Integer) => InfoArrayBuilder::ListInteger(Default::default()),
                    (_, Type::Float) => InfoArrayBuilder::ListFloat(Default::default()),
                    (_, Type::Flag) => todo!(),
                    (_, Type::Character) => todo!(),
                    (_, Type::String) => InfoArrayBuilder::ListString(Default::default()),
                };
                (name.as_str(), builder)
            })
            .collect();

        Self {
            info_builder,
            ..Default::default()
        }
    }

    #[allow(clippy::cognitive_complexity)]
    pub fn consume_record(&mut self, header: &Header, record: &mut Record) -> VortexResult<()> {
        self.CHROM_builder
            .append_value(record.reference_sequence_name());
        self.POS_builder.append_value(
            record
                .variant_start()
                .ok_or_else(|| vortex_err!("pos must not be null"))??
                .get() as u64,
        );
        self.ID_builder.append_value(record.ids().as_ref());
        self.REF_builder.append_value(record.reference_bases());
        self.ALT_builder.append_value(
            record
                .alternate_bases()
                .as_ref()
                .split(",")
                .map(|x| (x != ".").then_some(x)),
        );
        self.QUAL_builder
            .append_option(record.quality_score().transpose()?);
        self.FILTER_builder.append_value(
            record
                .filters()
                .as_ref()
                .split(";")
                .map(|x| (x != ".").then_some(x)),
        );

        self.consume_info(header, record.info())?;

        let samples = &record.samples();
        parse_genotype_format_field(samples, header, &mut self.GT_builder)?;

        parse_int32_format_field(samples, header, &mut self.GQ_builder, "GQ")?;
        parse_int32_format_field(samples, header, &mut self.DP_builder, "DP")?;
        parse_list_int32_format_field(samples, header, &mut self.AD_builder, "AD")?;
        parse_int32_format_field(samples, header, &mut self.MIN_DP_builder, "MIN_DP")?;
        parse_pgt_format_field(samples, header, &mut self.PGT_builder, "PGT")?;
        parse_string_format_field(samples, header, &mut self.PID_builder, "PID")?;
        parse_list_int32_format_field(samples, header, &mut self.PL_builder, "PL")?;
        parse_list_int32_format_field(samples, header, &mut self.SB_builder, "SB")?;

        Ok(())
    }

    #[allow(clippy::cognitive_complexity)]
    pub fn consume_info(&mut self, header: &Header, info: Info) -> VortexResult<()> {
        info.iter(header)
            .process_results(|iter| -> VortexResult<()> {
                for (name, value) in iter {
                    self.info_builder
                        .get_mut(name)
                        .expect("key must exist")
                        .push(value)?;
                }

                Ok(())
            })?
    }

    #[allow(clippy::cognitive_complexity)]
    pub fn finish(mut self) -> Result<RecordBatch, ArrowError> {
        let len = self.CHROM_builder.len();
        assert_eq!(len, self.POS_builder.len());
        assert_eq!(len, self.ID_builder.len());
        assert_eq!(len, self.REF_builder.len());
        assert_eq!(len, self.ALT_builder.len());
        assert_eq!(len, self.QUAL_builder.len());
        assert_eq!(len, self.FILTER_builder.len());

        for (field, builder) in self.info_builder.iter() {
            assert_eq!(len, builder.len(), "{field}");
        }

        assert_eq!(len, self.GT_builder.len());
        assert_eq!(len, self.GQ_builder.len());
        assert_eq!(len, self.DP_builder.len());
        assert_eq!(len, self.AD_builder.len());
        assert_eq!(len, self.MIN_DP_builder.len());
        assert_eq!(len, self.PGT_builder.len());
        assert_eq!(len, self.PID_builder.len());
        assert_eq!(len, self.PL_builder.len());
        assert_eq!(len, self.SB_builder.len());

        RecordBatch::try_new(
            SCHEMA.clone(),
            [
                Arc::new(self.CHROM_builder.finish()) as ArrayRef,
                Arc::new(self.POS_builder.finish()),
                Arc::new(self.ID_builder.finish()),
                Arc::new(self.REF_builder.finish()),
                Arc::new(self.ALT_builder.finish()),
                Arc::new(self.QUAL_builder.finish()),
                Arc::new(self.FILTER_builder.finish()),
            ]
            .into_iter()
            .chain(
                self.info_builder
                    .into_iter()
                    .map(|(_, builder)| builder.finish()),
            )
            .chain(
                [
                    Arc::new(self.GT_builder.finish()) as ArrayRef,
                    Arc::new(self.GQ_builder.finish()),
                    Arc::new(self.DP_builder.finish()),
                    Arc::new(self.AD_builder.finish()),
                    Arc::new(self.MIN_DP_builder.finish()),
                    Arc::new(self.PGT_builder.finish()),
                    Arc::new(self.PID_builder.finish()),
                    Arc::new(self.PL_builder.finish()),
                    Arc::new(self.SB_builder.finish()),
                ]
                .into_iter(),
            )
            .collect(),
        )
    }
}
