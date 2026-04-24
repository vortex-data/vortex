// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

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
use arrow_schema::SchemaRef;
use itertools::Itertools as _;
use noodles_vcf::Header;
use noodles_vcf::Record;
use noodles_vcf::record::Info;
use noodles_vcf::variant::record::info::field::Value;
use vortex::error::VortexExpect as _;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_err;
use vortex::utils::aliases::hash_map::HashMap;
use vortex::utils::aliases::hash_set::HashSet;

use super::vcf_conversion::*;

#[expect(non_snake_case)]
pub struct GnomADBuilder<'a> {
    /// The schema of the to-be-generated Parquet file.
    schema: SchemaRef,

    /// The contig on which this variant was found.
    ///
    /// Contig is short for contiguous. For VCFs containing Human data, it is an identifier for some
    /// contiguous segment of genetic material which may be a chromosome (1-22, X, or Y),
    /// mitchondrial circular DNA ("chrMT"), or a synthetic contig used for technical reasons.
    ///
    /// # Examples
    ///
    /// - `chr21`
    /// - `21`
    /// - `chrX`
    /// - `chrMT`
    pub CHROM_builder: StringBuilder,
    /// The 1-indexed position on the contig at which this variant was found.
    ///
    /// Typically the first 100,000 and last 100,000 positions are not recorded in a VCF because
    /// they are difficult to sequence. In particular, they contain long repetitive runs that are
    /// difficult to capture with "short read sequencing".
    pub POS_builder: UInt64Builder,
    /// A unique identifier for this variant.
    ///
    /// While contig, position, reference allele, and alternate alele uniquely identify a variant,
    /// the "reference" contig may change. A variant ID is, by definition, invariant to the
    /// reference "build".
    pub ID_builder: StringBuilder,
    /// The reference allele of this variant.
    ///
    /// # Examples
    ///
    /// - `A`
    /// - `ATG`
    pub REF_builder: StringBuilder,
    /// The list of alternate alleles of this variant.
    ///
    /// An empty list of alternate alleles is unusual in an analysis-ready, jointly-called VCF.
    ///
    /// If every alternate allele list in the VCF is length one, the dataset is called "biallelic"
    /// and/or "split".
    ///
    /// Variants with more than one alternate allele are called "multi-allelic variants".
    ///
    /// # Examples
    ///
    /// - `["A"]`
    /// - `["AA", "G", "ATG"]`
    pub ALT_builder: ListBuilder<StringBuilder>,
    /// The quality score of this variant.
    pub QUAL_builder: Float32Builder,
    /// A list of "filter" values of this variant.
    ///
    /// The header of the VCF lists possible FILTER values other than the string "PASS".
    pub FILTER_builder: ListBuilder<StringBuilder>,

    /// Metadata about the variant.
    ///
    /// The INFO field is effectively an arbitrary, per-variant key-value dictionary. The set of
    /// possible keys and their types are declared in the header, but each variant may have any
    /// possible subset of that dictionary.
    ///
    /// The VCF header may define certain variable-length INFO fields as having "A", "R", or "G"
    /// length. These indicate that the length of the field is, respectively, equal to the number of
    /// alternate alleles, the number of alternate alleles plus one, or the number of possible
    /// genotypes (which is [triangular number](https://en.wikipedia.org/wiki/Triangular_number) of
    /// R). We ignore this information.
    pub info_builder: HashMap<&'a str, InfoArrayBuilder>,

    /// The list of genotypes at this variant.
    ///
    /// Ignoring the sex chromosomes, human beings typically have two copies of each
    /// chromosome. Each copy may have a different allele. An individual with two copies of the
    /// reference allele at a given variant is encoded in the genotype array as the string `0/0`. An
    /// individual with one reference and one of the first alternate alleles is encoded as `0/1`. If
    /// one allele is the first alternate and the other is the third alternate, the encoding is:
    /// `1/3`.
    ///
    /// We do not use this string representation because our dataset is biallelic and admits a
    /// simpler, numeric representation.
    ///
    /// ```text
    /// 0/0  0/1
    ///      1/1
    /// ```
    /// ```text
    ///   0    1
    ///        2
    /// ```
    ///
    /// When one copy of the chromsome can be distinguished from the other, the genotype is called
    /// "phased". Such a genotype has four possible configurations: `0|0`, `0|1`, `1|0`, and
    /// `1|1`. We do not support these in the GT field.
    ///
    /// Every list is the same length; however, individual positions may be missing.
    pub GT_builder: ListBuilder<UInt64Builder>,
    /// The genotype quality.
    ///
    /// A small non-negative integer indicating our confidence in this genotype. It is usually the
    /// difference between the lowest and second lowest PL. Larger values indicate higher confidence.
    pub GQ_builder: ListBuilder<Int32Builder>,
    /// The genotype depth.
    ///
    /// Varies by sequencing technology and service provider, but typically the number of reads
    /// which influenced this genotype call.
    pub DP_builder: ListBuilder<Int32Builder>,
    /// The allele depth.
    ///
    /// For each alternate allele, how many reads contained this allele.
    ///
    /// The outer list is always equal to the number of samples and does not vary. The inner list is
    /// equal to the number of alternate alleles at this variant.
    pub AD_builder: ListBuilder<ListBuilder<Int32Builder>>,
    /// The minimum depth.
    ///
    /// From the VCF header: "Minimum DP observed within the GVCF block". I believe this is mostly
    /// relevant for homozygous reference calls.
    pub MIN_DP_builder: ListBuilder<Int32Builder>,
    /// A phased genotype.
    ///
    /// We encode them as `0|0`: 0, `0|1`: 1, `1|0`: 2, and `1|1`: 3. Almost all these values are
    /// null.
    pub PGT_builder: ListBuilder<Int32Builder>,
    /// The phase ID.
    ///
    /// An identifier used to reconstruct the two distinguished copies of the contig.
    pub PID_builder: ListBuilder<StringBuilder>,
    /// The phred-scaled (log) likelihood of each possible genotype call.
    ///
    /// Smaller is better. Has length equal to the number of possible unphased genotypes. For
    /// biallelic variants, that number is three: 0/0, 0/1, 1/1.
    pub PL_builder: ListBuilder<ListBuilder<Int32Builder>>,
    /// Per-sample component statistics which comprise the Fisher's Exact Test to detect strand bias.
    ///
    /// A single copy of a chromosome is a double-helix. Opposing positions on that double-helix are
    /// complementary. Sometimes when you're sequencing one of the two helices is preferred, for
    /// complicated chemical reasons. This is a test of how biased we were towards one or the other strand.
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
    pub fn new(header: &'a Header, schema: SchemaRef) -> Self {
        let info_builder: HashMap<&'a str, InfoArrayBuilder> = header
            .infos()
            .iter()
            .map(|(name, info)| {
                let builder = builder_from_info(info);
                (name.as_str(), builder)
            })
            .collect();

        Self {
            schema,
            info_builder,
            CHROM_builder: Default::default(),
            POS_builder: Default::default(),
            ID_builder: Default::default(),
            REF_builder: Default::default(),
            ALT_builder: Default::default(),
            QUAL_builder: Default::default(),
            FILTER_builder: Default::default(),
            GT_builder: Default::default(),
            GQ_builder: Default::default(),
            DP_builder: Default::default(),
            AD_builder: Default::default(),
            MIN_DP_builder: Default::default(),
            PGT_builder: Default::default(),
            PID_builder: Default::default(),
            PL_builder: Default::default(),
            SB_builder: Default::default(),
        }
    }

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

        for sample in record.samples().iter() {
            for result in sample.iter(header) {
                let (field, value) = result?;
                match field {
                    "GT" => self
                        .GT_builder
                        .values()
                        .append_option(parse_genotype(value)?),
                    "GQ" => self
                        .GQ_builder
                        .values()
                        .append_option(parse_int32_format(value)?),
                    "DP" => self
                        .DP_builder
                        .values()
                        .append_option(parse_int32_format(value)?),
                    "AD" => self
                        .AD_builder
                        .values()
                        .append_option(parse_list_int32_format(value)?),
                    "MIN_DP" => self
                        .MIN_DP_builder
                        .values()
                        .append_option(parse_int32_format(value)?),
                    "PGT" => self
                        .PGT_builder
                        .values()
                        .append_option(parse_pgt_format(value)?),
                    "PID" => self
                        .PID_builder
                        .values()
                        .append_option(parse_string_format(value)?),
                    "PL" => self
                        .PL_builder
                        .values()
                        .append_option(parse_list_int32_format(value)?),
                    "SB" => self
                        .SB_builder
                        .values()
                        .append_option(parse_list_int32_format(value)?),
                    _ => vortex_bail!("unknown field {field}"),
                }
            }
        }
        self.GT_builder.append(true);
        self.GQ_builder.append(true);
        self.DP_builder.append(true);
        self.AD_builder.append(true);
        self.MIN_DP_builder.append(true);
        self.PGT_builder.append(true);
        self.PID_builder.append(true);
        self.PL_builder.append(true);
        self.SB_builder.append(true);

        Ok(())
    }

    pub fn consume_info(&mut self, header: &Header, info: Info) -> VortexResult<()> {
        info.iter(header)
            .process_results(|iter| -> VortexResult<()> {
                let mut all_fields: HashSet<&str> = self.info_builder.keys().cloned().collect();
                for (name, value) in iter {
                    all_fields.remove(name);
                    self.info_builder
                        .get_mut(name)
                        .vortex_expect("key must exist")
                        .push(value)?;
                }

                for missing_field in all_fields {
                    self.info_builder
                        .get_mut(missing_field)
                        .vortex_expect("key must exist")
                        .push(None)?;
                }

                Ok(())
            })?
    }

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

        let variant_fields = [
            Arc::new(self.CHROM_builder.finish()) as ArrayRef,
            Arc::new(self.POS_builder.finish()),
            Arc::new(self.ID_builder.finish()),
            Arc::new(self.REF_builder.finish()),
            Arc::new(self.ALT_builder.finish()),
            Arc::new(self.QUAL_builder.finish()),
            Arc::new(self.FILTER_builder.finish()),
        ];
        let format_fields = [
            Arc::new(self.GT_builder.finish()) as ArrayRef,
            Arc::new(self.GQ_builder.finish()),
            Arc::new(self.DP_builder.finish()),
            Arc::new(self.AD_builder.finish()),
            Arc::new(self.MIN_DP_builder.finish()),
            Arc::new(self.PGT_builder.finish()),
            Arc::new(self.PID_builder.finish()),
            Arc::new(self.PL_builder.finish()),
            Arc::new(self.SB_builder.finish()),
        ];
        let info_fields = self.schema.fields()
            [variant_fields.len()..(self.schema.fields().len() - format_fields.len())]
            .iter()
            .map(|field| {
                self.info_builder
                    .remove(field.name().as_str())
                    .vortex_expect("field must exist")
                    .finish()
            });

        RecordBatch::try_new(
            Arc::clone(&self.schema),
            variant_fields
                .into_iter()
                .chain(info_fields)
                .chain(format_fields)
                .collect(),
        )
    }
}
