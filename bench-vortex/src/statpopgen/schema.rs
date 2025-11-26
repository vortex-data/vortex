// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_schema::DataType;
use arrow_schema::DataType::*;
use arrow_schema::Field;
use arrow_schema::Schema;
use arrow_schema::SchemaRef;
use noodles_vcf::Header;

use crate::statpopgen::vcf_conversion::data_type_from_info;

pub fn list(x: DataType) -> DataType {
    List(Arc::new(Field::new("item", x, true)))
}

pub fn schema_from_vcf_header(header: &Header) -> SchemaRef {
    let info_fields = header.infos().iter().map(|(name, info)| {
        let data_type = data_type_from_info(info);
        Arc::new(Field::new(name, data_type, true))
    });
    Arc::from(Schema::new(
        [
            Arc::new(Field::new("CHROM", Utf8, true)),
            Arc::new(Field::new("POS", UInt64, true)),
            Arc::new(Field::new("ID", Utf8, true)),
            Arc::new(Field::new("REF", Utf8, true)),
            Arc::new(Field::new("ALT", list(Utf8), true)),
            Arc::new(Field::new("QUAL", Float32, true)),
            Arc::new(Field::new("FILTER", list(Utf8), true)),
        ]
        .into_iter()
        .chain(info_fields)
        .chain([
            Arc::new(Field::new("GT", list(UInt64), true)),
            Arc::new(Field::new("GQ", list(Int32), true)),
            Arc::new(Field::new("DP", list(Int32), true)),
            Arc::new(Field::new("AD", list(list(Int32)), true)),
            Arc::new(Field::new("MIN_DP", list(Int32), true)),
            Arc::new(Field::new("PGT", list(Int32), true)),
            Arc::new(Field::new("PID", list(Utf8), true)),
            Arc::new(Field::new("PL", list(list(Int32)), true)),
            Arc::new(Field::new("SB", list(list(Int32)), true)),
        ])
        .collect::<Vec<_>>(),
    ))
}
