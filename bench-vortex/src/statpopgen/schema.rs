// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_schema::DataType::*;
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use noodles_vcf::Header;

use crate::statpopgen::vcf_conversion::data_type_from_info;

pub fn list(x: DataType) -> DataType {
    List(Arc::new(Field::new("item", x, true)))
}

pub fn fixed_size_list(x: DataType, size: i32) -> DataType {
    FixedSizeList(Arc::new(Field::new("item", x, true)), size)
}

pub fn schema_from_vcf_header(header: &Header, num_samples: i32) -> SchemaRef {
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
            Arc::new(Field::new("GT", fixed_size_list(UInt64, num_samples), true)),
            Arc::new(Field::new("GQ", fixed_size_list(Int32, num_samples), true)),
            Arc::new(Field::new("DP", fixed_size_list(Int32, num_samples), true)),
            Arc::new(Field::new(
                "AD",
                fixed_size_list(list(Int32), num_samples),
                true,
            )),
            Arc::new(Field::new(
                "MIN_DP",
                fixed_size_list(Int32, num_samples),
                true,
            )),
            Arc::new(Field::new("PGT", fixed_size_list(Int32, num_samples), true)),
            Arc::new(Field::new("PID", fixed_size_list(Utf8, num_samples), true)),
            Arc::new(Field::new(
                "PL",
                fixed_size_list(list(Int32), num_samples),
                true,
            )),
            Arc::new(Field::new(
                "SB",
                fixed_size_list(fixed_size_list(Int32, 4), num_samples),
                true,
            )),
        ])
        .collect::<Vec<_>>(),
    ))
}
