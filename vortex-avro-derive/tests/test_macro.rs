#![cfg(test)]
#![allow(unused)]
// TODO(aduffy): add tests for the macro

use apache_avro::schema::{Name, RecordField, RecordFieldOrder, RecordSchema};
use apache_avro::Schema;
use vortex_avro::{AvroValue, FromAvro, ToAvro};

// Test the derive macro defined by this crate.
//
// We create a struct and auto-derive the traits to convert to/from Avro binary format,
// checking that conversion is lossless.
#[test]
fn test_derive_macro() {
    #[derive(FromAvro, Clone, Debug, PartialEq, Eq)]
    enum Level {
        Low,
        Medium,
        High,
    }

    #[derive(FromAvro, ToAvro, Clone, Debug, PartialEq, Eq)]
    struct MyRecordType {
        a: i32,
        b: String,
    }

    // Convert into AvroValue.
    let original = MyRecordType {
        a: 1,
        b: "hello".to_string(),
    };

    let avro_value: AvroValue = original.clone().into();

    let record: MyRecordType = MyRecordType::try_from(avro_value).unwrap();
    assert_eq!(record, original);

    assert_eq!(
        MyRecordType::read_schema(),
        Schema::Record(RecordSchema {
            name: Name {
                name: "MyRecordType".to_string(),
                namespace: None,
            },
            doc: None,
            fields: vec![
                RecordField {
                    name: "a".to_string(),
                    doc: None,
                    schema: Schema::Int,
                    aliases: Default::default(),
                    default: Default::default(),
                    order: RecordFieldOrder::Ignore,
                    position: 0,
                    custom_attributes: Default::default(),
                },
                RecordField {
                    name: "b".to_string(),
                    doc: None,
                    schema: Schema::String,
                    aliases: None,
                    default: None,
                    order: RecordFieldOrder::Ignore,
                    position: 1,
                    custom_attributes: Default::default(),
                }
            ],
            aliases: Default::default(),
            lookup: Default::default(),
            attributes: Default::default(),
        })
    );

    assert_eq!(MyRecordType::write_schema(), MyRecordType::read_schema());
}
