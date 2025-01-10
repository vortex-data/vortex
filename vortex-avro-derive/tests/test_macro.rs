#![cfg(test)]
#![allow(unused)]
// TODO(aduffy): add tests for the macro

use std::io::Cursor;

use apache_avro::schema::{EnumSchema, Name, RecordField, RecordFieldOrder, RecordSchema};
use apache_avro::Schema;
use vortex_avro::{from_avro_binary, to_avro_binary, AvroValue, FromAvro, ToAvro};

// Test the derive macro defined by this crate.
//
// We create a struct and auto-derive the traits to convert to/from Avro binary format,
// checking that conversion is lossless.
#[test]
fn test_derive_macro() {
    #[derive(FromAvro, ToAvro, Clone, Debug, PartialEq, Eq)]
    struct UnitStruct;

    #[derive(FromAvro, ToAvro, Clone, Debug, PartialEq, Eq)]
    enum Level {
        Low,
        Medium,
        High,
    }

    #[derive(FromAvro, ToAvro, Clone, Debug, PartialEq, Eq)]
    struct MyRecordType {
        a: i32,
        b: String,
        level: Level,
        unit: UnitStruct,
    }

    // Convert into AvroValue.
    let original = MyRecordType {
        a: 1,
        b: "hello".to_string(),
        level: Level::Medium,
        unit: UnitStruct,
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
                },
                RecordField {
                    name: "level".to_string(),
                    doc: None,
                    schema: Schema::Enum(EnumSchema {
                        name: Name {
                            name: "Level".to_string(),
                            namespace: None,
                        },
                        aliases: Default::default(),
                        doc: None,
                        symbols: vec!["Low".to_string(), "Medium".to_string(), "High".to_string()],
                        default: None,
                        attributes: Default::default(),
                    }),
                    aliases: Default::default(),
                    default: Default::default(),
                    order: RecordFieldOrder::Ignore,
                    position: 2,
                    custom_attributes: Default::default(),
                },
                RecordField {
                    name: "unit".to_string(),
                    doc: None,
                    schema: Schema::Record(RecordSchema {
                        name: Name {
                            name: "UnitStruct".to_string(),
                            namespace: None,
                        },
                        aliases: Default::default(),
                        doc: None,
                        fields: vec![],
                        lookup: Default::default(),
                        attributes: Default::default(),
                    }),
                    aliases: Default::default(),
                    default: Default::default(),
                    order: RecordFieldOrder::Ignore,
                    position: 3,
                    custom_attributes: Default::default(),
                }
            ],
            aliases: Default::default(),
            lookup: Default::default(),
            attributes: Default::default(),
        })
    );

    assert_eq!(
        MyRecordType::write_schema("root"),
        MyRecordType::read_schema()
    );

    // Serialization test.
    let serialized = to_avro_binary(original.clone()).unwrap();
    let mut serialized = Cursor::new(serialized);
    let deserialized: MyRecordType =
        from_avro_binary(&MyRecordType::read_schema(), &mut serialized).unwrap();
    assert_eq!(deserialized, original);
}
