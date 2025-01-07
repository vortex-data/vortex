#![allow(unused)]
// TODO(aduffy): add tests for the macro

use apache_avro::schema::{Name, RecordField, RecordFieldOrder, RecordSchema};
use apache_avro::Schema;
use proc_macro_traits::{AvroValue, FromAvro};
use proc_macros::FromAvro;

#[test]
fn test_derive_macro() {
    #[derive(FromAvro)]
    struct MyRecordType {
        a: i32,
        b: String,
    }

    let value = AvroValue::Record(vec![
        ("a".to_string(), AvroValue::Int(1)),
        ("b".to_string(), AvroValue::String("hello".to_string())),
    ]);

    let record: MyRecordType = MyRecordType::try_from(value).unwrap();
    assert_eq!(record.a, 1);
    assert_eq!(record.b, "hello");

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
}
