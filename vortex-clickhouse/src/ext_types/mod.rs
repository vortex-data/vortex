// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! ClickHouse-specific Extension types for Vortex.
//!
//! This module provides Extension DTypes for ClickHouse types that don't have
//! direct Vortex equivalents, such as:
//! - Large integers (Int128, UInt128, Int256, UInt256)
//! - IP addresses (IPv4, IPv6)
//! - UUID
//! - Geo types (Point, Ring, LineString, Polygon, MultiLineString, MultiPolygon)
//! - Enum8/Enum16 (with name→value mappings)
//! - DateTime/DateTime64 (with precision and timezone)
//! - Date/Date32
//! - LowCardinality marker
//! - FixedString(N)

mod bigint;
pub mod date;
pub mod datetime;
pub mod enum_;
pub mod fixedstring;
mod geo;
mod ip;
pub mod lowcardinality;
mod uuid;

pub use bigint::{BigInt, BigIntMetadata, BigIntType};
pub use date::{ClickHouseDate, DATE_EXT_ID, DateMetadata};
pub use datetime::{ClickHouseDateTime, DATETIME_EXT_ID, DateTimeMetadata};
pub use enum_::{ClickHouseEnum, ENUM_EXT_ID, EnumEntry, EnumMetadata, EnumSize};
pub use fixedstring::{ClickHouseFixedString, FIXEDSTRING_EXT_ID, FixedStringMetadata};
pub use geo::{GEO_EXT_ID, Geo, GeoMetadata, GeoType};
pub use ip::{IPAddress, IPAddressMetadata, IPAddressType};
pub use lowcardinality::{ClickHouseLowCardinality, LOWCARDINALITY_EXT_ID, LowCardinalityMetadata};
pub use uuid::{UUID, UUID_BYTE_SIZE, UUID_EXT_ID, UUIDMetadata};
