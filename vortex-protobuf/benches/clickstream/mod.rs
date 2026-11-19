// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Seeded clickstream events of a web shop, a realistic protobuf schema for the benchmarks.

use std::mem;

use prost_reflect::DynamicMessage;
use prost_reflect::MessageDescriptor;
use prost_reflect::Value;
use protox::Compiler;
use protox::file::File;
use protox::file::FileResolver;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

/// The name of the schema file.
const SCHEMA_FILE: &str = "clickstream.proto";

/// A clickstream event of a web shop, with 30 leaf fields.
const SCHEMA: &str = r#"
syntax = "proto3";
package clickstream;

enum Platform {
  PLATFORM_UNSPECIFIED = 0;
  PLATFORM_WEB = 1;
  PLATFORM_IOS = 2;
  PLATFORM_ANDROID = 3;
}

message Geo {
  string country = 1;
  string region = 2;
  string city = 3;
  double latitude = 4;
  double longitude = 5;
}

message Device {
  Platform platform = 1;
  string os_version = 2;
  string app_version = 3;
  string model = 4;
  uint32 screen_width = 5;
  uint32 screen_height = 6;
}

message Item {
  string sku = 1;
  string name = 2;
  uint32 quantity = 3;
  int64 price_micros = 4;
}

message Attribute {
  string key = 1;
  string value = 2;
}

message Event {
  uint64 event_id = 1;
  int64 timestamp_ms = 2;
  string user_id = 3;
  string session_id = 4;
  string event_type = 5;
  string page_url = 6;
  string referrer = 7;
  Geo geo = 8;
  Device device = 9;
  repeated Item items = 10;
  repeated Attribute attributes = 11;
  repeated string tags = 12;
  int64 revenue_micros = 13;
  string currency = 14;
  bool is_test = 15;
  optional uint32 duration_ms = 16;
  float score = 17;
}
"#;

/// The prefix of the shop's page URLs.
const SHOP: &str = "https://shop.example.com";

/// The number of products in the shop.
const PRODUCTS: u32 = 5_000;

/// The number of users who visit the shop.
const USERS: u32 = 20_000;

/// The number of visits whose events interleave.
const CONCURRENT_VISITS: usize = 64;

/// The product categories, one per product noun.
const CATEGORIES: [&str; 8] = [
    "shoes",
    "shirts",
    "electronics",
    "books",
    "home",
    "toys",
    "garden",
    "beauty",
];

/// The nouns of product names, one per category.
const NOUNS: [&str; 8] = [
    "Sneaker",
    "T-Shirt",
    "Headphones",
    "Novel",
    "Lamp",
    "Puzzle",
    "Planter",
    "Serum",
];

/// The adjectives of product names.
const ADJECTIVES: [&str; 8] = [
    "Classic", "Organic", "Wireless", "Vintage", "Compact", "Deluxe", "Eco", "Smart",
];

/// The terms that visitors search for.
const SEARCHES: [&str; 8] = [
    "running+shoes",
    "wireless+headphones",
    "desk+lamp",
    "gift+ideas",
    "summer+shirt",
    "board+games",
    "face+serum",
    "garden+tools",
];

/// The referrers of the first page of a visit, where the empty string is direct traffic.
const ENTRY_REFERRERS: [&str; 6] = [
    "https://www.google.com/",
    "https://www.bing.com/",
    "https://duckduckgo.com/",
    "https://t.co/",
    "https://www.facebook.com/",
    "",
];

/// The keys of visit attributes, with their possible values.
const ATTRIBUTES: [(&str, &[&str]); 5] = [
    (
        "utm_source",
        &["google", "newsletter", "facebook", "partner"],
    ),
    ("utm_medium", &["cpc", "email", "social"]),
    (
        "utm_campaign",
        &["summer_sale", "back_to_school", "black_friday"],
    ),
    ("experiment", &["checkout_v2", "search_ranking", "new_nav"]),
    ("variant", &["control", "treatment"]),
];

/// The tags of visits.
const TAGS: [&str; 5] = ["new_user", "returning", "logged_in", "promo", "beta"];

/// The versions of the mobile apps.
const APP_VERSIONS: [&str; 3] = ["3.12.0", "3.12.1", "3.13.0"];

/// The number of `PLATFORM_WEB`.
const WEB: i32 = 1;

/// The number of `PLATFORM_IOS`.
const IOS: i32 = 2;

/// The number of `PLATFORM_ANDROID`.
const ANDROID: i32 = 3;

/// A city that visits come from, with the currency of its country.
struct City {
    country: &'static str,
    region: &'static str,
    name: &'static str,
    latitude: f64,
    longitude: f64,
    currency: &'static str,
}

/// The cities that visits come from.
const CITIES: [City; 12] = [
    City::new("US", "California", "San Francisco", 37.77, -122.42, "USD"),
    City::new("US", "California", "Los Angeles", 34.05, -118.24, "USD"),
    City::new("US", "New York", "New York", 40.71, -74.01, "USD"),
    City::new("US", "Texas", "Austin", 30.27, -97.74, "USD"),
    City::new("CA", "Ontario", "Toronto", 43.65, -79.38, "CAD"),
    City::new("GB", "England", "London", 51.51, -0.13, "GBP"),
    City::new("DE", "Berlin", "Berlin", 52.52, 13.40, "EUR"),
    City::new("DE", "Bavaria", "Munich", 48.14, 11.58, "EUR"),
    City::new("FR", "Ile-de-France", "Paris", 48.86, 2.35, "EUR"),
    City::new("JP", "Tokyo", "Tokyo", 35.68, 139.69, "JPY"),
    City::new("BR", "Sao Paulo", "Sao Paulo", -23.55, -46.63, "BRL"),
    City::new("IN", "Karnataka", "Bengaluru", 12.97, 77.59, "INR"),
];

impl City {
    const fn new(
        country: &'static str,
        region: &'static str,
        name: &'static str,
        latitude: f64,
        longitude: f64,
        currency: &'static str,
    ) -> Self {
        Self {
            country,
            region,
            name,
            latitude,
            longitude,
            currency,
        }
    }
}

/// A device model, with the OS versions that it runs and its screen size.
struct DeviceModel {
    platform: i32,
    model: &'static str,
    os_versions: &'static [&'static str],
    screen_width: u32,
    screen_height: u32,
}

/// The devices that visits come from.
const DEVICES: [DeviceModel; 8] = [
    DeviceModel::new(IOS, "iPhone15,2", &["17.5.1", "17.6", "18.0.1"], 1179, 2556),
    DeviceModel::new(IOS, "iPhone16,1", &["17.6", "18.0.1"], 1179, 2556),
    DeviceModel::new(IOS, "iPad13,4", &["17.5.1", "18.0.1"], 2048, 2732),
    DeviceModel::new(ANDROID, "Pixel 8", &["13", "14"], 1080, 2400),
    DeviceModel::new(ANDROID, "SM-S918B", &["12", "13", "14"], 1440, 3088),
    DeviceModel::new(WEB, "Chrome 126", &["Windows 11", "macOS 14.5"], 1920, 1080),
    DeviceModel::new(WEB, "Safari 17.5", &["macOS 14.5"], 1440, 900),
    DeviceModel::new(WEB, "Firefox 127", &["Windows 10", "Windows 11"], 1366, 768),
];

impl DeviceModel {
    const fn new(
        platform: i32,
        model: &'static str,
        os_versions: &'static [&'static str],
        screen_width: u32,
        screen_height: u32,
    ) -> Self {
        Self {
            platform,
            model,
            os_versions,
            screen_width,
            screen_height,
        }
    }
}

/// What a visitor does in an event.
#[derive(Clone, Copy)]
enum Action {
    PageView,
    Click,
    Search,
    Scroll,
    AddToCart,
    Purchase,
}

impl Action {
    /// The value of the `event_type` field.
    fn name(self) -> &'static str {
        match self {
            Self::PageView => "page_view",
            Self::Click => "click",
            Self::Search => "search",
            Self::Scroll => "scroll",
            Self::AddToCart => "add_to_cart",
            Self::Purchase => "purchase",
        }
    }
}

/// The message types of the schema.
struct Schema {
    event: MessageDescriptor,
    geo: MessageDescriptor,
    device: MessageDescriptor,
    item: MessageDescriptor,
    attribute: MessageDescriptor,
}

impl Schema {
    /// Compiles [`SCHEMA`].
    fn compile() -> VortexResult<Self> {
        let mut compiler = Compiler::with_file_resolver(SchemaResolver);
        compiler
            .open_file(SCHEMA_FILE)
            .map_err(|err| vortex_err!("Invalid benchmark schema: {err}"))?;
        let pool = compiler.descriptor_pool();
        let message = |name: &str| {
            pool.get_message_by_name(&format!("clickstream.{name}"))
                .ok_or_else(|| vortex_err!("The benchmark schema does not define {name}"))
        };
        Ok(Self {
            event: message("Event")?,
            geo: message("Geo")?,
            device: message("Device")?,
            item: message("Item")?,
            attribute: message("Attribute")?,
        })
    }
}

/// Resolves [`SCHEMA_FILE`] to [`SCHEMA`].
struct SchemaResolver;

impl FileResolver for SchemaResolver {
    fn open_file(&self, name: &str) -> Result<File, protox::Error> {
        if name == SCHEMA_FILE {
            File::from_source(name, SCHEMA)
        } else {
            Err(protox::Error::file_not_found(name))
        }
    }
}

/// Builds a message of type `descriptor` with the given fields.
fn message<'a>(
    descriptor: &MessageDescriptor,
    fields: impl IntoIterator<Item = (&'a str, Value)>,
) -> VortexResult<DynamicMessage> {
    let mut message = DynamicMessage::new(descriptor.clone());
    for (name, value) in fields {
        message
            .try_set_field_by_name(name, value)
            .map_err(|err| vortex_err!("Cannot set {name}: {err}"))?;
    }
    Ok(message)
}

/// The state that the events of one visit share.
struct Visit {
    /// The events that the visit still has.
    remaining: usize,
    user_id: String,
    session_id: String,
    geo: Option<DynamicMessage>,
    device: Option<DynamicMessage>,
    currency: &'static str,
    attributes: Vec<DynamicMessage>,
    tags: Vec<&'static str>,
    /// The page of the previous event, or the referrer of the visit before its first event.
    referrer: String,
    /// The product that the visitor looks at.
    product: u32,
}

/// Generates clickstream events in visits, deterministically from a seed.
///
/// The events of a visit share a user, a session, a location, a device and their attributes,
/// and each event's referrer is the page of the event before it. The events are in time order,
/// so the events of concurrent visits interleave, as in a server log.
pub(crate) struct Clickstream {
    rng: StdRng,
    schema: Schema,
    event_id: u64,
    timestamp_ms: i64,
}

impl Clickstream {
    /// Creates a generator whose choices follow from `seed`.
    ///
    /// # Errors
    ///
    /// Returns an error if the schema does not compile.
    pub(crate) fn new(seed: u64) -> VortexResult<Self> {
        Ok(Self {
            rng: StdRng::seed_from_u64(seed),
            schema: Schema::compile()?,
            event_id: 9_000_000_000,
            timestamp_ms: 1_730_000_000_000,
        })
    }

    /// The descriptor of the `Event` message.
    pub(crate) fn descriptor(&self) -> &MessageDescriptor {
        &self.schema.event
    }

    /// Generates the next `count` events.
    ///
    /// # Errors
    ///
    /// Returns an error if an event does not fit the schema, which is a bug of the generator.
    pub(crate) fn events(&mut self, count: usize) -> VortexResult<Vec<DynamicMessage>> {
        let mut visits = (0..CONCURRENT_VISITS)
            .map(|_| self.visit())
            .collect::<VortexResult<Vec<_>>>()?;
        let mut events = Vec::with_capacity(count);
        for _ in 0..count {
            let index = self.rng.random_range(0..visits.len());
            events.push(self.event(&mut visits[index])?);
            if visits[index].remaining == 0 {
                visits[index] = self.visit()?;
            }
        }
        Ok(events)
    }

    /// Returns a random element of `items`.
    fn pick<'a, T>(&mut self, items: &'a [T]) -> &'a T {
        &items[self.rng.random_range(0..items.len())]
    }

    /// Returns a random product, where products with lower numbers are more popular.
    fn product(&mut self) -> u32 {
        self.rng
            .random_range(0..PRODUCTS)
            .min(self.rng.random_range(0..PRODUCTS))
    }

    /// Starts a new visit.
    fn visit(&mut self) -> VortexResult<Visit> {
        let city = self.pick(&CITIES);
        let geo = if self.rng.random_bool(0.92) {
            let latitude = city.latitude + self.rng.random_range(-0.05..0.05);
            let longitude = city.longitude + self.rng.random_range(-0.05..0.05);
            Some(message(
                &self.schema.geo,
                [
                    ("country", Value::String(city.country.to_owned())),
                    ("region", Value::String(city.region.to_owned())),
                    ("city", Value::String(city.name.to_owned())),
                    ("latitude", Value::F64(latitude)),
                    ("longitude", Value::F64(longitude)),
                ],
            )?)
        } else {
            None
        };
        let device = if self.rng.random_bool(0.97) {
            let model = self.pick(&DEVICES);
            let mut fields = vec![
                ("platform", Value::EnumNumber(model.platform)),
                (
                    "os_version",
                    Value::String((*self.pick(model.os_versions)).to_owned()),
                ),
                ("model", Value::String(model.model.to_owned())),
                ("screen_width", Value::U32(model.screen_width)),
                ("screen_height", Value::U32(model.screen_height)),
            ];
            if model.platform != WEB {
                let version = *self.pick(&APP_VERSIONS);
                fields.push(("app_version", Value::String(version.to_owned())));
            }
            Some(message(&self.schema.device, fields)?)
        } else {
            None
        };
        let attribute_count = self.rng.random_range(0..=3);
        let mut attributes = Vec::with_capacity(attribute_count);
        for (key, values) in &ATTRIBUTES[..attribute_count] {
            let value = *self.pick(values);
            attributes.push(message(
                &self.schema.attribute,
                [
                    ("key", Value::String((*key).to_owned())),
                    ("value", Value::String(value.to_owned())),
                ],
            )?);
        }
        let tag_count = self.rng.random_range(0..=2);
        let tags = (0..tag_count).map(|_| *self.pick(&TAGS)).collect();
        let session_id = format!(
            "{:016x}{:016x}",
            self.rng.random::<u64>(),
            self.rng.random::<u64>()
        );
        Ok(Visit {
            remaining: self.rng.random_range(1..=24),
            user_id: format!("user-{:06}", self.rng.random_range(0..USERS)),
            session_id,
            geo,
            device,
            currency: city.currency,
            attributes,
            tags,
            referrer: (*self.pick(&ENTRY_REFERRERS)).to_owned(),
            product: self.product(),
        })
    }

    /// Returns a random action, weighted by how often visitors take it.
    fn action(&mut self) -> Action {
        match self.rng.random_range(0..100) {
            0..50 => Action::PageView,
            50..72 => Action::Click,
            72..82 => Action::Search,
            82..90 => Action::Scroll,
            90..97 => Action::AddToCart,
            _ => Action::Purchase,
        }
    }

    /// An item of `quantity` units of `product`.
    fn item(&self, product: u32, quantity: u32) -> VortexResult<DynamicMessage> {
        message(
            &self.schema.item,
            [
                ("sku", Value::String(format!("SKU-{product:05}"))),
                ("name", Value::String(product_name(product))),
                ("quantity", Value::U32(quantity)),
                ("price_micros", Value::I64(price_micros(product))),
            ],
        )
    }

    /// Generates the next event of `visit`.
    fn event(&mut self, visit: &mut Visit) -> VortexResult<DynamicMessage> {
        visit.remaining -= 1;
        self.event_id += 1;
        self.timestamp_ms += self.rng.random_range(0..120);

        let action = self.action();
        let mut items = Vec::new();
        let mut revenue_micros = 0;
        let page_url = match action {
            Action::Search => format!("{SHOP}/search?q={}", self.pick(&SEARCHES)),
            Action::AddToCart => {
                items.push(self.item(visit.product, 1)?);
                product_url(visit.product)
            }
            Action::Purchase => {
                for _ in 0..self.rng.random_range(1..=4) {
                    let product = self.product();
                    let quantity = self.rng.random_range(1..=3);
                    items.push(self.item(product, quantity)?);
                    revenue_micros += i64::from(quantity) * price_micros(product);
                }
                format!("{SHOP}/checkout/confirmation")
            }
            Action::PageView | Action::Click | Action::Scroll => {
                visit.product = self.product();
                product_url(visit.product)
            }
        };

        let mut fields = vec![
            ("event_id", Value::U64(self.event_id)),
            ("timestamp_ms", Value::I64(self.timestamp_ms)),
            ("user_id", Value::String(visit.user_id.clone())),
            ("session_id", Value::String(visit.session_id.clone())),
            ("event_type", Value::String(action.name().to_owned())),
            ("page_url", Value::String(page_url.clone())),
            ("score", Value::F32(self.rng.random())),
        ];
        // Fields with implicit presence are only set to values other than their defaults.
        let referrer = mem::replace(&mut visit.referrer, page_url);
        if !referrer.is_empty() {
            fields.push(("referrer", Value::String(referrer)));
        }
        if let Some(geo) = &visit.geo {
            fields.push(("geo", Value::Message(geo.clone())));
        }
        if let Some(device) = &visit.device {
            fields.push(("device", Value::Message(device.clone())));
        }
        if revenue_micros != 0 {
            fields.push(("revenue_micros", Value::I64(revenue_micros)));
        }
        if matches!(action, Action::AddToCart | Action::Purchase) {
            fields.push(("currency", Value::String(visit.currency.to_owned())));
        }
        if !items.is_empty() {
            fields.push((
                "items",
                Value::List(items.into_iter().map(Value::Message).collect()),
            ));
        }
        if !visit.attributes.is_empty() {
            let attributes = visit
                .attributes
                .iter()
                .cloned()
                .map(Value::Message)
                .collect();
            fields.push(("attributes", Value::List(attributes)));
        }
        if !visit.tags.is_empty() {
            let tags = visit
                .tags
                .iter()
                .map(|tag| Value::String((*tag).to_owned()))
                .collect();
            fields.push(("tags", Value::List(tags)));
        }
        if self.rng.random_bool(0.002) {
            fields.push(("is_test", Value::Bool(true)));
        }
        if self.rng.random_bool(0.7) {
            fields.push(("duration_ms", Value::U32(self.rng.random_range(20..60_000))));
        }
        message(&self.schema.event, fields)
    }
}

/// The name of `product`.
fn product_name(product: u32) -> String {
    let adjective = ADJECTIVES[product as usize % ADJECTIVES.len()];
    let noun = NOUNS[product as usize / ADJECTIVES.len() % NOUNS.len()];
    format!("{adjective} {noun}")
}

/// The page URL of `product`.
fn product_url(product: u32) -> String {
    let category = CATEGORIES[product as usize / ADJECTIVES.len() % CATEGORIES.len()];
    format!("{SHOP}/{category}/p/{product}")
}

/// The price of `product` in millionths of the currency unit, from 4.99 to 194.98.
fn price_micros(product: u32) -> i64 {
    (i64::from(product) * 7_919 % 19_000 + 499) * 10_000
}
