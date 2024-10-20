use std::collections::HashMap;

use lazy_static::lazy_static;

static SYMBOLS_V2: &str = include_str!("../resource/symbols_v2.json");

lazy_static! {
    pub static ref SYMBOLS: HashMap<String, i64> = serde_json::from_str(SYMBOLS_V2).unwrap();
}
