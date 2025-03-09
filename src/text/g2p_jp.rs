use lazy_static::lazy_static;
use std::collections::HashMap;

use jpreprocess::{
    kind::JPreprocessDictionaryKind, DefaultTokenizer, JPreprocess, JPreprocessConfig,
    SystemDictionaryConfig,
};

fn is_sentence_char(c: char) -> bool {
    matches!(c,
        'A'..='Z' | 'a'..='z' | '0'..='9' |           // Latin letters and digits
        '\u{3005}' |                                   // Ideographic iteration mark
        '\u{3040}'..='\u{30ff}' |                     // Hiragana and Katakana
        '\u{4e00}'..='\u{9fff}' |                     // CJK Unified Ideographs
        '\u{ff11}'..='\u{ff19}' |                     // Fullwidth digits
        '\u{ff21}'..='\u{ff3a}' |                     // Fullwidth Latin capital letters
        '\u{ff41}'..='\u{ff5a}' |                     // Fullwidth Latin small letters
        '\u{ff66}'..='\u{ff9d}'                       // Halfwidth Katakana
    )
}

struct TextSplitter<'t> {
    input: &'t str,
    index: usize,
}

impl<'t> TextSplitter<'t> {
    fn new(text: &'t str) -> Self {
        Self {
            input: text,
            index: 0,
        }
    }
}

impl<'t> Iterator for TextSplitter<'t> {
    type Item = (&'t str, bool);

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.input.len() {
            return None;
        }
        let start = self.index;
        let mut chars = self.input[start..].char_indices();
        let (_, c) = chars.next().unwrap(); // First character at start

        if is_sentence_char(c) {
            // Find the end of consecutive sentence characters
            let end = chars
                .find(|&(_, c)| !is_sentence_char(c))
                .map(|(i, _)| start + i)
                .unwrap_or(self.input.len());
            let sentence = &self.input[start..end];
            self.index = end;
            Some((sentence, true))
        } else {
            // Mark is a single character
            let mark_end = start + c.len_utf8();
            let mark = &self.input[start..mark_end];
            self.index = mark_end;
            Some((mark, false))
        }
    }
}

fn g2p_prosody(j_preprocess: &JPreprocess<DefaultTokenizer>, text: &str) -> Vec<String> {
    let labels = j_preprocess.make_label(j_preprocess.run_frontend(text).unwrap());
    let n_labels = labels.len();
    let mut phones = vec![];
    for (i, label) in labels.iter().enumerate() {
        let p3 = label.phoneme.c.as_ref().unwrap().as_str();
        if p3 == "sil" {
            assert!(i == 0 || i == n_labels - 1);
            continue;
        } else if p3 == "pau" {
            phones.push("_");
            continue;
        } else {
            phones.push(p3);
        }
        // we could assert 1 <= i < n_labels - 1
        let mora = label.mora.as_ref();
        let a1 = mora.map_or(127, |v| v.relative_accent_position);
        let a2 = mora.map_or(127, |v| v.position_forward);
        let a3 = mora.map_or(127, |v| v.position_backward);
        let f1 = label
            .accent_phrase_curr
            .as_ref()
            .map_or(127, |v| v.mora_count);
        let a2_next = labels[i + 1]
            .mora
            .as_ref()
            .map_or(127, |v| v.position_forward);

        if a3 == 1 && a2_next == 1 && "aeiouAEIOUNcl".contains(p3) {
            phones.push("#");
        } else if a1 == 0 && a2_next == a2 + 1 && a2 != f1 {
            phones.push("]");
        } else if a2 == 1 && a2_next == 2 {
            phones.push("[");
        }
    }
    phones
        .into_iter()
        .map(ToOwned::to_owned)
        .map(|s| {
            if "AEIOU".contains(&s) {
                s.to_lowercase()
            } else {
                s
            }
        })
        .collect()
}

fn preprocess_jap(j_preprocess: &JPreprocess<DefaultTokenizer>, text: &str) -> Vec<String> {
    let mut phones = vec![];
    for (part, is_sentence) in TextSplitter::new(text) {
        if is_sentence {
            phones.extend(g2p_prosody(j_preprocess, part));
        } else if part != " " {
            phones.push(part.replace(" ", ""));
        }
    }
    phones
}

lazy_static! {
    static ref REP_MAP: HashMap<&'static str, &'static str> = {
        let mut map = HashMap::new();
        map.insert("：", ",");
        map.insert("；", ",");
        map.insert("，", ",");
        map.insert("。", ".");
        map.insert("！", "!");
        map.insert("？", "?");
        map.insert("\n", ".");
        map.insert("·", ",");
        map.insert("、", ",");
        map.insert("...", "…");
        map
    };
}

fn post_replace_ph(phone: String) -> String {
    REP_MAP
        .get(phone.as_str())
        .copied()
        .map_or(phone, ToOwned::to_owned)
}

pub struct G2PJpConverter {
    j_preprocess: JPreprocess<DefaultTokenizer>,
}

impl G2PJpConverter {
    pub fn new() -> Self {
        let config = JPreprocessConfig {
            dictionary: SystemDictionaryConfig::Bundled(JPreprocessDictionaryKind::NaistJdic),
            user_dictionary: None,
        };
        let j_preprocess = JPreprocess::from_config(config).unwrap();
        Self { j_preprocess }
    }

    pub fn g2p(&self, text: &str) -> Vec<String> {
        let phones = preprocess_jap(&self.j_preprocess, text);
        phones.into_iter().map(post_replace_ph).collect()
    }
}
