use std::collections::{HashMap, LinkedList};

use pest::Parser;
use tch::{Kind, Tensor};
use tokenizers::Tokenizer;

use crate::GPTSovits;

pub mod dict;
pub mod num;

#[inline]
fn get_phone_symbol(symbols: &HashMap<String, i64>, ph: &str) -> i64 {
    // symbols[','] : 3
    symbols.get(ph).map(|id| *id).unwrap_or(3)
}

fn split_zh_ph(ph: &str) -> (&str, &str) {
    match ph {
        "a" => ("AA", "a5"),
        "a1" => ("AA", "a1"),
        "a2" => ("AA", "a2"),
        "a3" => ("AA", "a3"),
        "a4" => ("AA", "a4"),

        "ai" => ("AA", "ai5"),
        "ai1" => ("AA", "ai1"),
        "ai2" => ("AA", "ai2"),
        "ai3" => ("AA", "ai3"),
        "ai4" => ("AA", "ai4"),

        "an" => ("AA", "an5"),
        "an1" => ("AA", "an1"),
        "an2" => ("AA", "an2"),
        "an3" => ("AA", "an3"),
        "an4" => ("AA", "an4"),

        "ang" => ("AA", "ang5"),
        "ang1" => ("AA", "ang1"),
        "ang2" => ("AA", "ang2"),
        "ang3" => ("AA", "ang3"),
        "ang4" => ("AA", "ang4"),

        "ao" => ("AA", "ao5"),
        "ao1" => ("AA", "ao1"),
        "ao2" => ("AA", "ao2"),
        "ao3" => ("AA", "ao3"),
        "ao4" => ("AA", "ao4"),

        "chi" => ("ch", "ir5"),
        "chi1" => ("ch", "ir1"),
        "chi2" => ("ch", "ir2"),
        "chi3" => ("ch", "ir3"),
        "chi4" => ("ch", "ir4"),

        "ci" => ("c", "i05"),
        "ci1" => ("c", "i01"),
        "ci2" => ("c", "i02"),
        "ci3" => ("c", "i03"),
        "ci4" => ("c", "i04"),

        "e" => ("EE", "e5"),
        "e1" => ("EE", "e1"),
        "e2" => ("EE", "e2"),
        "e3" => ("EE", "e3"),
        "e4" => ("EE", "e4"),

        "ei" => ("EE", "ei5"),
        "ei1" => ("EE", "ei1"),
        "ei2" => ("EE", "ei2"),
        "ei3" => ("EE", "ei3"),
        "ei4" => ("EE", "ei4"),

        "en" => ("EE", "en5"),
        "en1" => ("EE", "en1"),
        "en2" => ("EE", "en2"),
        "en3" => ("EE", "en3"),
        "en4" => ("EE", "en4"),

        "eng" => ("EE", "eng5"),
        "eng1" => ("EE", "eng1"),
        "eng2" => ("EE", "eng2"),
        "eng3" => ("EE", "eng3"),
        "eng4" => ("EE", "eng4"),

        "er" => ("EE", "er5"),
        "er1" => ("EE", "er1"),
        "er2" => ("EE", "er2"),
        "er3" => ("EE", "er3"),
        "er4" => ("EE", "er4"),

        "ju" => ("j", "v5"),
        "ju1" => ("j", "v1"),
        "ju2" => ("j", "v2"),
        "ju3" => ("j", "v3"),
        "ju4" => ("j", "v4"),

        "juan" => ("j", "van5"),
        "juan1" => ("j", "van1"),
        "juan2" => ("j", "van2"),
        "juan3" => ("j", "van3"),
        "juan4" => ("j", "van4"),

        "jue" => ("j", "ve5"),
        "jue1" => ("j", "ve1"),
        "jue2" => ("j", "ve2"),
        "jue3" => ("j", "ve3"),
        "jue4" => ("j", "ve4"),

        "jun" => ("j", "vn5"),
        "jun1" => ("j", "vn1"),
        "jun2" => ("j", "vn2"),
        "jun3" => ("j", "vn3"),
        "jun4" => ("j", "vn4"),

        "o" => ("OO", "o5"),
        "o1" => ("OO", "o1"),
        "o2" => ("OO", "o2"),
        "o3" => ("OO", "o3"),
        "o4" => ("OO", "o4"),

        "ou" => ("OO", "ou5"),
        "ou1" => ("OO", "ou1"),
        "ou2" => ("OO", "ou2"),
        "ou3" => ("OO", "ou3"),
        "ou4" => ("OO", "ou4"),

        "qu" => ("q", "v5"),
        "qu1" => ("q", "v1"),
        "qu2" => ("q", "v2"),
        "qu3" => ("q", "v3"),
        "qu4" => ("q", "v4"),

        "quan" => ("q", "van5"),
        "quan1" => ("q", "van1"),
        "quan2" => ("q", "van2"),
        "quan3" => ("q", "van3"),
        "quan4" => ("q", "van4"),

        "que" => ("q", "ve5"),
        "que1" => ("q", "ve1"),
        "que2" => ("q", "ve2"),
        "que3" => ("q", "ve3"),
        "que4" => ("q", "ve4"),

        "qun" => ("q", "vn5"),
        "qun1" => ("q", "vn1"),
        "qun2" => ("q", "vn2"),
        "qun3" => ("q", "vn3"),
        "qun4" => ("q", "vn4"),

        "ri" => ("r", "ir5"),
        "ri1" => ("r", "ir1"),
        "ri2" => ("r", "ir2"),
        "ri3" => ("r", "ir3"),
        "ri4" => ("r", "ir4"),

        "xu" => ("x", "v5"),
        "xu1" => ("x", "v1"),
        "xu2" => ("x", "v2"),
        "xu3" => ("x", "v3"),
        "xu4" => ("x", "v4"),

        "xuan" => ("x", "van5"),
        "xuan1" => ("x", "van1"),
        "xuan2" => ("x", "van2"),
        "xuan3" => ("x", "van3"),
        "xuan4" => ("x", "van4"),

        "xue" => ("x", "ve5"),
        "xue1" => ("x", "ve1"),
        "xue2" => ("x", "ve2"),
        "xue3" => ("x", "ve3"),
        "xue4" => ("x", "ve4"),

        "xun" => ("x", "vn5"),
        "xun1" => ("x", "vn1"),
        "xun2" => ("x", "vn2"),
        "xun3" => ("x", "vn3"),
        "xun4" => ("x", "vn4"),

        "yan" => ("y", "En5"),
        "yan1" => ("y", "En1"),
        "yan2" => ("y", "En2"),
        "yan3" => ("y", "En3"),
        "yan4" => ("y", "En4"),

        "ye" => ("y", "E5"),
        "ye1" => ("y", "E1"),
        "ye2" => ("y", "E2"),
        "ye3" => ("y", "E3"),
        "ye4" => ("y", "E4"),

        "yu" => ("y", "v5"),
        "yu1" => ("y", "v1"),
        "yu2" => ("y", "v2"),
        "yu3" => ("y", "v3"),
        "yu4" => ("y", "v4"),

        "yuan" => ("y", "van5"),
        "yuan1" => ("y", "van1"),
        "yuan2" => ("y", "van2"),
        "yuan3" => ("y", "van3"),
        "yuan4" => ("y", "van4"),

        "yue" => ("y", "ve5"),
        "yue1" => ("y", "ve1"),
        "yue2" => ("y", "ve2"),
        "yue3" => ("y", "ve3"),
        "yue4" => ("y", "ve4"),

        "yun" => ("y", "vn5"),
        "yun1" => ("y", "vn1"),
        "yun2" => ("y", "vn2"),
        "yun3" => ("y", "vn3"),
        "yun4" => ("y", "vn4"),

        "zhi" => ("zh", "ir5"),
        "zhi1" => ("zh", "ir1"),
        "zhi2" => ("zh", "ir2"),
        "zhi3" => ("zh", "ir3"),
        "zhi4" => ("zh", "ir4"),

        "zi" => ("z", "i05"),
        "zi1" => ("z", "i01"),
        "zi2" => ("z", "i02"),
        "zi3" => ("z", "i03"),
        "zi4" => ("z", "i04"),

        "shi" => ("sh", "ir5"),
        "shi1" => ("sh", "ir1"),
        "shi2" => ("sh", "ir2"),
        "shi3" => ("sh", "ir3"),
        "shi4" => ("sh", "ir4"),

        "si" => ("s", "i05"),
        "si1" => ("s", "i01"),
        "si2" => ("s", "i02"),
        "si3" => ("s", "i03"),
        "si4" => ("s", "i04"),

        //['a', 'o', 'e', 'i', 'u', 'ü', 'ai', 'ei', 'ao', 'ou', 'ia', 'ie', 'ua', 'uo', 'üe', 'iao', 'iou', 'uai', 'uei', 'an', 'en', 'ang', 'eng', 'ian', 'in', 'iang', 'ing', 'uan', 'un', 'uang', 'ong', 'üan', 'ün', 'er']
        ph => match split_zh_ph_(ph) {
            (y, "ü") => (y, "v5"),
            (y, "ü1") => (y, "v1"),
            (y, "ü2") => (y, "v2"),
            (y, "ü3") => (y, "v3"),
            (y, "ü4") => (y, "v4"),

            (y, "üe") => (y, "ve5"),
            (y, "üe1") => (y, "ve1"),
            (y, "üe2") => (y, "ve2"),
            (y, "üe3") => (y, "ve3"),
            (y, "üe4") => (y, "ve4"),

            (y, "üan") => (y, "van5"),
            (y, "üan1") => (y, "van1"),
            (y, "üan2") => (y, "van2"),
            (y, "üan3") => (y, "van3"),
            (y, "üan4") => (y, "van4"),

            (y, "ün") => (y, "vn5"),
            (y, "ün1") => (y, "vn1"),
            (y, "ün2") => (y, "vn2"),
            (y, "ün3") => (y, "vn3"),
            (y, "ün4") => (y, "vn4"),

            (y, "a") => (y, "a5"),
            (y, "o") => (y, "o5"),
            (y, "e") => (y, "e5"),
            (y, "i") => (y, "i5"),
            (y, "u") => (y, "u5"),

            (y, "ai") => (y, "ai5"),
            (y, "ei") => (y, "ei5"),
            (y, "ao") => (y, "ao5"),
            (y, "ou") => (y, "ou5"),
            (y, "ia") => (y, "ia5"),
            (y, "ie") => (y, "ie5"),
            (y, "ua") => (y, "ua5"),
            (y, "uo") => (y, "uo5"),
            (y, "iao") => (y, "iao5"),
            (y, "iou") => (y, "iou5"),
            (y, "uai") => (y, "uai5"),
            (y, "uei") => (y, "uei5"),
            (y, "an") => (y, "an5"),
            (y, "en") => (y, "en5"),
            (y, "ang") => (y, "ang5"),
            (y, "eng") => (y, "eng5"),
            (y, "ian") => (y, "ian5"),
            (y, "in") => (y, "in5"),
            (y, "iang") => (y, "iang5"),
            (y, "ing") => (y, "ing5"),
            (y, "uan") => (y, "uan5"),
            (y, "un") => (y, "un5"),
            (y, "uang") => (y, "uang5"),
            (y, "ong") => (y, "ong5"),
            (y, "er") => (y, "er5"),

            (y, s) => (y, s),
        },
    }
}

fn split_zh_ph_(ph: &str) -> (&str, &str) {
    if ph.starts_with("zh") || ph.starts_with("ch") || ph.starts_with("sh") {
        ph.split_at(2)
    } else if ph.starts_with(&[
        'b', 'p', 'm', 'f', 'd', 't', 'n', 'l', 'g', 'k', 'h', 'j', 'q', 'x', 'r', 'z', 'c', 's',
        'y', 'w',
    ]) {
        // b p m f d t n l g k h j q x r z c s y w

        ph.split_at(1)
    } else {
        (ph, ph)
    }
}

pub fn get_phone_and_bert(gpts: &GPTSovits, text: &str) -> anyhow::Result<(Tensor, Tensor)> {
    let mut phone_seq = Vec::new();
    let mut bert_seq = Vec::new();

    let mut phone_builder = PhoneBuilder::new();
    phone_builder.push_text(&gpts.jieba, &gpts.symbols, text);
    // 加一个 \u{7a7a} '空' 避免音频戛然而止
    phone_builder.push_punctuation(&gpts.symbols, ".");

    for s in &phone_builder.sentence {
        match s {
            Sentence::Zh(zh) => {
                let (t, bert) = zh.build_phone_and_bert(gpts)?;
                phone_seq.push(t);
                bert_seq.push(bert);
            }
            Sentence::En(en) => {
                let (t, bert) = en.build_phone_and_bert(gpts)?;
                phone_seq.push(t);
                bert_seq.push(bert);
            }
            Sentence::Num(num) => {
                for s in num.to_phone_sentence(&gpts.symbols)? {
                    match s {
                        Sentence::Zh(zh) => {
                            let (t, bert) = zh.build_phone_and_bert(gpts)?;
                            phone_seq.push(t);
                            bert_seq.push(bert);
                        }
                        Sentence::En(en) => {
                            let (t, bert) = en.build_phone_and_bert(gpts)?;
                            phone_seq.push(t);
                            bert_seq.push(bert);
                        }
                        Sentence::Num(_) => unreachable!(),
                    }
                }
            }
        }
    }

    let phone_seq = Tensor::cat(&phone_seq, 1).to(gpts.device);
    let bert_seq = Tensor::cat(&bert_seq, 0).to(gpts.device);

    Ok((phone_seq, bert_seq))
}

pub struct CNBertModel {
    bert_and_tokenizer: Option<(tch::CModule, Tokenizer)>,
}

impl Default for CNBertModel {
    fn default() -> Self {
        Self {
            bert_and_tokenizer: None,
        }
    }
}

impl CNBertModel {
    pub fn new(bert: tch::CModule, tokenizer: Tokenizer) -> Self {
        Self {
            bert_and_tokenizer: Some((bert, tokenizer)),
        }
    }

    fn encode_text(
        tokenizer: &Tokenizer,
        text: &str,
        device: tch::Device,
    ) -> (Tensor, Tensor, Tensor) {
        let encoding = tokenizer.encode(text, true).unwrap();
        let ids = encoding
            .get_ids()
            .into_iter()
            .map(|x| (*x) as i64)
            .collect::<Vec<i64>>();
        let text_ids = Tensor::from_slice(&ids);
        let text_ids = text_ids.unsqueeze(0).to_device(device);

        let mask = encoding
            .get_attention_mask()
            .into_iter()
            .map(|x| (*x) as i64)
            .collect::<Vec<i64>>();
        let text_mask = Tensor::from_slice(&mask);
        let text_mask = text_mask.unsqueeze(0).to_device(device);

        let token_type_ids = encoding
            .get_type_ids()
            .into_iter()
            .map(|x| (*x) as i64)
            .collect::<Vec<i64>>();
        let text_token_type_ids = Tensor::from_slice(&token_type_ids);
        let text_token_type_ids = text_token_type_ids.unsqueeze(0).to_device(device);
        (text_ids, text_mask, text_token_type_ids)
    }

    pub fn get_text_bert(
        &self,
        text: &str,
        word2ph: &[i32],
        device: tch::Device,
    ) -> anyhow::Result<Tensor> {
        let bert = match &self.bert_and_tokenizer {
            Some((bert, tokenizer)) => {
                let (text_ids, text_mask, text_token_type_ids) =
                    Self::encode_text(tokenizer, text, device);
                let text_word2ph = Tensor::from_slice(word2ph).to_device(device);
                bert.forward_ts(&[&text_ids, &text_mask, &text_token_type_ids, &text_word2ph])?
            }
            None => {
                let len: i32 = word2ph.iter().sum();
                Tensor::zeros(&[len as i64, 1024], (Kind::Float, device))
            }
        };

        Ok(bert)
    }
}

#[derive(Debug)]
struct ZhSentence {
    phones_ids: Vec<i64>,
    phones: Vec<&'static str>,
    word2ph: Vec<i32>,
    zh_text: String,
}

impl ZhSentence {
    fn build_phone_and_bert(&self, gpts: &GPTSovits) -> anyhow::Result<(Tensor, Tensor)> {
        let bert = gpts
            .zh_bert
            .get_text_bert(&self.zh_text, &self.word2ph, gpts.device)?;

        let bert = bert.zero().to_device(gpts.device);

        let t = Tensor::from_slice(&self.phones_ids)
            .to_device(gpts.device)
            .unsqueeze(0);

        Ok((t, bert))
    }
}

#[derive(Debug)]
struct EnSentence {
    phones_ids: Vec<i64>,
    phones: Vec<&'static str>,
    en_text: String,
}

impl EnSentence {
    fn build_phone_and_bert(&self, gpts: &GPTSovits) -> anyhow::Result<(Tensor, Tensor)> {
        let t = Tensor::from_slice(&self.phones_ids)
            .to_device(gpts.device)
            .unsqueeze(0);
        let bert = Tensor::zeros(
            &[self.phones_ids.len() as i64, 1024],
            (Kind::Float, gpts.device),
        );

        Ok((t, bert))
    }
}

#[derive(Debug, Clone, Copy)]
enum Lang {
    Zh,
    En,
}

#[derive(Debug)]
struct NumSentence {
    num_text: String,
    lang: Lang,
}

impl NumSentence {
    fn to_phone_sentence(
        &self,
        symbols: &HashMap<String, i64>,
    ) -> anyhow::Result<LinkedList<Sentence>> {
        // match self.lang {
        //     Lang::Zh => text::num_to_zh_text(symbols, &self.num_text, last_char_is_punctuation),
        //     Lang::En => text::num_to_en_text(symbols, &self.num_text, last_char_is_punctuation),
        // }
        let mut builder = PhoneBuilder::new();
        let pairs = num::ExprParser::parse(num::Rule::all, &self.num_text)?;
        for pair in pairs {
            match self.lang {
                Lang::Zh => num::zh::parse_all(pair, symbols, &mut builder)?,
                Lang::En => num::en::parse_all(pair, symbols, &mut builder)?,
            }
        }

        Ok(builder.sentence)
    }
}

#[derive(Debug)]
enum Sentence {
    Zh(ZhSentence),
    En(EnSentence),
    Num(NumSentence),
}

#[derive(Debug)]
pub struct PhoneBuilder {
    sentence: LinkedList<Sentence>,
}

fn parse_punctuation(p: &str) -> Option<&'static str> {
    match p {
        "，" | "," => Some(","),
        "。" | "." => Some("."),
        "！" | "!" => Some("!"),
        "？" | "?" => Some("?"),
        "；" | ";" => Some("."),
        "：" | ":" => Some(","),
        "‘" | "’" | "'" => Some("'"),
        "“" | "”" | "\"" => Some("\""),
        "（" | "(" => Some("-"),
        "）" | ")" => Some("-"),
        "【" | "[" => Some("-"),
        "】" | "]" => Some("-"),
        "《" | "<" => Some("-"),
        "》" | ">" => Some("-"),
        "—" => Some("-"),
        "～" | "~" | "…" | "_" => Some("…"),
        "·" => Some(","),
        "、" => Some(","),
        "$" => Some("."),
        "/" => Some(","),
        "\n" => Some("."),
        // " " => Some("\u{7a7a}"),
        " " => Some("…"),
        _ => None,
    }
}

fn is_numeric(p: &str) -> bool {
    p.chars().any(|c| c.is_numeric())
        || p.contains(&['+', '-', '*', '×', '×', '/', '÷', '=', '%'])
        || p.to_lowercase().contains(&[
            'α', 'β', 'γ', 'δ', 'ε', 'ζ', 'η', 'θ', 'ι', 'κ', 'λ', 'μ', 'ν', 'ξ', 'ο', 'π', 'ρ',
            'σ', 'ς', 'τ', 'υ', 'φ', 'χ', 'ψ', 'ω',
        ])
}

impl PhoneBuilder {
    pub fn new() -> Self {
        Self {
            sentence: LinkedList::new(),
        }
    }

    pub fn push_text(
        &mut self,
        jieba: &jieba_rs::Jieba,
        symbols: &HashMap<String, i64>,
        text: &str,
    ) {
        let r = jieba.cut(text, true);
        for t in r {
            if is_numeric(t) {
                self.push_num_word(t);
            } else if let Some(p) = parse_punctuation(t) {
                self.push_punctuation(symbols, p);
            } else if t.is_ascii() {
                self.push_en_word(symbols, t);
            } else {
                self.push_zh_word(symbols, t);
            }
        }
    }

    pub fn push_punctuation(&mut self, symbols: &HashMap<String, i64>, p: &'static str) {
        match self.sentence.back_mut() {
            Some(Sentence::Zh(zh)) => {
                zh.phones.push(p);
                zh.phones_ids.push(get_phone_symbol(symbols, p));
                zh.zh_text.push_str(p);
                zh.word2ph.push(1);
            }
            Some(Sentence::En(en)) => {
                en.phones.push(p);
                en.en_text.push_str(p);
                en.phones_ids.push(get_phone_symbol(symbols, p));
            }
            Some(Sentence::Num(_)) => {
                self.sentence.push_back(Sentence::En(EnSentence {
                    phones_ids: vec![get_phone_symbol(symbols, p)],
                    phones: vec![p],
                    en_text: p.to_string(),
                }));
            }
            _ => {
                log::debug!("skip punctuation: {}", p);
            }
        }
    }

    pub fn push_en_word(&mut self, symbols: &HashMap<String, i64>, word: &str) {
        fn get_word_phone(symbols: &HashMap<String, i64>, word: &str, en: &mut EnSentence) {
            let arpabet = arpabet::load_cmudict();

            if let Some(v) = dict::en_word_dict(word) {
                for ph in v {
                    en.phones.push(ph);
                    en.phones_ids.push(get_phone_symbol(symbols, ph));
                }
            } else if let Some(v) = arpabet.get_polyphone_str(&word) {
                for ph in v {
                    en.phones.push(ph);
                    en.phones_ids.push(get_phone_symbol(symbols, ph));
                }
            } else {
                for c in word.chars() {
                    let mut b = [0; 4];
                    let c = c.encode_utf8(&mut b);

                    if let Some(v) = arpabet.get_polyphone_str(c) {
                        for ph in v {
                            en.phones.push(ph);
                            en.phones_ids.push(get_phone_symbol(symbols, ph));
                        }
                    }
                }
            }
            en.en_text.push_str(&word);
        }

        let word = word.to_ascii_lowercase();
        match self.sentence.back_mut() {
            Some(Sentence::En(en)) => {
                get_word_phone(symbols, &word, en);
            }
            _ => {
                let mut en = EnSentence {
                    phones_ids: vec![],
                    phones: vec![],
                    en_text: String::new(),
                };
                get_word_phone(symbols, &word, &mut en);
                self.sentence.push_back(Sentence::En(en));
            }
        }
    }

    pub fn push_zh_word(&mut self, symbols: &HashMap<String, i64>, word: &str) {
        use pinyin::ToPinyin;

        match self.sentence.back_mut() {
            Some(Sentence::Zh(zh)) => {
                zh.zh_text.push_str(word);
                match dict::zh_word_dict(word) {
                    Some(phones) => {
                        for p in phones {
                            let (y, s) = split_zh_ph(p);
                            zh.phones.push(y);
                            zh.phones_ids.push(get_phone_symbol(symbols, y));
                            zh.phones.push(s);
                            zh.phones_ids.push(get_phone_symbol(symbols, s));
                            zh.word2ph.push(2);
                        }
                    }
                    None => {
                        for c in word.chars() {
                            if let Some(p) = c.to_pinyin() {
                                let (y, s) = split_zh_ph(p.with_tone_num_end());
                                zh.phones.push(y);
                                zh.phones_ids.push(get_phone_symbol(symbols, y));
                                zh.phones.push(s);
                                zh.phones_ids.push(get_phone_symbol(symbols, s));
                                zh.word2ph.push(2);
                            } else {
                                log::debug!("illegal zh char: {}", c);
                            }
                        }
                    }
                }
            }
            _ => {
                let mut zh = ZhSentence {
                    phones_ids: Vec::new(),
                    phones: Vec::new(),
                    word2ph: Vec::new(),
                    zh_text: word.to_string(),
                };
                for c in word.chars() {
                    if let Some(p) = c.to_pinyin() {
                        let (y, s) = split_zh_ph(p.with_tone_num_end());
                        zh.phones.push(y);
                        zh.phones_ids.push(get_phone_symbol(symbols, y));
                        zh.phones.push(s);
                        zh.phones_ids.push(get_phone_symbol(symbols, s));
                        zh.word2ph.push(2);
                    } else {
                        log::debug!("illegal zh char: {}", c);
                    }
                }

                self.sentence.push_back(Sentence::Zh(zh));
            }
        }
    }

    fn push_num_word(&mut self, word: &str) {
        match self.sentence.back_mut() {
            Some(Sentence::Zh(_)) => {
                self.sentence.push_back(Sentence::Num(NumSentence {
                    num_text: word.to_string(),
                    lang: Lang::Zh,
                }));
            }
            Some(Sentence::En(_)) => {
                self.sentence.push_back(Sentence::Num(NumSentence {
                    num_text: word.to_string(),
                    lang: Lang::En,
                }));
            }
            Some(Sentence::Num(num)) => {
                num.num_text.push_str(word);
            }
            _ => {
                self.sentence.push_back(Sentence::Num(NumSentence {
                    num_text: word.to_string(),
                    lang: Lang::Zh,
                }));
            }
        }
    }
}

#[test]
fn test_cut() {
    // 分词
    use jieba_rs::Jieba;

    let target_text =
        "α-200,Good morning.我现在支持了英文和中文这两种语言。English and Chinese.I love Rust very much.我爱Rust！你知不知道1+1=多少？30%的人不知道哦.你可以问问 lisa_GPT-32 有-70+30=? 劫-98G";

    let jieba = Jieba::new();

    let mut phone_builder = PhoneBuilder::new();
    phone_builder.push_text(&jieba, &crate::symbols::SYMBOLS, target_text);

    for s in &phone_builder.sentence {
        match s {
            Sentence::Zh(zh) => {
                println!("###zh###");
                println!("phones: {:?}", zh.phones);
                println!("word2ph: {:?}", zh.word2ph);
                println!("zh_text: {:?}", zh.zh_text);
            }
            Sentence::En(en) => {
                println!("###en###");
                println!("phones: {:?}", en.phones);
                println!("en_text: {:?}", en.en_text);
            }
            Sentence::Num(num) => {
                println!("###num###");
                println!("num_text: {:?}|{:?}", num.num_text, num.lang);
                for s in num.to_phone_sentence(&crate::symbols::SYMBOLS).unwrap() {
                    match s {
                        Sentence::Zh(zh) => {
                            println!("###zh###");
                            println!("phones: {:?}", zh.phones);
                            println!("word2ph: {:?}", zh.word2ph);
                            println!("zh_text: {:?}", zh.zh_text);
                        }
                        Sentence::En(en) => {
                            println!("###en###");
                            println!("phones: {:?}", en.phones);
                            println!("en_text: {:?}", en.en_text);
                        }
                        Sentence::Num(_) => unreachable!(),
                    }
                }
            }
        }
    }
}

#[test]
fn phone_en() {
    let arpabet = arpabet::load_cmudict();

    let r = arpabet
        .get_polyphone_str(&"Chinese".to_ascii_lowercase())
        .unwrap();
    for r in r {
        println!("r: {:?}", r);
    }
}

#[test]
fn test_splite_text() {
    let text = "叹息声一声接着一声，木兰姑娘当门在织布。织机停下来不再作响，只听见姑娘在叹息。问姑娘在思念什么，问姑娘在惦记什么。我也没有在想什么，也没有在惦记什么。昨夜看见征兵的文书，知道君王在大规模征募兵士，那么多卷征兵文书，每卷上都有父亲的名字。父亲没有长大成人的儿子，木兰没有兄长，木兰愿意去买来马鞍和马匹，从此替父亲去出征。到各地集市买骏马，马鞍和鞍下的垫子，马嚼子和缰绳，马鞭。早上辞别父母上路，晚上宿营在黄河边，听不见父母呼唤女儿的声音，但能听到黄河汹涌奔流的声音。";
    // Maximum number of characters in a chunk
    let max_characters = 50;
    // Default implementation uses character count for chunk size
    let splitter = text_splitter::TextSplitter::new(max_characters);

    let chunks = splitter.chunks(text);
    for chunk in chunks {
        let s = chunk.chars().count();
        println!("chunk: {:?} {}", chunk, s);
    }
}

#[test]
fn test_jieba() {
    let jieba = jieba_rs::Jieba::empty();
    let tag = jieba.tag("-20000岁,问问lisa GPT-32是多少?我的ID:123564897.
    Good morning.我现在支持了英文翻译和中文这两种语言。English and Chinese.I love Rust very much.我爱Rust！
    2024年10月 2块 1+2=3 ", true);

    for t in tag {
        println!("{:?}", t);
    }
}
