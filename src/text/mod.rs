use std::{
    borrow::Cow,
    collections::{HashMap, LinkedList},
    fmt::Debug,
    sync::Arc,
};

use pest::Parser;
use tch::{Kind, Tensor};
use tokenizers::Tokenizer;

use crate::GPTSovits;

pub mod g2p_en;
pub mod g2p_jp;
pub mod g2pw;

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
        "a5" => ("AA", "a5"),

        "ai" => ("AA", "ai5"),
        "ai1" => ("AA", "ai1"),
        "ai2" => ("AA", "ai2"),
        "ai3" => ("AA", "ai3"),
        "ai4" => ("AA", "ai4"),
        "ai5" => ("AA", "ai5"),

        "an" => ("AA", "an5"),
        "an1" => ("AA", "an1"),
        "an2" => ("AA", "an2"),
        "an3" => ("AA", "an3"),
        "an4" => ("AA", "an4"),
        "an5" => ("AA", "an5"),

        "ang" => ("AA", "ang5"),
        "ang1" => ("AA", "ang1"),
        "ang2" => ("AA", "ang2"),
        "ang3" => ("AA", "ang3"),
        "ang4" => ("AA", "ang4"),
        "ang5" => ("AA", "ang5"),

        "ao" => ("AA", "ao5"),
        "ao1" => ("AA", "ao1"),
        "ao2" => ("AA", "ao2"),
        "ao3" => ("AA", "ao3"),
        "ao4" => ("AA", "ao4"),
        "ao5" => ("AA", "ao5"),

        "chi" => ("ch", "ir5"),
        "chi1" => ("ch", "ir1"),
        "chi2" => ("ch", "ir2"),
        "chi3" => ("ch", "ir3"),
        "chi4" => ("ch", "ir4"),
        "chi5" => ("ch", "ir5"),

        "ci" => ("c", "i05"),
        "ci1" => ("c", "i01"),
        "ci2" => ("c", "i02"),
        "ci3" => ("c", "i03"),
        "ci4" => ("c", "i04"),
        "ci5" => ("c", "i05"),

        "e" => ("EE", "e5"),
        "e1" => ("EE", "e1"),
        "e2" => ("EE", "e2"),
        "e3" => ("EE", "e3"),
        "e4" => ("EE", "e4"),
        "e5" => ("EE", "e5"),

        "ei" => ("EE", "ei5"),
        "ei1" => ("EE", "ei1"),
        "ei2" => ("EE", "ei2"),
        "ei3" => ("EE", "ei3"),
        "ei4" => ("EE", "ei4"),
        "ei5" => ("EE", "ei5"),

        "en" => ("EE", "en5"),
        "en1" => ("EE", "en1"),
        "en2" => ("EE", "en2"),
        "en3" => ("EE", "en3"),
        "en4" => ("EE", "en4"),
        "en5" => ("EE", "en5"),

        "eng" => ("EE", "eng5"),
        "eng1" => ("EE", "eng1"),
        "eng2" => ("EE", "eng2"),
        "eng3" => ("EE", "eng3"),
        "eng4" => ("EE", "eng4"),
        "eng5" => ("EE", "eng5"),

        "er" => ("EE", "er5"),
        "er1" => ("EE", "er1"),
        "er2" => ("EE", "er2"),
        "er3" => ("EE", "er3"),
        "er4" => ("EE", "er4"),
        "er5" => ("EE", "er5"),

        "ju" => ("j", "v5"),
        "ju1" => ("j", "v1"),
        "ju2" => ("j", "v2"),
        "ju3" => ("j", "v3"),
        "ju4" => ("j", "v4"),
        "ju5" => ("j", "v5"),

        "juan" => ("j", "van5"),
        "juan1" => ("j", "van1"),
        "juan2" => ("j", "van2"),
        "juan3" => ("j", "van3"),
        "juan4" => ("j", "van4"),
        "juan5" => ("j", "van5"),

        "jue" => ("j", "ve5"),
        "jue1" => ("j", "ve1"),
        "jue2" => ("j", "ve2"),
        "jue3" => ("j", "ve3"),
        "jue4" => ("j", "ve4"),
        "jue5" => ("j", "ve5"),

        "jun" => ("j", "vn5"),
        "jun1" => ("j", "vn1"),
        "jun2" => ("j", "vn2"),
        "jun3" => ("j", "vn3"),
        "jun4" => ("j", "vn4"),
        "jun5" => ("j", "vn5"),

        "o" => ("OO", "o5"),
        "o1" => ("OO", "o1"),
        "o2" => ("OO", "o2"),
        "o3" => ("OO", "o3"),
        "o4" => ("OO", "o4"),
        "o5" => ("OO", "o5"),

        "ou" => ("OO", "ou5"),
        "ou1" => ("OO", "ou1"),
        "ou2" => ("OO", "ou2"),
        "ou3" => ("OO", "ou3"),
        "ou4" => ("OO", "ou4"),
        "ou5" => ("OO", "ou5"),

        "qu" => ("q", "v5"),
        "qu1" => ("q", "v1"),
        "qu2" => ("q", "v2"),
        "qu3" => ("q", "v3"),
        "qu4" => ("q", "v4"),
        "qu5" => ("q", "v5"),

        "quan" => ("q", "van5"),
        "quan1" => ("q", "van1"),
        "quan2" => ("q", "van2"),
        "quan3" => ("q", "van3"),
        "quan4" => ("q", "van4"),
        "quan5" => ("q", "van5"),

        "que" => ("q", "ve5"),
        "que1" => ("q", "ve1"),
        "que2" => ("q", "ve2"),
        "que3" => ("q", "ve3"),
        "que4" => ("q", "ve4"),
        "que5" => ("q", "ve5"),

        "qun" => ("q", "vn5"),
        "qun1" => ("q", "vn1"),
        "qun2" => ("q", "vn2"),
        "qun3" => ("q", "vn3"),
        "qun4" => ("q", "vn4"),
        "qun5" => ("q", "vn5"),

        "ri" => ("r", "ir5"),
        "ri1" => ("r", "ir1"),
        "ri2" => ("r", "ir2"),
        "ri3" => ("r", "ir3"),
        "ri4" => ("r", "ir4"),
        "ri5" => ("r", "ir5"),

        "xu" => ("x", "v5"),
        "xu1" => ("x", "v1"),
        "xu2" => ("x", "v2"),
        "xu3" => ("x", "v3"),
        "xu4" => ("x", "v4"),
        "xu5" => ("x", "v5"),

        "xuan" => ("x", "van5"),
        "xuan1" => ("x", "van1"),
        "xuan2" => ("x", "van2"),
        "xuan3" => ("x", "van3"),
        "xuan4" => ("x", "van4"),
        "xuan5" => ("x", "van5"),

        "xue" => ("x", "ve5"),
        "xue1" => ("x", "ve1"),
        "xue2" => ("x", "ve2"),
        "xue3" => ("x", "ve3"),
        "xue4" => ("x", "ve4"),
        "xue5" => ("x", "ve5"),

        "xun" => ("x", "vn5"),
        "xun1" => ("x", "vn1"),
        "xun2" => ("x", "vn2"),
        "xun3" => ("x", "vn3"),
        "xun4" => ("x", "vn4"),
        "xun5" => ("x", "vn5"),

        "yan" => ("y", "En5"),
        "yan1" => ("y", "En1"),
        "yan2" => ("y", "En2"),
        "yan3" => ("y", "En3"),
        "yan4" => ("y", "En4"),
        "yan5" => ("y", "En5"),

        "ye" => ("y", "E5"),
        "ye1" => ("y", "E1"),
        "ye2" => ("y", "E2"),
        "ye3" => ("y", "E3"),
        "ye4" => ("y", "E4"),
        "ye5" => ("y", "E5"),

        "yu" => ("y", "v5"),
        "yu1" => ("y", "v1"),
        "yu2" => ("y", "v2"),
        "yu3" => ("y", "v3"),
        "yu4" => ("y", "v4"),
        "yu5" => ("y", "v5"),

        "yuan" => ("y", "van5"),
        "yuan1" => ("y", "van1"),
        "yuan2" => ("y", "van2"),
        "yuan3" => ("y", "van3"),
        "yuan4" => ("y", "van4"),
        "yuan5" => ("y", "van5"),

        "yue" => ("y", "ve5"),
        "yue1" => ("y", "ve1"),
        "yue2" => ("y", "ve2"),
        "yue3" => ("y", "ve3"),
        "yue4" => ("y", "ve4"),
        "yue5" => ("y", "ve5"),

        "yun" => ("y", "vn5"),
        "yun1" => ("y", "vn1"),
        "yun2" => ("y", "vn2"),
        "yun3" => ("y", "vn3"),
        "yun4" => ("y", "vn4"),
        "yun5" => ("y", "vn5"),

        "zhi" => ("zh", "ir5"),
        "zhi1" => ("zh", "ir1"),
        "zhi2" => ("zh", "ir2"),
        "zhi3" => ("zh", "ir3"),
        "zhi4" => ("zh", "ir4"),
        "zhi5" => ("zh", "ir5"),

        "zi" => ("z", "i05"),
        "zi1" => ("z", "i01"),
        "zi2" => ("z", "i02"),
        "zi3" => ("z", "i03"),
        "zi4" => ("z", "i04"),
        "zi5" => ("z", "i05"),

        "shi" => ("sh", "ir5"),
        "shi1" => ("sh", "ir1"),
        "shi2" => ("sh", "ir2"),
        "shi3" => ("sh", "ir3"),
        "shi4" => ("sh", "ir4"),
        "shi5" => ("sh", "ir5"),

        "si" => ("s", "i05"),
        "si1" => ("s", "i01"),
        "si2" => ("s", "i02"),
        "si3" => ("s", "i03"),
        "si4" => ("s", "i04"),
        "si5" => ("s", "i05"),

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

#[inline]
fn is_punctution(c: char) -> bool {
    matches!(
        c,
        // '。' | '.' | '?' | '？' | '!' | '！' | ',' | '，' | ';' | '；' | '\n'
        '。' | '.' | '?' | '？' | '!' | '！' | ';' | '；' | '\n'
    )
}

pub fn split_text(text: &str, max_chunk_size: usize) -> Vec<&str> {
    let is_en = text.is_ascii();

    let mut r = vec![];
    let mut start_text = text;

    let mut total_count = 0;
    let mut splite_index = 0;

    for s in text.split_inclusive(|c| is_punctution(c)) {
        let count = if is_en {
            s.split(" ").count()
        } else {
            s.chars().count()
        };
        log::trace!(
            "s: {:?}, count: {} total_count: {} splite_index: {}",
            s,
            count,
            total_count,
            splite_index
        );
        if s.chars().count() == 1 {
            splite_index += s.len();
            continue;
        }
        if total_count + count > max_chunk_size {
            let t = start_text.split_at(splite_index);
            let trim_s = t.0.trim();
            if !trim_s.is_empty() {
                r.push(trim_s);
            }
            start_text = t.1;
            total_count = count;
            splite_index = s.len();
        } else if s.ends_with(['\n']) {
            splite_index += s.len();
            let t = start_text.split_at(splite_index);
            let trim_s = t.0.trim();
            if !trim_s.is_empty() {
                r.push(trim_s);
            }
            start_text = t.1;
            total_count = 0;
            splite_index = 0;
        } else {
            total_count += count;
            splite_index += s.len();
        }
    }
    if !start_text.trim().is_empty() {
        r.push(start_text.trim());
    }
    r
}

pub fn get_phone_and_bert(gpts: &GPTSovits, text: &str) -> anyhow::Result<(Tensor, Tensor)> {
    let mut phone_seq = Vec::new();
    let mut bert_seq = Vec::new();

    let mut phone_builder = PhoneBuilder::new(gpts.enable_jp);
    phone_builder.push_text(&gpts.jieba, text);
    if !text.ends_with(['。', '.', '?', '？', '!', '！']) {
        phone_builder.push_punctuation(".");
    }

    fn helper<I: IntoIterator<Item = Sentence>>(
        i: I,
        gpts: &GPTSovits,
        phone_seq: &mut Vec<Tensor>,
        bert_seq: &mut Vec<Tensor>,
    ) -> anyhow::Result<()> {
        for s in i {
            match s {
                Sentence::Zh(mut zh) => {
                    log::trace!("zh text: {:?}", zh.zh_text);
                    log::trace!("zh phones: {:?}", zh.phones);
                    if zh.zh_text.trim().is_empty() {
                        log::trace!("get a empty zh text, skip");
                        continue;
                    }

                    zh.generate_pinyin(gpts);
                    match zh.build_phone_and_bert(gpts) {
                        Ok((t, bert)) => {
                            phone_seq.push(t);
                            bert_seq.push(bert);
                        }
                        Err(e) => {
                            if cfg!(debug_assertions) {
                                return Err(e);
                            } else {
                                log::warn!("get a error, skip: {}", zh.zh_text);
                                log::warn!("zh build_phone_and_bert error: {}", e);
                            }
                        }
                    };
                }
                Sentence::En(mut en) => {
                    log::trace!("en text: {:?}", en.en_text);
                    log::trace!("en phones: {:?}", en.phones);
                    en.generate_phones(gpts);
                    match en.build_phone_and_bert(gpts) {
                        Ok((t, bert)) => {
                            phone_seq.push(t);
                            bert_seq.push(bert);
                        }
                        Err(e) => {
                            if cfg!(debug_assertions) {
                                return Err(e);
                            } else {
                                log::warn!("get a error, skip: {:?}", en.en_text);
                                log::warn!("zh build_phone_and_bert error: {}", e);
                            }
                        }
                    };
                }
                Sentence::Jp(jp) => {
                    log::trace!("jp text: {:?}", jp.text);
                    match jp.build_phone_and_bert(gpts) {
                        Ok((t, bert)) => {
                            phone_seq.push(t);
                            bert_seq.push(bert);
                        }
                        Err(e) => {
                            if cfg!(debug_assertions) {
                                return Err(e);
                            } else {
                                log::warn!("get a error, skip: {:?}", jp.text);
                                log::warn!("jp build_phone_and_bert error: {}", e);
                            }
                        }
                    }
                }
                Sentence::Num(num) => helper(num.to_phone_sentence()?, gpts, phone_seq, bert_seq)?,
            }
        }
        Ok(())
    }

    helper(phone_builder.sentence, gpts, &mut phone_seq, &mut bert_seq)?;

    if phone_seq.is_empty() {
        return Err(anyhow::anyhow!("{text} get phone_seq is empty"));
    }
    if bert_seq.is_empty() {
        return Err(anyhow::anyhow!("{text} get bert_seq is empty"));
    }

    let phone_seq = Tensor::cat(&phone_seq, 1).to(gpts.device);
    let bert_seq = Tensor::cat(&bert_seq, 0).to(gpts.device);

    log::debug!(
        "phone_seq: {:?}",
        Vec::<i64>::try_from(phone_seq.shallow_clone().reshape(vec![-1])).unwrap()
    );
    log::debug!("bert_seq: {:?}", bert_seq);

    Ok((phone_seq, bert_seq))
}

#[derive(Debug, Clone)]
pub enum CNBertModel {
    None,
    TchBert(Arc<tch::CModule>, Arc<Tokenizer>),
}

impl Default for CNBertModel {
    fn default() -> Self {
        Self::None
    }
}

impl CNBertModel {
    pub fn new(bert: Arc<tch::CModule>, tokenizer: Arc<Tokenizer>) -> Self {
        Self::TchBert(bert, tokenizer)
    }

    pub fn tokenizer(&self) -> Option<Arc<Tokenizer>> {
        match self {
            CNBertModel::None => None,
            CNBertModel::TchBert(_, tokenizer) => Some(tokenizer.clone()),
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
            .map(|x| (*x) as i32)
            .collect::<Vec<i32>>();
        let text_ids = Tensor::from_slice(&ids);
        let text_ids = text_ids.unsqueeze(0).to_device(device);

        let mask = encoding
            .get_attention_mask()
            .into_iter()
            .map(|x| (*x) as i32)
            .collect::<Vec<i32>>();
        let text_mask = Tensor::from_slice(&mask);
        let text_mask = text_mask.unsqueeze(0).to_device(device);

        let token_type_ids = encoding
            .get_type_ids()
            .into_iter()
            .map(|x| (*x) as i32)
            .collect::<Vec<i32>>();
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
        let bert = match self {
            CNBertModel::None => {
                let len: i32 = word2ph.iter().sum();
                Tensor::zeros(&[len as i64, 1024], (Kind::Float, device))
            }
            CNBertModel::TchBert(bert, tokenizer) => {
                let (text_ids, text_mask, text_token_type_ids) =
                    Self::encode_text(tokenizer, text, device);
                let text_word2ph = Tensor::from_slice(word2ph).to_device(device);

                bert.forward_ts(&[&text_ids, &text_mask, &text_token_type_ids, &text_word2ph])?
                    .to_device(device)
            }
        };

        Ok(bert)
    }
}

#[derive(Debug)]
struct ZhSentence {
    phones_ids: Vec<i64>,
    phones: Vec<g2pw::G2PWOut>,
    word2ph: Vec<i32>,
    zh_text: String,
}

impl ZhSentence {
    fn generate_pinyin(&mut self, gpts: &GPTSovits) {
        let pinyin = match gpts.g2pw.get_pinyin(&self.zh_text) {
            Ok(pinyin) => pinyin,
            Err(e) => {
                log::warn!("get pinyin error: {}. try simple plan", e);
                gpts.g2pw.simple_get_pinyin(&self.zh_text)
            }
        };

        debug_assert_eq!(pinyin.len(), self.phones.len());

        log::debug!("pinyin: {:?}", pinyin);

        if pinyin.len() != self.phones.len() {
            log::warn!(
                "pinyin len not equal phones len: {} != {}",
                pinyin.len(),
                self.phones.len()
            );
            self.phones = pinyin;
        } else {
            for (i, out) in pinyin.iter().enumerate() {
                let p = &mut self.phones[i];
                if matches!(p, g2pw::G2PWOut::Pinyin("") | g2pw::G2PWOut::RawChar(_)) {
                    *p = *out
                }
            }
        }

        log::debug!("phones: {:?}", self.phones);

        for p in &self.phones {
            match p {
                g2pw::G2PWOut::Pinyin(p) => {
                    let (s, y) = split_zh_ph(&p);
                    self.phones_ids.push(get_phone_symbol(&gpts.symbols, s));
                    self.phones_ids.push(get_phone_symbol(&gpts.symbols, y));
                    self.word2ph.push(2);
                }
                g2pw::G2PWOut::RawChar(c) => {
                    self.phones_ids
                        .push(get_phone_symbol(&gpts.symbols, c.to_string().as_str()));
                    self.word2ph.push(1);
                }
            }
        }
    }

    fn build_phone_and_bert(&self, gpts: &GPTSovits) -> anyhow::Result<(Tensor, Tensor)> {
        let bert = gpts
            .zh_bert
            .get_text_bert(&self.zh_text, &self.word2ph, gpts.device)
            .map_err(|e| anyhow::anyhow!("get_text_bert error: {}", e))?;

        let t = Tensor::from_slice(&self.phones_ids)
            .to_device(gpts.device)
            .unsqueeze(0);

        Ok((t, bert))
    }
}

#[derive(PartialEq, Eq)]
enum EnWord {
    Word(String),
    Punctuation(&'static str),
}

impl Debug for EnWord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EnWord::Word(w) => write!(f, "\"{}\"", w),
            EnWord::Punctuation(p) => write!(f, "\"{}\"", p),
        }
    }
}

#[derive(Debug)]
struct EnSentence {
    phones_ids: Vec<i64>,
    phones: Vec<Cow<'static, str>>,
    en_text: Vec<EnWord>,
}

const SEPARATOR: &'static str = " ";

impl EnSentence {
    fn generate_phones(&mut self, gpts: &GPTSovits) {
        log::trace!("EnSentence text: {:?}", self.en_text);
        let symbols = &gpts.symbols;
        for word in &self.en_text {
            match word {
                EnWord::Word(word) => {
                    if let Some(v) = dict::en_word_dict(word) {
                        for ph in v {
                            self.phones.push(Cow::Borrowed(ph));
                            self.phones_ids.push(get_phone_symbol(symbols, ph));
                        }
                    } else if let (false, Ok(v)) = (
                        word.chars().all(char::is_uppercase),
                        gpts.g2p_en.get_phoneme(&word),
                    ) {
                        for ph in v.split_ascii_whitespace() {
                            self.phones.push(Cow::Owned(ph.to_string()));
                            self.phones_ids.push(get_phone_symbol(symbols, ph));
                        }
                    } else {
                        for c in word.chars() {
                            let mut b = [0; 4];
                            let c = c.encode_utf8(&mut b);

                            if let Ok(v) = gpts.g2p_en.get_phoneme(&c) {
                                for ph in v.split_ascii_whitespace() {
                                    self.phones.push(Cow::Owned(ph.to_string()));
                                    self.phones_ids.push(get_phone_symbol(symbols, ph));
                                }
                            }
                        }
                    }
                }
                EnWord::Punctuation(p) => {
                    self.phones.push(Cow::Borrowed(p));
                    self.phones_ids.push(get_phone_symbol(symbols, p));
                }
            }
        }
        log::trace!("EnSentence phones: {:?}", self.phones);
    }

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

#[derive(Debug)]
struct JpSentence {
    text: String,
}

impl JpSentence {
    fn build_phone_and_bert(&self, gpts: &GPTSovits) -> anyhow::Result<(Tensor, Tensor)> {
        let phones = gpts.g2p_jp.g2p(self.text.as_str());
        log::trace!("JpSentence phones: {:?}", phones);
        let symbols = &gpts.symbols;
        let phone_ids = phones
            .into_iter()
            .map(|v| get_phone_symbol(symbols, v.as_str()))
            .collect::<Vec<_>>();
        let t = Tensor::from_slice(&phone_ids)
            .to_device(gpts.device)
            .unsqueeze(0);
        let bert = Tensor::zeros(&[phone_ids.len() as i64, 1024], (Kind::Float, gpts.device));
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

static NUM_OP: [char; 8] = ['+', '-', '*', '×', '/', '÷', '=', '%'];

impl NumSentence {
    fn need_drop(&self) -> bool {
        let num_text = self.num_text.trim();
        num_text.is_empty() || num_text.chars().all(|c| NUM_OP.contains(&c))
    }

    fn is_link_symbol(&self) -> bool {
        self.num_text == "-"
    }

    fn to_phone_sentence(&self) -> anyhow::Result<LinkedList<Sentence>> {
        // match self.lang {
        //     Lang::Zh => text::num_to_zh_text(symbols, &self.num_text, last_char_is_punctuation),
        //     Lang::En => text::num_to_en_text(symbols, &self.num_text, last_char_is_punctuation),
        // }
        let mut builder = PhoneBuilder::new(false);
        let pairs = num::ExprParser::parse(num::Rule::all, &self.num_text)?;
        for pair in pairs {
            match self.lang {
                Lang::Zh => num::zh::parse_all(pair, &mut builder)?,
                Lang::En => num::en::parse_all(pair, &mut builder)?,
            }
        }

        Ok(builder.sentence)
    }
}

#[derive(Debug)]
enum Sentence {
    Zh(ZhSentence),
    En(EnSentence),
    Jp(JpSentence),
    Num(NumSentence),
}

#[derive(Debug)]
pub struct PhoneBuilder {
    sentence: LinkedList<Sentence>,
    enable_jp: bool,
}

fn parse_punctuation(p: &str) -> Option<&'static str> {
    match p {
        "，" | "," => Some(","),
        "。" | "." => Some("."),
        "！" | "!" => Some("!"),
        "？" | "?" => Some("."),
        "；" | ";" => Some("."),
        "：" | ":" => Some(","),
        "‘" | "’" => Some("'"),
        "'" => Some("'"),
        "“" | "”" | "\"" => Some("-"),
        "（" | "(" => Some("-"),
        "）" | ")" => Some("-"),
        "【" | "[" => Some("-"),
        "】" | "]" => Some("-"),
        "《" | "<" => Some("-"),
        "》" | ">" => Some("-"),
        "—" => Some("-"),
        "～" | "~" | "…" | "_" | "..." => Some("…"),
        "·" => Some(","),
        "、" => Some(","),
        "$" => Some("."),
        "/" => Some(","),
        "\n" => Some("."),
        " " => Some(" "),
        // " " => Some("\u{7a7a}"),
        _ => None,
    }
}

fn is_numeric(p: &str) -> bool {
    p.chars().any(|c| c.is_numeric())
        || p.contains(&NUM_OP)
        || p.to_lowercase().contains(&[
            'α', 'β', 'γ', 'δ', 'ε', 'ζ', 'η', 'θ', 'ι', 'κ', 'λ', 'μ', 'ν', 'ξ', 'ο', 'π', 'ρ',
            'σ', 'ς', 'τ', 'υ', 'φ', 'χ', 'ψ', 'ω',
        ])
}

fn is_jp_kana(p: &str) -> bool {
    p.chars().all(|v| {
        let code = v as u32;
        (0x3040..=0x30FF).contains(&code) || code == 0x3005 // 々
    })
}

impl PhoneBuilder {
    pub fn new(enable_jp: bool) -> Self {
        Self {
            sentence: LinkedList::new(),
            enable_jp,
        }
    }

    pub fn push_text(&mut self, jieba: &jieba_rs::Jieba, text: &str) {
        let r = jieba.cut(text, true);
        log::info!("jieba cut: {:?}", r);
        for t in r {
            if is_numeric(t) {
                self.push_num_word(t);
            } else if let Some(p) = parse_punctuation(t) {
                self.push_punctuation(p);
            } else if g2pw::str_is_chinese(t) {
                self.push_zh_word(t);
            } else if t.is_ascii() {
                self.push_en_word(t);
            } else if self.enable_jp && is_jp_kana(t) {
                self.push_jp_word(t);
            } else {
                log::warn!("skip word: {:?} in {}", t, text);
            }
        }
    }

    pub fn push_punctuation(&mut self, p: &'static str) {
        match self.sentence.back_mut() {
            Some(Sentence::Zh(zh)) => {
                zh.zh_text.push_str(if p == " " { "," } else { p });
                zh.phones
                    .push(g2pw::G2PWOut::RawChar(p.chars().next().unwrap()));
            }
            Some(Sentence::En(en)) => {
                if p == " "
                    && en
                        .en_text
                        .last()
                        .map(|w| match w {
                            EnWord::Word(p) => p == "a",
                            _ => false,
                        })
                        .unwrap_or(false)
                {
                    return;
                }
                en.en_text.push(EnWord::Punctuation(p));
            }
            Some(Sentence::Num(n)) => {
                if n.need_drop() {
                    self.sentence.pop_back();
                }
                self.sentence.push_back(Sentence::En(EnSentence {
                    phones_ids: vec![],
                    phones: vec![],
                    en_text: vec![EnWord::Punctuation(p)],
                }));
            }
            _ => {
                log::debug!("skip punctuation: {}", p);
            }
        }
    }

    pub fn push_en_word(&mut self, word: &str) {
        let word = word.to_string();
        match self.sentence.back_mut() {
            Some(Sentence::En(en)) => {
                if en
                    .en_text
                    .last()
                    .map(|w| w == &EnWord::Punctuation("'") || w == &EnWord::Punctuation("-"))
                    .unwrap_or(false)
                {
                    let p = en.en_text.pop().unwrap();
                    en.en_text.last_mut().map(|w| {
                        if let EnWord::Word(w) = w {
                            if let EnWord::Punctuation(p) = p {
                                w.push_str(p);
                            }
                            w.push_str(&word);
                        }
                    });
                } else if en
                    .en_text
                    .last()
                    .map(|w| match w {
                        EnWord::Word(w) => w == "a",
                        _ => false,
                    })
                    .unwrap_or(false)
                {
                    en.en_text.last_mut().map(|w| {
                        if let EnWord::Word(w) = w {
                            w.push_str(" ");
                            w.push_str(&word);
                        }
                    });
                } else {
                    en.en_text.push(EnWord::Word(word));
                }
            }
            Some(Sentence::Num(n)) if n.need_drop() => {
                let pop = self.sentence.pop_back().unwrap();
                if let Sentence::Num(n) = pop {
                    if n.is_link_symbol() {
                        self.push_punctuation("-");
                    }
                }
                self.push_en_word(&word)
            }
            _ => {
                let en = EnSentence {
                    phones_ids: vec![],
                    phones: vec![],
                    en_text: vec![EnWord::Word(word)],
                };
                self.sentence.push_back(Sentence::En(en));
            }
        }
    }

    pub fn push_zh_word(&mut self, word: &str) {
        fn h(zh: &mut ZhSentence, word: &str) {
            zh.zh_text.push_str(word);
            match dict::zh_word_dict(word) {
                Some(phones) => {
                    for p in phones {
                        zh.phones.push(g2pw::G2PWOut::Pinyin(p));
                    }
                }
                None => {
                    for _ in word.chars() {
                        zh.phones.push(g2pw::G2PWOut::Pinyin(""));
                    }
                }
            }
        }

        match self.sentence.back_mut() {
            Some(Sentence::Zh(zh)) => {
                h(zh, word);
            }
            Some(Sentence::Num(n)) if n.need_drop() => {
                self.sentence.pop_back();
                self.push_zh_word(word);
            }
            _ => {
                let mut zh = ZhSentence {
                    phones_ids: Vec::new(),
                    phones: Vec::new(),
                    word2ph: Vec::new(),
                    zh_text: String::new(),
                };
                h(&mut zh, word);
                self.sentence.push_back(Sentence::Zh(zh));
            }
        };
    }

    pub fn push_jp_word(&mut self, word: &str) {
        match self.sentence.back_mut() {
            Some(Sentence::Jp(jp)) => {
                jp.text.push_str(word);
            }
            _ => {
                let jp = JpSentence {
                    text: word.to_owned(),
                };
                self.sentence.push_back(Sentence::Jp(jp));
            }
        }
    }

    pub fn push_num_word(&mut self, word: &str) {
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

// cargo test --package gpt_sovits_rs --lib -- text::test_cut --exact --show-output
#[test]
fn test_cut() {
    env_logger::init();
    // 分词
    use jieba_rs::Jieba;

    // let target_text =
    //     "about 80% of Americans believed Thompson's killer had either \"a great deal\" or \"a moderate amount\" of responsibility for the murder,";

    let target_text = " Next up, we’re unraveling a tale as old as time – or at least as old as cautionary stories get. We're calling this one “The Wolf in Sheep’s Clothing…And Why Your Mom Is Always Right.” It’s the story of Little Red Riding Hood, but not the sanitized Disney version. We're going deep into the psychology, the symbolism, and honestly, the sheer *audacity* of that wolf. This isn't just a children's story; it's a masterclass in risk assessment, trust – or lack thereof – and why deviating from the established path can lead to becoming someone’s lunch. And joining me today to dissect this…well, frankly terrifying narrative is my friend, Elias. Welcome, Elias!";

    let jieba = Jieba::new();

    let mut phone_builder = PhoneBuilder::new(false);
    phone_builder.push_text(&jieba, target_text);

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
            Sentence::Jp(_) => unreachable!(),
            Sentence::Num(num) => {
                println!("###num###");
                println!("num_text: {:?}|{:?}", num.num_text, num.lang);
                for s in num.to_phone_sentence().unwrap() {
                    match s {
                        Sentence::Zh(zh) => {
                            println!("###num-zh###");
                            println!("phones: {:?}", zh.phones);
                            println!("word2ph: {:?}", zh.word2ph);
                            println!("zh_text: {:?}", zh.zh_text);
                        }
                        Sentence::En(en) => {
                            println!("###num-en###");
                            println!("phones: {:?}", en.phones);
                            println!("en_text: {:?}", en.en_text);
                        }
                        Sentence::Jp(_) | Sentence::Num(_) => unreachable!(),
                    }
                }
            }
        }
    }
}

// cargo test --package gpt_sovits_rs --lib -- text::test_splite_text --exact --show-output
#[test]
fn test_splite_text() {
    env_logger::init();
    let text = "叹息声一声接着一声，木兰姑娘当门在织布。织机停下来不再作响，只听见姑娘在叹息。\n问姑娘在思念什么，问姑娘在惦记什么。我也没有在想什么，也没有在惦记什么。\n昨夜看见征兵的文书，知道君王在大规模征募兵士，那么多卷征兵文书，每卷上都有父亲的名字。父亲没有长大成人的儿子，木兰没有兄长，木兰愿意去买来马鞍和马匹，从此替父亲去出征。到各地集市买骏马，马鞍和鞍下的垫子，马嚼子和缰绳，马鞭。早上辞别父母上路，晚上宿营在黄河边，听不见父母呼唤女儿的声音，但能听到黄河汹涌奔流的声音。";
    println!("text: {:?}", text.is_ascii());

    let chunks = split_text(text, 50);
    for chunk in chunks {
        let s = chunk.chars().count();
        println!("chunk: {:?} {}", chunk, s);
    }
}

// cargo test --package gpt_sovits_rs --lib -- text::test_splite_en_text --exact --show-output
#[test]
fn test_splite_en_text() {
    let text = r#"his story is called "The Farmer and the Snake." Every day, a farmer went to the city to sell his flowers and farm produce and then went home after selling all his things. One day, he left home very early, so early that when he arrived at the city, the gate was still closed. So he lay down to take a nap, when he awoke he found that the storage bin containing his farm produce had become empty except that there was a gold coin inside. Although all the things in the bin had vanished, the gold was much more valuable so he was still very happy. He thought most probably someone had taken his things and left the payment there, and went home happily with the money."#;
    println!("text: {:?}", text.is_ascii());

    let chunks = split_text(text, 50);
    for chunk in chunks {
        let s = chunk.split(" ").count();
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

#[test]
fn test_jp_enabled() {
    use jieba_rs::Jieba;
    let jieba = Jieba::new();

    let mut phone_builder = PhoneBuilder::new(true);
    phone_builder.push_text(&jieba, "你喜欢看アニメ吗");

    let mut iter = phone_builder.sentence.iter();
    assert!(matches!(iter.next().unwrap(), Sentence::Zh(_)));
    assert!(matches!(iter.next().unwrap(), Sentence::Jp(_)));
    assert!(matches!(iter.next().unwrap(), Sentence::Zh(_)));
    assert!(iter.next().is_none());

    let mut phone_builder = PhoneBuilder::new(true);
    phone_builder.push_text(&jieba, "昨天見た映画はとても感動的でした");

    let mut iter = phone_builder.sentence.iter();
    assert!(matches!(iter.next().unwrap(), Sentence::Zh(_)));
    assert!(matches!(iter.next().unwrap(), Sentence::Jp(_)));
    assert!(matches!(iter.next().unwrap(), Sentence::Zh(_)));
    assert!(matches!(iter.next().unwrap(), Sentence::Jp(_)));
    assert!(matches!(iter.next().unwrap(), Sentence::Zh(_)));
    assert!(matches!(iter.next().unwrap(), Sentence::Jp(_)));
    assert!(iter.next().is_none());

    let mut phone_builder = PhoneBuilder::new(true);
    phone_builder.push_text(
        &jieba,
        "我的名字是西野くまです。I am from Tokyo, 日本の首都。今天的天气非常好",
    );
    let mut iter = phone_builder.sentence.iter();
    assert!(matches!(iter.next().unwrap(), Sentence::Zh(_)));
    assert!(matches!(iter.next().unwrap(), Sentence::Jp(_)));
    assert!(matches!(iter.next().unwrap(), Sentence::En(_)));
    assert!(matches!(iter.next().unwrap(), Sentence::Zh(_)));
    assert!(matches!(iter.next().unwrap(), Sentence::Jp(_)));
    assert!(matches!(iter.next().unwrap(), Sentence::Zh(_)));
    assert!(iter.next().is_none());
}

#[test]
fn test_jp_disabled() {
    use jieba_rs::Jieba;
    let jieba = Jieba::new();

    let mut phone_builder = PhoneBuilder::new(false);
    phone_builder.push_text(&jieba, "你喜欢看アニメ吗");

    let mut iter = phone_builder.sentence.iter();
    assert!(matches!(iter.next().unwrap(), Sentence::Zh(_)));
    assert!(iter.next().is_none());

    let mut phone_builder = PhoneBuilder::new(false);
    phone_builder.push_text(&jieba, "昨天見た映画はとても感動的でした");

    let mut iter = phone_builder.sentence.iter();
    assert!(matches!(iter.next().unwrap(), Sentence::Zh(_)));
    assert!(iter.next().is_none());

    let mut phone_builder = PhoneBuilder::new(false);
    phone_builder.push_text(
        &jieba,
        "我的名字是西野くまです。I am from Tokyo, 日本の首都。今天的天气非常好",
    );
    let mut iter = phone_builder.sentence.iter();
    assert!(matches!(iter.next().unwrap(), Sentence::Zh(_)));
    assert!(matches!(iter.next().unwrap(), Sentence::En(_)));
    assert!(matches!(iter.next().unwrap(), Sentence::Zh(_)));
    assert!(iter.next().is_none());
}
