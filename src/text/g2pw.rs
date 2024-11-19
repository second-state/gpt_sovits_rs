use std::{collections::HashMap, fmt::Debug, sync::Arc};

static MONO_CHARS_DIST_STR: &str = include_str!("../../resource/g2pw/dict_mono_chars.json");
static POLY_CHARS_DIST_STR: &str = include_str!("../../resource/g2pw/dict_poly_chars.json");
static LABELS: &str = include_str!("../../resource/g2pw/dict_poly_index_list.json");

fn load_mono_chars() -> HashMap<char, MonoChar> {
    if let Ok(dir) = std::env::var("G2PW_DIST_DIR") {
        let s = std::fs::read_to_string(format!("{}/dict_mono_chars.json", dir))
            .expect("dict_mono_chars.json not found");
        serde_json::from_str(&s).expect("dict_mono_chars.json parse error")
    } else {
        serde_json::from_str(MONO_CHARS_DIST_STR).unwrap()
    }
}

fn load_poly_chars() -> HashMap<char, PolyChar> {
    if let Ok(dir) = std::env::var("G2PW_DIST_DIR") {
        let s = std::fs::read_to_string(format!("{}/dict_poly_chars.json", dir))
            .expect("dict_poly_chars.json not found");
        serde_json::from_str(&s).expect("dict_poly_chars.json parse error")
    } else {
        serde_json::from_str(POLY_CHARS_DIST_STR).unwrap()
    }
}

lazy_static::lazy_static! {
    static ref DICT_MONO_CHARS: HashMap<char, MonoChar> =load_mono_chars();
    static ref DICT_POLY_CHARS: HashMap<char, PolyChar> = load_poly_chars();
    static ref POLY_LABLES: Vec<String> = serde_json::from_str(LABELS).unwrap();
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PolyChar {
    index: usize,
    phones: Vec<String>,
    phones_index: Vec<usize>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MonoChar {
    phone: String,
}

#[test]
fn build_static_resource() {
    use std::collections::{BTreeMap, HashMap};
    let bopomofo_to_pinyin_dict =
        std::fs::read_to_string("resource/g2pw/bopomofo_to_pinyin_wo_tune_dict.json").unwrap();

    let dict = serde_json::from_str::<HashMap<String, String>>(&bopomofo_to_pinyin_dict).unwrap();

    let chars = std::fs::read_to_string("resource/g2pw/CHARS.txt").unwrap();
    let labels = std::fs::read_to_string("resource/g2pw/LABELS.txt").unwrap();

    let mut labels_index_map = BTreeMap::new();
    let mut lables_list = vec![];

    let mut mono_btree = BTreeMap::new();
    let mut poly_btree = BTreeMap::new();

    for (i, ph) in labels.lines().enumerate() {
        let (ph, tone) = ph.split_at(ph.len() - 1);
        let x = if let Some(ph) = dict.get(ph) {
            let ph_with_tone = format!("{}{}", ph, tone);
            lables_list.push(ph_with_tone.clone());
            labels_index_map.insert(ph_with_tone, i)
        } else {
            println!("{} py not found", ph);
            let ph_with_tone = format!("{}{}", ph, tone);
            lables_list.push(ph_with_tone.clone());
            labels_index_map.insert(ph_with_tone, i)
        };
        if x.is_some() {
            println!("{}{}repeat {:?}", ph, tone, x);
        }
    }
    let s = serde_json::to_string_pretty(&labels_index_map).unwrap();
    std::fs::write("resource/g2pw/dict_poly_index_map.json", s).unwrap();
    let s = serde_json::to_string_pretty(&lables_list).unwrap();
    std::fs::write("resource/g2pw/dict_poly_index_list.json", s).unwrap();

    for (i, c) in chars.lines().enumerate() {
        poly_btree.insert(
            c.to_string(),
            PolyChar {
                index: i,
                phones: vec![],
                phones_index: vec![],
            },
        );
    }

    let mono = std::fs::read_to_string("resource/g2pw/MONOPHONIC_CHARS.txt").unwrap();

    for (_, c) in mono.lines().enumerate() {
        let (c, ph) = c.split_once("\t").unwrap();
        let len = ph.len();
        let (ph, tone) = ph.split_at(len - 1);
        mono_btree.insert(
            c.to_string(),
            MonoChar {
                phone: format!("{}{}", dict[ph], tone),
            },
        );
    }

    let poly = std::fs::read_to_string("resource/g2pw/POLYPHONIC_CHARS.txt").unwrap();

    for c in poly.lines() {
        let (c, ph) = c.split_once("\t").unwrap();

        let (ph, tone) = ph.split_at(ph.len() - 1);

        let item = poly_btree.get_mut(c).unwrap();
        if let Some(ph) = dict.get(ph) {
            let phone_with_tone = format!("{}{}", ph, tone);
            if item.phones.contains(&phone_with_tone) {
                println!("poly_btree {c}:{} repeat", phone_with_tone);
                continue;
            }
            item.phones_index.push(labels_index_map[&phone_with_tone]);
            item.phones.push(phone_with_tone);
        } else {
            println!("skip {c}:{} , py not found", ph);
        }
    }

    // 繁体简体转换
    {
        let s = std::fs::read_to_string("resource/g2pw/bert-base-chinese_s2t_dict.txt").unwrap();
        for l in s.lines() {
            let (s, t) = l.split_once("\t").unwrap();
            if let Some(obj) = mono_btree.get(t) {
                mono_btree.insert(s.to_string(), obj.clone());
            } else if let Some(obj) = poly_btree.get(t) {
                poly_btree.insert(s.to_string(), obj.clone());
            }
            if mono_btree.contains_key(s) && poly_btree.contains_key(s) {
                // 合并繁简多音字
                let m = mono_btree.remove(s).unwrap();
                let p = poly_btree.get_mut(s).unwrap();
                if !p.phones.contains(&m.phone) {
                    p.phones_index.push(labels_index_map[&m.phone]);
                    p.phones.push(m.phone);
                }
            }
        }
    }

    let s = serde_json::to_string_pretty(&mono_btree).unwrap();
    std::fs::write("resource/g2pw/dict_mono_chars.json", s).unwrap();

    let s = serde_json::to_string_pretty(&poly_btree).unwrap();
    std::fs::write("resource/g2pw/dict_poly_chars.json", s).unwrap();
}

#[derive(Clone, Copy)]
pub enum G2PWOut {
    Pinyin(&'static str),
    RawChar(char),
}

impl Debug for G2PWOut {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Pinyin(s) => write!(f, "\"{}\"", s),
            Self::RawChar(s) => write!(f, "\"{}\"", s),
        }
    }
}

#[derive(Debug, Clone)]
pub struct G2PWConverter {
    model: Option<Arc<tch::CModule>>,
    tokenizers: Option<Arc<tokenizers::Tokenizer>>,
    device: crate::Device,
}

pub fn str_is_chinese(s: &str) -> bool {
    let mut r = true;
    for c in s.chars() {
        if !DICT_MONO_CHARS.contains_key(&c) && !DICT_POLY_CHARS.contains_key(&c) {
            r &= false;
        }
    }
    r
}

impl G2PWConverter {
    pub fn empty() -> Self {
        Self {
            model: None,
            tokenizers: None,
            device: crate::Device::Cpu,
        }
    }

    pub fn new(model_path: &str, tokenizer: Arc<tokenizers::Tokenizer>) -> anyhow::Result<Self> {
        let device = crate::Device::Cpu;
        Self::new_with_device(model_path, tokenizer, device)
    }

    pub fn new_with_device(
        model_path: &str,
        tokenizer: Arc<tokenizers::Tokenizer>,
        device: crate::Device,
    ) -> anyhow::Result<Self> {
        let mut model = tch::CModule::load_on_device(model_path, device)?;
        model.set_eval();
        Ok(Self {
            model: Some(Arc::new(model)),
            tokenizers: Some(tokenizer),
            device,
        })
    }

    pub fn get_pinyin<'s>(&self, text: &'s str) -> anyhow::Result<Vec<G2PWOut>> {
        if self.model.is_some() && self.tokenizers.is_some() {
            self.ml_get_pinyin(text)
        } else {
            Ok(self.simple_get_pinyin(text))
        }
    }

    pub fn simple_get_pinyin(&self, text: &str) -> Vec<G2PWOut> {
        let mut pre_data = vec![];
        for (_, c) in text.chars().enumerate() {
            if let Some(mono) = DICT_MONO_CHARS.get(&c) {
                pre_data.push(G2PWOut::Pinyin(&mono.phone));
            } else if let Some(poly) = DICT_POLY_CHARS.get(&c) {
                pre_data.push(G2PWOut::Pinyin(&poly.phones[0]));
            } else {
                pre_data.push(G2PWOut::RawChar(c));
            }
        }
        pre_data
    }

    fn ml_get_pinyin<'s>(&self, text: &'s str) -> anyhow::Result<Vec<G2PWOut>> {
        let c = self
            .tokenizers
            .as_ref()
            .unwrap()
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("encode error: {}", e))?;
        let input_ids = c.get_ids().iter().map(|x| *x as i64).collect::<Vec<i64>>();
        let token_type_ids = vec![0i64; input_ids.len()];
        let attention_mask = vec![1i64; input_ids.len()];

        let mut phoneme_masks = vec![];
        let mut pre_data = vec![];
        let mut query_id = vec![];
        let mut chars_id = vec![];

        for (i, c) in text.chars().enumerate() {
            if let Some(mono) = DICT_MONO_CHARS.get(&c) {
                pre_data.push(G2PWOut::Pinyin(&mono.phone));
            } else if let Some(poly) = DICT_POLY_CHARS.get(&c) {
                pre_data.push(G2PWOut::Pinyin(""));
                // 这个位置是 tokens 的位置，它的前后添加了 '[CLS]' 和 '[SEP]' 两个特殊字符
                query_id.push(i + 1);
                chars_id.push(poly.index);
                let mut phoneme_mask = vec![0f32; POLY_LABLES.len()];
                for i in &poly.phones_index {
                    phoneme_mask[*i] = 1.0;
                }
                phoneme_masks.push(phoneme_mask);
            } else {
                pre_data.push(G2PWOut::RawChar(c));
            }
        }

        let input_ids = tch::Tensor::from_slice(&input_ids)
            .unsqueeze(0)
            .to_device(self.device);
        let token_type_ids = tch::Tensor::from_slice(&token_type_ids)
            .unsqueeze(0)
            .to_device(self.device);
        let attention_mask = tch::Tensor::from_slice(&attention_mask)
            .unsqueeze(0)
            .to_device(self.device);

        for ((position_id, phoneme_mask), char_id) in query_id
            .iter()
            .zip(phoneme_masks.iter())
            .zip(chars_id.iter())
        {
            let phoneme_mask = tch::Tensor::from_slice(phoneme_mask)
                .unsqueeze(0)
                .to_device(self.device);
            let position_id_t =
                tch::Tensor::from_slice(&[*position_id as i64]).to_device(self.device);
            let char_id = tch::Tensor::from_slice(&[*char_id as i64]).to_device(self.device);

            let probs = tch::no_grad(|| {
                self.model.as_ref().unwrap().forward_ts(&[
                    &input_ids,
                    &token_type_ids,
                    &attention_mask,
                    &phoneme_mask,
                    &char_id,
                    &position_id_t,
                ])
            })?;

            let i = probs.argmax(-1, false).int64_value(&[]);

            pre_data[*position_id - 1] = G2PWOut::Pinyin(&POLY_LABLES[i as usize]);
        }

        Ok(pre_data)
    }
}
