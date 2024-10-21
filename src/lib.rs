use std::{collections::HashMap, usize};

use anyhow::Ok;
use tch::{IValue, Tensor};
use text::CNBertModel;

pub mod symbols;
pub mod text;

pub use tch::Device;
use text_splitter::{Characters, TextSplitter};

pub struct GPTSovitsConfig {
    pub cn_bert_path: Option<String>,
    pub tokenizer_path: Option<String>,
    pub gpt_sovits_path: String,
    pub ssl_path: String,
}

impl GPTSovitsConfig {
    pub fn new(gpt_sovits_path: String, ssl_path: String) -> Self {
        Self {
            cn_bert_path: None,
            tokenizer_path: None,
            gpt_sovits_path,
            ssl_path,
        }
    }

    pub fn with_cn_bert_path(mut self, cn_bert_path: String, tokenizer_path: String) -> Self {
        self.cn_bert_path = Some(cn_bert_path);
        self.tokenizer_path = Some(tokenizer_path);
        self
    }

    pub fn build(&self, device: Device) -> anyhow::Result<GPTSovits> {
        let mut gpt_sovits = tch::CModule::load_on_device(&self.gpt_sovits_path, device)?;
        gpt_sovits.set_eval();

        let cn_bert = match (&self.cn_bert_path, &self.tokenizer_path) {
            (Some(cn_bert_path), Some(tokenizer_path)) => {
                let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
                    .map_err(|e| anyhow::anyhow!("load tokenizer error: {}", e))?;

                let mut bert = tch::CModule::load_on_device(&cn_bert_path, device)?;
                bert.set_eval();
                CNBertModel::new(bert, tokenizer)
            }
            _ => CNBertModel::default(),
        };

        let mut ssl = tch::CModule::load_on_device(&self.ssl_path, device).unwrap();
        ssl.set_eval();

        Ok(GPTSovits {
            zh_bert: cn_bert,
            device,
            symbols: symbols::SYMBOLS.clone(),
            gpt_sovits,
            ssl,
            jieba: jieba_rs::Jieba::new(),
            text_splitter: TextSplitter::new(50),
        })
    }
}

pub struct GPTSovits {
    zh_bert: CNBertModel,
    device: tch::Device,
    symbols: HashMap<String, i64>,
    gpt_sovits: tch::CModule,
    ssl: tch::CModule,

    jieba: jieba_rs::Jieba,
    text_splitter: TextSplitter<Characters>,
}

impl GPTSovits {
    pub fn new(
        zh_bert: CNBertModel,
        device: tch::Device,
        symbols: HashMap<String, i64>,
        gpt_sovits: tch::CModule,
        ssl: tch::CModule,
        jieba: jieba_rs::Jieba,
        text_splitter: TextSplitter<Characters>,
    ) -> Self {
        Self {
            zh_bert,
            device,
            symbols,
            gpt_sovits,
            ssl,
            jieba,
            text_splitter,
        }
    }

    pub fn resample(&self, audio: &Tensor, sr: usize, target_sr: usize) -> anyhow::Result<Tensor> {
        let resample = self.ssl.method_is(
            "resample",
            &[
                &IValue::Tensor(audio.shallow_clone()),
                &IValue::Int(sr as i64),
                &IValue::Int(target_sr as i64),
            ],
        )?;
        match resample {
            IValue::Tensor(resample) => Ok(resample),
            _ => unreachable!(),
        }
    }

    /// generate a audio tensor from text
    /// sr: 32000
    pub fn infer(
        &self,
        ref_audio_samples: &[f32],
        ref_audio_sr: usize,
        ref_text: &str,
        target_text: &str,
    ) -> anyhow::Result<Tensor> {
        log::debug!("start infer");
        tch::no_grad(|| {
            let ref_audio = Tensor::from_slice(ref_audio_samples)
                .to_device(self.device)
                .unsqueeze(0);

            let ref_audio_16k = self.resample(&ref_audio, ref_audio_sr, 16000)?;
            let ref_audio_sr = self.resample(&ref_audio, ref_audio_sr, 32000)?;

            let ssl_content = self.ssl.forward_ts(&[&ref_audio_16k])?;

            let (ref_phone_seq, ref_bert_seq) = text::get_phone_and_bert(self, ref_text)?;
            let (phone_seq, bert_seq) = text::get_phone_and_bert(self, target_text)?;

            let audio = self.gpt_sovits.forward_ts(&[
                &ssl_content,
                &ref_audio_sr,
                &ref_phone_seq,
                &phone_seq,
                &ref_bert_seq,
                &bert_seq,
            ])?;

            Ok(audio)
        })
    }
}
