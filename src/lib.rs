use std::{collections::HashMap, str::FromStr, sync::Arc, usize};

use tch::{IValue, Tensor};
use text::{g2p_en::G2PEnConverter, g2p_jp::G2PJpConverter, g2pw::G2PWConverter, CNBertModel};

pub mod symbols;
pub mod text;
pub use tch::Device;
pub mod gsv;

pub struct GPTSovitsConfig {
    pub cn_setting: Option<(String, String)>,
    pub g2p_en_path: String,
    pub ssl_path: String,
    pub enable_jp: bool,
}

impl GPTSovitsConfig {
    pub fn new(ssl_path: String, g2p_en_path: String) -> Self {
        Self {
            cn_setting: None,
            g2p_en_path,
            ssl_path,
            enable_jp: false,
        }
    }

    pub fn with_chinese(mut self, g2pw_path: String, cn_bert_path: String) -> Self {
        self.cn_setting = Some((g2pw_path, cn_bert_path));
        self
    }

    pub fn with_jp(self, enable_jp: bool) -> Self {
        Self { enable_jp, ..self }
    }

    pub fn build(&self, device: Device) -> anyhow::Result<GPTSovits> {
        let (cn_bert, g2pw) = match &self.cn_setting {
            Some((g2pw_path, cn_bert_path)) => {
                let tokenizer = tokenizers::Tokenizer::from_str(text::g2pw::G2PW_TOKENIZER)
                    .map_err(|e| anyhow::anyhow!("load tokenizer error: {}", e))?;
                let tokenizer = Arc::new(tokenizer);

                let mut bert = tch::CModule::load_on_device(&cn_bert_path, device)?;
                bert.set_eval();

                let cn_bert_model = CNBertModel::new(Arc::new(bert), tokenizer.clone());
                let g2pw = G2PWConverter::new_with_device(g2pw_path, tokenizer.clone(), device)?;

                (cn_bert_model, g2pw)
            }
            _ => (CNBertModel::default(), G2PWConverter::empty()),
        };

        let mut ssl = tch::CModule::load_on_device(&self.ssl_path, device).unwrap();
        ssl.set_eval();
        let ssl = Arc::new(ssl);

        Ok(GPTSovits {
            zh_bert: cn_bert,
            g2pw,
            g2p_en: G2PEnConverter::new(&self.g2p_en_path),
            g2p_jp: G2PJpConverter::new(),
            device,
            symbols: symbols::SYMBOLS.clone(),
            ssl,
            jieba: jieba_rs::Jieba::new(),
            speakers: HashMap::new(),

            enable_jp: self.enable_jp,
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Version {
    V2_0,
    // 由于 v3 的改动，V2.1 需要传入 top_k
    V2_1,
    #[deprecated(since = "0.6.0", note = "v3v4 has been abandoned")]
    V3,
    #[deprecated(since = "0.6.0", note = "v3v4 has been abandoned")]
    V4,
    V2Pro,
}

#[derive(Debug)]
pub struct Speaker {
    name: String,
    gpt_sovits_path: String,
    gpt_sovits: Arc<tch::CModule>,
    ref_text: String,
    ssl_content: Tensor,
    ref_audio_32k: Tensor,
    ref_phone_seq: Tensor,
    ref_bert_seq: Tensor,
    version: Version,
    pub top_k: Option<i64>,
    pub sample_steps: Option<i64>,
}

impl Speaker {
    pub fn get_name(&self) -> &str {
        &self.name
    }

    pub fn version(&self) -> Version {
        self.version
    }

    pub fn set_top_k(&mut self, top_k: Option<i64>) {
        self.top_k = top_k;
    }

    pub fn set_sample_steps(&mut self, sample_steps: Option<i64>) {
        self.sample_steps = sample_steps;
    }

    pub fn get_ref_text(&self) -> &str {
        &self.ref_text
    }

    pub fn get_ref_audio_32k(&self) -> &Tensor {
        &self.ref_audio_32k
    }

    pub fn infer_v2(&self, text_phone_seq: &Tensor, bert_seq: &Tensor) -> anyhow::Result<Tensor> {
        let audio = self.gpt_sovits.forward_ts(&[
            &self.ssl_content,
            &self.ref_audio_32k,
            &self.ref_phone_seq,
            &text_phone_seq,
            &self.ref_bert_seq,
            &bert_seq,
        ])?;

        Ok(audio)
    }

    pub fn infer_v2_1(&self, text_phone_seq: &Tensor, bert_seq: &Tensor) -> anyhow::Result<Tensor> {
        let top_k = self.top_k.unwrap_or(15);
        let top_k = Tensor::from_slice(&[top_k]);

        let audio = self.gpt_sovits.forward_ts(&[
            &self.ssl_content,
            &self.ref_audio_32k,
            &self.ref_phone_seq,
            &text_phone_seq,
            &self.ref_bert_seq,
            &bert_seq,
            &top_k,
        ])?;

        Ok(audio)
    }

    pub fn infer_v2_pro(
        &self,
        text_phone_seq: &Tensor,
        bert_seq: &Tensor,
    ) -> anyhow::Result<Tensor> {
        let top_k = self.top_k.unwrap_or(15);
        let top_k = Tensor::from_slice(&[top_k]);

        let audio = self.gpt_sovits.forward_ts(&[
            &self.ssl_content,
            &self.ref_audio_32k,
            &self.ref_phone_seq,
            &text_phone_seq,
            &self.ref_bert_seq,
            &bert_seq.internal_cast_half(false),
            &top_k,
        ])?;

        let audio = audio.to_dtype(tch::Kind::Float, false, false);
        let size = 32000.0 * 0.3;
        let zero = tch::Tensor::zeros([size as i64], (tch::Kind::Float, audio.device()));

        Ok(tch::Tensor::cat(&[audio, zero], 0))
    }

    #[deprecated(since = "0.6.0", note = "v3v4 has been abandoned")]
    pub fn infer_v3(&self, text_phone_seq: &Tensor, bert_seq: &Tensor) -> anyhow::Result<Tensor> {
        let top_k = self.top_k.unwrap_or(15);
        let top_k = Tensor::from_slice(&[top_k]);

        let sample_steps = self.sample_steps.unwrap_or(8);
        let sample_steps = Tensor::from_slice(&[sample_steps]);

        let is_cpu = bert_seq.device() == Device::Cpu;

        let audio = if !is_cpu {
            self.gpt_sovits.forward_ts(&[
                &self.ssl_content.internal_cast_half(false),
                &self.ref_audio_32k,
                &self.ref_phone_seq,
                &text_phone_seq,
                &self.ref_bert_seq.internal_cast_half(false),
                &bert_seq.internal_cast_half(false),
                &top_k,
                &sample_steps,
            ])?
        } else {
            self.gpt_sovits.forward_ts(&[
                &self.ssl_content,
                &self.ref_audio_32k,
                &self.ref_phone_seq,
                &text_phone_seq,
                &self.ref_bert_seq,
                &bert_seq,
                &top_k,
                &sample_steps,
            ])?
        };

        let audio = audio.to_dtype(tch::Kind::Float, false, false);
        let size = 24000.0 * 0.3;
        let zero = tch::Tensor::zeros([size as i64], (tch::Kind::Float, audio.device()));

        Ok(tch::Tensor::cat(&[audio, zero], 0))
    }

    #[deprecated(since = "0.6.0", note = "v3v4 has been abandoned")]
    pub fn infer_v4(&self, text_phone_seq: &Tensor, bert_seq: &Tensor) -> anyhow::Result<Tensor> {
        let top_k = self.top_k.unwrap_or(15);
        let top_k = Tensor::from_slice(&[top_k]);

        let sample_steps = self.sample_steps.unwrap_or(8);
        let sample_steps = Tensor::from_slice(&[sample_steps]);

        let is_cpu = bert_seq.device() == Device::Cpu;

        let audio = if !is_cpu {
            self.gpt_sovits.forward_ts(&[
                &self.ssl_content.internal_cast_half(false),
                &self.ref_audio_32k,
                &self.ref_phone_seq,
                &text_phone_seq,
                &self.ref_bert_seq.internal_cast_half(false),
                &bert_seq.internal_cast_half(false),
                &top_k,
                &sample_steps,
            ])?
        } else {
            self.gpt_sovits.forward_ts(&[
                &self.ssl_content,
                &self.ref_audio_32k,
                &self.ref_phone_seq,
                &text_phone_seq,
                &self.ref_bert_seq,
                &bert_seq,
                &top_k,
                &sample_steps,
            ])?
        };

        let audio = audio.to_dtype(tch::Kind::Float, false, false);
        let size = 48000.0 * 0.3;
        let zero = tch::Tensor::zeros([size as i64], (tch::Kind::Float, audio.device()));

        Ok(tch::Tensor::cat(&[audio, zero], 0))
    }

    pub fn infer(&self, text_phone_seq: &Tensor, bert_seq: &Tensor) -> anyhow::Result<Tensor> {
        match self.version {
            Version::V2_0 => self.infer_v2(text_phone_seq, bert_seq),
            Version::V2_1 => self.infer_v2_1(text_phone_seq, bert_seq),
            Version::V3 => self.infer_v3(text_phone_seq, bert_seq),
            Version::V4 => self.infer_v4(text_phone_seq, bert_seq),
            Version::V2Pro => self.infer_v2_pro(text_phone_seq, bert_seq),
        }
    }
}

pub struct GPTSovits {
    zh_bert: CNBertModel,
    g2pw: G2PWConverter,
    g2p_en: G2PEnConverter,
    g2p_jp: G2PJpConverter,
    pub device: tch::Device,
    symbols: HashMap<String, i64>,
    pub ssl: Arc<tch::CModule>,

    speakers: HashMap<String, Speaker>,

    jieba: jieba_rs::Jieba,

    enable_jp: bool,
}

impl GPTSovits {
    pub fn new(
        zh_bert: CNBertModel,
        g2pw: G2PWConverter,
        g2p_en: G2PEnConverter,
        g2p_jp: G2PJpConverter,
        device: tch::Device,
        symbols: HashMap<String, i64>,
        ssl: Arc<tch::CModule>,
        jieba: jieba_rs::Jieba,
        enable_jp: bool,
    ) -> Self {
        Self {
            zh_bert,
            g2pw,
            g2p_en,
            g2p_jp,
            device,
            symbols,
            speakers: HashMap::new(),
            ssl,
            jieba,
            enable_jp,
        }
    }

    fn find_gpt_sovits_by_path(&self, path: &str) -> Option<Arc<tch::CModule>> {
        for speaker in self.speakers.values() {
            if speaker.gpt_sovits_path == path {
                return Some(speaker.gpt_sovits.clone());
            }
        }
        None
    }

    fn find_gpt_sovits_by_path_or_load(&self, path: &str) -> anyhow::Result<Arc<tch::CModule>> {
        if let Some(gpt_sovits) = self.find_gpt_sovits_by_path(path) {
            Ok(gpt_sovits)
        } else {
            let mut gpt_sovits;
            if self.device == Device::Mps {
                gpt_sovits = tch::CModule::load(path)?;
                gpt_sovits.to(self.device, tch::Kind::Half, false)
            } else {
                gpt_sovits = tch::CModule::load_on_device(path, self.device)?;
            }
            gpt_sovits.set_eval();
            Ok(Arc::new(gpt_sovits))
        }
    }

    pub fn create_speaker(
        &mut self,
        name: &str,
        gpt_sovits_path: &str,
        ref_audio_samples: &[f32],
        ref_audio_sr: usize,
        ref_text: &str,
    ) -> anyhow::Result<()> {
        tch::no_grad(|| {
            let gpt_sovits = self.find_gpt_sovits_by_path_or_load(gpt_sovits_path)?;

            // 避免句首吞字
            let ref_text = if !ref_text.ends_with(['。', '.']) {
                ref_text.to_string() + "."
            } else {
                ref_text.to_string()
            };

            let ref_audio = Tensor::from_slice(ref_audio_samples)
                .to_device(self.device)
                .unsqueeze(0);

            let ref_audio_16k = self.resample(&ref_audio, ref_audio_sr, 16000)?;
            let ref_audio_32k = self.resample(&ref_audio, ref_audio_sr, 32000)?;

            let ssl_content = self.ssl.forward_ts(&[&ref_audio_16k])?;

            let (ref_phone_seq, ref_bert_seq) = text::get_phone_and_bert(self, &ref_text)?;

            let speaker = Speaker {
                name: name.to_string(),
                gpt_sovits_path: gpt_sovits_path.to_string(),
                gpt_sovits,
                ref_text,
                ssl_content,
                ref_audio_32k,
                ref_phone_seq,
                ref_bert_seq,
                version: Version::V2_0,
                top_k: None,
                sample_steps: None,
            };

            self.speakers.insert(name.to_string(), speaker);
            Ok(())
        })
    }

    pub fn create_speaker_v2_1(
        &mut self,
        name: &str,
        gpt_sovits_path: &str,
        ref_audio_samples: &[f32],
        ref_audio_sr: usize,
        ref_text: &str,
        top_k: Option<i64>,
    ) -> anyhow::Result<()> {
        tch::no_grad(|| {
            let gpt_sovits = self.find_gpt_sovits_by_path_or_load(gpt_sovits_path)?;

            // 避免句首吞字
            let ref_text = if !ref_text.ends_with(['。', '.']) {
                ref_text.to_string() + "."
            } else {
                ref_text.to_string()
            };

            let ref_audio = Tensor::from_slice(ref_audio_samples)
                .to_device(self.device)
                .unsqueeze(0);

            let ref_audio_16k = self.resample(&ref_audio, ref_audio_sr, 16000)?;
            let ref_audio_32k = self.resample(&ref_audio, ref_audio_sr, 32000)?;

            let ssl_content = self.ssl.forward_ts(&[&ref_audio_16k])?;

            let (ref_phone_seq, ref_bert_seq) = text::get_phone_and_bert(self, &ref_text)?;

            let speaker = Speaker {
                name: name.to_string(),
                gpt_sovits_path: gpt_sovits_path.to_string(),
                gpt_sovits,
                ref_text,
                ssl_content,
                ref_audio_32k,
                ref_phone_seq,
                ref_bert_seq,
                version: Version::V2_1,
                top_k,
                sample_steps: None,
            };

            self.speakers.insert(name.to_string(), speaker);
            Ok(())
        })
    }

    pub fn create_speaker_v2_pro(
        &mut self,
        name: &str,
        gpt_sovits_path: &str,
        ref_audio_samples: &[f32],
        ref_audio_sr: usize,
        ref_text: &str,
        top_k: Option<i64>,
    ) -> anyhow::Result<()> {
        tch::no_grad(|| {
            let gpt_sovits = self.find_gpt_sovits_by_path_or_load(gpt_sovits_path)?;

            // 避免句首吞字
            let ref_text = if !ref_text.ends_with(['。', '.']) {
                ref_text.to_string() + "."
            } else {
                ref_text.to_string()
            };

            let ref_audio = Tensor::from_slice(ref_audio_samples)
                .to_device(self.device)
                .unsqueeze(0);

            let ref_audio_16k = self.resample(&ref_audio, ref_audio_sr, 16000)?;
            let ref_audio_32k = self
                .resample(&ref_audio, ref_audio_sr, 32000)?
                .internal_cast_half(false);

            let ssl_content = self
                .ssl
                .forward_ts(&[&ref_audio_16k])?
                .internal_cast_half(false);

            let (ref_phone_seq, ref_bert_seq) = text::get_phone_and_bert(self, &ref_text)?;
            let ref_bert_seq = ref_bert_seq.internal_cast_half(false);

            let speaker = Speaker {
                name: name.to_string(),
                gpt_sovits_path: gpt_sovits_path.to_string(),
                gpt_sovits,
                ref_text,
                ssl_content,
                ref_audio_32k,
                ref_phone_seq,
                ref_bert_seq,
                version: Version::V2Pro,
                top_k,
                sample_steps: None,
            };

            self.speakers.insert(name.to_string(), speaker);
            Ok(())
        })
    }

    #[deprecated(since = "0.6.0", note = "v3v4 has been abandoned")]
    pub fn create_speaker_v3(
        &mut self,
        name: &str,
        gpt_sovits_path: &str,
        ref_audio_samples: &[f32],
        ref_audio_sr: usize,
        ref_text: &str,
        top_k: Option<i64>,
        sample_steps: Option<i64>,
    ) -> anyhow::Result<()> {
        tch::no_grad(|| {
            let gpt_sovits = self.find_gpt_sovits_by_path_or_load(gpt_sovits_path)?;

            // 避免句首吞字
            let ref_text = if !ref_text.ends_with(['。', '.']) {
                ref_text.to_string() + "."
            } else {
                ref_text.to_string()
            };

            let mut ref_audio = Tensor::from_slice(ref_audio_samples)
                .to_device(self.device)
                .unsqueeze(0);
            if self.device == Device::Mps {
                ref_audio = ref_audio.totype(tch::Kind::Half);
            }

            let ref_audio_16k = self.resample(&ref_audio, ref_audio_sr, 16000)?;
            let ref_audio_32k = self.resample(&ref_audio, ref_audio_sr, 32000)?;

            let ssl_content = self.ssl.forward_ts(&[&ref_audio_16k])?;

            let (ref_phone_seq, ref_bert_seq) = text::get_phone_and_bert(self, &ref_text)?;

            let speaker = Speaker {
                name: name.to_string(),
                gpt_sovits_path: gpt_sovits_path.to_string(),
                gpt_sovits,
                ref_text,
                ssl_content,
                ref_audio_32k,
                ref_phone_seq,
                ref_bert_seq,
                version: Version::V3,
                top_k,
                sample_steps,
            };

            self.speakers.insert(name.to_string(), speaker);
            Ok(())
        })
    }

    #[deprecated(since = "0.6.0", note = "v3v4 has been abandoned")]
    pub fn create_speaker_v4(
        &mut self,
        name: &str,
        gpt_sovits_path: &str,
        ref_audio_samples: &[f32],
        ref_audio_sr: usize,
        ref_text: &str,
        top_k: Option<i64>,
        sample_steps: Option<i64>,
    ) -> anyhow::Result<()> {
        // v3 和 v4 只有输出的采样率不同，输入相同
        self.create_speaker_v3(
            name,
            gpt_sovits_path,
            ref_audio_samples,
            ref_audio_sr,
            ref_text,
            top_k,
            sample_steps,
        )?;
        self.speakers
            .get_mut(name)
            .ok_or_else(|| anyhow::anyhow!("speaker not found"))?
            .version = Version::V4;
        Ok(())
    }

    pub fn resample(&self, audio: &Tensor, sr: usize, target_sr: usize) -> anyhow::Result<Tensor> {
        tch::no_grad(|| {
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
        })
    }

    /// Only V2.1 and V3 support top_k
    pub fn set_top_k(&mut self, speaker: &str, top_k: Option<i64>) -> anyhow::Result<()> {
        let speaker = self
            .speakers
            .get_mut(speaker)
            .ok_or_else(|| anyhow::anyhow!("speaker not found"))?;
        speaker.set_top_k(top_k);
        Ok(())
    }

    /// Only V3 V4 support sample_steps
    pub fn set_sample_steps(
        &mut self,
        speaker: &str,
        sample_steps: Option<i64>,
    ) -> anyhow::Result<()> {
        let speaker = self
            .speakers
            .get_mut(speaker)
            .ok_or_else(|| anyhow::anyhow!("speaker not found"))?;
        speaker.set_sample_steps(sample_steps);
        Ok(())
    }

    pub fn get_version(&self, speaker: &str) -> anyhow::Result<Version> {
        let speaker = self
            .speakers
            .get(speaker)
            .ok_or_else(|| anyhow::anyhow!("speaker not found"))?;
        Ok(speaker.version)
    }

    /// generate a audio tensor from text
    pub fn infer(&self, speaker: &str, target_text: &str) -> anyhow::Result<Tensor> {
        log::debug!("start infer");
        tch::no_grad(|| {
            let speaker = self
                .speakers
                .get(speaker)
                .ok_or_else(|| anyhow::anyhow!("speaker not found"))?;

            let (phone_seq, bert_seq) = text::get_phone_and_bert(self, target_text)?;
            let audio = speaker.infer(&phone_seq, &bert_seq)?;
            Ok(audio)
        })
    }

    pub fn segment_infer(
        &self,
        speaker: &str,
        target_text: &str,
        split_chunk_size: usize,
    ) -> anyhow::Result<Tensor> {
        tch::no_grad(|| {
            let mut audios = vec![];
            let split_chunk_size = if split_chunk_size == 0 {
                50
            } else {
                split_chunk_size
            };
            let chunks = crate::text::split_text(target_text, split_chunk_size);
            log::debug!("segment_infer split_text result: {:#?}", chunks);
            for target_text in chunks {
                match self.infer(speaker, target_text) {
                    Ok(audio) => {
                        audios.push(audio);
                    }
                    Err(e) => {
                        log::warn!("SKIP segment_infer chunk:{target_text} error: {:?}", e);
                    }
                }
            }
            if !audios.is_empty() {
                Ok(Tensor::cat(&audios, 0))
            } else {
                Err(anyhow::anyhow!("no audio generated"))
            }
        })
    }
}
