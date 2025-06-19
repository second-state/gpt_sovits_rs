use std::sync::Arc;

use anyhow::Ok;
use tch::{Device, IValue, Tensor};

pub struct T2S {
    pub file_path: String,
    pub model: tch::CModule,
}

unsafe impl Send for T2S {}
unsafe impl Sync for T2S {}

impl T2S {
    pub const EOS: i64 = 1024;
    pub fn new(file_path: &str, device: Device) -> anyhow::Result<Self> {
        let mut model = tch::CModule::load_on_device(file_path, device).map_err(|e| {
            log::debug!("Failed to load T2S model from {}: {}", file_path, e);
            anyhow::anyhow!("Failed to load T2S model from {}", file_path)
        })?;
        model.set_eval();
        let file_path = file_path.to_string();
        Ok(T2S { file_path, model })
    }

    /// return: (y_len, y, xy_pos, k_cache, v_cache)
    pub fn pre_infer(
        &self,
        prompts: Tensor,
        ref_seq: Tensor,
        text_seq: Tensor,
        ref_bert: Tensor,
        text_bert: Tensor,
        top_k: i64,
    ) -> anyhow::Result<(i64, Tensor, Tensor, Vec<Tensor>, Vec<Tensor>)> {
        let result = self
            .model
            .method_is(
                "pre_infer",
                &[
                    &IValue::Tensor(prompts),
                    &IValue::Tensor(ref_seq),
                    &IValue::Tensor(text_seq),
                    &IValue::Tensor(ref_bert),
                    &IValue::Tensor(text_bert),
                    &IValue::Int(top_k),
                ],
            )
            .map_err(|e| {
                log::debug!("Failed to pre-infer T2S model: {}", e);
                anyhow::anyhow!("Failed to pre-infer T2S model")
            })?;
        if let IValue::Tuple(mut result) = result {
            let v_cache = result.pop().ok_or(anyhow::anyhow!(
                "Take v_cache from T2S model pre-inference result failed"
            ))?;
            let k_cache = result.pop().ok_or(anyhow::anyhow!(
                "Take k_cache from T2S model pre-inference result failed"
            ))?;
            let xy_pos = result.pop().ok_or(anyhow::anyhow!(
                "Take xy_pos from T2S model pre-inference result failed"
            ))?;
            let y = result.pop().ok_or(anyhow::anyhow!(
                "Take y from T2S model pre-inference result failed"
            ))?;
            let y_len = result.pop().ok_or(anyhow::anyhow!(
                "Take y_len from T2S model pre-inference result failed"
            ))?;

            if let (
                IValue::Int(y_len),
                IValue::Tensor(y),
                IValue::Tensor(xy_pos),
                IValue::TensorList(k_cache),
                IValue::TensorList(v_cache),
            ) = (y_len, y, xy_pos, k_cache, v_cache)
            {
                Ok((y_len, y, xy_pos, k_cache, v_cache))
            } else {
                Err(anyhow::anyhow!(
                    "Unexpected types in T2S model pre-inference result"
                ))
            }
        } else {
            Err(anyhow::anyhow!(
                "Expected a tuple from T2S model pre-inference result"
            ))
        }
    }

    /// idx: mast start from 1!
    ///
    /// return: (y, xy_pos, is_end, k_cache, v_cache)
    pub fn decode_next_token(
        &self,
        idx: i64,
        top_k: i64,
        y_len: i64,
        y: Tensor,
        xy_pos: Tensor,
        k_cache: Vec<Tensor>,
        v_cache: Vec<Tensor>,
    ) -> anyhow::Result<(Tensor, Tensor, i64, Vec<Tensor>, Vec<Tensor>)> {
        let result = self
            .model
            .forward_is(&[
                &IValue::Int(idx),
                &IValue::Int(top_k),
                &IValue::Int(y_len),
                &IValue::Tensor(y),
                &IValue::Tensor(xy_pos),
                &IValue::TensorList(k_cache),
                &IValue::TensorList(v_cache),
            ])
            .map_err(|e| {
                log::debug!("Failed to decode next token in T2S model: {}", e);
                anyhow::anyhow!("Failed to decode next token in T2S model")
            })?;
        if let IValue::Tuple(mut result) = result {
            let v_cache = result.pop().ok_or(anyhow::anyhow!(
                "Take v_cache from T2S model decode result failed"
            ))?;
            let k_cache = result.pop().ok_or(anyhow::anyhow!(
                "Take k_cache from T2S model decode result failed"
            ))?;

            let last_token = result.pop().ok_or(anyhow::anyhow!(
                "Take last_token from T2S model decode result failed"
            ))?;
            let xy_pos = result.pop().ok_or(anyhow::anyhow!(
                "Take xy_pos from T2S model decode result failed"
            ))?;
            let y = result.pop().ok_or(anyhow::anyhow!(
                "Take y from T2S model decode result failed"
            ))?;

            if let (
                IValue::Tensor(y),
                IValue::Tensor(xy_pos),
                IValue::Int(last_token),
                IValue::TensorList(k_cache),
                IValue::TensorList(v_cache),
            ) = (y, xy_pos, last_token, k_cache, v_cache)
            {
                Ok((y, xy_pos, last_token, k_cache, v_cache))
            } else {
                Err(anyhow::anyhow!(
                    "Unexpected types in T2S model decode result"
                ))
            }
        } else {
            Err(anyhow::anyhow!(
                "Expected a tuple from T2S model decode result"
            ))
        }
    }
}

pub struct Vits {
    pub file_path: String,
    pub model: tch::CModule,
}

unsafe impl Send for Vits {}
unsafe impl Sync for Vits {}

impl Vits {
    pub fn new(file_path: &str, device: Device) -> anyhow::Result<Self> {
        let mut model = tch::CModule::load_on_device(file_path, device).map_err(|e| {
            log::debug!("Failed to load VITS model from {}: {}", file_path, e);
            anyhow::anyhow!("Failed to load VITS model from {}", file_path)
        })?;
        model.set_eval();
        let file_path = file_path.to_string();
        Ok(Vits { file_path, model })
    }

    /// return: (refer, sv_emb)
    pub fn ref_handle(&self, ref_audio_32k: Tensor) -> anyhow::Result<(Tensor, Tensor)> {
        let result = self
            .model
            .method_is("ref_handle", &[&IValue::Tensor(ref_audio_32k)])
            .map_err(|e| {
                log::debug!("Failed to handle reference in VITS model: {}", e);
                anyhow::anyhow!("Failed to handle reference in VITS model")
            })?;
        if let IValue::Tuple(mut result) = result {
            // refer, sv_emb
            let sv_emb = result.pop().ok_or(anyhow::anyhow!(
                "Take sv_emb from VITS model ref_handle result failed"
            ))?;
            let refer = result.pop().ok_or(anyhow::anyhow!(
                "Take refer from VITS model ref_handle result failed"
            ))?;

            if let (IValue::Tensor(refer), IValue::Tensor(sv_emb)) = (refer, sv_emb) {
                Ok((refer, sv_emb))
            } else {
                Err(anyhow::anyhow!(
                    "Unexpected types in VITS model ref_handle result"
                ))
            }
        } else {
            Err(anyhow::anyhow!(
                "Expected a tuple from VITS model ref_handle result"
            ))
        }
    }

    pub fn extract_latent(&self, ssl_content: Tensor) -> anyhow::Result<Tensor> {
        let r = self
            .model
            .method_ts("extract_latent", &[ssl_content])
            .map_err(|e| {
                log::debug!("Failed to extract latent in VITS model: {}", e);
                anyhow::anyhow!("Failed to extract latent in VITS model")
            })?;
        Ok(r)
    }

    pub fn decode(
        &self,
        pred_semantic: Tensor,
        text_seq: Tensor,
        refer: Tensor,
        sv_emb: Tensor,
    ) -> anyhow::Result<Tensor> {
        let r = self
            .model
            .forward_ts(&[pred_semantic, text_seq, refer, sv_emb])
            .map_err(|e| {
                log::debug!("Failed to generate in VITS model: {}", e);
                anyhow::anyhow!("Failed to generate in VITS model")
            })?;
        Ok(r)
    }
}

pub struct SSL {
    pub ssl: tch::CModule,
}

unsafe impl Send for SSL {}
unsafe impl Sync for SSL {}

impl SSL {
    pub fn new(file_path: &str, device: Device) -> anyhow::Result<Self> {
        let mut ssl = tch::CModule::load_on_device(file_path, device).map_err(|e| {
            log::debug!("Failed to load SSL model from {}: {}", file_path, e);
            anyhow::anyhow!("Failed to load SSL model from {}", file_path)
        })?;
        ssl.set_eval();
        Ok(SSL { ssl })
    }

    /// return: ssl_content
    pub fn to_ssl_content(&self, audio_16k: Tensor) -> anyhow::Result<Tensor> {
        let r = self.ssl.forward_ts(&[audio_16k]).map_err(|e| {
            log::debug!("Failed to forward SSL model: {}", e);
            anyhow::anyhow!("Failed to forward SSL model")
        })?;
        Ok(r)
    }

    pub fn resample(&self, audio: &Tensor, sr: usize, target_sr: usize) -> anyhow::Result<Tensor> {
        tch::no_grad(|| {
            let resample = self
                .ssl
                .method_is(
                    "resample",
                    &[
                        &IValue::Tensor(audio.shallow_clone()),
                        &IValue::Int(sr as i64),
                        &IValue::Int(target_sr as i64),
                    ],
                )
                .map_err(|e| {
                    log::debug!("Failed to resample audio: {}", e);
                    anyhow::anyhow!("Failed to resample audio")
                })?;
            match resample {
                IValue::Tensor(resample) => Ok(resample),
                _ => unreachable!(),
            }
        })
    }
}

pub struct SpeakerV2Pro {
    pub name: String,
    pub t2s: T2S,
    pub vits: Arc<Vits>,
    pub ssl: Arc<SSL>,
}

impl SpeakerV2Pro {
    pub fn new(name: &str, t2s: T2S, vits: Arc<Vits>, ssl: Arc<SSL>) -> Self {
        Self {
            name: name.to_string(),
            t2s,
            vits,
            ssl,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// return : (prompts, refer, sv_emb)
    pub fn pre_handle_ref(
        &self,
        ref_audio_32k: Tensor,
    ) -> anyhow::Result<(Tensor, Tensor, Tensor)> {
        let ref_audio_16k = self.ssl.resample(&ref_audio_32k, 32000, 16000)?;

        let mut ssl_content = self.ssl.to_ssl_content(ref_audio_16k)?;

        if ref_audio_32k.kind() == tch::Kind::Half {
            ssl_content = ssl_content.internal_cast_half(false);
        }

        let prompts = self.vits.extract_latent(ssl_content)?;
        let (refer, sv_emb) = self.vits.ref_handle(ref_audio_32k)?;
        Ok((prompts, refer, sv_emb))
    }

    pub fn infer(
        &self,
        ref_params: (Tensor, Tensor, Tensor),
        ref_seq: Tensor,
        text_seq: Tensor,
        ref_bert: Tensor,
        text_bert: Tensor,
        top_k: i64,
    ) -> anyhow::Result<Tensor> {
        let (prompts, refer, sv_emb) = ref_params;

        let (y_len, mut y, mut xy_pos, mut k_cache, mut v_cache) = self.t2s.pre_infer(
            prompts,
            ref_seq,
            text_seq.shallow_clone(),
            ref_bert,
            text_bert,
            top_k,
        )?;

        let mut i = 1500;
        for idx in 1..1500 {
            // log::debug!("Decoding next token: idx={}/1500", idx);
            let (y_, xy_pos_, last_token, k_cache_, v_cache_) = self
                .t2s
                .decode_next_token(idx, top_k, y_len, y, xy_pos, k_cache, v_cache)?;
            y = y_;
            xy_pos = xy_pos_;
            (k_cache, v_cache) = (k_cache_, v_cache_);
            if last_token == T2S::EOS {
                i = idx;
                break;
            }
        }

        let audio = self.vits.decode(
            y.slice(1, -i, None, 1).unsqueeze(0),
            text_seq,
            refer,
            sv_emb,
        )?;

        let audio = audio.to_dtype(tch::Kind::Float, false, false);
        let size = 32000.0 * 0.3;
        let zero = tch::Tensor::zeros([size as i64], (tch::Kind::Float, audio.device()));

        Ok(tch::Tensor::cat(&[audio, zero], 0))
    }
}

pub struct StreamSpeaker<'a> {
    speaker: &'a SpeakerV2Pro,
    idx: i64,
    last_chunk_idx: i64,
    output_n: usize,
    top_k: i64,
    y_len: i64,
    y: Tensor,
    xy_pos: Tensor,
    is_end: bool,

    text_seq: Tensor,
    refer: Tensor,
    sv_emb: Tensor,

    cache: Option<(Vec<Tensor>, Vec<Tensor>)>,
}

impl<'a> StreamSpeaker<'a> {
    /// max_cut_token: when token < max_cut_token, it will split the audio chunk
    ///
    /// chunk_token_nums: the number of tokens in each chunk, used to determine when to split
    /// 25 token = 1s
    pub fn next_chunk(
        &mut self,
        max_cut_token: i64,
        chunk_token_nums: &[i64],
    ) -> anyhow::Result<Option<Tensor>> {
        if self.is_end {
            return Ok(None);
        }

        let (mut k_cache, mut v_cache) = self.cache.take().unwrap();
        let mut cut_idx = 0;

        let chunk_token_size = chunk_token_nums.len();

        loop {
            let mut idx = self.idx;
            // y.shape = [1,N]
            let (y_, xy_pos_, last_token, k_cache_, v_cache_) =
                self.speaker.t2s.decode_next_token(
                    idx,
                    self.top_k,
                    self.y_len,
                    self.y.shallow_clone(),
                    self.xy_pos.shallow_clone(),
                    k_cache,
                    v_cache,
                )?;

            self.y = y_;
            self.xy_pos = xy_pos_;
            self.idx += 1;
            k_cache = k_cache_;
            v_cache = v_cache_;

            if self.idx > 1500 || last_token == T2S::EOS {
                self.is_end = true;
            }

            let st = std::time::Instant::now();
            if last_token < max_cut_token
                && (idx - self.last_chunk_idx > {
                    if self.output_n >= chunk_token_size {
                        chunk_token_nums[chunk_token_size - 1]
                    } else {
                        chunk_token_nums[self.output_n]
                    }
                })
                && idx > cut_idx
            {
                cut_idx = idx + 7;
            }

            if self.is_end {
                idx = idx - 1;
            }

            if (idx == cut_idx) || self.is_end {
                log::debug!("{idx} * {:?}", st.elapsed());
                let audio = self.speaker.vits.decode(
                    self.y.slice(1, -idx, None, 1).unsqueeze(0),
                    self.text_seq.shallow_clone(),
                    self.refer.shallow_clone(),
                    self.sv_emb.shallow_clone(),
                )?;
                let (start, end) = if self.output_n == 0 {
                    (0, -1280 * 8)
                } else {
                    ((self.last_chunk_idx - 8) * 1280, -1280 * 8)
                };

                let audio = if self.is_end {
                    self.is_end = true;
                    const SIZE: f64 = 32000.0 * 0.3;

                    let zero =
                        tch::Tensor::zeros([SIZE as i64], (tch::Kind::Float, audio.device()));

                    let audio = audio
                        .to_dtype(tch::Kind::Float, false, false)
                        .slice(0, start, None, 1);

                    tch::Tensor::cat(&[audio, zero], 0)
                } else {
                    audio
                        .to_dtype(tch::Kind::Float, false, false)
                        .slice(0, start, end, 1)
                };

                self.output_n += 1;
                self.last_chunk_idx = idx;
                self.cache = Some((k_cache, v_cache));

                return Ok(Some(audio));
            } else {
                log::debug!("{idx} {:?}", st.elapsed());
            }
        }
    }
}

impl SpeakerV2Pro {
    /// Create a new streaming inference session.
    /// This method is still experimental, and its current performance has some flaws.
    pub fn stream_infer<'a>(
        &'a self,
        ref_params: (Tensor, Tensor, Tensor),
        ref_seq: Tensor,
        text_seq: Tensor,
        ref_bert: Tensor,
        text_bert: Tensor,
        top_k: i64,
    ) -> anyhow::Result<StreamSpeaker<'a>> {
        let (prompts, refer, sv_emb) = ref_params;

        let (y_len, y, xy_pos, k_cache, v_cache) = self.t2s.pre_infer(
            prompts,
            ref_seq,
            text_seq.shallow_clone(),
            ref_bert,
            text_bert,
            top_k,
        )?;

        Ok(StreamSpeaker {
            speaker: self,
            idx: 1,
            last_chunk_idx: 0,
            output_n: 0,
            top_k,
            y_len,
            y,
            xy_pos,
            is_end: false,

            text_seq,
            refer,
            sv_emb,
            cache: Some((k_cache, v_cache)),
        })
    }
}
