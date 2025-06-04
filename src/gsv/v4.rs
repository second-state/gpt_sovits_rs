use std::{fmt::Debug, sync::Arc};

use tch::{IValue, Tensor};

#[derive(Clone)]
pub struct GPTSoVITSV4Half(Arc<tch::CModule>);
impl Debug for GPTSoVITSV4Half {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GPTSoVITSV4Half")
    }
}

impl GPTSoVITSV4Half {
    pub fn new(gpt_sovits_v4_half: Arc<tch::CModule>) -> Self {
        Self(gpt_sovits_v4_half)
    }

    pub fn get_fea_ref(
        &self,
        ssl_content: Tensor,
        ref_audio_32k: Tensor,
        ref_phone_seq: Tensor,
    ) -> anyhow::Result<(Tensor, Tensor, Tensor, Tensor)> {
        tch::no_grad(|| {
            let tensors = self.0.method_is(
                "get_fea_ref",
                &[
                    &IValue::Tensor(ssl_content),
                    &IValue::Tensor(ref_audio_32k),
                    &IValue::Tensor(ref_phone_seq),
                ],
            )?;
            if let IValue::Tuple(mut tensors) = tensors {
                // codes, ge, fea_ref, mel2
                let mel2 = tensors.pop();
                let fea_ref = tensors.pop();
                let ge = tensors.pop();
                let codes = tensors.pop();
                if let (
                    Some(IValue::Tensor(codes)),
                    Some(IValue::Tensor(ge)),
                    Some(IValue::Tensor(fea_ref)),
                    Some(IValue::Tensor(mel2)),
                ) = (codes, ge, fea_ref, mel2)
                {
                    return Ok((codes, ge, fea_ref, mel2));
                } else {
                    return Err(anyhow::anyhow!("return value is not enough tensors"));
                }
            } else {
                return Err(anyhow::anyhow!("expected a tuple of tensors"));
            }
        })
    }

    pub fn get_fea_todo(
        &self,
        codes: Tensor,
        ge: Tensor,
        ref_phone_seq: Tensor,
        text_phone_seq: Tensor,
        ref_bert_seq: Tensor,
        bert_seq: Tensor,
        top_k: Tensor,
    ) -> anyhow::Result<Tensor> {
        // get_fea_todo(self, codes, ge, phoneme_ids0, phoneme_ids1, bert1, bert2, top_k):

        tch::no_grad(|| {
            let fea_todo = self.0.method_is(
                "get_fea_todo",
                &[
                    IValue::Tensor(codes),
                    IValue::Tensor(ge),
                    IValue::Tensor(ref_phone_seq),
                    IValue::Tensor(text_phone_seq),
                    IValue::Tensor(ref_bert_seq),
                    IValue::Tensor(bert_seq),
                    IValue::Tensor(top_k),
                ],
            )?;
            if let IValue::Tensor(fea_todo) = fea_todo {
                Ok(fea_todo)
            } else {
                Err(anyhow::anyhow!("expected a tensor"))
            }
        })
    }
}

pub struct StreamSpeakerV4 {
    gpt_sovits_v4_half: GPTSoVITSV4Half,
    cfm: Arc<tch::CModule>,
    hifigan: Arc<tch::CModule>,

    codes: Tensor,
    ge: Tensor,
    fea_ref: Tensor,
    mel2: Tensor,

    ref_phone_seq: Tensor,
    ref_bert_seq: Tensor,
    pub top_k: Option<i64>,
    pub sample_steps: Option<i64>,
}

impl StreamSpeakerV4 {
    pub fn new(
        gpt_sovits_v4_half: Arc<tch::CModule>,
        cfm: Arc<tch::CModule>,
        hifigan: Arc<tch::CModule>,
        ssl_content: Tensor,
        ref_audio_32k: Tensor,
        ref_phone_seq: Tensor,
        ref_bert_seq: Tensor,
        top_k: Option<i64>,
        sample_steps: Option<i64>,
    ) -> anyhow::Result<Self> {
        let gpt_sovits_v4_half = GPTSoVITSV4Half::new(gpt_sovits_v4_half);

        let (codes, ge, fea_ref, mel2) = gpt_sovits_v4_half.get_fea_ref(
            ssl_content.internal_cast_half(false),
            ref_audio_32k,
            ref_phone_seq.shallow_clone(),
        )?;

        let ref_bert_seq = ref_bert_seq.internal_cast_half(false);

        Ok(Self {
            gpt_sovits_v4_half,
            cfm,
            hifigan,
            ref_bert_seq,
            top_k,
            sample_steps,

            codes,
            ge,
            fea_ref,
            mel2,
            ref_phone_seq,
        })
    }

    fn get_fea_todo(
        &self,
        text_phone_seq: &Tensor,
        bert_seq: &Tensor,
        top_k: Tensor,
    ) -> anyhow::Result<Tensor> {
        // get_fea_todo(self, codes, ge, phoneme_ids0, phoneme_ids1, bert1, bert2, top_k):
        self.gpt_sovits_v4_half.get_fea_todo(
            self.codes.shallow_clone(),
            self.ge.shallow_clone(),
            self.ref_phone_seq.shallow_clone(),
            text_phone_seq.shallow_clone(),
            self.ref_bert_seq.shallow_clone(),
            bert_seq.internal_cast_half(false),
            top_k,
        )
    }

    /// return mel2_, fea_ref_, cfm_res
    fn cfm_forward(
        &self,
        fea_ref: Tensor,
        fea_todo_chunk: Tensor,
        mel2: Tensor,
        sample_steps: Tensor,
    ) -> anyhow::Result<(Tensor, Tensor, Tensor)> {
        if let IValue::Tuple(mut tuple) = self.cfm.forward_is(&[
            &IValue::Tensor(fea_ref),
            &IValue::Tensor(fea_todo_chunk),
            &IValue::Tensor(mel2),
            &IValue::Tensor(sample_steps),
        ])? {
            let mel2_ = tuple.pop();
            let fea_ref_ = tuple.pop();
            let cfm_res = tuple.pop();

            if let (
                Some(IValue::Tensor(mel2_)),
                Some(IValue::Tensor(fea_ref_)),
                Some(IValue::Tensor(cfm_res)),
            ) = (mel2_, fea_ref_, cfm_res)
            {
                Ok((mel2_, fea_ref_, cfm_res))
            } else {
                Err(anyhow::anyhow!("cfm return not enough tensors"))
            }
        } else {
            Err(anyhow::anyhow!("expected a tuple of tensors"))
        }
    }

    pub fn infer(
        &self,
        text_phone_seq: &Tensor,
        bert_seq: &Tensor,
        mut chunk_len: i64,
        mut callback: impl FnMut(tch::Tensor) -> anyhow::Result<()>,
    ) -> anyhow::Result<()> {
        log::info!("start stream infer");
        let top_k = self.top_k.unwrap_or(15);
        let top_k = Tensor::from_slice(&[top_k]);

        let sample_steps = self.sample_steps.unwrap_or(8);
        let sample_steps = Tensor::from_slice(&[sample_steps]);

        // fea_ref, fea_todo, mel2 = self.gpt_sovits_half(
        //     ssl_content, ref_audio_32k, phoneme_ids0, phoneme_ids1, bert1, bert2, top_k
        // )
        log::info!("start get_fea_todo");
        let mut mel2 = self.mel2.shallow_clone();
        let mut fea_ref = self.fea_ref.shallow_clone();
        let fea_todo = self.get_fea_todo(text_phone_seq, bert_seq, top_k)?;

        log::info!("end get_fea_todo");

        {
            if chunk_len <= 0 {
                chunk_len = 1000 - fea_ref.size3()?.2;
            }

            let fea_todo_size = fea_todo.size3()?.2;

            let mut idx = 0;
            loop {
                let max_chunk_len = fea_todo_size - idx;

                let mut fea_todo_chunk = fea_todo.narrow(2, idx, chunk_len.min(max_chunk_len));

                let complete_len = chunk_len - fea_todo_chunk.size3()?.2;
                if complete_len != 0 {
                    let kind = fea_todo_chunk.kind();
                    let device = fea_todo_chunk.device();

                    fea_todo_chunk = tch::Tensor::cat(
                        &[
                            fea_todo_chunk,
                            tch::Tensor::zeros([1, 512, complete_len], (kind, device)),
                        ],
                        2,
                    );
                }
                let (mel2_, fea_ref_, cfm_res) =
                    self.cfm_forward(fea_ref, fea_todo_chunk, mel2, sample_steps.shallow_clone())?;

                idx += chunk_len;

                // denorm_spec
                let cfm_res = (cfm_res + 1) / 2 * (2 - -12) + -12;

                let hifigen_res = self.hifigan.forward_ts(&[cfm_res])?.squeeze();

                mel2 = mel2_;
                fea_ref = fea_ref_;

                log::info!(
                    "chunk_len: {}, complete_len: {}, idx: {}",
                    chunk_len,
                    complete_len,
                    idx
                );
                let hifigen_res = hifigen_res.to_dtype(tch::Kind::Float, false, false);

                if complete_len > 0 {
                    let hifigen_res =
                        hifigen_res.narrow(0, 0, (chunk_len - complete_len - 5) * 480);
                    let device = hifigen_res.device();
                    let kind = hifigen_res.kind();

                    callback(tch::Tensor::cat(
                        &[
                            hifigen_res,
                            tch::Tensor::zeros([(0.3 * 48000.0) as i64], (kind, device)),
                        ],
                        0,
                    ))?;

                    return Ok(());
                } else {
                    callback(hifigen_res)?;
                }
            }
        }
    }
}
