// model from cisco-ai/mini-bart-g2p

use std::{str::FromStr, sync::Arc};

static MINI_BART_G2P_TOKENIZER: &str = include_str!("../../resource/tokenizer.mini-bart-g2p.json");

static DECODER_START_TOKEN_ID: u32 = 2;

#[allow(unused)]
static BOS_TOKEN: &str = "<s>";
#[allow(unused)]
static EOS_TOKEN: &str = "</s>";

#[allow(unused)]
static BOS_TOKEN_ID: u32 = 0;
static EOS_TOKEN_ID: u32 = 2;

pub struct G2PEnConverter {
    model: Arc<tch::CModule>,
    tokenizer: Arc<tokenizers::Tokenizer>,
    device: tch::Device,
}

impl G2PEnConverter {
    pub fn new(model_path: &str) -> Self {
        let device = tch::Device::Cpu;
        Self::new_with_device(model_path, device)
    }

    fn new_with_device(model_path: &str, device: tch::Device) -> Self {
        let tokenizer = tokenizers::Tokenizer::from_str(MINI_BART_G2P_TOKENIZER)
            .map_err(|e| anyhow::anyhow!("load g2p_en tokenizer error: {}", e))
            .unwrap();
        let tokenizer = Arc::new(tokenizer);

        let mut model = tch::CModule::load_on_device(model_path, device)
            .map_err(|e| anyhow::anyhow!("load g2p_en model error: {}", e))
            .unwrap();
        model.set_eval();

        Self {
            model: Arc::new(model),
            tokenizer,
            device,
        }
    }

    pub fn get_phoneme(&self, text: &str) -> anyhow::Result<String> {
        let c = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("encode error: {}", e))?;
        let input_ids = c.get_ids().iter().map(|x| *x as i64).collect::<Vec<i64>>();
        let mut decoder_input_ids = vec![DECODER_START_TOKEN_ID as i64];

        for _ in 0..50 {
            let input = tch::Tensor::from_slice(&input_ids)
                .to_device(self.device)
                .unsqueeze(0);
            let decoder_input = tch::Tensor::from_slice(&decoder_input_ids)
                .to_device(self.device)
                .unsqueeze(0);

            let output = self
                .model
                .forward_ts(&[input, decoder_input])
                .map_err(|e| anyhow::anyhow!("g2p_en forward error: {}", e))?;

            let next_token_logits = output.get(0).get(-1);

            let next_token_id = next_token_logits.argmax(0, true).int64_value(&[]);
            decoder_input_ids.push(next_token_id);
            if next_token_id == EOS_TOKEN_ID as i64 {
                break;
            }
        }

        let decoder_input_ids = decoder_input_ids
            .iter()
            .map(|x| *x as u32)
            .collect::<Vec<u32>>();
        Ok(self
            .tokenizer
            .decode(&decoder_input_ids, true)
            .map_err(|e| anyhow::anyhow!("g2p_en decode error: {}", e))?)
    }
}

// cargo test --package gpt_sovits_rs --lib -- text::g2p_en::test_g2p_en_converter --exact --show-output
#[test]
fn test_g2p_en_converter() {
    let g2p_en = G2PEnConverter::new("./resource/mini-bart-g2p.pt");
    let phoneme = g2p_en.get_phoneme("a hello world").unwrap();
    println!("{}", phoneme);
    let phoneme = g2p_en.get_phoneme("hello world").unwrap();
    println!("{}", phoneme);
    let phoneme = g2p_en.get_phoneme("a").unwrap();
    println!("{}", phoneme);
    let phoneme = g2p_en.get_phoneme("near-team").unwrap();
    println!("{}", phoneme);
}
