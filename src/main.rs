use std::collections::HashMap;

use tch::{IValue, Kind, Tensor};
use tokenizers::Tokenizer;

fn main1() {
    use pinyin::ToPinyin;
    let x = "声音,是有温度的.夜晚的声音,会发光";
    for c in x.chars() {
        if let Some(p) = c.to_pinyin() {
            let s = p.with_tone_num_end();
            let (y, s) = split_zh_ph(s);
            print!("'{y}','{s}',")
        } else {
            print!("','")
        }
    }
    println!()
}

fn split_zh_ph(ph: &str) -> (&str, &str) {
    if ph.starts_with("zh") || ph.starts_with("ch") || ph.starts_with("sh") {
        ph.split_at(2)
    } else {
        ph.split_at(1)
    }
}

fn phones(symbels: &HashMap<String, i64>, text: &str, device: tch::Device) -> (Tensor, Tensor) {
    use pinyin::ToPinyin;

    let mut word2ph = Vec::new();

    let mut sequence = Vec::new();

    for c in text.chars() {
        if let Some(p) = c.to_pinyin() {
            let (y, s) = split_zh_ph(p.with_tone_num_end());
            sequence.push(symbels.get(y).map(|id| *id).unwrap_or(0));
            sequence.push(symbels.get(s).map(|id| *id).unwrap_or(0));
            word2ph.push(2);
        } else {
            let s = c.to_string();
            sequence.push(symbels.get(&s).map(|id| *id).unwrap_or(0));
            word2ph.push(1);
        }
    }

    let word2ph = Tensor::from_slice(&word2ph);
    let t = Tensor::from_slice(&sequence).to_device(device);
    (t.unsqueeze(0), word2ph)
}

fn infer(
    tokenizer: &Tokenizer,
    symbels: &HashMap<String, i64>,
    t2s: &tch::CModule,
    vits: &tch::CModule,
    bert: &tch::CModule,
    text: &str,
    prompts: &Tensor,
    ref_seq: &Tensor,
    ref_bert: &Tensor,
    ref_audio_sr: &Tensor,
    output: &str,
    device: tch::Device,
) {
    let (text_ids, text_mask, text_token_type_ids) = encode_text(text, &tokenizer, device);
    let (text_seq, text_word2ph) = phones(&symbels, text, device);

    let text_bert = bert
        .forward_ts(&[text_ids, text_mask, text_token_type_ids, text_word2ph])
        .unwrap();

    println!("start t2s");
    let pred_semantic = t2s
        .forward_ts(&[prompts, ref_seq, &text_seq, ref_bert, &text_bert])
        .unwrap();

    // audio = vits(text_seq, pred_semantic, ref_audio_sr)
    println!("start vits");
    let audio = vits
        .forward_ts(&[&text_seq, &pred_semantic, ref_audio_sr])
        .unwrap();

    let audio_size = audio.size1().unwrap() as usize;
    println!("audio size: {}", audio_size);

    println!("start save audio");
    let mut samples = vec![0f32; audio_size];
    audio.f_copy_data(&mut samples, audio_size).unwrap();

    println!("start write file");
    let mut file_out = std::fs::File::create(output).unwrap();
    let header = wav_io::new_header(32000, 16, false, true);
    wav_io::write_to_file(&mut file_out, &header, &samples).unwrap();
}

fn run_ssl(ssl: &tch::CModule, ref_audio: &IValue) -> (Tensor, Tensor) {
    let r = ssl.forward_is(&[ref_audio]).unwrap();
    if let IValue::Tuple(mut r) = r {
        let ref_audio_sr = r.pop().unwrap();
        let prompts = r.pop().unwrap();
        if let (IValue::Tensor(prompts), IValue::Tensor(ref_audio_sr)) = (prompts, ref_audio_sr) {
            (prompts, ref_audio_sr)
        } else {
            unreachable!()
        }
    } else {
        unimplemented!()
    }
}

fn main() {
    let tokenizer =
        Tokenizer::from_file("./pretrained_models/chinese-roberta-wwm-ext-large/tokenizer.json")
            .unwrap();

    let symbels = load_symbel();

    let device = tch::Device::cuda_if_available();
    println!("device: {:?}", device);

    let ref_text = "声音,是有温度的.夜晚的声音,会发光~";
    let (ref_text_ids, ref_text_mask, ref_text_token_type_ids) =
        encode_text(ref_text, &tokenizer, device);
    let (ref_seq, ref_text_word2ph) = phones(&symbels, ref_text, device);

    let ref_audio = load_ref_audio().to_device(device);
    println!("ref_audio size: {:?}", ref_audio.size());
    let ref_audio = IValue::Tensor(ref_audio);

    let (bert, ssl, t2s, vits) = check_vits_model(device);
    std::thread::sleep(std::time::Duration::from_secs(1));
    println!("start infer");

    let (prompts, ref_audio_sr) = tch::no_grad(|| run_ssl(&ssl, &ref_audio));

    println!("done ssl");
    let ref_bert = bert
        .forward_ts(&[
            &ref_text_ids,
            &ref_text_mask,
            &ref_text_token_type_ids,
            &ref_text_word2ph,
        ])
        .unwrap();

    println!("start infer");
    let text = "我有一个奇怪的问题.你们谁知道什么是春晚?";
    tch::no_grad(|| {
        infer(
            &tokenizer,
            &symbels,
            &t2s,
            &vits,
            &bert,
            &text,
            &prompts,
            &ref_seq,
            &ref_bert,
            &ref_audio_sr,
            "output.wav",
            device,
        );
    });

    println!("start infer 2");
    let text = "你们谁知道什么是春晚吗？第一年的春晚是什么时候?";
    tch::no_grad(|| {
        infer(
            &tokenizer,
            &symbels,
            &t2s,
            &vits,
            &bert,
            &text,
            &prompts,
            &ref_seq,
            &ref_bert,
            &ref_audio_sr,
            "output1.wav",
            device,
        );
    });
}

fn encode_text(text: &str, tokenizer: &Tokenizer, device: tch::Device) -> (Tensor, Tensor, Tensor) {
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

fn check_vits_model(
    device: tch::Device,
) -> (tch::CModule, tch::CModule, tch::CModule, tch::CModule) {
    let mut t2s = tch::CModule::load_on_device(
        "/home/csh/ai/python/GPT-SoVITS/onnx/xw/t2s_model.pt",
        device,
    )
    .unwrap();
    t2s.set_eval();
    println!("load t2s_model model success");

    let mut bert =
        tch::CModule::load_on_device("/home/csh/ai/python/GPT-SoVITS/onnx/bert_model.pt", device)
            .unwrap();
    bert.set_eval();
    println!("load bert_model model success");

    let mut ssl = tch::CModule::load_on_device(
        "/home/csh/ai/python/GPT-SoVITS/onnx/xw/ssl_model.pt",
        device,
    )
    .unwrap();
    ssl.set_eval();
    println!("load ssl_model model success");

    let mut vits = tch::CModule::load_on_device(
        "/home/csh/ai/python/GPT-SoVITS/onnx/xw/vits_model.pt",
        device,
    )
    .unwrap();
    vits.set_eval();
    println!("load vits_model model success");
    (bert, ssl, t2s, vits)
}

fn load_symbel() -> HashMap<String, i64> {
    let f = std::fs::File::open("/home/csh/ai/python/GPT-SoVITS/onnx/symbols_v2.json").unwrap();
    serde_json::from_reader(f).unwrap()
}

fn load_ref_audio() -> Tensor {
    let ref_path = "/home/csh/ai/python/GPT-SoVITS/chen1_ref.t.wav";
    let file = std::fs::File::open(ref_path).unwrap();

    let (head, samples) = wav_io::read_from_file(file).unwrap();

    let t = Tensor::from_slice(&samples);
    t.unsqueeze(0)
}
