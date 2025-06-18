use std::{sync::Arc, vec};

use gpt_sovits_rs::{gsv, text::G2PConfig};

// If you want to run this example, you need to download some model files.
// Look at the README.md for more details.

fn main() {
    env_logger::init();
    let g2p_conf = G2PConfig::new("./resource/mini-bart-g2p.pt".to_string()).with_chinese(
        "path/to/g2pw.pt".to_string(),
        "path/to/bert_model.pt".to_string(),
    );

    let device = gpt_sovits_rs::Device::cuda_if_available();
    log::info!("device: {:?}", device);

    let g2p = g2p_conf.build(device).unwrap();

    // ref audio and ref text
    let ref_text = "说真的，这件衣服才配得上本小姐嘛。";
    let ref_path = "path/to/ref.wav";

    let file = std::fs::File::open(ref_path).unwrap();
    let (head, mut ref_audio_samples) = wav_io::read_from_file(file).unwrap();
    log::info!("head: {:?}", head);
    if head.sample_rate != 32000 {
        log::info!("ref audio sample rate: {}, need 32000", head.sample_rate);
        ref_audio_samples = wav_io::resample::linear(ref_audio_samples, 1, head.sample_rate, 32000);
    }

    log::info!("load ht ref done");

    let ssl = gsv::SSL::new("path/to/ssl_model.pt", device).unwrap();
    let t2s = gsv::T2S::new("path/to/t2s.pt", device).unwrap();
    let vits = gsv::Vits::new("path/to/vits.pt", device).unwrap();

    let mut speaker =
        gpt_sovits_rs::gsv::SpeakerV2Pro::new("ht", t2s, Arc::new(vits), Arc::new(ssl));

    log::info!("start write file");

    let text = "这是一个简单的示例，真没想到这么简单就完成了。真的神奇。接下来我们说说狐狸,可能这就是狐狸吧.它有长长的尾巴，尖尖的耳朵，传说中还有九条尾巴。你觉得狐狸神奇吗？";

    let (text_seq, text_bert) = gpt_sovits_rs::text::get_phone_and_bert(&g2p, text).unwrap();
    let text_bert = text_bert.internal_cast_half(false);

    let (ref_seq, ref_bert) = gpt_sovits_rs::text::get_phone_and_bert(&g2p, ref_text).unwrap();
    let ref_bert = ref_bert.internal_cast_half(false);

    let ref_audio_32k = tch::Tensor::from_slice(&ref_audio_samples)
        .internal_cast_half(false)
        .to_device(device)
        .unsqueeze(0);

    let _g = tch::no_grad_guard();

    let header = wav_io::new_header(32000, 16, false, true);

    let (prompts, refer, sv_emb) = speaker.pre_handle_ref(ref_audio_32k).unwrap();

    let st = std::time::Instant::now();
    let audio = speaker
        .infer(
            (prompts, refer, sv_emb),
            ref_seq,
            text_seq,
            ref_bert,
            text_bert,
            15,
        )
        .unwrap();
    log::info!("infer done, cost: {:?}", st.elapsed());

    let output = "out.wav";
    let audio_size = audio.size1().unwrap() as usize;
    println!("audio size: {}", audio_size);

    let mut samples = vec![0f32; audio_size];
    audio.f_copy_data(&mut samples, audio_size).unwrap();
    let mut file_out = std::fs::File::create(output).unwrap();
    wav_io::write_to_file(&mut file_out, &header, &samples).unwrap();
    log::info!("save wav done");
}
