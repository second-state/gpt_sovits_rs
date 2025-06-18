use std::{sync::Arc, vec};

use gpt_sovits_rs::{gsv, text::G2PConfig};

fn main() {
    env_logger::init();

    // let t = tch::Tensor::from_slice(&[2i64, 3, 10]).unsqueeze(0);
    // let t = t.get(0).get(2);

    // let mut data = [true];
    // println!("t: {}", t);
    // println!("x: {}", t.greater(30).is_nonzero());
    // println!("x: {}", t.greater(1).is_nonzero());
    // println!("{:?}", data);
    // return;

    // "../python/GPT-SoVITS/onnx/xww/gpt_sovits_model.pt".to_string(),
    let g2p_conf = G2PConfig::new("./resource/mini-bart-g2p.pt".to_string()).with_chinese(
        "../python/g2pW/onnx/g2pw.pt".to_string(),
        "../python/GPT-SoVITS/onnx/bert_model.pt".to_string(),
    );

    let device = gpt_sovits_rs::Device::cuda_if_available();
    log::info!("device: {:?}", device);

    let g2p = g2p_conf.build(device).unwrap();

    // let ref_text = "声音,是有温度的.夜晚的声音,会发光.";
    // let ref_text = "在参加rust春晚的时候，听到有人问了这样一个问题.";
    // let ref_path = "../python/GPT-SoVITS/onnx/xww/ref.wav";
    // let file = std::fs::File::open(ref_path).unwrap();
    // let (head, ref_audio_samples) = wav_io::read_from_file(file).unwrap();

    // log::info!("load wx ref done");

    // gpt_sovits
    //     .create_speaker(
    //         "xw",
    //         "../python/GPT-SoVITS/onnx/xww/gpt_sovits_model.pt",
    //         &ref_audio_samples,
    //         head.sample_rate as usize,
    //         ref_text,
    //     )
    //     .unwrap();

    // log::info!("init wx done");

    let ref_text = "说真的，这件衣服才配得上本小姐嘛。";
    let ref_path = "../python/GPT-SoVITS/output/denoise_opt/ht/ht.mp4_0000026560_0000147200.wav";
    // let ref_path = "../python/GPT-SoVITS/onnx/trump/ref.wav";
    // let ref_text = "It said very simply, we can do it.";

    let file = std::fs::File::open(ref_path).unwrap();
    let (head, mut ref_audio_samples) = wav_io::read_from_file(file).unwrap();
    log::info!("head: {:?}", head);
    if head.sample_rate != 32000 {
        log::info!("ref audio sample rate: {}, need 32000", head.sample_rate);
        ref_audio_samples = wav_io::resample::linear(ref_audio_samples, 1, head.sample_rate, 32000);
    }

    log::info!("load ht ref done");

    let ssl = gsv::SSL::new("../python/GPT-SoVITS/onnx/xww/ssl_model.pt", device).unwrap();
    let t2s = gsv::T2S::new("../python/GPT-SoVITS/streaming/t2s.pt", device).unwrap();
    let vits = gsv::Vits::new("../python/GPT-SoVITS/streaming/vits.pt", device).unwrap();

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
    let mut stream = speaker
        .stream_infer(
            (
                prompts.shallow_clone(),
                refer.shallow_clone(),
                sv_emb.shallow_clone(),
            ),
            ref_seq.shallow_clone(),
            text_seq.shallow_clone(),
            ref_bert.shallow_clone(),
            text_bert.shallow_clone(),
            15,
        )
        .unwrap();

    let mut audios = Vec::new();
    let mut n = 0;
    while let Some(a) = stream.next_chunk(25, &[25, 25, 50, 100]).unwrap() {
        log::info!(
            "stream chunk: {} {} {:?}",
            a.size()[0] / 32000,
            n,
            st.elapsed()
        );
        audios.push(a);
        n += 1;
    }

    log::info!("stream done, total chunks: {}", n);
    let audio = tch::Tensor::cat(&audios, 0).to_device(device);
    log::info!("stream audio size: {:?}", audio.size());
    let audio_size = audio.size1().unwrap() as usize;
    println!("stream audio size: {}", audio_size);
    let mut samples = vec![0f32; audio_size];
    audio.f_copy_data(&mut samples, audio_size).unwrap();
    println!("start write stream file");
    let mut file_out = std::fs::File::create("stream_out.wav").unwrap();
    wav_io::write_to_file(&mut file_out, &header, &samples).unwrap();
    log::info!("stream write file done");

    for x in 0..10 {
        let st0 = std::time::Instant::now();
        let mut st = std::time::Instant::now();
        let mut stream = speaker
            .stream_infer(
                (
                    prompts.shallow_clone(),
                    refer.shallow_clone(),
                    sv_emb.shallow_clone(),
                ),
                ref_seq.shallow_clone(),
                text_seq.shallow_clone(),
                ref_bert.shallow_clone(),
                text_bert.shallow_clone(),
                15,
            )
            .unwrap();

        let mut audios = Vec::new();
        let mut n = 0;
        while let Some(a) = stream.next_chunk(25, &[25, 25, 50, 100]).unwrap() {
            log::info!(
                "stream chunk: {} {} {:?}",
                a.size()[0] as f32 / 32000.0,
                n,
                st.elapsed()
            );
            audios.push(a);
            n += 1;
            st = std::time::Instant::now();
        }
        log::info!("stream done, cost: {:?}", st0.elapsed());

        log::info!("stream done, total chunks: {}", n);
        let audio = tch::Tensor::cat(&audios, 0).to_device(device);
        log::info!("stream audio size: {:?}", audio.size());
        let audio_size = audio.size1().unwrap() as usize;
        println!("stream audio size: {}", audio_size);
        let mut samples = vec![0f32; audio_size];
        audio.f_copy_data(&mut samples, audio_size).unwrap();
        println!("start write stream file");
        let mut file_out = std::fs::File::create(format!("stream_out.{x}.wav")).unwrap();
        wav_io::write_to_file(&mut file_out, &header, &samples).unwrap();
        log::info!("stream write file done");
    }

    let st = std::time::Instant::now();
    let audio = speaker
        .infer(
            (
                prompts.shallow_clone(),
                refer.shallow_clone(),
                sv_emb.shallow_clone(),
            ),
            ref_seq.shallow_clone(),
            text_seq.shallow_clone(),
            ref_bert.shallow_clone(),
            text_bert.shallow_clone(),
            15,
        )
        .unwrap();
    log::info!("infer done, cost: {:?}", st.elapsed());

    let output = "out.wav";
    let audio_size = audio.size1().unwrap() as usize;
    println!("audio size: {}", audio_size);

    println!("start save audio {output}");
    let mut samples = vec![0f32; audio_size];
    audio.f_copy_data(&mut samples, audio_size).unwrap();
    println!("start write file {output}");
    let mut file_out = std::fs::File::create(output).unwrap();
    wav_io::write_to_file(&mut file_out, &header, &samples).unwrap();
    log::info!("write file done");
}
