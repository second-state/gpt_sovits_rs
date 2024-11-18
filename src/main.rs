use std::vec;

use gpt_sovits_rs::GPTSovitsConfig;
use tch::Tensor;

fn main() {
    env_logger::init();

    // "../python/GPT-SoVITS/onnx/xww/gpt_sovits_model.pt".to_string(),
    let gpt_config = GPTSovitsConfig::new(
        "../python/GPT-SoVITS/onnx/xww/ssl_model.pt".to_string(),
    ).with_chinese("../python/g2pW/onnx/g2pw.pt".to_string(),
    "../python/GPT-SoVITS/onnx/bert_model.pt".to_string(), 
    "../python/GPT-SoVITS/GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large/tokenizer.json".to_string());

    let device = gpt_sovits_rs::Device::cuda_if_available();
    log::info!("device: {:?}", device);

    let mut gpt_sovits = gpt_config.build(device).unwrap();

    // let ref_text = "声音,是有温度的.夜晚的声音,会发光.";
    let ref_text = "在参加rust春晚的时候，听到有人问了这样一个问题.";
    let ref_path = "../python/GPT-SoVITS/onnx/xww/ref.wav";
    let file = std::fs::File::open(ref_path).unwrap();
    let (head, ref_audio_samples) = wav_io::read_from_file(file).unwrap();

    log::info!("load wx ref done");

    gpt_sovits
        .create_speaker(
            "xw",
            "../python/GPT-SoVITS/onnx/xww/gpt_sovits_model.pt",
            &ref_audio_samples,
            head.sample_rate as usize,
            ref_text,
        )
        .unwrap();

    log::info!("init wx done");

    let ref_text = "说真的，这件衣服才配得上本小姐嘛。";
    let ref_path = "../python/GPT-SoVITS/output/denoise_opt/ht/ht.mp4_0000026560_0000147200.wav";
    let file = std::fs::File::open(ref_path).unwrap();
    let (head, ref_audio_samples) = wav_io::read_from_file(file).unwrap();

    log::info!("load ht ref done");

    gpt_sovits
        .create_speaker(
            "ht",
            "../python/GPT-SoVITS/onnx/htt/gpt_sovits_model.pt",
            &ref_audio_samples,
            head.sample_rate as usize,
            ref_text,
        )
        .unwrap();

    log::info!("init ht done");

    let text = std::fs::read_to_string("./input.txt").unwrap();

    let text_splitter = text_splitter::TextSplitter::new(50);
    let header = wav_io::new_header(32000, 16, false, true);
    let mut audios = vec![];
    let mut audios2 = vec![];

    for target_text in text_splitter.chunks(&text) {
        println!("text: {}", target_text);
        if target_text == "。" {
            continue;
        }

        let audio = gpt_sovits.infer("xw", target_text).unwrap();
        audios.push(audio);
        let audio = gpt_sovits.infer("ht", target_text).unwrap();
        audios2.push(audio);
    }

    log::info!("start write file");

    let output = "out.wav";
    let output2 = "out2.wav";

    let audio = Tensor::cat(&audios, 0);
    let audio2 = Tensor::cat(&audios2, 0);

    let audio_size = audio.size1().unwrap() as usize;
    let audio2_size = audio2.size1().unwrap() as usize;
    println!("audio size: {}", audio_size);
    println!("audio2 size: {}", audio2_size);

    println!("start save audio {output}");
    let mut samples = vec![0f32; audio_size];
    audio.f_copy_data(&mut samples, audio_size).unwrap();
    println!("start write file {output}");
    let mut file_out = std::fs::File::create(output).unwrap();
    wav_io::write_to_file(&mut file_out, &header, &samples).unwrap();
    log::info!("write file done");

    println!("start save audio {output2}");
    let mut samples2 = vec![0f32; audio2_size];
    audio2.f_copy_data(&mut samples2, audio2_size).unwrap();
    println!("start write file {output2}");
    let mut file_out2 = std::fs::File::create(output2).unwrap();
    wav_io::write_to_file(&mut file_out2, &header, &samples2).unwrap();
    log::info!("write file done");
}
