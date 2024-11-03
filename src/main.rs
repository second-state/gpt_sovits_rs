use gpt_sovits_rs::GPTSovitsConfig;
use tch::Tensor;

fn main() {
    env_logger::init();

    let gpt_config = GPTSovitsConfig::new(
        "../python/GPT-SoVITS/onnx/xww/gpt_sovits_model.pt".to_string(),
        "../python/GPT-SoVITS/onnx/xww/ssl_model.pt".to_string(),
    ).with_cn_bert_path("/home/csh/ai/python/GPT-SoVITS/onnx/bert_model.pt".to_string(), "/home/csh/ai/python/GPT-SoVITS/GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large/tokenizer.json".to_string());

    let device = gpt_sovits_rs::Device::cuda_if_available();
    log::info!("device: {:?}", device);

    let gpt_sovits = gpt_config.build(device).unwrap();

    log::info!("init done");

    // let ref_text = "声音,是有温度的.夜晚的声音,会发光.";
    let ref_text = "在参加rust春晚的时候，听到有人问了这样一个问题.";

    let ref_path = "../python/GPT-SoVITS/onnx/xww/ref.wav";
    let file = std::fs::File::open(ref_path).unwrap();

    let (head, ref_audio_samples) = wav_io::read_from_file(file).unwrap();

    log::info!("load ref done");

    let text = std::fs::read_to_string("./input.txt").unwrap();

    let text_splitter = text_splitter::TextSplitter::new(50);
    let header = wav_io::new_header(32000, 16, false, true);
    let mut audios = vec![];

    for target_text in text_splitter.chunks(&text) {
        println!("text: {}", target_text);
        if target_text == "。" {
            continue;
        }

        let audio = gpt_sovits
            .infer(
                &ref_audio_samples,
                head.sample_rate as usize,
                ref_text,
                target_text,
            )
            .unwrap();

        audios.push(audio);
    }

    log::info!("start write file");

    let output = "out.wav";

    let audio = Tensor::cat(&audios, 0);

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
