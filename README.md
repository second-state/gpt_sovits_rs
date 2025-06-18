# GPT-SoVITS-rs

## Overview
This Rust project provides a library for integrating [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) inference in Rust applications.  GPT-Sovits is a powerful tool for speech synthesis and voice conversion.  By using this library, developers can easily leverage the capabilities of GPT-Sovits within their Rust projects.

## Prerequisites
Before you can use the GPT-SoVits-rs Inference Library, you must ensure that libtorch 2.4.0 is installed on your system. libtorch is the C++ frontend for PyTorch, which is required for running the GPT-Sovits models.

You can download and install libtorch 2.4.0 from the official PyTorch website:

Go to the [PyTorch website](https://pytorch.org/).
Select the appropriate options for your system (OS, Package Manager, Python version, etc.), making sure to choose the "LibTorch" option.
Scroll down to the "Install" section and run the provided command to download and install libtorch 2.4.0.

For example, on a Linux system with CUDA support, the command might look like this:

```bash
wget https://download.pytorch.org/libtorch/cu124/libtorch-cxx11-abi-shared-with-deps-2.4.0%2Bcu124.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.4.0+cu124.zip
```

After downloading and extracting the library, you may need to set environment variables to include the libtorch library path. For example:
```bash
export LIBTORCH=/path/to/libtorch
export LD_LIBRARY_PATH=$LIBTORCH/lib:$LD_LIBRARY_PATH
```
Replace /path/to/libtorch with the actual path where you extracted libtorch.

## Installation
To use this library, add the following dependency to your `Cargo.toml` file:
```toml
[dependencies]
gpt_sovits_rs = "0.7.0"
```
Replace "0.7.0" with the latest version of the library.

## Usage

### Download Model
You can download all common model from my Hugging Face [repo](https://huggingface.co/L-jasmine/GPT_Sovits/tree/main)
* [SSL And Bert](https://huggingface.co/L-jasmine/GPT_Sovits/resolve/main/resource.zip)
* [Base T2S (v2pro)](https://huggingface.co/L-jasmine/GPT_Sovits/resolve/main/v2pro/t2s.pt)
* [Base VITS (v2pro)](https://huggingface.co/L-jasmine/GPT_Sovits/resolve/main/v2pro/vits.pt)
* [g2pw](https://huggingface.co/L-jasmine/GPT_Sovits/resolve/main/g2pw.pt)
* [mini-bert-g2p](https://huggingface.co/L-jasmine/GPT_Sovits/resolve/main/mini-bart-g2p.pt)

### CODE
```rust
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

```

### Turn on Japanese support

> [!NOTE]
> Currently the text frontend could only parse kana as Japanese. However, the Japanese g2p should support kanji as well. You might want to fork this repo if your use case have no Chinese input.

```toml
# Cargo.toml
...
gpt_sovits_rs = { version = "*", features = ["enable_jp"] }
...
```
```rust
use gpt_sovits_rs::{gsv, text::G2PConfig};

fn main() {
    env_logger::init();

   let g2p_conf = G2PConfig::new("./resource/mini-bart-g2p.pt".to_string())
    .with_chinese(
        "path/to/g2pw.pt".to_string(),
        "path/to/bert_model.pt".to_string(),
    )
    .with_jp(true);

    // init gpt_sovits with config
    // ...
}
```

## Exporting GPT-Sovits Training Results
After completing the training of a GPT-Sovits model, you might need to export the training results to a .pt (PyTorch) file for use in other environments. Below are the detailed steps to export the trained model:

#### Step 1: Confirm Training Completion
Ensure that your GPT-Sovits model has finished training and that you have a trained model file available.

#### Step 2: Run the Export Script
Use the following command to run the export script and export the training results to vits.pt and t2s.pt:
```bash
# Change the corresponding file paths in the script for export.
python GPT_SoVITS/export_torch_script.py
```

Now you can find `vits.pt`, `t2s.pt`, in the `output` directory.

The `ssl_model.pt` and `bert_model.pt` are common model files, determined by the option `--export_common_model` for whether to export, and they are not related to the trained model. Therefore, they do not need to be exported every time. 
If you do not wish to export, you can go to Hugging Face to [download](https://huggingface.co/L-jasmine/GPT_Sovits/tree/main) the resource.zip that I have already exported. And remove the `--export_common_model` when export model.

You can download `g2pw.pt` from my Hugging Face [repo](https://huggingface.co/L-jasmine/GPT_Sovits/tree/main)

#### Warn
Currently, only CUDA is supported. MPS and CPU will be compatible later.
