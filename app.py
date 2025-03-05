import os
import numpy as np
import shutil
import warnings
import torch
import yaml
import time
import torchaudio
import librosa
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Union
import tempfile
import uuid
from pathlib import Path
import uvicorn

# Suppress warnings
warnings.simplefilter('ignore')

# Import the required modules
os.environ['HF_HUB_CACHE'] = './checkpoints/hf_cache'
from modules.commons import *
from modules.commons import str2bool
from hf_utils import load_custom_model_from_hf

import torch.multiprocessing as mp
from fastapi.concurrency import BackgroundTasks
import threading

# Global lock for GPU operations
gpu_lock = threading.Lock()

# Initialize FastAPI app
app = FastAPI(
    title="Voice Conversion API", 
    description="API for converting voice characteristics from a source audio to match a target voice",
    version="1.0.0"
)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Global variables to store models
models = None
device = None

# Pydantic models for API requests
class VoiceConversionRequest(BaseModel):
    diffusion_steps: int = 30
    length_adjust: float = 1.0
    inference_cfg_rate: float = 0.7
    f0_condition: bool = False
    auto_f0_adjust: bool = False
    semi_tone_shift: int = 0
    fp16: bool = True

# Helper functions
def adjust_f0_semitones(f0_sequence, n_semitones):
    factor = 2 ** (n_semitones / 12)
    return f0_sequence * factor

def crossfade(chunk1, chunk2, overlap):
    fade_out = np.cos(np.linspace(0, np.pi / 2, overlap)) ** 2
    fade_in = np.cos(np.linspace(np.pi / 2, 0, overlap)) ** 2
    if len(chunk2) < overlap:
        chunk2[:overlap] = chunk2[:overlap] * fade_in[:len(chunk2)] + (chunk1[-overlap:] * fade_out)[:len(chunk2)]
    else:
        chunk2[:overlap] = chunk2[:overlap] * fade_in + chunk1[-overlap:] * fade_out
    return chunk2

# Load models function
def load_models(f0_condition=False, checkpoint_path=None, config_path=None, fp16=True):
    global device
    
    # Set device and enable eager mode to avoid inference tensor issues
    torch._C._jit_set_profiling_executor(False)
    torch._C._jit_set_profiling_mode(False)
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    class Args:
        def __init__(self, f0_condition, checkpoint_path, config_path, fp16):
            self.f0_condition = f0_condition
            self.checkpoint_path = checkpoint_path
            self.config_path = config_path
            self.fp16 = fp16
    
    args = Args(f0_condition, checkpoint_path, config_path, fp16)
    
    # Original load_models implementation
    if not args.f0_condition:
        dit_checkpoint_path, dit_config_path = load_custom_model_from_hf(
            "Plachta/Seed-VC",
            "DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth",
            "config_dit_mel_seed_uvit_whisper_small_wavenet.yml"
        )
        f0_fn = None
    else:
        if args.checkpoint_path is None:
            dit_checkpoint_path, dit_config_path = load_custom_model_from_hf(
                "Plachta/Seed-VC",
                "DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth",
                "config_dit_mel_seed_uvit_whisper_base_f0_44k.yml"
            )
        else:
            dit_checkpoint_path = args.checkpoint_path
            dit_config_path = args.config_path
        # f0 extractor
        from modules.rmvpe import RMVPE

        model_path = load_custom_model_from_hf("lj1995/VoiceConversionWebUI", "rmvpe.pt", None)
        f0_extractor = RMVPE(model_path, is_half=False, device=device)
        f0_fn = f0_extractor.infer_from_audio

    config = yaml.safe_load(open(dit_config_path, "r"))
    model_params = recursive_munch(config["model_params"])
    model_params.dit_type = 'DiT'
    model = build_model(model_params, stage="DiT")
    hop_length = config["preprocess_params"]["spect_params"]["hop_length"]
    sr = config["preprocess_params"]["sr"]

    # Load checkpoints
    model, _, _, _ = load_checkpoint(
        model,
        None,
        dit_checkpoint_path,
        load_only_params=True,
        ignore_modules=[],
        is_distributed=False,
    )
    for key in model:
        model[key].eval()
        model[key].to(device)
    model.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)

    # Load additional modules
    from modules.campplus.DTDNN import CAMPPlus

    campplus_ckpt_path = load_custom_model_from_hf(
        "funasr/campplus", "campplus_cn_common.bin", config_filename=None
    )
    campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
    campplus_model.load_state_dict(torch.load(campplus_ckpt_path, map_location="cpu"))
    campplus_model.eval()
    campplus_model.to(device)

    vocoder_type = model_params.vocoder.type

    if vocoder_type == 'bigvgan':
        from modules.bigvgan import bigvgan
        bigvgan_name = model_params.vocoder.name
        bigvgan_model = bigvgan.BigVGAN.from_pretrained(bigvgan_name, use_cuda_kernel=False)
        # remove weight norm in the model and set to eval mode
        bigvgan_model.remove_weight_norm()
        bigvgan_model = bigvgan_model.eval().to(device)
        vocoder_fn = bigvgan_model
    elif vocoder_type == 'hifigan':
        from modules.hifigan.generator import HiFTGenerator
        from modules.hifigan.f0_predictor import ConvRNNF0Predictor
        hift_config = yaml.safe_load(open('configs/hifigan.yml', 'r'))
        hift_gen = HiFTGenerator(**hift_config['hift'], f0_predictor=ConvRNNF0Predictor(**hift_config['f0_predictor']))
        hift_path = load_custom_model_from_hf("FunAudioLLM/CosyVoice-300M", 'hift.pt', None)
        hift_gen.load_state_dict(torch.load(hift_path, map_location='cpu'))
        hift_gen.eval()
        hift_gen.to(device)
        vocoder_fn = hift_gen
    elif vocoder_type == "vocos":
        vocos_config = yaml.safe_load(open(model_params.vocoder.vocos.config, 'r'))
        vocos_path = model_params.vocoder.vocos.path
        vocos_model_params = recursive_munch(vocos_config['model_params'])
        vocos = build_model(vocos_model_params, stage='mel_vocos')
        vocos_checkpoint_path = vocos_path
        vocos, _, _, _ = load_checkpoint(
            vocos, None, vocos_checkpoint_path,
            load_only_params=True, ignore_modules=[], is_distributed=False
        )
        _ = [vocos[key].eval().to(device) for key in vocos]
        _ = [vocos[key].to(device) for key in vocos]
        total_params = sum(sum(p.numel() for p in vocos[key].parameters() if p.requires_grad) for key in vocos.keys())
        print(f"Vocoder model total parameters: {total_params / 1_000_000:.2f}M")
        vocoder_fn = vocos.decoder
    else:
        raise ValueError(f"Unknown vocoder type: {vocoder_type}")

    speech_tokenizer_type = model_params.speech_tokenizer.type
    if speech_tokenizer_type == 'whisper':
        # whisper
        from transformers import AutoFeatureExtractor, WhisperModel
        whisper_name = model_params.speech_tokenizer.name
        whisper_model = WhisperModel.from_pretrained(whisper_name, torch_dtype=torch.float16).to(device)
        del whisper_model.decoder
        whisper_feature_extractor = AutoFeatureExtractor.from_pretrained(whisper_name)

        def semantic_fn(waves_16k):
            ori_inputs = whisper_feature_extractor(
                [waves_16k.squeeze(0).cpu().numpy()],
                return_tensors="pt",
                return_attention_mask=True
            )
            ori_input_features = whisper_model._mask_input_features(
                ori_inputs.input_features, attention_mask=ori_inputs.attention_mask
            ).to(device)
            with torch.no_grad():
                ori_outputs = whisper_model.encoder(
                    ori_input_features.to(whisper_model.encoder.dtype),
                    head_mask=None,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                )
            S_ori = ori_outputs.last_hidden_state.to(torch.float32)
            S_ori = S_ori[:, :waves_16k.size(-1) // 320 + 1]
            return S_ori
    elif speech_tokenizer_type == 'cnhubert':
        from transformers import (
            Wav2Vec2FeatureExtractor,
            HubertModel,
        )
        hubert_model_name = config['model_params']['speech_tokenizer']['name']
        hubert_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(hubert_model_name)
        hubert_model = HubertModel.from_pretrained(hubert_model_name)
        hubert_model = hubert_model.to(device)
        hubert_model = hubert_model.eval()
        hubert_model = hubert_model.half()

        def semantic_fn(waves_16k):
            ori_waves_16k_input_list = [
                waves_16k[bib].cpu().numpy()
                for bib in range(len(waves_16k))
            ]
            ori_inputs = hubert_feature_extractor(
                ori_waves_16k_input_list,
                return_tensors="pt",
                return_attention_mask=True,
                padding=True,
                sampling_rate=16000
            ).to(device)
            with torch.no_grad():
                ori_outputs = hubert_model(
                    ori_inputs.input_values.half(),
                )
            S_ori = ori_outputs.last_hidden_state.float()
            return S_ori
    elif speech_tokenizer_type == 'xlsr':
        from transformers import (
            Wav2Vec2FeatureExtractor,
            Wav2Vec2Model,
        )
        model_name = config['model_params']['speech_tokenizer']['name']
        output_layer = config['model_params']['speech_tokenizer']['output_layer']
        wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        wav2vec_model = Wav2Vec2Model.from_pretrained(model_name)
        wav2vec_model.encoder.layers = wav2vec_model.encoder.layers[:output_layer]
        wav2vec_model = wav2vec_model.to(device)
        wav2vec_model = wav2vec_model.eval()
        wav2vec_model = wav2vec_model.half()

        def semantic_fn(waves_16k):
            ori_waves_16k_input_list = [
                waves_16k[bib].cpu().numpy()
                for bib in range(len(waves_16k))
            ]
            ori_inputs = wav2vec_feature_extractor(
                ori_waves_16k_input_list,
                return_tensors="pt",
                return_attention_mask=True,
                padding=True,
                sampling_rate=16000
            ).to(device)
            with torch.no_grad():
                ori_outputs = wav2vec_model(
                    ori_inputs.input_values.half(),
                )
            S_ori = ori_outputs.last_hidden_state.float()
            return S_ori
    else:
        raise ValueError(f"Unknown speech tokenizer type: {speech_tokenizer_type}")
    
    # Generate mel spectrograms
    mel_fn_args = {
        "n_fft": config['preprocess_params']['spect_params']['n_fft'],
        "win_size": config['preprocess_params']['spect_params']['win_length'],
        "hop_size": config['preprocess_params']['spect_params']['hop_length'],
        "num_mels": config['preprocess_params']['spect_params']['n_mels'],
        "sampling_rate": sr,
        "fmin": config['preprocess_params']['spect_params'].get('fmin', 0),
        "fmax": None if config['preprocess_params']['spect_params'].get('fmax', "None") == "None" else 8000,
        "center": False
    }
    from modules.audio import mel_spectrogram

    to_mel = lambda x: mel_spectrogram(x, **mel_fn_args)

    return (
        model,
        semantic_fn,
        f0_fn,
        vocoder_fn,
        campplus_model,
        to_mel,
        mel_fn_args,
    )

# Voice conversion function
async def process_voice_conversion(source_path, target_path, output_path, params):
    global models, device
    
    # Acquire GPU lock to prevent multiple concurrent GPU operations
    with gpu_lock:
        if models is None:
            raise ValueError("Models not loaded")
            
        model, semantic_fn, f0_fn, vocoder_fn, campplus_model, mel_fn, mel_fn_args = models
        
        # Check that f0_fn is available if f0_condition is True
        if params.f0_condition and f0_fn is None:
            raise ValueError("f0_condition is True but f0 extractor was not loaded. Please reload models with f0_condition=True")
        
        sr = mel_fn_args['sampling_rate']
        f0_condition = params.f0_condition
        auto_f0_adjust = params.auto_f0_adjust
        pitch_shift = params.semi_tone_shift
        diffusion_steps = params.diffusion_steps
        length_adjust = params.length_adjust
        inference_cfg_rate = params.inference_cfg_rate
        
        print(f"Loading audio files with sampling rate {sr}")
        # Load audio files
        try:
            source_audio = librosa.load(source_path, sr=sr)[0]
            print(f"Source audio loaded, length: {len(source_audio)}")
            ref_audio = librosa.load(target_path, sr=sr)[0]
            print(f"Target audio loaded, length: {len(ref_audio)}")
        except Exception as e:
            print(f"Error loading audio files: {e}")
            raise

    sr = 22050 if not f0_condition else 44100
    hop_length = 256 if not f0_condition else 512
    max_context_window = sr // hop_length * 30
    overlap_frame_len = 16
    overlap_wave_len = overlap_frame_len * hop_length

    # Process audio
    with torch.no_grad():
        source_audio = torch.tensor(source_audio).unsqueeze(0).float().to(device)
        ref_audio = torch.tensor(ref_audio[:sr * 25]).unsqueeze(0).float().to(device)

    time_vc_start = time.time()
    # Resample
    with torch.no_grad():
        converted_waves_16k = torchaudio.functional.resample(source_audio, sr, 16000)
    # if source audio less than 30 seconds, whisper can handle in one forward
    with torch.no_grad():
        if converted_waves_16k.size(-1) <= 16000 * 30:
            S_alt = semantic_fn(converted_waves_16k)
        else:
            overlapping_time = 5  # 5 seconds
            S_alt_list = []
            buffer = None
            traversed_time = 0
            while traversed_time < converted_waves_16k.size(-1):
                if buffer is None:  # first chunk
                    chunk = converted_waves_16k[:, traversed_time:traversed_time + 16000 * 30]
                else:
                    chunk = torch.cat(
                        [buffer, converted_waves_16k[:, traversed_time:traversed_time + 16000 * (30 - overlapping_time)]],
                        dim=-1)
                S_alt_chunk = semantic_fn(chunk)
                if traversed_time == 0:
                    S_alt_list.append(S_alt_chunk)
                else:
                    S_alt_list.append(S_alt_chunk[:, 50 * overlapping_time:])
                buffer = chunk[:, -16000 * overlapping_time:]
                traversed_time += 30 * 16000 if traversed_time == 0 else chunk.size(-1) - 16000 * overlapping_time
            S_alt = torch.cat(S_alt_list, dim=1)

        ori_waves_16k = torchaudio.functional.resample(ref_audio, sr, 16000)
        S_ori = semantic_fn(ori_waves_16k)

    with torch.no_grad():
        mel = mel_fn(source_audio.to(device).float())
        mel2 = mel_fn(ref_audio.to(device).float())

        target_lengths = torch.LongTensor([int(mel.size(2) * length_adjust)]).to(mel.device)
        target2_lengths = torch.LongTensor([mel2.size(2)]).to(mel2.device)

        feat2 = torchaudio.compliance.kaldi.fbank(ori_waves_16k,
                                              num_mel_bins=80,
                                              dither=0,
                                              sample_frequency=16000)
        feat2 = feat2 - feat2.mean(dim=0, keepdim=True)
        style2 = campplus_model(feat2.unsqueeze(0))

    if f0_condition:
        with torch.no_grad():
            if f0_fn is None:
                raise ValueError("f0_condition is True but f0_fn is None. Please reload models with f0_condition=True")
                
            print("Extracting F0 features...")
            F0_ori = f0_fn(ori_waves_16k[0], thred=0.03)
            F0_alt = f0_fn(converted_waves_16k[0], thred=0.03)

            F0_ori = torch.from_numpy(F0_ori).to(device)[None]
            F0_alt = torch.from_numpy(F0_alt).to(device)[None]

            voiced_F0_ori = F0_ori[F0_ori > 1]
            voiced_F0_alt = F0_alt[F0_alt > 1]

            log_f0_alt = torch.log(F0_alt + 1e-5)
            voiced_log_f0_ori = torch.log(voiced_F0_ori + 1e-5)
            voiced_log_f0_alt = torch.log(voiced_F0_alt + 1e-5)
            
            if len(voiced_log_f0_ori) == 0 or len(voiced_log_f0_alt) == 0:
                print("Warning: No voiced frames detected. Using default F0 values.")
                median_log_f0_ori = torch.tensor(5.0).to(device)
                median_log_f0_alt = torch.tensor(5.0).to(device)
            else:
                median_log_f0_ori = torch.median(voiced_log_f0_ori)
                median_log_f0_alt = torch.median(voiced_log_f0_alt)

            # shift alt log f0 level to ori log f0 level
            shifted_log_f0_alt = log_f0_alt.clone()
            if auto_f0_adjust:
                shifted_log_f0_alt[F0_alt > 1] = log_f0_alt[F0_alt > 1] - median_log_f0_alt + median_log_f0_ori
            shifted_f0_alt = torch.exp(shifted_log_f0_alt)
            if pitch_shift != 0:
                shifted_f0_alt[F0_alt > 1] = adjust_f0_semitones(shifted_f0_alt[F0_alt > 1], pitch_shift)
            print("F0 features extracted successfully")
    else:
        F0_ori = None
        F0_alt = None
        shifted_f0_alt = None

    # Length regulation
    with torch.no_grad():
        cond, _, codes, commitment_loss, codebook_loss = model.length_regulator(
            S_alt, 
            ylens=target_lengths,
            n_quantizers=3,
            f0=shifted_f0_alt
        )
        
        prompt_condition, _, codes, commitment_loss, codebook_loss = model.length_regulator(
            S_ori,
            ylens=target2_lengths,
            n_quantizers=3,
            f0=F0_ori
        )

    max_source_window = max_context_window - mel2.size(2)
    # split source condition (cond) into chunks
    processed_frames = 0
    generated_wave_chunks = []
    # generate chunk by chunk and stream the output
    while processed_frames < cond.size(1):
        with torch.no_grad():
            chunk_cond = cond[:, processed_frames:processed_frames + max_source_window]
            is_last_chunk = processed_frames + max_source_window >= cond.size(1)
            cat_condition = torch.cat([prompt_condition, chunk_cond], dim=1)
            cat_len_tensor = torch.LongTensor([cat_condition.size(1)]).to(mel2.device)
            
            # Clone tensors before using them in autocast context to avoid inference tensor issues
            cat_condition_clone = cat_condition.clone().detach()
            cat_len_tensor_clone = cat_len_tensor.clone().detach()
            mel2_clone = mel2.clone().detach()
            style2_clone = style2.clone().detach()
            
        with torch.autocast(device_type=device.type, dtype=torch.float16 if params.fp16 else torch.float32):
            with torch.no_grad():
                # Voice Conversion
                vc_target = model.cfm.inference(
                    cat_condition_clone,
                    cat_len_tensor_clone,
                    mel2_clone, style2_clone, None, diffusion_steps,
                    inference_cfg_rate=inference_cfg_rate
                )
                vc_target = vc_target[:, :, mel2.size(-1):]
                vc_wave = vocoder_fn(vc_target.float()).squeeze()
                vc_wave = vc_wave[None, :]
        with torch.no_grad():
            if processed_frames == 0:
                if is_last_chunk:
                    output_wave = vc_wave[0].cpu().numpy()
                    generated_wave_chunks.append(output_wave)
                    break
                output_wave = vc_wave[0, :-overlap_wave_len].cpu().numpy()
                generated_wave_chunks.append(output_wave)
                previous_chunk = vc_wave[0, -overlap_wave_len:].cpu()
                processed_frames += vc_target.size(2) - overlap_frame_len
            elif is_last_chunk:
                output_wave = crossfade(previous_chunk.numpy(), vc_wave[0].cpu().numpy(), overlap_wave_len)
                generated_wave_chunks.append(output_wave)
                processed_frames += vc_target.size(2) - overlap_frame_len
                break
            else:
                output_wave = crossfade(previous_chunk.numpy(), vc_wave[0, :-overlap_wave_len].cpu().numpy(),
                                    overlap_wave_len)
                generated_wave_chunks.append(output_wave)
                previous_chunk = vc_wave[0, -overlap_wave_len:].cpu()
                processed_frames += vc_target.size(2) - overlap_frame_len
    
    vc_wave = torch.tensor(np.concatenate(generated_wave_chunks))[None, :].float()
    time_vc_end = time.time()
    rtf = (time_vc_end - time_vc_start) / vc_wave.size(-1) * sr
    
    # Clean up GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    # Save the output file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torchaudio.save(output_path, vc_wave.cpu(), sr)
    
    return {"output_path": output_path, "rtf": rtf}

# Cleanup function to remove temporary files
def cleanup_temp_files(file_paths):
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Cleaned up temporary file: {file_path}")
        except Exception as e:
            print(f"Error cleaning up {file_path}: {e}")
            
# Add a new endpoint to handle direct binary data if needed
@app.post("/convert_voice_raw/")
async def convert_voice_raw(
    background_tasks: BackgroundTasks,
    request: Request
):
    """
    Alternative endpoint that accepts raw binary data for file uploads
    This can be useful for some frontend frameworks that have trouble with multipart/form-data
    """
    try:
        # Create temp directory for files
        temp_dir = Path("./temp")
        temp_dir.mkdir(exist_ok=True)
        
        # Generate unique filenames
        unique_id = uuid.uuid4().hex
        source_path = temp_dir / f"source_{unique_id}.wav"
        target_path = temp_dir / f"target_{unique_id}.wav"
        output_path = temp_dir / f"output_{unique_id}.wav"
        
        # Parse the request body
        body = await request.json()
        
        # Get parameters
        params_dict = body.get("params", {})
        params = VoiceConversionRequest(
            diffusion_steps=params_dict.get("diffusion_steps", 30),
            length_adjust=params_dict.get("length_adjust", 1.0),
            inference_cfg_rate=params_dict.get("inference_cfg_rate", 0.7),
            f0_condition=params_dict.get("f0_condition", False),
            auto_f0_adjust=params_dict.get("auto_f0_adjust", False),
            semi_tone_shift=params_dict.get("semi_tone_shift", 0),
            fp16=params_dict.get("fp16", True)
        )
        
        # Get and decode base64 audio data
        import base64
        source_data = base64.b64decode(body.get("source_audio", ""))
        target_data = base64.b64decode(body.get("target_audio", ""))
        
        if not source_data or not target_data:
            raise HTTPException(status_code=400, detail="Source or target audio data is missing or empty")
        
        # Save audio data to temp files
        with open(source_path, "wb") as f:
            f.write(source_data)
        
        with open(target_path, "wb") as f:
            f.write(target_data)
        
        # Process voice conversion
        result = await process_voice_conversion(
            str(source_path),
            str(target_path),
            str(output_path),
            params
        )
        
        # Schedule cleanup of temporary files
        background_tasks.add_task(
            cleanup_temp_files, 
            [str(source_path), str(target_path)]
        )
        
        # Read the output file and return as base64
        with open(output_path, "rb") as f:
            output_data = f.read()
        
        # Schedule cleanup of output file
        background_tasks.add_task(
            cleanup_temp_files, 
            [str(output_path)]
        )
        
        # Return the base64 encoded output
        return {
            "converted_audio": base64.b64encode(output_data).decode("utf-8")
        }
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error during raw voice conversion: {str(e)}\n{error_details}")
        
        # Clean up files in case of error
        cleanup_temp_files([str(source_path), str(target_path), str(output_path)])
        raise HTTPException(status_code=500, detail=f"Error during voice conversion: {str(e)}")

# API endpoints
@app.on_event("startup")
async def startup_event():
    """Load models on startup with both f0_condition options"""
    global models
    print("Loading models...")
    try:
        # Load models with f0_condition=True for singing voice conversion
        models = load_models(f0_condition=True)
        print("Models loaded successfully with f0_condition=True")
    except Exception as e:
        print(f"Error loading models: {e}")
        raise

@app.get("/")
async def root():
    """Root endpoint to check if the API is running"""
    return {"message": "Voice Conversion API is running"}

@app.post("/convert_voice/")
async def convert_voice(
    background_tasks: BackgroundTasks,
    source_file: Optional[UploadFile] = File(None),
    target_file: Optional[UploadFile] = File(None),
    source_url: Optional[str] = Form(None),
    target_url: Optional[str] = Form(None),
    diffusion_steps: Union[int, None] = Form(30),
    length_adjust: Union[float, None] = Form(1.0),
    inference_cfg_rate: Union[float, None] = Form(0.7),
    f0_condition: Union[bool, None] = Form(True),  # Changed default to True
    auto_f0_adjust: Union[bool, None] = Form(False),
    semi_tone_shift: Union[int, None] = Form(0),
    fp16: Union[bool, None] = Form(True)
):
    """
    Convert voice from source audio to match target voice characteristics
    
    - **source_file**: The source audio file upload (optional if source_url is provided)
    - **target_file**: The target voice reference file upload (optional if target_url is provided)
    - **source_url**: URL to the source audio file (optional if source_file is provided)
    - **target_url**: URL to the target voice reference file (optional if target_file is provided)
    - **diffusion_steps**: Number of diffusion steps (default: 30)
    - **length_adjust**: Length adjustment factor (default: 1.0)
    - **inference_cfg_rate**: Inference configuration rate (default: 0.7)
    - **f0_condition**: Whether to use f0 conditioning (default: True)
    - **auto_f0_adjust**: Auto-adjust f0 (default: False)
    - **semi_tone_shift**: Semitone shift for pitch (default: 0)
    - **fp16**: Use FP16 precision (default: True)
    """
    # Create temp directory for files
    temp_dir = Path("./temp")
    temp_dir.mkdir(exist_ok=True)
    
    # Generate unique filenames
    unique_id = uuid.uuid4().hex
    source_path = temp_dir / f"source_{unique_id}.wav"
    target_path = temp_dir / f"target_{unique_id}.wav"
    output_path = temp_dir / f"output_{unique_id}.wav"
    
    files_to_cleanup = []
    
    try:
        # Process boolean values properly
        f0_condition = str(f0_condition).lower() == 'true' if isinstance(f0_condition, str) else bool(f0_condition)
        auto_f0_adjust = str(auto_f0_adjust).lower() == 'true' if isinstance(auto_f0_adjust, str) else bool(auto_f0_adjust)
        fp16 = str(fp16).lower() == 'true' if isinstance(fp16, str) else bool(fp16)
        
        # Convert other parameters to correct types
        diffusion_steps = int(diffusion_steps) if diffusion_steps is not None else 30
        length_adjust = float(length_adjust) if length_adjust is not None else 1.0
        inference_cfg_rate = float(inference_cfg_rate) if inference_cfg_rate is not None else 0.7
        semi_tone_shift = int(semi_tone_shift) if semi_tone_shift is not None else 0
        
        # Handle file uploads or URLs
        source_content = None
        target_content = None
        
        # Process source file
        if source_file:
            source_content = await source_file.read()
            print(f"Source file: {source_file.filename}, size: {len(source_content)} bytes")
            with open(source_path, "wb") as buffer:
                buffer.write(source_content)
            files_to_cleanup.append(str(source_path))
        elif source_url:
            print(f"Downloading source from URL: {source_url}")
            import requests
            response = requests.get(source_url)
            response.raise_for_status()
            source_content = response.content
            print(f"Source from URL, size: {len(source_content)} bytes")
            with open(source_path, "wb") as buffer:
                buffer.write(source_content)
            files_to_cleanup.append(str(source_path))
        else:
            raise HTTPException(status_code=400, detail="Either source_file or source_url must be provided")
        
        # Process target file
        if target_file:
            target_content = await target_file.read()
            print(f"Target file: {target_file.filename}, size: {len(target_content)} bytes")
            with open(target_path, "wb") as buffer:
                buffer.write(target_content)
            files_to_cleanup.append(str(target_path))
        elif target_url:
            print(f"Downloading target from URL: {target_url}")
            import requests
            response = requests.get(target_url)
            response.raise_for_status()
            target_content = response.content
            print(f"Target from URL, size: {len(target_content)} bytes")
            with open(target_path, "wb") as buffer:
                buffer.write(target_content)
            files_to_cleanup.append(str(target_path))
        else:
            raise HTTPException(status_code=400, detail="Either target_file or target_url must be provided")
        
        if not os.path.exists(source_path) or not os.path.getsize(source_path):
            raise HTTPException(status_code=400, detail="Source file is empty or could not be created")
            
        if not os.path.exists(target_path) or not os.path.getsize(target_path):
            raise HTTPException(status_code=400, detail="Target file is empty or could not be created")
        
        # Create params object
        params = VoiceConversionRequest(
            diffusion_steps=diffusion_steps,
            length_adjust=length_adjust,
            inference_cfg_rate=inference_cfg_rate,
            f0_condition=f0_condition,
            auto_f0_adjust=auto_f0_adjust,
            semi_tone_shift=semi_tone_shift,
            fp16=fp16
        )
        
        print(f"Conversion parameters: {params}")
        
        # Process voice conversion
        result = await process_voice_conversion(
            str(source_path),
            str(target_path),
            str(output_path),
            params
        )
        
        # Schedule cleanup of temporary files
        background_tasks.add_task(
            cleanup_temp_files, 
            files_to_cleanup
        )
        
        # Check if output file exists
        if not os.path.exists(output_path):
            raise HTTPException(status_code=500, detail="Output file was not generated")
        
        print(f"Conversion complete. Output file size: {os.path.getsize(output_path)} bytes")
        
        # Return the output file with proper CORS headers
        return FileResponse(
            path=str(output_path),
            media_type="audio/wav",
            filename=f"converted_voice_{unique_id}.wav",
            background=background_tasks.add_task(cleanup_temp_files, [str(output_path)])
        )
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error during voice conversion: {str(e)}\n{error_details}")
        
        # Clean up files in case of error
        cleanup_temp_files(files_to_cleanup + [str(output_path)])
        raise HTTPException(status_code=500, detail=f"Error during voice conversion: {str(e)}")

@app.post("/load_custom_models/")
async def load_custom_models(
    f0_condition: bool = Form(False),
    checkpoint_path: Optional[str] = Form(None),
    config_path: Optional[str] = Form(None),
    fp16: bool = Form(True)
):
    """
    Load custom models with specific configuration
    
    - **f0_condition**: Whether to use f0 conditioning
    - **checkpoint_path**: Path to custom checkpoint
    - **config_path**: Path to custom config
    - **fp16**: Use FP16 precision
    """
    global models
    
    try:
        models = load_models(
            f0_condition=f0_condition,
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            fp16=fp16
        )
        return {"message": "Custom models loaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading custom models: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if models is None:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "message": "Models not loaded"}
        )
    return {"status": "healthy", "message": "API is running and models are loaded"}

# Run the application
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
