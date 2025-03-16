from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
import shutil
import os
import tempfile
import numpy as np
import uuid
import torch
import torchaudio
import librosa
from modules.commons import build_model, load_checkpoint, recursive_munch
import yaml
from hf_utils import load_custom_model_from_hf
from pydub import AudioSegment

# Create a FastAPI app
app = FastAPI(title="Voice Conversion API")

# Create a temporary directory for storing uploaded and processed files
TEMP_DIR = tempfile.mkdtemp()

# Device selection
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Load models and configurations
# DiT model without F0 conditioning
dit_checkpoint_path, dit_config_path = load_custom_model_from_hf(
    "Plachta/Seed-VC",
    "DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth",
    "config_dit_mel_seed_uvit_whisper_small_wavenet.yml"
)
config = yaml.safe_load(open(dit_config_path, 'r'))
model_params = recursive_munch(config['model_params'])
model = build_model(model_params, stage='DiT')
hop_length = config['preprocess_params']['spect_params']['hop_length']
sr = config['preprocess_params']['sr']

# Load checkpoints
model, _, _, _ = load_checkpoint(model, None, dit_checkpoint_path,
                                load_only_params=True, ignore_modules=[], is_distributed=False)
for key in model:
    model[key].eval()
    model[key].to(device)
model.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)

# Load CAMP+ model
from modules.campplus.DTDNN import CAMPPlus

campplus_ckpt_path = load_custom_model_from_hf("funasr/campplus", "campplus_cn_common.bin", config_filename=None)
campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
campplus_model.load_state_dict(torch.load(campplus_ckpt_path, map_location="cpu"))
campplus_model.eval()
campplus_model.to(device)

# Load BigVGAN model
from modules.bigvgan import bigvgan

bigvgan_model = bigvgan.BigVGAN.from_pretrained('nvidia/bigvgan_v2_22khz_80band_256x', use_cuda_kernel=False)
bigvgan_model.remove_weight_norm()
bigvgan_model = bigvgan_model.eval().to(device)

# Load Whisper model
from transformers import AutoFeatureExtractor, WhisperModel

whisper_name = model_params.speech_tokenizer.whisper_name if hasattr(model_params.speech_tokenizer,
                                                                    'whisper_name') else "openai/whisper-small"
whisper_model = WhisperModel.from_pretrained(whisper_name, torch_dtype=torch.float16).to(device)
del whisper_model.decoder
whisper_feature_extractor = AutoFeatureExtractor.from_pretrained(whisper_name)

# Generate mel spectrograms configuration
mel_fn_args = {
    "n_fft": config['preprocess_params']['spect_params']['n_fft'],
    "win_size": config['preprocess_params']['spect_params']['win_length'],
    "hop_size": config['preprocess_params']['spect_params']['hop_length'],
    "num_mels": config['preprocess_params']['spect_params']['n_mels'],
    "sampling_rate": sr,
    "fmin": 0,
    "fmax": None,
    "center": False
}
from modules.audio import mel_spectrogram
to_mel = lambda x: mel_spectrogram(x, **mel_fn_args)

# Load F0-conditioned model
dit_checkpoint_path, dit_config_path = load_custom_model_from_hf(
    "Plachta/Seed-VC",
    "DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth",
    "config_dit_mel_seed_uvit_whisper_base_f0_44k.yml"
)

config = yaml.safe_load(open(dit_config_path, 'r'))
model_params = recursive_munch(config['model_params'])
model_f0 = build_model(model_params, stage='DiT')
hop_length_f0 = config['preprocess_params']['spect_params']['hop_length']
sr_f0 = config['preprocess_params']['sr']

# Load F0 model checkpoints
model_f0, _, _, _ = load_checkpoint(model_f0, None, dit_checkpoint_path,
                                load_only_params=True, ignore_modules=[], is_distributed=False)
for key in model_f0:
    model_f0[key].eval()
    model_f0[key].to(device)
model_f0.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)

# Load F0 extractor
from modules.rmvpe import RMVPE

model_path = load_custom_model_from_hf("lj1995/VoiceConversionWebUI", "rmvpe.pt", None)
rmvpe = RMVPE(model_path, is_half=False, device=device)

# F0 model mel spectrogram configuration
mel_fn_args_f0 = {
    "n_fft": config['preprocess_params']['spect_params']['n_fft'],
    "win_size": config['preprocess_params']['spect_params']['win_length'],
    "hop_size": config['preprocess_params']['spect_params']['hop_length'],
    "num_mels": config['preprocess_params']['spect_params']['n_mels'],
    "sampling_rate": sr_f0,
    "fmin": 0,
    "fmax": None,
    "center": False
}
to_mel_f0 = lambda x: mel_spectrogram(x, **mel_fn_args_f0)

# BigVGAN 44k model for F0 condition
bigvgan_44k_model = bigvgan.BigVGAN.from_pretrained('nvidia/bigvgan_v2_44khz_128band_512x', use_cuda_kernel=False)
bigvgan_44k_model.remove_weight_norm()
bigvgan_44k_model = bigvgan_44k_model.eval().to(device)

# Utility functions
def adjust_f0_semitones(f0_sequence, n_semitones):
    """Adjust F0 values by a number of semitones."""
    factor = 2 ** (n_semitones / 12)
    return f0_sequence * factor

def crossfade(chunk1, chunk2, overlap):
    """Crossfade between two audio chunks."""
    fade_out = np.cos(np.linspace(0, np.pi / 2, overlap)) ** 2
    fade_in = np.cos(np.linspace(np.pi / 2, 0, overlap)) ** 2
    if len(chunk2) < overlap:
        chunk2[:overlap] = chunk2[:overlap] * fade_in[:len(chunk2)] + (chunk1[-overlap:] * fade_out)[:len(chunk2)]
    else:
        chunk2[:overlap] = chunk2[:overlap] * fade_in + chunk1[-overlap:] * fade_out
    return chunk2

# Non-streaming voice conversion function for API
@torch.no_grad()
@torch.inference_mode()
def voice_conversion_api(source, target, diffusion_steps, length_adjust, inference_cfg_rate, f0_condition, auto_f0_adjust, pitch_shift):
    """
    Convert voice from source to target without streaming.
    Returns a tuple of (sample_rate, audio_data)
    """
    # Set model parameters based on f0_condition
    inference_module = model if not f0_condition else model_f0
    mel_fn = to_mel if not f0_condition else to_mel_f0
    bigvgan_fn = bigvgan_model if not f0_condition else bigvgan_44k_model
    sample_rate = 22050 if not f0_condition else 44100
    current_hop_length = hop_length if not f0_condition else hop_length_f0
    
    # Constants
    overlap_frame_len = 16
    max_context_window = sample_rate // current_hop_length * 30
    overlap_wave_len = overlap_frame_len * current_hop_length
    
    # Load audio
    source_audio = librosa.load(source, sr=sample_rate)[0]
    ref_audio = librosa.load(target, sr=sample_rate)[0]

    # Process audio
    source_audio = torch.tensor(source_audio).unsqueeze(0).float().to(device)
    ref_audio = torch.tensor(ref_audio[:sample_rate * 25]).unsqueeze(0).float().to(device)

    # Resample to 16kHz for Whisper
    ref_waves_16k = torchaudio.functional.resample(ref_audio, sample_rate, 16000)
    converted_waves_16k = torchaudio.functional.resample(source_audio, sample_rate, 16000)
    
    # Process with Whisper
    if converted_waves_16k.size(-1) <= 16000 * 30:
        # Single forward pass for shorter audio
        alt_inputs = whisper_feature_extractor([converted_waves_16k.squeeze(0).cpu().numpy()],
                                            return_tensors="pt",
                                            return_attention_mask=True,
                                            sampling_rate=16000)
        alt_input_features = whisper_model._mask_input_features(
            alt_inputs.input_features, attention_mask=alt_inputs.attention_mask).to(device)
        alt_outputs = whisper_model.encoder(
            alt_input_features.to(whisper_model.encoder.dtype),
            head_mask=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        S_alt = alt_outputs.last_hidden_state.to(torch.float32)
        S_alt = S_alt[:, :converted_waves_16k.size(-1) // 320 + 1]
    else:
        # Process longer audio in chunks
        overlapping_time = 5  # 5 seconds
        S_alt_list = []
        buffer = None
        traversed_time = 0
        while traversed_time < converted_waves_16k.size(-1):
            if buffer is None:  # first chunk
                chunk = converted_waves_16k[:, traversed_time:traversed_time + 16000 * 30]
            else:
                chunk = torch.cat([buffer, converted_waves_16k[:, traversed_time:traversed_time + 16000 * (30 - overlapping_time)]], dim=-1)
            alt_inputs = whisper_feature_extractor([chunk.squeeze(0).cpu().numpy()],
                                                return_tensors="pt",
                                                return_attention_mask=True,
                                                sampling_rate=16000)
            alt_input_features = whisper_model._mask_input_features(
                alt_inputs.input_features, attention_mask=alt_inputs.attention_mask).to(device)
            alt_outputs = whisper_model.encoder(
                alt_input_features.to(whisper_model.encoder.dtype),
                head_mask=None,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
            S_alt = alt_outputs.last_hidden_state.to(torch.float32)
            S_alt = S_alt[:, :chunk.size(-1) // 320 + 1]
            if traversed_time == 0:
                S_alt_list.append(S_alt)
            else:
                S_alt_list.append(S_alt[:, 50 * overlapping_time:])
            buffer = chunk[:, -16000 * overlapping_time:]
            traversed_time += 30 * 16000 if traversed_time == 0 else chunk.size(-1) - 16000 * overlapping_time
        S_alt = torch.cat(S_alt_list, dim=1)

    # Process reference audio
    ori_waves_16k = torchaudio.functional.resample(ref_audio, sample_rate, 16000)
    ori_inputs = whisper_feature_extractor([ori_waves_16k.squeeze(0).cpu().numpy()],
                                        return_tensors="pt",
                                        return_attention_mask=True)
    ori_input_features = whisper_model._mask_input_features(
        ori_inputs.input_features, attention_mask=ori_inputs.attention_mask).to(device)
    with torch.no_grad():
        ori_outputs = whisper_model.encoder(
            ori_input_features.to(whisper_model.encoder.dtype),
            head_mask=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
    S_ori = ori_outputs.last_hidden_state.to(torch.float32)
    S_ori = S_ori[:, :ori_waves_16k.size(-1) // 320 + 1]

    # Generate mel spectrograms
    mel = mel_fn(source_audio.to(device).float())
    mel2 = mel_fn(ref_audio.to(device).float())

    # Calculate target lengths
    target_lengths = torch.LongTensor([int(mel.size(2) * length_adjust)]).to(mel.device)
    target2_lengths = torch.LongTensor([mel2.size(2)]).to(mel2.device)

    # Extract speaker embedding for target voice
    feat2 = torchaudio.compliance.kaldi.fbank(ori_waves_16k,
                                            num_mel_bins=80,
                                            dither=0,
                                            sample_frequency=16000)
    feat2 = feat2 - feat2.mean(dim=0, keepdim=True)
    style2 = campplus_model(feat2.unsqueeze(0))

    # Handle F0 conditioning if enabled
    if f0_condition:
        F0_ori = rmvpe.infer_from_audio(ori_waves_16k[0], thred=0.03)
        F0_alt = rmvpe.infer_from_audio(converted_waves_16k[0], thred=0.03)

        if device.type == "mps":
            F0_ori = torch.from_numpy(F0_ori).float().to(device)[None]
            F0_alt = torch.from_numpy(F0_alt).float().to(device)[None]
        else:
            F0_ori = torch.from_numpy(F0_ori).to(device)[None]
            F0_alt = torch.from_numpy(F0_alt).to(device)[None]

        voiced_F0_ori = F0_ori[F0_ori > 1]
        voiced_F0_alt = F0_alt[F0_alt > 1]

        log_f0_alt = torch.log(F0_alt + 1e-5)
        voiced_log_f0_ori = torch.log(voiced_F0_ori + 1e-5)
        voiced_log_f0_alt = torch.log(voiced_F0_alt + 1e-5)
        median_log_f0_ori = torch.median(voiced_log_f0_ori)
        median_log_f0_alt = torch.median(voiced_log_f0_alt)

        # Shift alt log f0 level to ori log f0 level
        shifted_log_f0_alt = log_f0_alt.clone()
        if auto_f0_adjust:
            shifted_log_f0_alt[F0_alt > 1] = log_f0_alt[F0_alt > 1] - median_log_f0_alt + median_log_f0_ori
        shifted_f0_alt = torch.exp(shifted_log_f0_alt)
        if pitch_shift != 0:
            shifted_f0_alt[F0_alt > 1] = adjust_f0_semitones(shifted_f0_alt[F0_alt > 1], pitch_shift)
    else:
        F0_ori = None
        F0_alt = None
        shifted_f0_alt = None

    # Length regulation
    cond, _, _, _, _ = inference_module.length_regulator(S_alt, ylens=target_lengths, n_quantizers=3, f0=shifted_f0_alt)
    prompt_condition, _, _, _, _ = inference_module.length_regulator(S_ori, ylens=target2_lengths, n_quantizers=3, f0=F0_ori)

    # Process audio in chunks and collect the results
    max_source_window = max_context_window - mel2.size(2)
    processed_frames = 0
    generated_wave_chunks = []
    
    # Process chunk by chunk but collect all chunks instead of streaming
    while processed_frames < cond.size(1):
        chunk_cond = cond[:, processed_frames:processed_frames + max_source_window]
        is_last_chunk = processed_frames + max_source_window >= cond.size(1)
        cat_condition = torch.cat([prompt_condition, chunk_cond], dim=1)
        
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            # Voice Conversion
            vc_target = inference_module.cfm.inference(cat_condition,
                                                    torch.LongTensor([cat_condition.size(1)]).to(mel2.device),
                                                    mel2, style2, None, diffusion_steps,
                                                    inference_cfg_rate=inference_cfg_rate)
            vc_target = vc_target[:, :, mel2.size(-1):]
        
        vc_wave = bigvgan_fn(vc_target.float())[0]
        
        if processed_frames == 0:
            if is_last_chunk:
                output_wave = vc_wave[0].cpu().numpy()
                generated_wave_chunks.append(output_wave)
                break
            output_wave = vc_wave[0, :-overlap_wave_len].cpu().numpy()
            generated_wave_chunks.append(output_wave)
            previous_chunk = vc_wave[0, -overlap_wave_len:]
            processed_frames += vc_target.size(2) - overlap_frame_len
        elif is_last_chunk:
            output_wave = crossfade(previous_chunk.cpu().numpy(), vc_wave[0].cpu().numpy(), overlap_wave_len)
            generated_wave_chunks.append(output_wave)
            break
        else:
            output_wave = crossfade(previous_chunk.cpu().numpy(), vc_wave[0, :-overlap_wave_len].cpu().numpy(), overlap_wave_len)
            generated_wave_chunks.append(output_wave)
            previous_chunk = vc_wave[0, -overlap_wave_len:]
            processed_frames += vc_target.size(2) - overlap_frame_len
    
    # Concatenate all audio chunks
    final_audio = np.concatenate(generated_wave_chunks)

    # Clean up GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
            
    return sample_rate, final_audio

# FastAPI endpoints
@app.post("/convert/", summary="Convert voice from source to target")
async def convert_voice(
    background_tasks: BackgroundTasks,
    source_audio: UploadFile = File(..., description="The source audio file to convert"),
    target_audio: UploadFile = File(..., description="The target voice reference audio file"),
    diffusion_steps: int = Form(20, description="Number of diffusion steps (higher = better quality but slower)"),
    length_adjust: float = Form(1.0, description="Adjust output length (1.0 = original length)"),
    inference_cfg_rate: float = Form(0.0, description="Classifier-free guidance rate"),
    f0_condition: bool = Form(False, description="Whether to use F0 conditioning for pitch preservation"),
    auto_f0_adjust: bool = Form(True, description="Automatically adjust F0 to match target voice"),
    pitch_shift: int = Form(0, description="Shift pitch in semitones (positive = higher, negative = lower)")
):
    """
    Convert the voice in source_audio to sound like the voice in target_audio.
    
    Returns a converted audio file in MP3 format.
    """
    # Input validation
    if diffusion_steps < 1 or diffusion_steps > 100:
        raise HTTPException(status_code=400, detail="diffusion_steps must be between 1 and 100")
    
    if length_adjust < 0.5 or length_adjust > 2.0:
        raise HTTPException(status_code=400, detail="length_adjust must be between 0.5 and 2.0")
    
    if pitch_shift < -24 or pitch_shift > 24:
        raise HTTPException(status_code=400, detail="pitch_shift must be between -24 and 24")
    
    # Create unique filenames
    source_path = os.path.join(TEMP_DIR, f"{uuid.uuid4()}{os.path.splitext(source_audio.filename)[1]}")
    target_path = os.path.join(TEMP_DIR, f"{uuid.uuid4()}{os.path.splitext(target_audio.filename)[1]}")
    output_path = os.path.join(TEMP_DIR, f"{uuid.uuid4()}.mp3")
    
    # Save uploaded files
    try:
        with open(source_path, "wb") as buffer:
            shutil.copyfileobj(source_audio.file, buffer)
        
        with open(target_path, "wb") as buffer:
            shutil.copyfileobj(target_audio.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving uploaded files: {str(e)}")
    
    try:
        # Process voice conversion
        sr, audio_data = voice_conversion_api(
            source_path,
            target_path,
            diffusion_steps,
            length_adjust,
            inference_cfg_rate,
            f0_condition,
            auto_f0_adjust,
            pitch_shift
        )
        
        # Save as MP3
        audio_array = (audio_data * 32768.0).astype(np.int16)
        AudioSegment(
            audio_array.tobytes(),
            frame_rate=sr,
            sample_width=audio_array.dtype.itemsize,
            channels=1
        ).export(output_path, format="mp3", bitrate="320k")
        
        # Schedule cleanup of temporary files after response is sent
        background_tasks.add_task(cleanup_files, source_path, target_path, output_path)
        
        return FileResponse(
            path=output_path,
            filename=f"converted_audio.mp3",
            media_type="audio/mpeg"
        )
        
    except Exception as e:
        # Clean up files in case of error
        cleanup_files(source_path, target_path)
        raise HTTPException(status_code=500, detail=f"Voice conversion failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "device": device.type}

def cleanup_files(*file_paths):
    """Remove temporary files after they're no longer needed."""
    for file_path in file_paths:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except:
                pass

@app.on_event("shutdown")
def cleanup_temp_dir():
    """Remove the temporary directory on shutdown."""
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8020)
