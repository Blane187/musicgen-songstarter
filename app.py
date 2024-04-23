import gradio as gr
import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import spaces
import logging
import os 
import uuid
from torch.cuda.amp import autocast
import torch

ZERO_GPU_PATCH_TORCH_DEVICE = 1

# Configura o logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logging.info("Loading the pre-trained model.")
model = MusicGen.get_pretrained('nateraw/musicgen-songstarter-v0.2')
model.set_generation_params(duration=30)

@spaces.GPU(duration=120)
def generate_music(description, melody_audio):
    with autocast():
        logging.info("Starting the music generation.")
        if description:
            description = [description]
            if melody_audio:
                logging.info(f"Loading the audio melody of: {melody_audio}")
                melody, sr = torchaudio.load(melody_audio)
                logging.info("Generating music with description and melody.")
                wav = model.generate_with_chroma(description, melody[None], sr)
            else:
                logging.info("Generating music solely from description.")
                wav = model.generate(description)
        else:
            logging.info("Generating music unconditionally.")
            wav = model.generate_unconditional(1)
        filename = f'{str(uuid.uuid4())}'
        logging.info(f"Salvando a música gerada com o nome: {filename}")
        path = audio_write(filename, wav[0].cpu().to(torch.float32), model.sample_rate, strategy="loudness", loudness_compressor=True)
        print("Music saved in", path, ".")
        # Verifica a forma do tensor de áudio e se foi salvo corretamente
        logging.info(f"The shape of the generated audio tensor: {wav[0].shape}")
        logging.info("Music generated and saved successfully.")
        if not os.path.exists(path):
            raise ValueError(f'Failed to save audio to {path}')

        return path
    

    
    title="",
    

with gr.Blocks(title="demo app") as demo:
    gr.Markdown("# MusicGen Demo")
    melody_audio = gr.Audio(label="Melody Audio (optional)", type="filepath")
    output_path = gr.Audio(label="Generated Music", type="filepath")
    greet_btn = gr.Button("Greet")
    greet_btn.click(fn=generate_music, inputs=[description, melody_audio], outputs=output_path,)
    examples=[
        ["trap, synthesizer, songstarters, dark, G# minor, 140 bpm", "./assets/kalhonaho.mp3"],
        ["sega genesis, 8bit, dark, 140 bpm", None],
        ["upbeat, electronic, synth, dance, 120 bpm", None]
    ]

demo.launch()
