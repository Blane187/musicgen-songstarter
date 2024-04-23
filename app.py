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

logging.info("Carregando o modelo pré-treinado.")
model = MusicGen.get_pretrained('nateraw/musicgen-songstarter-v0.2')
model.set_generation_params(duration=30)

@spaces.GPU(duration=120)
def generate_music(description, melody_audio):
    with autocast():
        logging.info("Iniciando a geração de música.")
        if description:
            description = [description]
            if melody_audio:
                logging.info(f"Carregando a melodia de áudio de: {melody_audio}")
                melody, sr = torchaudio.load(melody_audio)
                logging.info("Gerando música com descrição e melodia.")
                wav = model.generate_with_chroma(description, melody[None], sr)
            else:
                logging.info("Gerando música apenas com descrição.")
                wav = model.generate(description)
        else:
            logging.info("Gerando música de forma incondicional.")
            wav = model.generate_unconditional(1)
        filename = f'{str(uuid.uuid4())}'
        logging.info(f"Salvando a música gerada com o nome: {filename}")
        path = audio_write(filename, wav[0].cpu().to(torch.float32), model.sample_rate, strategy="loudness", loudness_compressor=True)
        print("Música salva em", path, ".")
        # Verifica a forma do tensor de áudio e se foi salvo corretamente
        logging.info(f"A forma do tensor de áudio gerado: {wav[0].shape}")
        logging.info("Música gerada e salva com sucesso.")
        if not os.path.exists(path):
            raise ValueError(f'Failed to save audio to {path}')

        return path
    
# Define a interface Gradio
description = gr.Textbox(label="Description", placeholder="acoustic, guitar, melody, trap, d minor, 90 bpm")
melody_audio = gr.Audio(label="Melody Audio (optional)", type="filepath")
output_path = gr.Audio(label="Generated Music", type="filepath")

gr.Interface(
    fn=generate_music,
    inputs=[description, melody_audio],
    outputs=output_path,
    title="MusicGen Demo",
    description="Generate music using the MusicGen model by Nateraw.\n\n"
                "Model: musicgen-songstarter-v0.2\n"
                "Download the model [here](https://huggingface.co/nateraw/musicgen-songstarter-v0.2).\n\n"
                "musicgen-songstarter-v0.2 is a musicgen-stereo-melody-large fine-tuned on a dataset of melody loops from Nateraw's Splice sample library. "
                "It's intended to be used to generate song ideas that are useful for music producers. It generates stereo audio in 32khz.\n\n"
                "Compared to musicgen-songstarter-v0.1, this new version:\n"
                "- Was trained on 3x more unique, manually-curated samples that Nateraw painstakingly purchased on Splice\n"
                "- Is twice the size, bumped up from size medium ➡️ large transformer LM\n\n"
                "If you find this model interesting, please consider:\n"
                "- Following Nateraw on [GitHub](https://github.com/nateraw)\n"
                "- Following Nateraw on [Twitter](https://twitter.com/nateraw)\n\n"
                "Space created by [artificialguybr](https://twitter.com/artificialguybr) on Twitter.",
    examples=[
        ["trap, synthesizer, songstarters, dark, G# minor, 140 bpm", "./assets/kalhonaho.mp3"],
        ["upbeat, electronic, synth, dance, 120 bpm", None]
    ]
).launch()
