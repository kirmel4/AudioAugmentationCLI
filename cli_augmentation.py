from scipy.signal import resample
import soundfile as sf
import numpy as np
import click

def audio_augmentation(audio):
    
    def time_stretch(audio):
        stretch_rate = np.random.uniform(0.85, 1.25)
        return resample(audio, int(audio.shape[-1]*stretch_rate), axis = -1)
    
    def white_noise(audio):
        noise = np.random.normal(0, audio.std(), audio.shape[-1]).astype('float32')
        noise_rate = np.random.uniform(0.01, 0.3)
        return audio + noise * noise_rate

    def time_shift(audio):
        time_rate = np.random.rand()
        return np.roll(audio, int(audio.shape[-1]*time_rate), axis = -1)

    def random_gain(audio):
        gain_rate = np.random.uniform(0.5, 1.5)
        return np.clip(audio * gain_rate, -1.0, 1.0)

    def invert_polarity(audio):
        return audio * -1
    
    augmentations = {time_stretch: 0.75, white_noise: 0.75, time_shift: 0.5 , random_gain: 0.75, invert_polarity: 0.5}
    
    for aug in augmentations.keys():
        if np.random.rand() < augmentations[aug]:
            audio = aug(audio)
            
    return audio

@click.command()
@click.argument('audio_path')

def cli(audio_path):
    audio, sample_rate = sf.read(audio_path, dtype='float32')
    augmented_audio = audio_augmentation(audio.T)
    sf.write(f"augmented_audio.wav", augmented_audio.T, sample_rate)

if __name__ == '__main__':
    cli()

