from pathlib import Path
from pedalboard import Pedalboard, load_plugin
from pedalboard.io import AudioFile
import numpy as np

# This script will test all plugins in the plugins folder


def main():
    SAMPLE_RATE = 44100
    with AudioFile("elvis_short.mp3").resampled_to(SAMPLE_RATE) as f:
        audio = f.read(f.frames)
        mono_audio = audio.sum(axis=0) / 2
        root = Path("/content/plugins")
        plugins = root.glob("**/*.vst3")
        for plugin in plugins:
            print(f"Testing {plugin.name}")
            test_load(plugin)
            test_effect(plugin)
            test_audio_mono(plugin, mono_audio, SAMPLE_RATE)
            test_audio_stereo(plugin, audio, SAMPLE_RATE)
            test_audio_param_randomization(plugin, audio, SAMPLE_RATE)
            print(f"Passed {plugin.name}")


def test_load(plugin_path: Path):
    try:
        effect = load_plugin(str(plugin_path))
    except Exception as e:
        assert False, f"Failed to load {plugin_path}: {e}"


def test_effect(plugin):
    assert plugin.is_effect, f"{plugin.name} is not an effect"


def test_audio_mono(plugin, audio, sample_rate):
    try:
        out = plugin(audio, sample_rate)
    except:
        assert False, f"{plugin.name} failed to process mono audio"
    assert out.shape == audio.shape, f"{plugin.name} changed the shape of the audio"


def test_audio_stereo(plugin, audio, sample_rate):
    try:
        out = plugin(audio, sample_rate)
    except:
        assert False, f"{plugin.name} failed to process stereo audio"
    assert out.shape == audio.shape, f"{plugin.name} changed the shape of the audio"


def test_audio_param_randomization(plugin, audio, sample_rate, tolerance=1e-3):
    for param_name in plugin.parameters.keys():
        param = plugin._get_parameter_by_python_name(param_name)
        new_val = np.random.random()
        param.raw_value = new_val
        print(
            f"Setting {param_name} to {param.string_value}, raw_value = {round(new_val, 3)}, range = {param.range}"
        )
    out = plugin(audio, sample_rate)
    mse = ((out - audio) ** 2).mean(axis=1)
    np.all(mse > tolerance)


if __name__ == "__main__":
    main()
