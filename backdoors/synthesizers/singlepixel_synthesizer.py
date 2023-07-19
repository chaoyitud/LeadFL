import torch
from backdoors.synthesizers.pattern_synthesizer import PatternSynthesizer


class SinglePixelSynthesizer(PatternSynthesizer):
    pattern_tensor = torch.tensor([[1.]])
