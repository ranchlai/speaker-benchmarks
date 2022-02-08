import torch
import torch.nn as nn
from speechbrain.pretrained import EncoderClassifier
from torchaudio.functional import amplitude_to_DB
from torchaudio.transforms import MelSpectrogram

from models_torch import ResNetSE34V2

# class SpeakerBase(nn.Module):
#     def __init__(self, ):
#         super().__init__()

#     @abc.abstractclassmethod
#     def forward(self, x):
#         pass


class SpeakerSB(nn.Module):
    def __init__(self) -> None:
        super(SpeakerSB, self).__init__()

        self.classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb")

    def forward(self, x):
        feat = self.classifier.encode_batch(x)
        return feat[:, 0, :]


class SpeakerSV(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transform = MelSpectrogram(sample_rate=16000,
                                        n_fft=512,
                                        win_length=400,
                                        hop_length=160,
                                        window_fn=torch.hamming_window,
                                        n_mels=80,
                                        f_min=20,
                                        f_max=7600,
                                        norm='slaney')
        state = torch.load('./resnetse34_epoch92_eer0.00931.pth')
        self.model = ResNetSE34V2(nOut=256, n_mels=80)
        self.model.load_state_dict(state)
        self.model.eval()

    @torch.no_grad()
    def forward(self, x):

        EPS = 1e-8
        amax = torch.max(torch.abs(x))
        factor = 1.0 / (amax + EPS)
        x *= factor

        x = self.transform(x)
        x = amplitude_to_DB(x,
                            multiplier=10,
                            amin=1e-5,
                            db_multiplier=0,
                            top_db=75)
        feat = self.model(x[:, None, :, :])
        feat = torch.nn.functional.normalize(feat)

        return feat


if __name__ == '__main__':
    model = SpeakerSV()
    print(model)
    model = SpeakerSB()
    print(model)
