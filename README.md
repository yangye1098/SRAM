
Author's implementation of SRAM described in paper "Syllable-Rate-Adjusted-Modulation (SRAM) Predicts Clear and Conversational Speech Intelligibility"

# Introduction
SRAM was inspired from speech-based STI but differs in two ways: (1) SRAM uses a window of 1 sec while speech-based STI uses 16 sec; (2) modulation energy was calculated with a lower frequency limit at the syllable rate. 

# Usage

## Calculate SRAM with known syllable rate
```python
from sram import sram

# speech is the target speech audio
# syllable_rate is the calculated syllable rate for the speech. 
d = sram(speech, sample_rate, sr=syllable_rate, use_sr_estimation=False)
```

## Calculate SRAM with unknown syllable rate

```python
from sram import sram

# speech is the target speech audio
# reference is the clean audio containing the same sentence as speech
d = sram(speech, sample_rate, reference=reference_audio, sr=None, use_sr_estimation=True):

```
