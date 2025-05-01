# %%
from moisesdb.dataset import MoisesDB

# %%
db = MoisesDB(
    data_path='./moisesdb',
    sample_rate=44100
)

# %%
n_songs = len(db)
n_songs

# %%
import os
from glob import glob

# %%
track = db[0]

import numpy as np
from scipy.io.wavfile import write

rate = 44100
data = track.audio
print(track.name)


from IPython.display import Audio
wave_audio = np.sin(data)
Audio(wave_audio, rate=44100)

# %%
# Cell A: Show exactly what folders/files live under "./moisesdb"
import os

root = "./moisesdb"
print("Exists?       ", os.path.isdir(root))
print("Top-level contents of", root, ":")
for name in sorted(os.listdir(root)):
    print("  ", name)

# %%
# Cell 1: Inspect one track’s directory structure

import os

# Adjust this to your root
moises_root = "./moisesdb/moisesdb_v0.1"

# List all track folders
tracks = [d for d in os.listdir(moises_root)
          if os.path.isdir(os.path.join(moises_root, d))]
print(f"Found {len(tracks)} track folders.\n")

# Peek into the first folder
first = tracks[0]
first_path = os.path.join(moises_root, first)
print(f"Contents of track folder '{first}':")
for name in sorted(os.listdir(first_path)):
    print("  ", name)

# %%
# Cell 2: Build an index treating each subfolder as a stem

import os
from glob import glob

def build_moises_index(data_path: str = "./moisesdb"):
    """
    Scans MoisesDB (moisesdb_v0.1/) and returns:
      - track_id   : folder name
      - stem_dirs  : {stem_name: path_to_stem_folder}
    No mix.wav—will later sum stems to get the mix.
    """
    root = os.path.join(data_path, "moisesdb_v0.1")
    if not os.path.isdir(root):
        root = data_path
    index = []

    for td in os.listdir(root):
        tdir = os.path.join(root, td)
        if not os.path.isdir(tdir):
            continue

        # Find all subdirectories that contain .wav files
        stems = {}
        for sub in os.listdir(tdir):
            subp = os.path.join(tdir, sub)
            if os.path.isdir(subp) and glob(os.path.join(subp, "*.wav")):
                stems[sub] = subp

        if not stems:
            continue

        index.append({
            "track_id": td,
            "stem_dirs": stems
        })

    return index

# Usage
moises_index = build_moises_index()
print(f"Indexed {len(moises_index)} tracks; each with stems:", 
      list(moises_index[0]['stem_dirs'].keys()) if moises_index else [])

# %%
print(f"Indexed {len(moises_index)} tracks; each with stems:", 
      list(moises_index[100]['stem_dirs'].keys()) if moises_index else [])

# %%
#Preprocessing function (resample + pad + mix synthesis)
import numpy as np
import soundfile as sf
from scipy.signal import resample

def preprocess_and_save_fixed(track_entry, output_root,
                              target_sr=44100, segment_length=None):
    """
    - Loads each stem, resamples & converts to mono
    - Pads/truncates each stem to the same length
      (either segment_length or the longest stem)
    - Synthesizes mix by summing all uniformly-sized stems
    - Saves mix.wav + stems/*.wav under output_root/track_id/
    """
    tid     = track_entry['track_id']
    stems   = track_entry['stem_dirs']
    out_dir = os.path.join(output_root, tid)
    stems_out = os.path.join(out_dir, 'stems')
    os.makedirs(stems_out, exist_ok=True)

    processed = []
    # 1) Load/resample/mono all stems, record their lengths
    for stem, stem_dir in stems.items():
        wavs = [f for f in os.listdir(stem_dir) if f.lower().endswith('.wav')]
        if not wavs: 
            continue
        # pick first wav
        y, orig_sr = sf.read(os.path.join(stem_dir, wavs[0]), dtype='float32')
        # resample if needed
        if orig_sr != target_sr:
            num = int(len(y) * target_sr / orig_sr)
            y = resample(y, num)
        # mono
        if y.ndim > 1:
            y = y.mean(axis=1)
        processed.append((stem, y))

    # 2) Determine uniform length
    if segment_length is not None:
        L = segment_length
    else:
        L = max(y.shape[0] for _, y in processed)

    # 3) Pad/truncate and save each stem
    for stem, y in processed:
        if y.shape[0] < L:
            y_p = np.pad(y, (0, L - y.shape[0]))
        else:
            y_p = y[:L]
        # save stem
        sf.write(os.path.join(stems_out, f"{stem}.wav"), y_p, target_sr)

    # 4) Synthesize mix by summing padded stems
    mix = np.zeros(L, dtype=np.float32)
    for _, y in processed:
        y_p = y if y.shape[0]==L else (np.pad(y, (0, L - y.shape[0])) if len(y)<L else y[:L])
        mix += y_p

    # save mix
    sf.write(os.path.join(out_dir, 'mix.wav'), mix, target_sr)

# %%
# Cell 2:run preprocessing over all tracks with the fixed function

OUTPUT_ROOT    = "./processed/moisesdb"
TARGET_SR      = 44100
SEGMENT_LENGTH = None   

for entry in moises_index:
    preprocess_and_save_fixed(
        entry,
        output_root=OUTPUT_ROOT,
        target_sr=TARGET_SR,
        segment_length=SEGMENT_LENGTH
    )

print("Re-preprocessing complete. All stems and mixes are now uniform length.")

# %%
# Cell 1: Build index over your processed folder
import os

processed_root = "./processed/moisesdb"
def build_processed_index(processed_root):
    index = []
    for tid in sorted(os.listdir(processed_root)):
        track_dir = os.path.join(processed_root, tid)
        mix_path  = os.path.join(track_dir, "mix.wav")
        stems_dir = os.path.join(track_dir, "stems")
        if os.path.isdir(track_dir) and os.path.isfile(mix_path) and os.path.isdir(stems_dir):
            stems = sorted(glob(os.path.join(stems_dir, "*.wav")))
            index.append({
                "track_id":   tid,
                "mix_path":   mix_path,
                "stem_paths": stems
            })
    return index

processed_index = build_processed_index(processed_root)
print(f"Indexed {len(processed_index)} tracks in '{processed_root}'.")

# %%
def check_processed_consistency(index, target_sr=44100):
    """
    For each track:
     1) mix.wav exists, is target_sr & mono
     2) each stem exists, is target_sr & mono
     3) all stem lengths == mix length
    """
    errors = []
    for e in index:
        tid = e["track_id"]
        mix, sr_mix = sf.read(e["mix_path"], dtype="float32")
        if sr_mix != target_sr or mix.ndim != 1:
            errors.append(f"{tid}-mix: SR={sr_mix}, ndim={mix.ndim}")
            continue
        mix_len = len(mix)
        for stem_path in e["stem_paths"]:
            y, sr_y = sf.read(stem_path, dtype="float32")
            if sr_y != target_sr or y.ndim != 1 or len(y) != mix_len:
                stem = os.path.basename(stem_path)
                errors.append(
                    f"{tid}-{stem}: SR={sr_y}, ndim={y.ndim}, len={len(y)} != mix len={mix_len}"
                )
        if not any(err.startswith(tid+"-") for err in errors):
            print(f"[OK] {tid}: all stems match mix (len={mix_len})")

    if errors:
        print("\nErrors detected:")
        for err in errors:
            print(" •", err)
    else:
        print("\nAll processed tracks passed consistency check.")

# Run the consistency check
check_processed_consistency(processed_index)

# %%
#inspecting musdb
musdb_root = "./musdb_traindb"
# Peek into the first track folder
first = sorted(os.listdir(musdb_root))[0]
print(f"\nContents of track '{first}':")
for item in sorted(os.listdir(os.path.join(musdb_root, first))):
    print("   ", item)

# %%
#Build index over musdb_traindb

def build_musdb_index(data_root="./musdb_traindb"):
    """
    Scans musdb_traindb/ and returns a list of dicts, one per track:
      - track_id   : str
      - mix_path   : str   (detects 'mixture.wav' or first .wav)
      - stem_paths : Dict[str, str] (all other .wav files)
    """
    idx = []
    for tid in sorted(os.listdir(data_root)):
        td = os.path.join(data_root, tid)
        if not os.path.isdir(td):
            continue

        # Gather all WAVs in the track root
        wav_files = sorted(glob(os.path.join(td, "*.wav")))
        if not wav_files:
            continue

        # 1) Identify mix: look for 'mixture.wav' first, else fallback to first .wav
        mix_candidates = [w for w in wav_files if os.path.basename(w).lower() == "mixture.wav"]
        mix_path = mix_candidates[0] if mix_candidates else wav_files[0]

        # 2) Everything else is a stem
        stem_paths = {
            os.path.splitext(os.path.basename(w))[0]: w
            for w in wav_files
            if w != mix_path
        }

        if not stem_paths:
            continue

        idx.append({
            "track_id":   tid,
            "mix_path":   mix_path,
            "stem_paths": stem_paths
        })

    return idx

musdb_index = build_musdb_index()
print(f"Indexed {len(musdb_index)} tracks from musdb_traindb.")
musdb_index[:2]  # preview first two entries


# %%
# Cell 2: Preprocessing + save into processed folder

MUSDB_PROC_ROOT = "./processed/musdb_traindb"  # adjust if needed
TARGET_SR       = 44100
SEGMENT_LENGTH  = None 

def preprocess_and_save_musdb(entry, output_root=MUSDB_PROC_ROOT,
                              target_sr=TARGET_SR, segment_length=SEGMENT_LENGTH):
    tid    = entry["track_id"]
    mix, sr_mix = sf.read(entry["mix_path"], dtype="float32")
    # Resample mix
    if sr_mix != target_sr:
        mix = resample(mix, int(len(mix) * target_sr / sr_mix))
    if mix.ndim > 1:
        mix = mix.mean(axis=1)
    mix_len = len(mix)

    # Load & preprocess stems
    stems_proc = {}
    for name, path in entry["stem_paths"].items():
        y, sr = sf.read(path, dtype="float32")
        if sr != target_sr:
            y = resample(y, int(len(y) * target_sr / sr))
        if y.ndim > 1:
            y = y.mean(axis=1)
        stems_proc[name] = y

    # Determine final length
    L = segment_length or max(mix_len, *(len(y) for y in stems_proc.values()))

    # Prepare output dirs
    out_dir   = os.path.join(output_root, tid)
    stems_dir = os.path.join(out_dir, "stems")
    os.makedirs(stems_dir, exist_ok=True)

    # Pad/trim and save mix
    mix_p = mix[:L] if len(mix)>=L else np.pad(mix, (0, L-len(mix)))
    sf.write(os.path.join(out_dir, "mix.wav"), mix_p, target_sr)

    # Pad/trim and save stems
    for name, y in stems_proc.items():
        y_p = y[:L] if len(y)>=L else np.pad(y, (0, L-len(y)))
        sf.write(os.path.join(stems_dir, f"{name}.wav"), y_p, target_sr)


# Run preprocessing on all indexed tracks
for entry in musdb_index:
    preprocess_and_save_musdb(entry)

print("Preprocessed MUSDB into:", MUSDB_PROC_ROOT)

# %%
# Cell 3: Index the processed MUSDB folder

def build_processed_musdb_index(proc_root=MUSDB_PROC_ROOT):
    idx = []
    for tid in sorted(os.listdir(proc_root)):
        td = os.path.join(proc_root, tid)
        mix = os.path.join(td, "mix.wav")
        stems_dir = os.path.join(td, "stems")
        if os.path.isdir(td) and os.path.isfile(mix) and os.path.isdir(stems_dir):
            stems = {
                os.path.splitext(fn)[0]: os.path.join(stems_dir, fn)
                for fn in os.listdir(stems_dir) if fn.lower().endswith(".wav")
            }
            idx.append({"track_id": tid, "mix_path": mix, "stem_paths": stems})
    return idx

processed_musdb_index = build_processed_musdb_index()
print(f"Indexed {len(processed_musdb_index)} processed MUSDB tracks.")

# %%
# Cell 4: Consistency check on processed MUSDB
def check_processed_consistency(index, target_sr=TARGET_SR):
    for e in index:
        tid = e["track_id"]
        mix, sr_mix = sf.read(e["mix_path"], dtype="float32")
        assert sr_mix == target_sr, f"{tid}: mix SR={sr_mix}"
        assert mix.ndim == 1,        f"{tid}: mix not mono"
        L = len(mix)
        for name, path in e["stem_paths"].items():
            y, sr = sf.read(path, dtype="float32")
            assert sr == target_sr, f"{tid}-{name}: SR={sr}"
            assert y.ndim == 1,       f"{tid}-{name}: not mono"
            assert len(y) == L,       f"{tid}-{name}: len={len(y)} != mix len={L}"
        print(f"[OK] {tid}: all stems match mix (len={L})")

check_processed_consistency(processed_musdb_index)

# %%
#next phase - building a Unified Dataset Class reads mix/stem files and supports variable query lists
#loading libraries required
import random
import torch
from torch.utils.data import Dataset, DataLoader


# %%
# class QuerySepDataset(Dataset):
#     def __init__(self, root, split, queries, segment_length=None):
#         """
#         Args:
#           root (str): path to processed dataset, e.g. "./processed/musdb18"
#           split (str): one of "train"/"dev"/"test"
#           queries (List[str]): list of stem names to load, e.g. ["vocals","drums","bass","other"]
#           segment_length (int or None): if set, crop or pad each sample to exactly this many frames
#         """
#         self.queries = queries
#         self.seg_len = segment_length

#         # 1) build self.index: list of dicts with mix + stem paths
#         split_dir = os.path.join(root, split)
#         self.index = []
#         for tid in sorted(os.listdir(split_dir)):
#             td = os.path.join(split_dir, tid)
#             mix_path = os.path.join(td, "mix.wav")
#             stems_dir = os.path.join(td, "stems")
#             if not (os.path.isfile(mix_path) and os.path.isdir(stems_dir)):
#                 continue
#             # gather only the requested queries
#             stem_paths = {}
#             for q in queries:
#                 p = os.path.join(stems_dir, f"{q}.wav")
#                 if os.path.isfile(p):
#                     stem_paths[q] = p
#                 else:
#                     raise FileNotFoundError(f"Expected stem '{q}.wav' in {stems_dir}")
#             self.index.append({
#                 "track_id": tid,
#                 "mix_path": mix_path,
#                 "stem_paths": stem_paths
#             })

#     def __len__(self):
#         return len(self.index)

#     def __getitem__(self, idx):
#         rec = self.index[idx]
#         # 2) load mix
#         mix, sr = sf.read(rec["mix_path"], dtype="float32")
#         mix = self._to_mono(mix)
#         # 3) load each stem
#         stems = {}
#         for q, p in rec["stem_paths"].items():
#             y, _ = sf.read(p, dtype="float32")
#             stems[q] = self._to_mono(y)

#         # 4) align lengths & crop/pad to segment_length
#         mix, stems = self._crop_or_pad(mix, stems)

#         # 5) to torch
#         mix = torch.from_numpy(mix)
#         stems = {q: torch.from_numpy(y) for q, y in stems.items()}

#         return {
#             "track_id": rec["track_id"],
#             "mix": mix, 
#             "stems": stems
#         }

#     def _to_mono(self, y: np.ndarray) -> np.ndarray:
#         if y.ndim > 1:
#             return y.mean(axis=1)
#         return y

#     def _crop_or_pad(self, mix: np.ndarray, stems: dict) -> (np.ndarray, dict):
#         # determine current length
#         L = mix.shape[0]
#         if self.seg_len is not None:
#             target = self.seg_len
#             if L >= target:
#                 # random crop
#                 start = np.random.randint(0, L - target + 1)
#                 mix = mix[start:start + target]
#                 for q in stems:
#                     stems[q] = stems[q][start:start + target]
#             else:
#                 # pad
#                 pad_width = target - L
#                 mix = np.pad(mix, (0, pad_width))
#                 for q in stems:
#                     stems[q] = np.pad(stems[q], (0, pad_width))
#         else:
#             # no segment_length → just ensure all stems match mix length
#             for q in stems:
#                 y = stems[q]
#                 if y.shape[0] < L:
#                     stems[q] = np.pad(y, (0, L - y.shape[0]))
#                 else:
#                     stems[q] = y[:L]
#         return mix, stems


# %%
def discover_union_stems(root, split):
    split_dir = os.path.join(root, split)
    union = set()
    for tid in os.listdir(split_dir):
        stems_dir = os.path.join(split_dir, tid, "stems")
        if not os.path.isdir(stems_dir): 
            continue
        union |= {
            os.path.splitext(fn)[0]
            for fn in os.listdir(stems_dir) if fn.lower().endswith(".wav")
        }
    return sorted(union)

# %%
class PresenceSepDataset(Dataset):
    def __init__(self, root, split, queries=None, segment_length=None):
        """
        - root: path to processed dataset (e.g. "./processed/musdb18")
        - split: "train"/"dev"/"test"
        - queries: list of stem names, or None to auto-discover union
        - segment_length: fixed # of samples per example, or None
        """
        split_dir = os.path.join(root, split)
        self.queries = queries or discover_union_stems(root, split)
        self.seg_len  = segment_length

        if self.seg_len is not None and (self.seg_len % 8) != 0:
            raise ValueError(f"segment_length ({self.seg_len}) must be multiple of stride=8")

        # build index, allowing missing stems
        self.index = []
        for tid in sorted(os.listdir(split_dir)):
            td       = os.path.join(split_dir, tid)
            mix_path = os.path.join(td, "mix.wav")
            stems_dir= os.path.join(td, "stems")
            if not (os.path.isfile(mix_path) and os.path.isdir(stems_dir)):
                continue

            stem_paths = {
                q: os.path.join(stems_dir, f"{q}.wav")
                for q in self.queries
            }
            self.index.append({
                "track_id":   tid,
                "mix_path":   mix_path,
                "stem_paths": stem_paths
            })

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        rec = self.index[idx]
        # load mix
        mix, _ = sf.read(rec["mix_path"], dtype="float32")
        mix     = self._to_mono(mix)

        # 0) ----- ensure length is multiple of stride -----
        STRIDE = 8
        L0 = mix.shape[0]
        if L0 % STRIDE != 0:
            L_pad = ((L0 + STRIDE - 1) // STRIDE) * STRIDE
            pad_width = L_pad - L0
            mix = np.pad(mix, (0, pad_width))
        # same for stems:
        stems, presence = {}, []
        for q, path in rec["stem_paths"].items():
            if os.path.isfile(path):
                y, _ = sf.read(path, dtype="float32")
                y     = self._to_mono(y)
                presence.append(1)
            else:
                y     = np.zeros_like(mix)
                presence.append(0)

            # pad this stem to L_pad if needed
            if y.shape[0] < mix.shape[0]:
                y = np.pad(y, (0, mix.shape[0] - y.shape[0]))
            stems[q] = y


        # # load or zero each stem
        # for q, path in rec["stem_paths"].items():
        #     if os.path.isfile(path):
        #         y, _ = sf.read(path, dtype="float32")
        #         y     = self._to_mono(y)
        #         presence.append(1)
        #     else:
        #         y     = np.zeros_like(mix)
        #         presence.append(0)
        #     stems[q] = y

        # crop or pad to fixed length
        mix, stems = self._crop_or_pad(mix, stems)

        # to torch tensors
        mix_t   = torch.from_numpy(mix)
        stems_t = torch.stack([torch.from_numpy(stems[q]) 
                               for q in self.queries], dim=0)  # (n_src, L)
        pres_t  = torch.tensor(presence, dtype=torch.float32)      # (n_src,)

        return {
            "track_id": rec["track_id"],
            "mix":       mix_t,
            "stems":     stems_t,
            "presence":  pres_t
        }

    def _to_mono(self, y: np.ndarray) -> np.ndarray:
        return y.mean(axis=1) if y.ndim > 1 else y

    def _crop_or_pad(self, mix: np.ndarray, stems: dict):
        L = mix.shape[0]
        if self.seg_len is not None:
            target = self.seg_len
            if L >= target:
                start = np.random.randint(0, L - target + 1)
                mix   = mix[start:start+target]
                for q in stems:
                    stems[q] = stems[q][start:start+target]
            else:
                pad_w = target - L
                mix   = np.pad(mix, (0, pad_w))
                for q in stems:
                    stems[q] = np.pad(stems[q], (0, pad_w))
        else:
            # just align stems to mix length
            for q in stems:
                y = stems[q]
                if len(y) < L:
                    stems[q] = np.pad(y, (0, L - len(y)))
                else:
                    stems[q] = y[:L]
        return mix, stems


# %%
def presence_collate(batch):
    track_ids = [b["track_id"] for b in batch]
    mixes      = torch.stack([b["mix"]      for b in batch], dim=0)  # (B, L)
    stems      = torch.stack([b["stems"]    for b in batch], dim=0)  # (B, n_src, L)
    presence   = torch.stack([b["presence"] for b in batch], dim=0)  # (B, n_src)
    return {
        "track_id": track_ids,
        "mix":       mixes,
        "stems":     stems,
        "presence":  presence
    }

# %%
# 4) Example DataLoader setup
SEG_LEN = 44100 * 4  # 4-second clips

# %%
# pretrain_ds = PresenceSepDataset(
#     root="./processed/moisesdb",
#     split="",
#     queries=None,
#     segment_length=SEG_LEN
# )
# pretrain_loader = DataLoader(
#     pretrain_ds,
#     batch_size=8,
#     shuffle=True,
#     num_workers=4,
#     collate_fn=presence_collate
# )

# # MUSDB18 fine-tune loader (auto-discovers its stems)
# finetune_ds = PresenceSepDataset(
#     root="./processed/musdb_traindb",
#     split="",
#     queries=None,
#     segment_length=SEG_LEN
# )
# finetune_loader = DataLoader(
#     finetune_ds,
#     batch_size=4,
#     shuffle=True,
#     num_workers=4,
#     collate_fn=presence_collate
# )

# %%
import torch.nn as nn
import torch.nn.functional as F

class TemporalBlock(nn.Module):
    def __init__(self, channels, hidden_channels, kernel_size, dilation):
        """
        A single TCN block with:
          - depthwise dilated conv → PReLU → Norm
          - pointwise conv → PReLU → Norm
        Residual adds input→output.
        """
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2

        # Depthwise dilated conv
        self.conv_dilated = nn.Conv1d(
            in_channels=channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            groups=channels
        )
        self.prelu1 = nn.PReLU()
        self.norm1  = nn.BatchNorm1d(hidden_channels)

        # Pointwise conv back to channels
        self.conv_1x1 = nn.Conv1d(
            in_channels=hidden_channels,
            out_channels=channels,
            kernel_size=1
        )
        self.prelu2 = nn.PReLU()
        self.norm2  = nn.BatchNorm1d(channels)

    def forward(self, x):
        # x: (B, channels, T)
        out = self.conv_dilated(x)
        out = self.prelu1(out)
        out = self.norm1(out)
        out = self.conv_1x1(out)
        out = self.prelu2(out)
        out = self.norm2(out)
        return out + x


# %%
class ConvTasNetWithPresence(nn.Module):
    def __init__(
        self,
        n_src: int,
        enc_channels: int = 512,
        hidden_channels: int = 1024,
        kernel_size: int = 16,
        stride: int = 8,
        num_blocks: int = 8
    ):
        """
        Args:
          n_src: number of possible stems (union of all stems)
          enc_channels: channels in encoder feature space
          hidden_channels: intermediate channels in TCN blocks
          kernel_size, stride: for encoder & decoder convs
          num_blocks: how many TemporalBlocks to stack
        """
        super().__init__()
        self.n_src = n_src

        # 1) Encoder: raw waveform → features
        self.encoder = nn.Conv1d(
            in_channels=1,
            out_channels=enc_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - stride) // 2
        )
        self.norm_enc = nn.BatchNorm1d(enc_channels)

        # 2) Separator: stack of dilated residual blocks
        blocks = []
        for i in range(num_blocks):
            dilation = 2 ** i
            blocks.append(TemporalBlock(
                channels=enc_channels,
                hidden_channels=hidden_channels,
                kernel_size=3,
                dilation=dilation
            ))
        self.separator = nn.Sequential(*blocks)

        # 3) Mask generator: 1×1 conv → n_src × enc_channels masks
        self.mask_conv = nn.Conv1d(
            in_channels=enc_channels,
            out_channels=n_src * enc_channels,
            kernel_size=1
        )

        # 4) Decoder: features → reconstructed waveform
        self.decoder = nn.ConvTranspose1d(
            in_channels=enc_channels,
            out_channels=1,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - stride) // 2
        )

        # 5) Presence head: pool features → n_src logits
        self.presence_pool = nn.AdaptiveAvgPool1d(1)
        self.presence_fc   = nn.Linear(enc_channels, n_src)

    def forward(self, mix: torch.Tensor):
        """
        Args:
          mix: (B, L) raw waveform batch
        Returns:
          est_sources: (B, n_src, L) separated waveforms
          pres_logits: (B, n_src)   presence logits per source
        """
        B, L = mix.size()
        x = mix.unsqueeze(1)                    # → (B, 1, L)
        w = self.encoder(x)                     # → (B, enc_channels, T)
        w = self.norm_enc(w)
        

        # separator
        w_sep = self.separator(w)               # → (B, enc_channels, T)

        # produce masks
        masks = self.mask_conv(w_sep)           # → (B, n_src*enc, T)
        masks = masks.view(B, self.n_src, -1, w_sep.size(2))
        masks = F.relu(masks)                   # enforce non‐negativity

        # apply masks and decode each source
        est_sources = []
        for i in range(self.n_src):
            feat_i = masks[:, i] * w            # (B, enc_channels, T)
            wav_i  = self.decoder(feat_i)       # → (B, 1, L)
            wav_i  = wav_i[..., :L]
            est_sources.append(wav_i.squeeze(1))# → (B, L)
        est_sources = torch.stack(est_sources, dim=1)  # → (B, n_src, L)

        # presence detection
        pres_feat  = self.presence_pool(w_sep).squeeze(-1)  # → (B, enc_channels)
        pres_logits = self.presence_fc(pres_feat)           # → (B, n_src)

        return est_sources, pres_logits


# %%
# # assume `batch` comes from your DataLoader with presence_collate()
# batch = next(iter(pretrain_loader))
# mix       = batch["mix"]       # (B, L)
# targets   = batch["stems"]     # (B, n_src, L)
# presence  = batch["presence"]  # (B, n_src)

# model = ConvTasNetWithPresence(n_src=targets.size(1))
# est, pres_logits = model(mix)

# # est:   (B, n_src, L)    compare to `targets` with L1/L2 loss
# # pres_logits: (B, n_src) compare to `presence` with BCEWithLogitsLoss
# sep_loss = F.l1_loss(est, targets)  
# pres_loss = F.binary_cross_entropy_with_logits(pres_logits, presence)

# print(sep_loss)
# print(pres_loss)

# %%
import torch.optim as optim

def train_epoch(model, loader, optimizer, device, lambda_pres):
    sep_crit  = nn.L1Loss()
    pres_crit = nn.BCEWithLogitsLoss()
    model.train()
    total_loss = 0.0

    for batch in loader:
        mix      = batch["mix"].to(device)       # (B,L)
        targets  = batch["stems"].to(device)     # (B,n_src,L)
        presence = batch["presence"].to(device)  # (B,n_src)

        optimizer.zero_grad()
        est, pres_logits = model(mix)
        loss_sep  = sep_crit(est, targets)
        loss_pres = pres_crit(pres_logits, presence)
        loss      = loss_sep + lambda_pres * loss_pres
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        total_loss += loss.item() * mix.size(0)

    return total_loss / len(loader.dataset)


def eval_epoch(model, loader, device, lambda_pres):
    sep_crit  = nn.L1Loss()
    pres_crit = nn.BCEWithLogitsLoss()
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            mix      = batch["mix"].to(device)
            targets  = batch["stems"].to(device)
            presence = batch["presence"].to(device)

            est, pres_logits = model(mix)
            loss_sep  = sep_crit(est, targets)
            loss_pres = pres_crit(pres_logits, presence)
            total_loss += (loss_sep + lambda_pres * loss_pres).item() * mix.size(0)

    return loss_sep, loss_pres, total_loss / len(loader.dataset)

# %%

# ──────────────────────────────────────────────────────────────────────────────
# 3) Hyperparameters & data loaders
# ──────────────────────────────────────────────────────────────────────────────
device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEG_LEN     = 44100 * 4
BATCH_PRE   = 8
BATCH_FT    = 4
LR_PRE      = 1e-3
LR_FT       = 5e-4
EPOCHS_PRE  = 10
EPOCHS_FT   = 5
LAMBDA_PRES = 0.1

# assume PresenceSepDataset and presence_collate are already defined above
pretrain_ds = PresenceSepDataset(
    root="./processed/moisesdb_final", split="train",
    queries=None, segment_length=SEG_LEN
)
pretrain_loader = DataLoader(
    pretrain_ds, batch_size=BATCH_PRE,
    shuffle=True, num_workers=4,
    pin_memory=True, collate_fn=presence_collate
)
val_pre_ds = PresenceSepDataset(
    root="./processed/moisesdb_final", split="test",
    queries=None, segment_length=SEG_LEN
)
val_pre_loader = DataLoader(
    val_pre_ds, batch_size=BATCH_PRE,
    shuffle=False, num_workers=2,
    pin_memory=True, collate_fn=presence_collate
)

QUERIES = ["vocals", "bass", "drums", "guitar", "percussion", "piano", "bowed_strings", "other_keys", "wind", "other", "other_plucked"]

# MUSDB loaders for evaluation & fine-tuning
musdb_train_ds = PresenceSepDataset(
    root="./processed/musdb_traindb", split="",
    queries=QUERIES, segment_length=SEG_LEN
)
musdb_train_loader = DataLoader(
    musdb_train_ds, batch_size=BATCH_FT,
    shuffle=True, num_workers=4,
    pin_memory=True, collate_fn=presence_collate
)
musdb_val_ds = PresenceSepDataset(
    root="./processed/moisesdb_final", split="test",
    queries=QUERIES, segment_length=SEG_LEN
)
musdb_val_loader = DataLoader(
    musdb_val_ds, batch_size=BATCH_FT,
    shuffle=False, num_workers=2,
    pin_memory=True, collate_fn=presence_collate
)

%%
──────────────────────────────────────────────────────────────────────────────
4) Pre-training on MoisesDB
──────────────────────────────────────────────────────────────────────────────
n_src_pre = len(pretrain_ds.queries)
model_pre = ConvTasNetWithPresence(n_src_pre).to(device)

opt_pre = optim.Adam(model_pre.parameters(), lr=LR_PRE)
sched_pre = optim.lr_scheduler.StepLR(opt_pre, step_size=10, gamma=0.5)

for epoch in range(1, EPOCHS_PRE+1):
    train_loss = train_epoch(model_pre, pretrain_loader, opt_pre, device, LAMBDA_PRES)
    sep_val, pres_val, tot_val   = eval_epoch(model_pre, val_pre_loader, device, LAMBDA_PRES)
    print(f"[Pretrain Epoch {epoch:02d}] "
          f"Train Loss: {train_loss:.4f} │ "
          f"Val Sep: {sep_val:.4f} │ "
          f"Val Pres: {pres_val:.4f} │ "
          f"Val Total: {tot_val:.4f}")
    sched_pre.step()

# save pretrained-only model
torch.save(model_pre.state_dict(), "model_pretrained.pth")

%%
──────────────────────────────────────────────────────────────────────────────
5) Evaluate pretrained-only on MUSDB18
──────────────────────────────────────────────────────────────────────────────
sep_only, pres_only, tot_only = eval_epoch(model_pre, musdb_val_loader, device, LAMBDA_PRES)
print("Pretrained‐only on MUSDB18 →",
      f"Sep: {sep_only:.4f}, Pres: {pres_only:.4f}, Total: {tot_only:.4f}")

# %%
# ──────────────────────────────────────────────────────────────────────────────
# 6) Fine-tuning on MUSDB18
# ──────────────────────────────────────────────────────────────────────────────
n_src_ft = len(musdb_train_ds.queries)
model_ft = ConvTasNetWithPresence(n_src_ft).to(device)

# load encoder + separator + decoder weights; skip mask & presence heads
pretrained = torch.load("model_pretrained.pth", map_location=device)

filtered = {
    k: v
    for k, v in pretrained.items()
    if not (k.startswith("mask_conv") or k.startswith("presence_fc"))
}

model_ft.load_state_dict(filtered, strict=False)

# optionally freeze everything except the new heads
for name, p in model_ft.named_parameters():
    if "mask_conv" not in name and "presence_fc" not in name:
        p.requires_grad = False

opt_ft = optim.Adam(filter(lambda p: p.requires_grad, model_ft.parameters()), lr=LR_FT)
sched_ft = optim.lr_scheduler.StepLR(opt_ft, step_size=5, gamma=0.5)

# unfreeze after 5 epochs
UNFREEZE_AFTER = 5


# 6) Define masked separation loss
def masked_l1_loss(est, target, presence):
    """
    est, target: (B, n_src, L)
    presence:    (B, n_src)  binary mask
    """
    mask = presence.unsqueeze(-1)                  # (B, n_src, 1)
    mask = mask.expand(-1, -1, est.size(2))        # (B, n_src, L)
    abs_err = torch.abs(est - target) * mask       # zero-out absent channels
    return abs_err.sum() / mask.sum().clamp_min(1.0)

# 7) Fine-tuning loop
pres_crit = nn.BCEWithLogitsLoss()

for epoch in range(1, EPOCHS_FT + 1):
    # unfreeze after UNFREEZE_AFTER epochs
    if epoch == UNFREEZE_AFTER + 1:
        for p in model_ft.parameters():
            p.requires_grad = True
        opt_ft = optim.Adam(model_ft.parameters(), lr=LR_FT)
        sched_ft = optim.lr_scheduler.StepLR(opt_ft, step_size=5, gamma=0.5)

    # train
    model_ft.train()
    running_loss = 0.0
    for batch in musdb_train_loader:
        mix      = batch["mix"].to(device)       # (B, L)
        targets  = batch["stems"].to(device)     # (B, n_src_pre, L)
        presence = batch["presence"].to(device)  # (B, n_src_pre)

        opt_ft.zero_grad()
        est, pres_logits = model_ft(mix)

        sep_loss  = masked_l1_loss(est, targets, presence)
        pres_loss = pres_crit(pres_logits, presence)
        loss      = sep_loss + LAMBDA_PRES * pres_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_ft.parameters(), 5)
        opt_ft.step()

        running_loss += loss.item() * mix.size(0)

    train_loss = running_loss / len(musdb_train_loader.dataset)

    # validate
    model_ft.eval()
    sum_sep = sum_pres = sum_tot = 0.0
    with torch.no_grad():
        for batch in musdb_val_loader:
            mix      = batch["mix"].to(device)
            targets  = batch["stems"].to(device)
            presence = batch["presence"].to(device)

            est, pres_logits = model_ft(mix)
            sep_l  = masked_l1_loss(est, targets, presence)
            pres_l = pres_crit(pres_logits, presence)
            tot_l  = sep_l + LAMBDA_PRES * pres_l

            B = mix.size(0)
            sum_sep  += sep_l.item()  * B
            sum_pres += pres_l.item() * B
            sum_tot  += tot_l.item()  * B

    n_val = len(musdb_val_loader.dataset)
    sep_val  = sum_sep  / n_val
    pres_val = sum_pres / n_val
    tot_val  = sum_tot  / n_val

    print(f"[Finetune Epoch {epoch:02d}] "
          f"Train: {train_loss:.4f} │ "
          f"Val Sep: {sep_val:.4f} │ "
          f"Val Pres: {pres_val:.4f} │ "
          f"Val Total: {tot_val:.4f}")

    sched_ft.step()

# 8) Save fine-tuned model
torch.save(model_ft.state_dict(), "model_finetuned_full_union.pth")








# ##############################
# for epoch in range(1, EPOCHS_FT+1):
#     if epoch == UNFREEZE_AFTER+1:
#         for p in model_ft.parameters():
#             p.requires_grad = True
#         # switch optimizer to include all params
#         opt_ft = optim.Adam(model_ft.parameters(), lr=LR_FT)
#         sched_ft = optim.lr_scheduler.StepLR(opt_ft, step_size=5, gamma=0.5)

#     train_loss = train_epoch(model_ft, musdb_train_loader, opt_ft, device, LAMBDA_PRES)
#     sep_val, pres_val, tot_val   = eval_epoch(model_ft, musdb_val_loader, device, LAMBDA_PRES)
#     print(f"[Finetune Epoch {epoch:02d}] "
#           f"Train Loss: {train_loss:.4f} │ "
#           f"Val Sep: {sep_val:.4f} │ "
#           f"Val Pres: {pres_val:.4f} │ "
#           f"Val Total: {tot_val:.4f}")
#     sched_ft.step()

# # save finetuned model
# torch.save(model_ft.state_dict(), "model_finetuned.pth")

# %%
# ──────────────────────────────────────────────────────────────────────────────
# 7) Evaluate finetuned model
# ──────────────────────────────────────────────────────────────────────────────
# sep_ft, pres_ft, tot_ft = eval_epoch(model_ft, musdb_val_loader, device, LAMBDA_PRES)
# print("Finetuned on MUSDB18 →",
#       f"Sep: {sep_ft:.4f}, Pres: {pres_ft:.4f}, Total: {tot_ft:.4f}")


#ANALYSIS AND EVALUATION SEGMENT
import torch, numpy as np, matplotlib.pyplot as plt, scipy.stats as st
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_fscore_support

# 0) Configuration
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEG_LEN    = 44100 * 4
BATCH_EVAL = 2
MUSDB_QUERIES = ["vocals","drums","bass","other"]

# SI-SDR helper
def si_sdr(est, ref, eps=1e-8):
    ref, est = ref - ref.mean(), est - est.mean()
    proj = np.sum(ref*est) / (np.sum(ref**2)+eps) * ref
    noise = est - proj
    return 10*np.log10((np.sum(proj**2)+eps)/(np.sum(noise**2)+eps))

# Load models
n_src = len(pretrain_ds.queries)
model_pre = ConvTasNetWithPresence(n_src).to(DEVICE)
model_pre.load_state_dict(torch.load("model_pretrained.pth", map_location=DEVICE))
model_pre.eval()
model_ft = ConvTasNetWithPresence(n_src).to(DEVICE)
model_ft.load_state_dict(torch.load("model_finetuned_full_union.pth", map_location=DEVICE))
model_ft.eval()

# Test loader on MUSDB18
#test_ds = PresenceSepDataset("./processed/musdb_testdb","",MUSDB_QUERIES,SEG_LEN)
#test_loader = DataLoader(test_ds, BATCH_EVAL, False, 2, True, presence_collate)



# 1) Evaluate
metrics = {k:{"sdr":[],"true":[],"prob":[]} for k in ["pre","ft"]}
with torch.no_grad():
    for batch in musdb_val_loader:
        mix = batch["mix"].to(DEVICE)
        targ = batch["stems"].cpu().numpy()
        pres = batch["presence"].numpy()
        for key,model in [("pre",model_pre),("ft",model_ft)]:
            est,logits = model(mix)
            est_np  = est.cpu().numpy()
            prob    = torch.sigmoid(logits).cpu().numpy()
            metrics[key]["true"].append(pres)
            metrics[key]["prob"].append(prob)
            B,C,L = est_np.shape
            for b in range(B):
                for c in range(C):
                    if pres[b,c]>0.5:
                        metrics[key]["sdr"].append(si_sdr(est_np[b,c],targ[b,c]))

for key in ["pre","ft"]:
    metrics[key]["true"] = np.vstack(metrics[key]["true"]).ravel()
    metrics[key]["prob"] = np.vstack(metrics[key]["prob"]).ravel()

# Summary
for key,name in [("pre","Pretrained"),("ft","Fine-tuned")]:
    m_sdr = np.mean(metrics[key]["sdr"])
    auc   = roc_auc_score(metrics[key]["true"], metrics[key]["prob"])
    p,r,f,_ = precision_recall_fscore_support(metrics[key]["true"],
                                              (metrics[key]["prob"]>0.5).astype(int),
                                              average="binary")
    print(f"{name}: SI-SDR={m_sdr:.2f} dB | AUC={auc:.3f} | Prec={p:.3f} | Rec={r:.3f} | F1={f:.3f}")

# Paired t-test
t,p = st.ttest_rel(metrics["ft"]["sdr"], metrics["pre"]["sdr"])
print(f"SI-SDR t={t:.3f}, p={p:.3e}")

# 2) Bar plot (overall)
plt.figure()
plt.bar([0,1], [np.mean(metrics["pre"]["sdr"]), np.mean(metrics["ft"]["sdr"])],
        width=0.4)
plt.xticks([0,1],["Pretrained","Fine-tuned"])
plt.ylabel("Mean SI-SDR (dB)")
plt.title("Overall SI-SDR on MUSDB18 Test")
plt.tight_layout()
plt.savefig("muscdb_overall_sdr.png")
plt.show()

# 3) ROC plot
fpr_pre,tpr_pre,_ = roc_curve(metrics["pre"]["true"], metrics["pre"]["prob"])
fpr_ft,tpr_ft,_   = roc_curve(metrics["ft"]["true"],  metrics["ft"]["prob"])
plt.figure()
plt.plot(fpr_pre,tpr_pre,label="Pretrained")
plt.plot(fpr_ft, tpr_ft, label="Fine-tuned")
plt.plot([0,1],[0,1],"--",label="Chance")
plt.xlabel("FPR"); plt.ylabel("TPR")
plt.title("ROC: Stem Presence (MUSDB18)")
plt.legend()
plt.tight_layout()
plt.savefig("muscdb_presence_roc.png")
plt.show()