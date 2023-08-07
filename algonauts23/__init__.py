import os
from pathlib import Path

ALGONAUTS_DIR = Path(
    os.environ.get("ALGONAUTS_DIR", Path(__file__).parent.parent.absolute())
)

ALGONAUTS_DATA_DIR = Path(
    os.environ.get("ALGONAUTS_DATA_DIR", ALGONAUTS_DIR / "dataset")
)

ALGONAUTS_RAW_DIR = Path(
    os.environ.get(
        "ALGONAUTS_RAW_DIR",
        ALGONAUTS_DATA_DIR / "algonauts_2023_challenge_data",
    )
)

SUBS = tuple([f"subj{ii:02d}" for ii in range(1, 9)])
NUM_SUBS = len(SUBS)
FMRI_DIM = 39548

ROI_GROUPS = {
    "streams": "Anatomical streams",
    "prf-visualrois": "Early retinotopic",
    "floc-bodies": "Body-selective",
    "floc-faces": "Face-selective",
    "floc-places": "Place-selective",
    "floc-words": "Word-selective",
}
