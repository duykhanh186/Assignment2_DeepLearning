from pathlib import Path
import sys
import zipfile


ROOT = Path(__file__).resolve().parent

if len(sys.argv) != 2:
    raise SystemExit(
        "Cách dùng: python collect_submission.py <MSSV>"
    )

student_id = sys.argv[1].strip()

required = [
    "Q1_CNN_from_scratch.ipynb",
    "Q2_Normalization_Dropout.ipynb",
    "Q3_PyTorch_CNN.ipynb",
    "Q4_RNN_fundamentals.ipynb",
    "Q5_RNN_captioning.ipynb",
    "best_cnn.pt",
    "caption_rnn.pt",
]

missing = [
    rel
    for rel in required
    if not (ROOT / rel).exists()
]

if missing:
    raise SystemExit(
        "Thiếu file:\n"
        + "\n".join(
            f" - {p}"
            for p in missing
        )
    )

out = ROOT / f"{student_id}_assignment2.zip"

with zipfile.ZipFile(
    out,
    "w",
    zipfile.ZIP_DEFLATED,
) as z:
    for rel in required:
        z.write(
            ROOT / rel,
            arcname=rel,
        )

print(f"Đã tạo: {out.name}")
