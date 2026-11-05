from pathlib import Path
import re
import nbformat


ROOT = Path(__file__).resolve().parent

NOTEBOOKS = [
    "Q1_CNN_from_scratch.ipynb",
    "Q2_Normalization_Dropout.ipynb",
    "Q3_PyTorch_CNN.ipynb",
    "Q4_RNN_fundamentals.ipynb",
    "Q5_RNN_captioning.ipynb",
]

fatal = []
warnings = []

for name in NOTEBOOKS:
    path = ROOT / name

    if not path.exists():
        fatal.append(
            f"Thiếu notebook: {name}"
        )
        continue

    try:
        nb = nbformat.read(
            path,
            as_version=4,
        )
    except Exception as exc:
        fatal.append(
            f"{name}: không đọc được notebook: {exc}"
        )
        continue

    unresolved = []

    for idx, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue

        if "raise NotImplementedError" in cell.source:
            unresolved.append(idx)

    if unresolved:
        fatal.append(
            f"{name}: còn NotImplementedError ở các code cell {unresolved}"
        )

    answer_cells = [
        cell
        for cell in nb.cells
        if (
            cell.cell_type == "markdown"
            and "### ✍️ Câu trả lời của bạn" in cell.source
        )
    ]

    for i, cell in enumerate(
        answer_cells,
        start=1,
    ):
        body = cell.source.replace(
            "### ✍️ Câu trả lời của bạn",
            "",
        )
        body = re.sub(
            r"<!--.*?-->",
            "",
            body,
            flags=re.S,
        ).strip()

        if not body:
            warnings.append(
                f"{name}: câu trả lời #{i} có vẻ đang để trống"
            )

    n_outputs = sum(
        len(cell.get("outputs", []))
        for cell in nb.cells
        if cell.cell_type == "code"
    )

    if n_outputs == 0:
        warnings.append(
            f"{name}: chưa lưu đầu ra của các code cell"
        )

for required_file in [
    "best_cnn.pt",
    "caption_rnn.pt",
]:
    if not (ROOT / required_file).exists():
        fatal.append(
            f"Thiếu {required_file}"
        )

print("=" * 72)
print("KIỂM TRA CẤU TRÚC BÀI NỘP — ASSIGNMENT 2")
print("=" * 72)

if warnings:
    print("\nCẢNH BÁO:")
    for item in warnings:
        print(" -", item)

if fatal:
    print("\nLỖI CẦN SỬA:")
    for item in fatal:
        print(" -", item)
    raise SystemExit(1)

print(
    "\nKiểm tra cấu trúc đã hoàn tất. "
    "Kết quả này không đảm bảo code chắc chắn đúng."
)
