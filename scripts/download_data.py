from __future__ import annotations

import os
import zipfile

FOLDER = "https://drive.google.com/drive/folders/1gXlpzEduvb1RZAGlyTLsitZC2WPWw2I4"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    try:
        import gdown
    except ImportError as exc: 
        raise SystemExit("Please `pip install gdown` first.") from exc

    gdown.download_folder(FOLDER, output=DATA_DIR, quiet=False, use_cookies=False)
    for name in ("photos.csv.zip", "queries.csv.zip"):
        path = os.path.join(DATA_DIR, name)
        if os.path.exists(path):
            with zipfile.ZipFile(path) as zf:
                for member in zf.namelist():
                    if member.startswith("__MACOSX") or member.endswith("/"):
                        continue
                    zf.extract(member, DATA_DIR)
            print("unzipped", name)
    print("Data ready in", DATA_DIR)


if __name__ == "__main__":
    main()
