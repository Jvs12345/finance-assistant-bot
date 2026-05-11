from pathlib import Path
import argparse
import subprocess
import shutil

from pypdf import PdfReader


def has_readable_text(input_pdf: Path, max_pages: int = 5, min_chars: int = 80) -> bool:
    try:
        reader = PdfReader(str(input_pdf))
        parts = []
        for page in reader.pages[:max_pages]:
            parts.append((page.extract_text() or "").strip())
        return len(" ".join(parts).strip()) >= min_chars
    except Exception:
        return False


def _run_ocr_command(command, language: str):
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        msg = f"{exc.stdout or ''}\n{exc.stderr or ''}".lower()
        if language != "eng" and "language data" in msg:
            lang_index = command.index(language)
            command[lang_index] = "eng"
            subprocess.run(command, check=True)
        else:
            raise


def make_searchable_pdf(input_pdf: Path, output_pdf: Path, language: str = "eng"):
    if not input_pdf.exists():
        raise FileNotFoundError(f"Input PDF not found: {input_pdf}")
    if input_pdf.suffix.lower() != ".pdf":
        raise ValueError("Input file must be a PDF.")

    ocrmypdf_exe = shutil.which("ocrmypdf")
    if ocrmypdf_exe:
        command = [
            ocrmypdf_exe,
            "--force-ocr",
            "--deskew",
            "--clean",
            "--optimize",
            "1",
            "-l",
            language,
            str(input_pdf),
            str(output_pdf),
        ]
        _run_ocr_command(command, language=language)
        print(f"Searchable PDF created: {output_pdf}")
        return

    docker_exe = shutil.which("docker")
    if not docker_exe:
        raise RuntimeError("Neither ocrmypdf nor Docker is available on this machine.")

    work_dir = str(input_pdf.parent.resolve())
    command = [
        docker_exe,
        "run",
        "--rm",
        "-v",
        f"{work_dir}:/work",
        "jbarlow83/ocrmypdf",
        "--force-ocr",
        "--deskew",
        "--clean",
        "--optimize",
        "1",
        "-l",
        language,
        f"/work/{input_pdf.name}",
        f"/work/{output_pdf.name}",
    ]
    _run_ocr_command(command, language=language)
    print(f"Searchable PDF created: {output_pdf}")


def process_directory(directory: Path, language: str, force: bool):
    pdfs = sorted(p for p in directory.glob("*.pdf") if p.is_file())
    if not pdfs:
        print(f"No PDFs found in {directory}")
        return

    for pdf in pdfs:
        if pdf.stem.lower().endswith("_searchable"):
            continue
        output_pdf = pdf.with_name(pdf.stem + "_searchable.pdf")
        if not force and has_readable_text(pdf):
            print(f"Skip (already readable): {pdf.name}")
            continue
        make_searchable_pdf(pdf, output_pdf, language=language)


def main():
    parser = argparse.ArgumentParser(
        description="Convert non-readable PDFs to searchable OCR PDFs."
    )
    parser.add_argument("input_pdf", nargs="?", help="Path to one scanned/non-readable PDF")
    parser.add_argument("-o", "--output", help="Output searchable PDF path")
    parser.add_argument(
        "-l",
        "--language",
        default="eng",
        help="OCR language, for example: eng, nld, deu, or eng+nld",
    )
    parser.add_argument("--dir", help="Process all PDFs in this directory")
    parser.add_argument("--force", action="store_true", help="Force OCR even if text is already readable")

    args = parser.parse_args()

    if args.dir:
        process_directory(Path(args.dir), language=args.language, force=args.force)
        return

    if not args.input_pdf:
        raise ValueError("Provide an input PDF path or use --dir.")

    input_pdf = Path(args.input_pdf)
    output_pdf = Path(args.output) if args.output else input_pdf.with_name(input_pdf.stem + "_searchable.pdf")
    make_searchable_pdf(input_pdf, output_pdf, args.language)


if __name__ == "__main__":
    main()
