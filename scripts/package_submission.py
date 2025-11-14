"""
Package repository code + dataset into a ZIP suitable for LMS submission.
- Includes: python code (*.py), web app files, README*, requirements.txt, dataset/ (configurable)
- Excludes: checkpoints/ (by default), large files > size threshold (configurable)

Usage:
    python scripts/package_submission.py --output ../submission_code_dataset.zip

Run this from repository root (where this script lives at ./scripts)
"""

import os
import zipfile
import argparse
from pathlib import Path

# Configuration: adjust as needed
INCLUDE_PATTERNS = ["*.py", "README*", "requirements.txt", "*.md"]
INCLUDE_DIRS = ["dataset", "app.py", "gradcam.py", "train.py", "train_anti_overfitting.py", "utils.py", "compare.py", "evaluate.py", "."]
EXCLUDE_DIRS = ["checkpoints", "outputs", ".git", "__pycache__"]
MAX_FILE_SIZE_BYTES = 200 * 1024 * 1024  # 200 MB (skip files larger than this by default)


def should_exclude(path: Path):
    # Exclude by directory name
    for ex in EXCLUDE_DIRS:
        if ex in path.parts:
            return True
    # Exclude model checkpoint binaries
    if path.suffix in ['.pth', '.pt', '.h5']:
        return True
    # Exclude large files
    try:
        if path.is_file() and path.stat().st_size > MAX_FILE_SIZE_BYTES:
            return True
    except Exception:
        pass
    return False


def add_file_to_zip(z: zipfile.ZipFile, file_path: Path, base_dir: Path):
    arcname = file_path.relative_to(base_dir)
    z.write(file_path, arcname)


def gather_files(base_dir: Path, exclude_dataset: bool = False):
    files = set()
    # Include the configured directories and patterns
    for d in INCLUDE_DIRS:
        dpath = base_dir / d
        # Optionally skip dataset directory entirely
        if exclude_dataset and dpath.name.lower() == 'dataset':
            continue
        # If d is an explicit file and exists
        if dpath.exists() and dpath.is_file():
            if not should_exclude(dpath):
                files.add(dpath)
            continue
        if not dpath.exists():
            # allow patterns or skip
            continue
        if dpath.is_dir():
            for p in dpath.rglob('*'):
                if p.is_file() and not should_exclude(p):
                    if p.suffix.lower() in ['.py', '.md', '.txt', '.png', '.jpg', '.jpeg', '.csv'] or p.name.lower().startswith('readme'):
                        files.add(p)
        else:
            # ignore
            pass

    # Also include top-level python files by pattern
    for pattern in INCLUDE_PATTERNS:
        for p in base_dir.glob(pattern):
            if p.is_file() and not should_exclude(p):
                files.add(p)

    # Always include this script
    script_file = base_dir / 'scripts' / 'package_submission.py'
    if script_file.exists():
        files.add(script_file)

    return sorted(files)


def make_zip(output: Path, base_dir: Path, exclude_dataset: bool = False, code_only: bool = False):
    # If code_only is requested, build a tight include list to avoid scanning large media folders
    if code_only:
        # Select typical backend/core files and folders only
        code_include = ['app.py', 'gradcam.py', 'train.py', 'train_anti_overfitting.py', 'utils.py', 'evaluate.py', 'compare.py', 'scripts', 'requirements.txt', 'README.md', 'README_SUBMISSION.md']
        # Temporarily override INCLUDE_DIRS for a focused gather
        global INCLUDE_DIRS
        old_include = INCLUDE_DIRS
        INCLUDE_DIRS = code_include
        try:
            files = gather_files(base_dir, exclude_dataset=exclude_dataset)
        finally:
            INCLUDE_DIRS = old_include
    else:
        files = gather_files(base_dir, exclude_dataset=exclude_dataset)
    if not files:
        raise RuntimeError('No files found to include. Check INCLUDE_DIRS and repository structure.')
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_STORED) as z:
        for f in files:
            try:
                add_file_to_zip(z, f, base_dir)
            except Exception as e:
                print(f"Skipping {f} due to error: {e}")
    print(f"Created {output} with {len(files)} files")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Package repo code and dataset into a submission ZIP')
    parser.add_argument('--output', '-o', type=str, default='../submission_code_dataset.zip', help='Output zip path (relative to repo root)')
    parser.add_argument('--base', '-b', type=str, default='.', help='Repository base dir')
    parser.add_argument('--exclude-dataset', action='store_true', help='Exclude the dataset directory and related files from the ZIP')
    parser.add_argument('--code-only', action='store_true', help='Package only core backend code files and scripts (no dataset, no media)')
    args = parser.parse_args()

    base = Path(args.base).resolve()
    out = (base / args.output).resolve()
    print(f"Packaging from {base} into {out}")
    make_zip(out, base, exclude_dataset=args.exclude_dataset, code_only=args.code_only)
    print('Done.')
