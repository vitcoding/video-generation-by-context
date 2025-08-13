#!/usr/bin/env python3
"""
Temporary validator for b_roll transcript chunks.

Validates the following for files under b_roll/data/video_generation/broll_prompts/chunks_temp:
1) Interval contiguity (no gaps/overlaps) between sub-chunks for early and remaining groups
2) Target segment sums across groups and sub-groups
3) Word timestamps fall within chunk boundaries (with small tolerance)
   - Relaxed ONLY for boundary-crossing duplicate words across adjacent chunks
4) early_end equals total_duration * EARLY_DURATION_RATIO (within tolerance)
5) Segment starts in workflow_generated_prompts.json align with nearest word start in transcription
6) Segment durations match end-start and target segment duration

Usage:
    python temp_broll_prompt_validate.py [--root <project_root>] [--chunks_dir <path>]

By default, project_root is the current file's parent directory and chunks_dir is:
    b_roll/data/video_generation/broll_prompts/chunks_temp
"""

import argparse
import bisect
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

# --------------------------- Data Models ---------------------------


@dataclass
class Chunk:
    file_path: Path
    chunk_id: int
    chunk_type: str
    start_time: float
    end_time: float
    duration: float
    target_segments: int
    word_timestamps: List[Dict]

    @property
    def is_early(self) -> bool:
        return self.chunk_type.startswith("early")

    @property
    def is_remaining(self) -> bool:
        return self.chunk_type.startswith("remaining")


# --------------------------- Helpers ---------------------------


def load_constants(project_root: Path) -> Tuple[float]:
    """Load EARLY_DURATION_RATIO from b_roll/constants.py by file path to avoid package side-effects."""
    constants_path = project_root / "b_roll" / "constants.py"
    if not constants_path.exists():
        raise RuntimeError(f"constants.py not found at {constants_path}")

    spec = importlib.util.spec_from_file_location(
        "_validator_constants", constants_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to create import spec for constants.py")

    module = importlib.util.module_from_spec(spec)
    sys.modules["_validator_constants"] = module
    spec.loader.exec_module(module)

    if not hasattr(module, "EARLY_DURATION_RATIO"):
        raise RuntimeError("EARLY_DURATION_RATIO not defined in constants.py")

    return (float(getattr(module, "EARLY_DURATION_RATIO")),)


def read_chunks_from_dir(directory: Path) -> List[Chunk]:
    """Read all JSON chunk files from the provided directory into Chunk objects."""
    if not directory.exists() or not directory.is_dir():
        return []

    chunks: List[Chunk] = []
    for json_file in sorted(directory.glob("*.json")):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        chunks.append(
            Chunk(
                file_path=json_file,
                chunk_id=int(data.get("chunk_id", -1)),
                chunk_type=str(data.get("chunk_type", "")),
                start_time=float(data.get("start_time", 0.0)),
                end_time=float(data.get("end_time", 0.0)),
                duration=float(data.get("duration", 0.0)),
                target_segments=int(data.get("target_segments", 0)),
                word_timestamps=list(data.get("word_timestamps", [])),
            )
        )
    return chunks


def group_subchunks(chunks: List[Chunk]) -> Tuple[List[Chunk], List[Chunk]]:
    """Split chunks into early_* and remaining_* groups and sort by start_time."""
    early = sorted(
        [c for c in chunks if c.is_early], key=lambda c: c.start_time
    )
    rem = sorted(
        [c for c in chunks if c.is_remaining], key=lambda c: c.start_time
    )
    return early, rem


def approximately_equal(a: float, b: float, tol: float = 1e-2) -> bool:
    return abs(a - b) <= tol


# --------------------------- Validators ---------------------------


def validate_contiguity(
    chunks: List[Chunk],
    expected_start: float,
    expected_end: float,
    label: str,
    tol: float,
    issues: List[str],
) -> bool:
    """Validate sub-chunks are contiguous and exactly cover [expected_start, expected_end]."""
    ok = True
    if not chunks:
        issues.append(f"[{label}] No sub-chunks found")
        return False

    # Start boundary
    first = chunks[0]
    if not approximately_equal(first.start_time, expected_start, tol):
        issues.append(
            f"[{label}] First sub-chunk start {first.start_time:.3f} != expected {expected_start:.3f}"
        )
        ok = False

    # Adjacency
    for prev, curr in zip(chunks, chunks[1:]):
        if not approximately_equal(prev.end_time, curr.start_time, tol):
            issues.append(
                f"[{label}] Gap/overlap: prev.end {prev.end_time:.3f} != curr.start {curr.start_time:.3f}"
            )
            ok = False

    # End boundary
    last = chunks[-1]
    if not approximately_equal(last.end_time, expected_end, tol):
        issues.append(
            f"[{label}] Last sub-chunk end {last.end_time:.3f} != expected {expected_end:.3f}"
        )
        ok = False

    return ok


def validate_target_segment_sums(
    initial_early: Chunk,
    initial_remaining: Chunk,
    early_sub: List[Chunk],
    rem_sub: List[Chunk],
    issues: List[str],
) -> bool:
    """Validate target_segments sums across initial and final groups."""
    ok = True
    early_sum = sum(c.target_segments for c in early_sub)
    rem_sum = sum(c.target_segments for c in rem_sub)

    if early_sum != initial_early.target_segments:
        issues.append(
            f"[segments] Early sub-chunks sum {early_sum} != initial early {initial_early.target_segments}"
        )
        ok = False
    if rem_sum != initial_remaining.target_segments:
        issues.append(
            f"[segments] Remaining sub-chunks sum {rem_sum} != initial remaining {initial_remaining.target_segments}"
        )
        ok = False

    total_initial = (
        initial_early.target_segments + initial_remaining.target_segments
    )
    total_final = early_sum + rem_sum
    if total_initial != total_final:
        issues.append(
            f"[segments] Total initial {total_initial} != total final {total_final}"
        )
        ok = False

    return ok


def validate_word_boundaries(
    chunks: List[Chunk], tol: float, issues: List[str], max_report: int = 10
) -> bool:
    """Validate words lie within chunk boundaries; allow boundary-crossing duplicates."""
    violations = 0
    for chunk in chunks:
        for w in chunk.word_timestamps:
            ws = float(w.get("start", 0.0))
            we = float(w.get("end", 0.0))
            # Accept if strictly inside (with tolerance)
            inside_ok = (ws >= chunk.start_time - tol) and (
                we <= chunk.end_time + tol
            )
            if inside_ok:
                continue
            # Relaxation ONLY for duplicates that cross chunk boundaries
            crosses_start = (ws < chunk.start_time) and (we > chunk.start_time)
            crosses_end = (ws < chunk.end_time) and (we > chunk.end_time)
            if crosses_start or crosses_end:
                continue  # allowed boundary-crossing duplicate

            if violations < max_report:
                issues.append(
                    f"[words] {chunk.file_path.name}: word [{ws:.3f}, {we:.3f}] outside chunk [{chunk.start_time:.3f}, {chunk.end_time:.3f}]"
                )
            violations += 1
            if violations >= max_report:
                break
        if violations >= max_report:
            break

    if violations > 0:
        issues.append(f"[words] Total word boundary violations: {violations}")
        return False
    return True


def validate_early_boundary_alignment(
    initial_early: Chunk,
    initial_remaining: Chunk,
    early_duration_ratio: float,
    issues: List[str],
    tol: float,
) -> bool:
    """Validate early_end equals total_duration * EARLY_DURATION_RATIO."""
    total_duration = max(initial_early.end_time, initial_remaining.end_time)
    expected_early_end = total_duration * early_duration_ratio
    if not approximately_equal(
        initial_early.end_time, expected_early_end, tol
    ):
        issues.append(
            f"[early_end] early_end {initial_early.end_time:.3f} != duration*ratio {expected_early_end:.3f} (ratio={early_duration_ratio})"
        )
        return False
    return True


def load_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_prompts_vs_transcript(
    project_root: Path,
    issues: List[str],
    word_tol: float = 0.5,
    duration_tol: float = 0.05,
) -> Tuple[bool, bool]:
    """
    Validate that:
    - Each segment start_time in workflow_generated_prompts.json aligns with a word.start in transcription within word_tol
    - Each segment duration matches end-start and target segment duration within duration_tol

    Returns:
    (starts_ok, durations_ok)
    """
    prompts_path = (
        project_root
        / "b_roll/data/video_generation/broll_prompts/workflow_generated_prompts.json"
    )
    transcript_path = (
        project_root
        / "b_roll/data/video_generation/audio_transcript/transcription_verbose_to_json.json"
    )

    if not prompts_path.exists():
        issues.append(f"[prompts] Not found: {prompts_path}")
        return (False, False)
    if not transcript_path.exists():
        issues.append(f"[transcript] Not found: {transcript_path}")
        return (False, False)

    prompts = load_json(prompts_path)
    transcript = load_json(transcript_path)

    segments = prompts.get("broll_segments", [])
    gen_info = prompts.get("generation_info", {})
    target_seg_duration = float(gen_info.get("segment_duration", 0.0))

    words = transcript.get("words", [])
    word_starts = sorted(
        float(w.get("start", 0.0)) for w in words if "start" in w
    )

    def nearest_word_delta(t: float) -> float:
        if not word_starts:
            return float("inf")
        idx = bisect.bisect_left(word_starts, t)
        candidates = []
        if idx < len(word_starts):
            candidates.append(abs(word_starts[idx] - t))
        if idx > 0:
            candidates.append(abs(word_starts[idx - 1] - t))
        return min(candidates) if candidates else float("inf")

    starts_ok = True
    durations_ok = True

    for i, seg in enumerate(segments, 1):
        s = float(seg.get("start_time", 0.0))
        e = float(seg.get("end_time", 0.0))
        d = float(seg.get("duration", 0.0))

        # Start alignment with nearest word start
        delta = nearest_word_delta(s)
        if delta > word_tol:
            issues.append(
                f"[prompts:start] Segment {i} start {s:.3f} not aligned to word start (min Δ={delta:.3f} > {word_tol})"
            )
            starts_ok = False

        # Duration checks: end-start vs reported duration, and vs target
        computed = e - s
        if not approximately_equal(computed, d, duration_tol):
            issues.append(
                f"[prompts:duration] Segment {i} duration mismatch: end-start={computed:.3f} vs reported {d:.3f}"
            )
            durations_ok = False
        if target_seg_duration > 0 and not approximately_equal(
            computed, target_seg_duration, duration_tol
        ):
            issues.append(
                f"[prompts:duration] Segment {i} end-start {computed:.3f} != target {target_seg_duration:.3f}"
            )
            durations_ok = False

    return (starts_ok, durations_ok)


# --------------------------- Main ---------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate b_roll chunks in chunks_temp"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).parent,
        help="Project root path (default: this script's directory)",
    )
    parser.add_argument(
        "--chunks_dir",
        type=Path,
        default=None,
        help="Path to chunks_temp directory (default: inferred from project root)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-2,
        help="Time tolerance in seconds",
    )
    args = parser.parse_args()

    project_root: Path = args.root.resolve()
    chunks_dir: Path = (
        args.chunks_dir
        if args.chunks_dir is not None
        else project_root
        / "b_roll"
        / "data"
        / "video_generation"
        / "broll_prompts"
        / "chunks_temp"
    )

    initial_dir = chunks_dir / "initial"
    final_dir = chunks_dir / "final"

    print(f"Project root: {project_root}")
    print(f"Chunks dir:   {chunks_dir}")

    # Load constants
    try:
        (early_duration_ratio,) = load_constants(project_root)
    except Exception as e:
        print(f"ERROR: {e}")
        return 1

    # Read chunks
    initial_chunks = read_chunks_from_dir(initial_dir)
    final_chunks = read_chunks_from_dir(final_dir)

    if not initial_chunks:
        print(
            "ERROR: No initial chunks found. Run the analyzer to generate chunks first."
        )
        return 1

    # Identify initial early and remaining
    initial_early_list = [c for c in initial_chunks if c.chunk_type == "early"]
    initial_remaining_list = [
        c for c in initial_chunks if c.chunk_type == "remaining"
    ]

    if len(initial_early_list) != 1 or len(initial_remaining_list) != 1:
        print(
            "ERROR: Expected exactly one 'early' and one 'remaining' chunk in initial/"
        )
        return 1

    initial_early = initial_early_list[0]
    initial_remaining = initial_remaining_list[0]

    issues: List[str] = []

    # 1) early_end alignment with EARLY_DURATION_RATIO
    early_ok = validate_early_boundary_alignment(
        initial_early,
        initial_remaining,
        early_duration_ratio,
        issues,
        args.tolerance,
    )
    print(
        f"[OK] early_end alignment"
        if early_ok
        else "[FAIL] early_end alignment"
    )

    # 2) Word timestamps within chunk boundaries (initial and final), allowing duplicates at boundaries
    words_initial_ok = validate_word_boundaries(
        initial_chunks, args.tolerance, issues
    )
    words_final_ok = validate_word_boundaries(
        final_chunks, args.tolerance, issues
    )
    words_ok = words_initial_ok and words_final_ok
    print(
        f"[OK] word timestamps within chunk boundaries (duplicates allowed)"
        if words_ok
        else "[FAIL] word timestamps contain out-of-bound entries"
    )

    # If no final chunks, skip further checks
    contiguity_ok = True
    segments_sum_ok = True
    if not final_chunks:
        print(
            "WARNING: No final chunks found. Skipping contiguity and sub-sum checks."
        )
    else:
        early_sub, rem_sub = group_subchunks(final_chunks)

        # 3) Contiguity and coverage for early and remaining sub-chunks
        contiguity_early_ok = validate_contiguity(
            early_sub,
            initial_early.start_time,
            initial_early.end_time,
            "early",
            args.tolerance,
            issues,
        )
        contiguity_rem_ok = validate_contiguity(
            rem_sub,
            initial_remaining.start_time,
            initial_remaining.end_time,
            "remaining",
            args.tolerance,
            issues,
        )
        contiguity_ok = contiguity_early_ok and contiguity_rem_ok
        print(
            "[OK] contiguity early+remaining"
            if contiguity_ok
            else "[FAIL] contiguity"
        )

        # 4) Target segment sums
        segments_sum_ok = validate_target_segment_sums(
            initial_early, initial_remaining, early_sub, rem_sub, issues
        )
        print(
            "[OK] target_segments sums"
            if segments_sum_ok
            else "[FAIL] target_segments sums"
        )

    # 5) Validate prompts vs transcript: starts alignment and durations
    starts_ok, durations_ok = validate_prompts_vs_transcript(
        project_root, issues
    )
    print(
        "[OK] prompts segment starts aligned to words"
        if starts_ok
        else "[FAIL] prompts segment starts alignment"
    )
    print(
        "[OK] prompts segment durations"
        if durations_ok
        else "[FAIL] prompts segment durations"
    )

    # Summary
    all_ok = all(
        [
            early_ok,
            words_ok,
            contiguity_ok,
            segments_sum_ok,
            starts_ok,
            durations_ok,
        ]
    )
    if issues:
        print(
            "\nVALIDATION RESULT: FAIL"
            if not all_ok
            else "\nVALIDATION RESULT: PASS (with notes)"
        )
        print(f"Issues found: {len(issues)}")
        for i, msg in enumerate(issues, 1):
            print(f" {i:02d}. {msg}")
        return 0 if all_ok else 2
    else:
        print("\nVALIDATION RESULT: PASS")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
