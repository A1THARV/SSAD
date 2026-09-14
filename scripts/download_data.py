"""Fetch and verify every dataset the prototype needs (PRD §7, M0).

Design rule, from PRD §7: **never silently skip a source**. Each source is a
discrete, resumable step with a size/checksum sanity check, and an unreachable
source fails loudly with an actionable message.

The manual drop-in contract
---------------------------
Some sources (the scidb.cn bundles) sit behind an account wall that no script
can pass. For those, this script does not pretend to download anything. It
prints the dataset page, the exact path the file is expected at, and then exits
non-zero. A file downloaded by hand and dropped at that path passes
verification on the next run and the pipeline proceeds identically -- so a
registration wall costs five minutes rather than blocking a milestone
(§11 edge case 14).

Usage::

    python scripts/download_data.py              # fetch everything possible
    python scripts/download_data.py --list       # show sources and status
    python scripts/download_data.py --verify     # check only, never download
    python scripts/download_data.py --only salinas abu
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data"

CHUNK = 1 << 20  # 1 MiB


# --------------------------------------------------------------------------
# Source definitions
# --------------------------------------------------------------------------


@dataclass
class Source:
    """One downloadable (or hand-placed) artifact.

    Attributes:
        key: Selector for ``--only``.
        group: Which branch needs it, for the summary table.
        target: Path relative to ``data/``.
        url: Direct download URL, or ``None`` for manual-only sources.
        page: Human-facing landing page, printed when manual action is needed.
        min_bytes: Floor below which the file is treated as truncated. Catches
            the classic failure where a login redirect saves an HTML error page
            under a ``.mat`` filename.
        sha256: Pinned digest once known. ``None`` means unpinned -- the script
            computes and prints the digest so it can be recorded here.
        manual_reason: Why automation cannot fetch this one.
        required_for: Milestones that block without it.
    """

    key: str
    group: str
    target: str
    url: str | None = None
    page: str = ""
    min_bytes: int = 1024
    sha256: str | None = None
    manual_reason: str = ""
    required_for: str = ""
    kind: str = "file"  # file | dir | git | gdrive
    notes: str = ""

    @property
    def path(self) -> Path:
        return DATA_ROOT / self.target


SOURCES: list[Source] = [
    # ---------------- HAD100 benchmark (PRD v2, change C1) ----------------
    # Primary hyperspectral dataset. The anomaly-free training split is the
    # point: change C2 moves autoencoder training off the test scene and onto
    # background-only data, which is what makes Eq. 1's reconstruction-error
    # premise defensible.
    Source(
        key="had100",
        group="hyperspectral",
        target="hyperspectral/had100/had100_data.zip",
        kind="gdrive",
        url="10Z6f-8ELatxdAi31IDg2Mgbl92_4EnCJ",
        page="https://zhaoxuli123.github.io/HAD100/",
        min_bytes=3 * 1024 * 1024 * 1024,
        sha256="6f91035543b7bc7806ebc555cd5411d320f2810f0af8dfcf20fe4f331227c19f",
        required_for="M1, M2 (primary benchmark)",
        notes=(
            "3.75 GB. Verified contents: 94 annotated AVIRIS-NG source scenes that "
            "HAD100's own main.py crops into exactly 100 test images at 64x64 (six "
            "scenes yield two crops each -- hence the name), plus 522 anomaly-free "
            "training scenes (260 AVIRIS-NG at 100x100x425, 262 AVIRIS-Classic at "
            "81x81x224) that crop into 2,088 training patches. "
            "Run HAD100/main.py after extracting: its band selection and crop "
            "protocol are what make the change-C3 baseline comparison valid. "
            "Cite: Li et al., IEEE TGRS 61, 2023, doi:10.1109/TGRS.2023.3258067."
        ),
    ),
    Source(
        key="had100_baselines",
        group="hyperspectral",
        target="hyperspectral/had100/had100_baselines.zip",
        kind="gdrive",
        url="1d4Lr1jqD7hzhTpIWIU-49smFa14RREEe",
        page="https://zhaoxuli123.github.io/HAD100/",
        min_bytes=100 * 1024 * 1024,
        sha256="f50133d95760ab15e1d0547028364a7b647069a4d41a3349d9cee38517b4a954",
        required_for="M2 comparison (change C3)",
        notes=(
            "17 published baseline methods + 2 variants, with results and evaluation "
            "code. Consumed as published numbers -- the baselines are NOT "
            "re-implemented (change C3)."
        ),
    ),
    Source(
        key="hyperad",
        group="hyperspectral",
        target="hyperspectral/hyperad",
        kind="git",
        url="https://github.com/WiseGamgee/HyperAD.git",
        page="https://github.com/WiseGamgee/HyperAD",
        min_bytes=1024 * 100,
        required_for="Cited-method comparison (paper reference [6], ERX)",
        notes=(
            "Garske et al.'s ERX repository, with three annotated .npy datasets "
            "(beach / synthetic-AVIRIS / Sequoia wildfire). Running against its own "
            "data turns a citation into a direct comparison."
        ),
    ),
    # ---------------- Hyperspectral scenes (PRD §7.1) ----------------
    # The canonical ehu.eus host returns 403 to any scripted request (checked
    # with browser User-Agent and Referer alike), so these come from the Zenodo
    # mirror the PRD §7.1 names. The scene contents are identical.
    Source(
        key="salinas",
        group="hyperspectral",
        target="hyperspectral/salinas/Salinas_corrected.mat",
        url="https://zenodo.org/records/15771735/files/Salinas_corrected.mat?download=1",
        page="https://zenodo.org/records/15771735",
        min_bytes=20 * 1024 * 1024,
        sha256="5ec1c0d22f56d18ecd336f8e35735863c0f160682e04e0c18ef3f89a3334d87d",
        required_for="M1, M2",
        notes=(
            "512x217x204 int16 (224 bands less 20 water-absorption bands), AVIRIS. "
            "The .mat is compressed, hence ~26 MB on disk for a ~45 MB array."
        ),
    ),
    Source(
        key="salinas",
        group="hyperspectral",
        target="hyperspectral/salinas/Salinas_gt.mat",
        url="https://zenodo.org/records/15771735/files/Salinas_gt.mat?download=1",
        page="https://zenodo.org/records/15771735",
        min_bytes=2 * 1024,
        sha256="ecfab4d31ef5553f097943235d8ea502038eb4a2067b2ad10b33e37c949955e2",
        required_for="§11 edge case 3 fixture (change C5)",
        notes=(
            "Confirmed 17 values (background + 16 land-cover classes) -- a class map, "
            "NOT an anomaly mask. Change C5 resolves this by NOT manufacturing anomaly "
            "labels from it: Salinas is now the zero-anomaly fixture used to measure "
            "false-positive rate at the chosen threshold."
        ),
    ),
    # Salinas-A was option (b) of the Salinas labelling decision. Change C5
    # resolves that decision differently -- Salinas becomes the zero-anomaly
    # fixture and no anomaly labels are manufactured from it -- so the subscene
    # is no longer needed and its ehu.eus 403 no longer blocks anything.
    Source(
        key="abu",
        group="hyperspectral",
        target="hyperspectral/abu",
        kind="git",
        url="https://github.com/sxt1996/Airport-Beach-Urban-ABU.git",
        page="https://github.com/sxt1996/Airport-Beach-Urban-ABU",
        min_bytes=1024 * 1024,
        required_for="M1, M2 (cross-scene generalization)",
        notes="13 scenes with binary anomaly masks; band counts vary 189-205.",
    ),
    # San Diego was dropped by change C1: registration-walled on scidb.cn, and
    # now redundant -- abu-airport-1 is the same 100x100 AVIRIS airport framing
    # with a real binary mask. See REJECTED at the bottom of this file.

    # ---------------- Thermal (change C8, resolved) ----------------
    # FLAME 3 supersedes the aerial/satellite split: one aerial wildfire dataset
    # serves BOTH Objective 3 (detection, with fire masks) and Objective 4
    # (Eq. 5, via radiometric TIFFs carrying per-pixel temperature). It also
    # pairs with HyperAD's Sequoia wildfire scene on the hyperspectral side, so
    # both branches detect the same class of event and Fig. 4's fusion is
    # meaningful rather than two unrelated outputs.
    Source(
        key="flame3",
        group="thermal",
        target="thermal/flame3_cv_subset",
        kind="dir",
        url=None,
        page="https://www.kaggle.com/datasets/brycehopkins/flame-3-computer-vision-subset-sycan-marsh",
        min_bytes=1024 * 1024 * 1024,
        manual_reason=(
            "Both hosts are account-gated. Kaggle needs an API token at "
            "~/.kaggle/kaggle.json; IEEE DataPort needs a free account (no paid "
            "subscription, no IEEE membership)."
        ),
        required_for="M4 and M5 -- the entire thermal branch",
        notes=(
            "CV subset only (~6.23 GB): 738 image quartets = 2,952 images, of which "
            "622 fire and 116 no-fire sets. Each quartet is raw RGB, raw thermal JPEG, "
            "FOV-corrected RGB, and a RADIOMETRIC thermal TIFF at 640x512. The TIFF is "
            "the one that matters -- per-pixel temperature is what unblocks Eq. 5. "
            "With a Kaggle token: kaggle datasets download -d "
            "brycehopkins/flame-3-computer-vision-subset-sycan-marsh. "
            "Alternative host: IEEE DataPort, DOI 10.21227/w0mz-aq48. "
            "Cite: Hopkins et al., arXiv:2412.02831.\n"
            "AFTER EXTRACTING: Kaggle unpacks to 'FLAME 3 CV Dataset (Sycan Marsh)'. "
            "Rename it to flame3_cv_subset -- the spaces and parentheses break shell "
            "quoting in enough places to be worth the one-time rename.\n"
            "VERIFIED ON DISK 2026-09-04: 7,535,535,691 bytes, 2,952 files, laid out as "
            "{Fire,No Fire}/{RGB/{Raw,Corrected FOV},Thermal/{Raw JPG,Celsius TIFF}} -- "
            "622 fire and 116 no-fire quartets. The Celsius TIFF is radiometric as "
            "claimed: mode F float32 640x512, and a fire frame spans -22.8 to 573.0 C "
            "against a no-fire frame's -40.6 to 25.6 C (2.78% of pixels above 100 C "
            "versus 0.00%). That is per-pixel temperature, so Eq. 5 is unblocked."
        ),
    ),
    # NADIR plot set is deliberately NOT listed. It is 32.59 GB against ~51 GB
    # free with HAD100 already down, and preprocessed float32 cubes multiply
    # that. Add it only when the LSTM temporal path actually needs it, after
    # freeing disk or taking a partial burn rather than the full set.

    # The entry below is retained for one narrow purpose only.
    Source(
        key="thermal_negative_fixture",
        group="thermal",
        target="thermal/thermal_images_equip",
        kind="git",
        url="https://github.com/mohnaj-nit/thermal-images-equip.git",
        page="https://github.com/mohnaj-nit/thermal-images-equip",
        min_bytes=1024 * 100,
        required_for="§11 edge case 9 (negative fixture only)",
        notes=(
            "NOT a detection dataset for this project. Verified 320x240 8-bit RGB "
            "false-colour -- non-radiometric. Its role under change C8 is to prove the "
            "Planck module REFUSES such input rather than fabricating a temperature."
        ),
    ),
    # ---------------- Spectral libraries (PRD §7.2, change C4) ----------------
    Source(
        key="speclib_ecostress",
        group="speclib",
        # The FILE, not its directory. An earlier version pointed at
        # `speclib/ecostress` with the default kind="file", so verify() called
        # is_file() on a directory and reported "missing" wherever the download
        # was actually placed -- the same defect that hid FLAME 3. A source's
        # target must name the thing being verified.
        target="speclib/ecostress/spectra_all.txt",
        url=None,
        page="https://speclib.jpl.nasa.gov/download",
        # The full library is thousands of spectra of a few hundred samples
        # each, so a valid export is tens of MB. 1 MB is a floor that catches a
        # login page or a truncated transfer without guessing the exact size;
        # the digest is unpinned, so the first successful verify prints it for
        # recording here.
        min_bytes=1024 * 1024,
        manual_reason=(
            "Spectra are selected through the site's search interface at "
            "https://speclib.jpl.nasa.gov/library, so the subset is chosen by hand.\n"
            "ORDERED 2026-09-14: order ecospeclib-1789396065385, one item, "
            "spectra_all.txt. JPL emails a download link once the order is "
            "processed; drop the file at the target path below.\n"
            "(The earlier 2026-09-04 to 2026-09-13 block was scheduled maintenance on "
            "the ordering page, now cleared.)"
        ),
        required_for="M3 (SAM) and M5 (the epsilon prior) -- two subsets, one file",
        notes=(
            "PRIMARY library under change C4: 3,400+ spectra, and a superset here -- it "
            "already incorporates the USGS-Reston, JHU and JPL sub-libraries. "
            "Cite: Meerdink et al. (2019) RSE 230:111196; Baldridge et al. (2009) RSE "
            "113:711-715.\n"
            "TWO SUBSETS, NOT ONE. scripts/build_speclib.py extracts both from this "
            "single file: (a) a REFLECTANCE subset for Objective 2, minerals with SWIR "
            "absorption features, sized for a SWIR-diagnostic scene per RESULTS.md "
            "section 3 rather than for HAD100; (b) an LWIR EMISSIVITY subset for "
            "Objective 4, vegetation/char/ash/soil plus a metal kept deliberately as "
            "the out-of-class counterexample behind the 307 K error in section 13. The "
            "second turns Eq. 5's emissivity prior from a hardcoded constant into a "
            "cited measurement.\n"
            "NOTE: spectra_all.txt bundles the JHU, JPL and USGS sub-libraries on "
            "different wavelength grids, so many LWIR entries never reach SWIR and vice "
            "versa. build_speclib.py checks coverage PER ENTRY at build time and reports "
            "what it dropped -- otherwise resample_to_grid's (correct) refusal to "
            "extrapolate surfaces mid-match instead."
        ),
    ),
    Source(
        key="speclib_usgs",
        group="speclib",
        target="speclib/usgs_splib07",
        url=None,
        page="https://doi.org/10.5066/F7RR1WDJ",
        min_bytes=1024 * 1024,
        manual_reason=(
            "Optional under change C4 -- only needed for a mineral ECOSTRESS lacks. "
            "Use the DOI or the splib07-loader package; the ScienceBase catalogue page "
            "is JavaScript-rendered, so scraping its HTML returns an empty shell."
        ),
        required_for="M3, optional supplement to ECOSTRESS",
        notes="Loader: https://github.com/brianschubert/splib07-loader",
    ),
]

#: Sources deliberately removed, recorded so they are not re-suggested later.
REJECTED = {
    "scidb San Diego bundle": "Registration wall; redundant with abu-airport-1 (C1).",
    "scidb thermal equipment set": "Registration wall, and wrong domain (C8).",
    "ehu.eus scenes page": "403s all scripted requests, including browser UA + Referer.",
    "HAD100 Baidu mirror": "Requires a Baidu account; the Google Drive link is equivalent.",
    "HIT-UAV": "Verified 8-bit: source video is 640x512 uint8 greyscale H.264. Superseded by FLAME 3.",
    "M3OT": "Video-derived from a Mavic 3T, so 8-bit; urban tracking, not wildfire. Superseded by FLAME 3.",
    "ECOSTRESS L2T_LSTE": "Satellite thermal product, no longer needed -- FLAME 3 is radiometric (C8).",
    # CORRECTED 2026-09-04, from the files rather than the dataset description.
    # The CV subset ships NO pixel masks -- no mask images, no label CSV, only
    # the Fire/ vs No Fire/ folder split, i.e. IMAGE-level labels. The earlier
    # note here claimed per-pixel fire masks and was wrong.
    "MODIS/VIIRS FIRMS": (
        "Still not needed, but for a different reason than first recorded. FLAME 3's CV "
        "subset gives image-level fire/no-fire labels, not masks, so M4 is an "
        "image-level detection result. FIRMS is 375 m/px satellite thermal -- far too "
        "coarse to supply pixel masks for 640x512 UAV frames, so it could not repair "
        "the gap even if added. Pixel-level maps are reported qualitatively and "
        "labelled as unvalidated at pixel level."
    ),
    "LTIR": "Ground-level tracking benchmark, non-radiometric, labels need reframing (C8).",
}

# NOTE: the ECOSTRESS *spectral library* (speclib.jpl.nasa.gov) is a different
# thing from the ECOSTRESS L2T_LSTE satellite product rejected above. The
# library stays -- it is Objective 2's SAM reference under change C4.


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------


@dataclass
class Status:
    source: Source
    ok: bool
    detail: str
    digest: str | None = None


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def dir_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _nearby_archive(path: Path) -> Path | None:
    """Find an un-extracted archive sitting where a directory was expected.

    The likeliest way a hand-placed dataset fails: the zip is downloaded to the
    right place but never unpacked. Saying so beats "directory not found".
    """
    parent = path.parent
    if not parent.is_dir():
        return None
    for candidate in sorted(parent.iterdir()):
        if candidate.is_file() and candidate.suffix.lower() in {".zip", ".tar", ".gz", ".7z"}:
            return candidate
    return None


def looks_like_html(path: Path) -> bool:
    """Detect a login page or error page saved under a data filename."""
    try:
        head = path.open("rb").read(512).lstrip().lower()
    except OSError:
        return False
    return head.startswith(b"<!doctype html") or head.startswith(b"<html")


def verify(source: Source) -> Status:
    path = source.path

    if source.kind == "git":
        if not path.is_dir():
            return Status(source, False, "not cloned")
        size = dir_size(path)
        if size < source.min_bytes:
            return Status(source, False, f"clone looks empty ({size:,} bytes)")
        return Status(source, True, f"{size:,} bytes")

    if source.kind == "dir":
        # A hand-extracted archive, not a single file. Checked by tree size and
        # file count rather than by checksum: the archive's own digest says
        # nothing once it is unpacked, and re-zipping to verify would cost more
        # than the check is worth.
        if not path.is_dir():
            hint = ""
            archive = _nearby_archive(path)
            if archive is not None:
                hint = (
                    f" -- found {archive.name} beside it; extract it INTO "
                    f"{path.name}/ rather than leaving it zipped"
                )
            return Status(source, False, f"directory not found{hint}")
        size = dir_size(path)
        n_files = sum(1 for p in path.rglob("*") if p.is_file())
        if size < source.min_bytes:
            return Status(
                source,
                False,
                f"only {size:,} bytes in {n_files:,} files, expected at least "
                f"{source.min_bytes:,} -- extraction may be incomplete",
            )
        return Status(source, True, f"{size:,} bytes across {n_files:,} files")

    if not path.is_file():
        return Status(source, False, "missing")

    size = path.stat().st_size
    if looks_like_html(path):
        return Status(
            source,
            False,
            "file is HTML, not data -- the server returned a login or error page",
        )
    if size < source.min_bytes:
        return Status(
            source, False, f"truncated: {size:,} bytes, expected at least {source.min_bytes:,}"
        )

    digest = sha256_of(path)
    if source.sha256 and digest != source.sha256:
        return Status(source, False, f"checksum mismatch (got {digest[:12]}...)", digest)
    if source.sha256 is None:
        return Status(source, True, f"{size:,} bytes, unpinned sha256={digest[:16]}...", digest)
    return Status(source, True, f"{size:,} bytes, checksum verified")


# --------------------------------------------------------------------------
# Fetching
# --------------------------------------------------------------------------


def fetch_file(source: Source, timeout: int = 60) -> None:
    """Download to a ``.part`` file, then move into place once complete.

    Downloading to a temporary name means an interrupted run never leaves a
    truncated file that later looks valid.
    """
    import requests

    path = source.path
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")

    headers = {"User-Agent": "anomaly-prototype/0.1 (research prototype)"}
    resume_from = part.stat().st_size if part.exists() else 0
    if resume_from:
        headers["Range"] = f"bytes={resume_from}-"

    with requests.get(source.url, headers=headers, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        mode = "ab" if resume_from and resp.status_code == 206 else "wb"
        if mode == "wb":
            resume_from = 0
        total = int(resp.headers.get("Content-Length", 0)) + resume_from
        written = resume_from
        with part.open(mode) as fh:
            for chunk in resp.iter_content(CHUNK):
                fh.write(chunk)
                written += len(chunk)
                if total:
                    pct = 100 * written / total
                    print(f"\r    {written / 1e6:7.1f} / {total / 1e6:.1f} MB ({pct:5.1f}%)",
                          end="", flush=True)
        if total:
            print()

    part.replace(path)


def fetch_gdrive(source: Source) -> None:
    """Download a Google Drive file by id.

    ``gdown`` rather than ``requests`` because Drive interposes a
    confirmation-token page on large files, which a plain GET silently saves as
    HTML -- exactly the failure ``looks_like_html`` exists to catch.
    """
    import gdown

    path = source.path
    path.parent.mkdir(parents=True, exist_ok=True)
    result = gdown.download(id=source.url, output=str(path), quiet=False, resume=True)
    if result is None:
        raise RuntimeError(
            "gdown returned no file -- Drive may be rate-limiting this id, or the "
            "share link may have changed"
        )


def fetch_git(source: Source) -> None:
    path = source.path
    path.parent.mkdir(parents=True, exist_ok=True)
    if not shutil.which("git"):
        raise RuntimeError("git is not on PATH, so this source cannot be cloned")
    if path.is_dir():
        shutil.rmtree(path)
    subprocess.run(
        ["git", "clone", "--depth", "1", source.url, str(path)],
        check=True,
        capture_output=True,
        text=True,
    )


def fetch(source: Source) -> Status:
    if source.url is None:
        return Status(source, False, "manual download required")
    try:
        if source.kind == "git":
            fetch_git(source)
        elif source.kind == "gdrive":
            fetch_gdrive(source)
        else:
            fetch_file(source)
    except Exception as exc:  # noqa: BLE001 -- the reason is reported, not swallowed
        detail = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
        return Status(source, False, f"download failed: {detail}")
    return verify(source)


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def print_manual_instructions(statuses: list[Status]) -> None:
    print("\n" + "=" * 78)
    print("ACTION REQUIRED -- these sources could not be fetched automatically")
    print("=" * 78)
    for st in statuses:
        s = st.source
        print(f"\n  {s.key}  ({s.group})")
        print(f"    why      : {s.manual_reason or st.detail}")
        print(f"    get it   : {s.page}")
        print(f"    place at : {s.path}")
        print(f"    needed by: {s.required_for or 'n/a'}")
        if s.notes:
            print(f"    note     : {s.notes}")
    print(
        "\n  Drop each file at the path above and re-run this script; it verifies\n"
        "  the file and the pipeline then behaves identically to an automated fetch."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", nargs="+", metavar="KEY", help="fetch only these source keys")
    parser.add_argument("--verify", action="store_true", help="check what is present; download nothing")
    parser.add_argument("--list", action="store_true", help="list sources and exit")
    args = parser.parse_args()

    selected = SOURCES
    if args.only:
        wanted = set(args.only)
        unknown = wanted - {s.key for s in SOURCES}
        if unknown:
            print(f"Unknown source key(s): {sorted(unknown)}", file=sys.stderr)
            print(f"Known keys: {sorted({s.key for s in SOURCES})}", file=sys.stderr)
            return 2
        selected = [s for s in SOURCES if s.key in wanted]

    if args.list:
        for s in SOURCES:
            mode = "manual" if s.url is None else s.kind
            print(f"  {s.key:24} {s.group:14} {mode:7} -> data/{s.target}")
        print("\n  Rejected sources (do not re-suggest):")
        for name, why in REJECTED.items():
            print(f"    {name:30} {why}")
        return 0

    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    results: list[Status] = []

    for source in selected:
        label = f"{source.key} -> data/{source.target}"
        status = verify(source)
        if status.ok:
            print(f"[ok]      {label}\n            {status.detail}")
        elif args.verify:
            print(f"[MISSING] {label}\n            {status.detail}")
        else:
            print(f"[fetch]   {label}")
            status = fetch(source)
            marker = "[ok]     " if status.ok else "[FAILED] "
            print(f"{marker} {label}\n            {status.detail}")
        results.append(status)

    ok = [r for r in results if r.ok]
    failed = [r for r in results if not r.ok]

    print("\n" + "-" * 78)
    print(f"{len(ok)} of {len(results)} sources present and verified.")

    if failed:
        print_manual_instructions(failed)
        blocking = [r for r in failed if r.source.group != "speclib"]
        if blocking:
            print(
                f"\n{len(blocking)} source(s) still block a milestone. "
                "Nothing was faked or skipped silently."
            )
        return 1

    print("All sources verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
