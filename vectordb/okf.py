"""
Ingest Open Knowledge Format (OKF) bundles into a VectorDB.

OKF is Google Cloud's open spec (v0.1/v0.2, published June 2026) for
representing knowledge as a directory of markdown files with YAML
frontmatter: https://github.com/GoogleCloudPlatform/knowledge-catalog

This module walks an OKF bundle, parses each concept document, and loads
it into a VectorDB via add_text() (semantic + keyword indexing), so you can
hybrid_search() across an OKF knowledge base by meaning, not just by
following its markdown cross-links by hand.

Built and tested against Google's real reference bundles (not a guess at
the spec) — see GoogleCloudPlatform/knowledge-catalog/okf/bundles/.

Spec notes this module relies on:
  - A bundle is a directory tree of .md files.
  - `index.md` and `log.md` are reserved filenames (navigation / history),
    not concept documents, at any directory level (SPEC §3.1, §8, §9).
  - Every other .md file is a concept: YAML frontmatter (only `type` is
    required) + a markdown body (SPEC §4).
  - Concept ID = the file's path within the bundle, with `.md` stripped
    (SPEC §2).
  - Cross-links are standard markdown links, absolute bundle-relative
    (`/tables/x.md`) or relative (`./x.md`) — there is no `links:`
    frontmatter field (SPEC §6.1).
"""
import os
import re
import datetime
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml

RESERVED_FILENAMES = {"index.md", "log.md"}

# Matches markdown links whose target looks like a .md file, so we can
# recover the concept graph (SPEC §6.1) even though OKF has no links: field.
_LINK_RE = re.compile(r"\]\(([^)\s]+\.md)(?:#[^)]*)?\)")


@dataclass
class OKFConcept:
    concept_id: str  # bundle-relative path, no .md suffix (SPEC §2)
    path: str  # bundle-relative path, with .md suffix
    type: str
    title: Optional[str]
    description: Optional[str]
    resource: Optional[str]
    tags: List[str]
    status: str  # draft | stable | deprecated (absent -> "stable", SPEC §5.4)
    stale_after: Optional[str]
    body: str
    links: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)  # any other frontmatter keys

    def searchable_text(self) -> str:
        """Text used for embedding + keyword indexing. Frontloads title and
        description since they're the highest-signal summary of a concept."""
        parts = [p for p in (self.title, self.description) if p]
        parts.append(self.body)
        return "\n\n".join(parts)

    def metadata(self) -> Dict[str, Any]:
        """Flat dict suitable for VectorDB metadata / filtering."""
        return {
            "okf_type": self.type,
            "title": self.title,
            "description": self.description,
            "resource": self.resource,
            "tags": self.tags,
            "status": self.status,
            "stale_after": self.stale_after,
            "path": self.path,
            "links": self.links,
            **{f"okf_{k}": v for k, v in self.extra.items() if _is_flat(v)},
        }


def _is_flat(v: Any) -> bool:
    """Keep metadata filterable (dict/list-of-dict values from things like
    `sources`/`generated` won't work well as exact-match filter targets, so
    we keep them out of the flat metadata dict but they're still in body
    text indirectly via title/description)."""
    return isinstance(v, (str, int, float, bool)) or v is None


def _stringify_date(v: Any) -> Optional[str]:
    """YAML auto-parses bare YYYY-MM-DD values into datetime.date objects,
    which aren't JSON-serializable for our metadata store. Normalize to
    ISO-format strings (SPEC §5.5 defines stale_after as an absolute date
    anyway, so string round-trips it losslessly)."""
    if v is None:
        return None
    if isinstance(v, (datetime.date, datetime.datetime)):
        return v.isoformat()
    return str(v)


def _normalize_tags(raw) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(t).strip() for t in raw]
    if isinstance(raw, str):
        # some real-world producers write "tag1, tag2" as a single string
        # despite the spec saying tags should be a YAML list -- tolerate it
        return [t.strip() for t in raw.split(",") if t.strip()]
    return [str(raw)]


def parse_concept(bundle_root: str, rel_path: str) -> Optional[OKFConcept]:
    """Parse one concept document. Returns None if the file has no
    frontmatter or no `type` field (SPEC §4.1: `type` is the only always-
    required key) -- such files are treated as non-conformant and skipped
    rather than raising, per the spec's tolerance requirements (§4.1, §11)."""
    full_path = os.path.join(bundle_root, rel_path)
    with open(full_path, "r", encoding="utf-8") as f:
        text = f.read()

    if not text.startswith("---"):
        return None
    end = text.find("\n---", 3)
    if end == -1:
        return None
    frontmatter_raw = text[3:end].strip()
    body = text[end + 4:].lstrip("\n")

    try:
        frontmatter = yaml.safe_load(frontmatter_raw) or {}
    except yaml.YAMLError:
        return None
    except RecursionError:
        # A pathologically deeply-nested (but otherwise valid) YAML
        # structure can exceed Python's recursion limit inside PyYAML's
        # recursive-descent parser. This must NOT propagate: one malformed
        # or malicious concept file would otherwise crash the entire
        # load_bundle()/ingest_okf_bundle() call, taking down every other
        # concept in the bundle along with it -- directly contradicting
        # the spec's requirement that consumers tolerate malformed content
        # (§4.1, §11). Treat it the same as any other unparseable frontmatter.
        return None
    if not isinstance(frontmatter, dict) or "type" not in frontmatter:
        return None

    known = {"type", "title", "description", "resource", "tags", "status", "stale_after"}
    extra = {k: v for k, v in frontmatter.items() if k not in known}

    links = _LINK_RE.findall(body)

    return OKFConcept(
        concept_id=rel_path[:-3] if rel_path.endswith(".md") else rel_path,
        path=rel_path,
        type=str(frontmatter["type"]),
        title=frontmatter.get("title"),
        description=frontmatter.get("description"),
        resource=frontmatter.get("resource"),
        tags=_normalize_tags(frontmatter.get("tags")),
        status=frontmatter.get("status", "stable"),
        stale_after=_stringify_date(frontmatter.get("stale_after")),
        body=body,
        links=links,
        extra=extra,
    )


def _is_safe_bundle_file(bundle_root: str, dirpath: str, fn: str) -> bool:
    """Reject symlinked files whose real target resolves outside the
    bundle. Without this, a bundle containing a symlink to an OKF-shaped
    file elsewhere on disk (e.g. /etc/some_file crafted with frontmatter,
    or any other file the process can read) gets silently read and
    indexed -- a real file-exfiltration vector for a bundle from an
    untrusted source (a cloned repo, an uploaded archive, etc). Symlinked
    *directories* are already safe by default since os.walk uses
    followlinks=False; this closes the remaining per-file gap."""
    full_path = os.path.join(dirpath, fn)
    if not os.path.islink(full_path):
        return True
    bundle_real = os.path.realpath(bundle_root)
    target_real = os.path.realpath(full_path)
    try:
        return os.path.commonpath([bundle_real, target_real]) == bundle_real
    except ValueError:
        return False  # different drives on Windows, etc -- treat as unsafe


def load_bundle(bundle_root: str) -> List[OKFConcept]:
    """Walk an OKF bundle directory and parse every concept document.
    Reserved filenames (index.md, log.md) are skipped at every level, per
    SPEC §3.1. Files with no frontmatter/type are silently skipped, per the
    spec's requirement that consumers tolerate non-conformant content.
    Symlinks pointing outside the bundle are skipped (see
    _is_safe_bundle_file) rather than followed."""
    concepts = []
    for dirpath, _dirnames, filenames in os.walk(bundle_root):
        for fn in filenames:
            if not fn.endswith(".md") or fn in RESERVED_FILENAMES:
                continue
            if not _is_safe_bundle_file(bundle_root, dirpath, fn):
                continue
            rel_path = os.path.relpath(os.path.join(dirpath, fn), bundle_root)
            rel_path = rel_path.replace(os.sep, "/")  # stable IDs across OSes
            concept = parse_concept(bundle_root, rel_path)
            if concept is not None:
                concepts.append(concept)
    return concepts


def ingest_okf_bundle(
    db,
    bundle_root: str,
    include_deprecated: bool = False,
    batch_size: int = 500,
) -> Dict[str, int]:
    """
    Load an OKF bundle into a VectorDB via add_text() -- requires db to have
    an embedder configured (VectorDB(..., embedder=...)) since this enables
    both semantic search and hybrid_search() over the bundle.

    include_deprecated: by default, concepts with status: deprecated (SPEC
      §5.4 -- "kept for links and history; no longer current") are skipped,
      since surfacing deprecated info in search results is usually wrong.
      Set True to index them anyway (e.g. for an archival/audit use case).

    Returns {"indexed": N, "skipped_deprecated": N, "skipped_malformed": N}.
    """
    if db.embedder is None:
        raise RuntimeError(
            "ingest_okf_bundle requires a VectorDB with an embedder "
            "configured, e.g. VectorDB(path, embedder=TextEmbedder())."
        )

    all_files = []
    for dirpath, _dirnames, filenames in os.walk(bundle_root):
        for fn in filenames:
            if fn.endswith(".md") and fn not in RESERVED_FILENAMES:
                if not _is_safe_bundle_file(bundle_root, dirpath, fn):
                    continue
                all_files.append(os.path.relpath(os.path.join(dirpath, fn), bundle_root))

    indexed = 0
    skipped_deprecated = 0
    skipped_malformed = 0

    batch_ids, batch_texts, batch_metas = [], [], []

    def flush():
        nonlocal batch_ids, batch_texts, batch_metas
        if batch_ids:
            db.add_text(batch_ids, batch_texts, batch_metas)
            batch_ids, batch_texts, batch_metas = [], [], []

    for rel_path in all_files:
        rel_path = rel_path.replace(os.sep, "/")
        concept = parse_concept(bundle_root, rel_path)
        if concept is None:
            skipped_malformed += 1
            continue
        if concept.status == "deprecated" and not include_deprecated:
            skipped_deprecated += 1
            continue

        batch_ids.append(concept.concept_id)
        batch_texts.append(concept.searchable_text())
        batch_metas.append(concept.metadata())
        indexed += 1

        if len(batch_ids) >= batch_size:
            flush()

    flush()
    return {
        "indexed": indexed,
        "skipped_deprecated": skipped_deprecated,
        "skipped_malformed": skipped_malformed,
    }
