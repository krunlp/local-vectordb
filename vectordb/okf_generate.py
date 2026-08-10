"""
Generate an OKF (Open Knowledge Format) bundle from raw documents.

This is the reverse direction of vectordb.okf (which reads OKF bundles):
this module takes arbitrary text documents and writes them out as a valid,
spec-conformant OKF bundle on disk -- markdown files with YAML frontmatter,
index.md navigation, and a log.md provenance record -- so any AI agent
(Claude Code, Cursor, Gemini CLI, etc.) can read it without knowing
anything about vectordb.

Every concept written here round-trips cleanly through vectordb.okf's own
parser (verified in the test suite), and the output has been checked
against Google's real reference bundles' actual index.md style, not just
the spec description.

Typical use:
    from vectordb.okf_generate import documents_to_okf

    result = documents_to_okf(
        "/path/to/my/text/files",   # a directory of .txt/.md files, OR
        "/path/to/output/bundle",
        default_type="Document",
    )
"""
import datetime
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import yaml

_HEADING_RE = re.compile(r"^\s*#{1,6}\s+(.+?)\s*$", re.MULTILINE)
_SENTENCE_END_RE = re.compile(r"[.!?](\s|$)")


@dataclass
class ConceptInput:
    text: str
    id: Optional[str] = None  # bundle-relative path, no .md suffix; auto-derived if absent
    title: Optional[str] = None  # derived from first heading/line if absent
    type: str = "Document"
    description: Optional[str] = None  # derived from body if absent
    tags: Optional[List[str]] = None
    resource: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


def _safe_concept_id(concept_id: str) -> str:
    """Reject concept IDs that could escape output_dir when joined into a
    filesystem path -- absolute paths, '..' components, or empty segments
    (e.g. from a leading '/' producing a '' split). This matters because
    concept IDs can come directly from caller-supplied dicts (the
    documents_to_okf list-of-dicts input path), not just from our own
    filename-derived slugs, so a caller (or a malicious/buggy upstream
    document source) could otherwise write files outside output_dir."""
    normalized = concept_id.replace("\\", "/")
    if normalized.startswith("/") or normalized.startswith("~"):
        raise ValueError(
            f"Unsafe concept id (absolute path not allowed): {concept_id!r}"
        )
    parts = normalized.split("/")
    if any(p in ("..", "") for p in parts):
        raise ValueError(
            f"Unsafe concept id (path traversal / empty segment): {concept_id!r}"
        )
    return normalized


def _slugify(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-") or "concept"


def _prettify_filename(id_: str) -> str:
    """Fallback title from a filename/id, e.g. 'deploy-process' -> 'Deploy Process'."""
    base = os.path.basename(id_)
    words = re.split(r"[-_]+", base)
    return " ".join(w.capitalize() for w in words if w)


def _derive_title(text: str, fallback: str) -> "tuple[str, bool]":
    """Returns (title, consumed_first_line) -- consumed_first_line tells the
    caller whether the title came from the body's first line, so that line
    can be excluded when deriving the description (otherwise a plain-text
    document's title line ends up duplicated at the start of its own
    description)."""
    heading = _HEADING_RE.match(text.strip())
    if heading:
        return heading.group(1).strip(), True
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Only treat the first line as a title if it actually looks like
        # one: short, and not a full sentence (no terminal punctuation).
        # Otherwise a plain-text document with no real title just donates
        # its entire opening sentence as a garbled "title".
        if (
            len(line) <= 80
            and not line.rstrip().endswith((".", "!", "?", ":", ";"))
            and not re.search(r"[.!?]\s", line)  # a period/etc mid-line means
            # this "line" is actually one or more full sentences, just not
            # yet wrapped onto multiple lines -- not a title.
        ):
            return line, True
        break
    return fallback, False


def _derive_description(text: str, strip_first_line: bool, max_len: int = 200) -> str:
    body = text.strip()
    if strip_first_line:
        lines = body.splitlines()
        # drop the first non-empty line (the one used as the title)
        for i, line in enumerate(lines):
            if line.strip():
                lines = lines[i + 1:]
                break
        body = "\n".join(lines)
    body = _HEADING_RE.sub("", body)  # in case a heading appears later too
    body = " ".join(body.split())
    if not body:
        return ""
    if len(body) <= max_len:
        return body
    truncated = body[:max_len]
    m = None
    for m in _SENTENCE_END_RE.finditer(truncated):
        pass  # walk to the last sentence boundary within the limit
    if m:
        return truncated[: m.end()].strip()
    return truncated.rsplit(" ", 1)[0].strip() + "…"


def render_concept_markdown(
    concept: ConceptInput,
    producer: str = "vectordb.okf_generate",
) -> str:
    """Render one ConceptInput into a valid OKF concept document (SPEC §4)."""
    frontmatter: Dict[str, Any] = {"type": concept.type}
    if concept.title:
        frontmatter["title"] = concept.title
    if concept.description:
        frontmatter["description"] = concept.description
    if concept.resource:
        frontmatter["resource"] = concept.resource
    if concept.tags:
        frontmatter["tags"] = list(concept.tags)
    frontmatter["generated"] = {
        "at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "by": producer,
    }
    frontmatter.update(concept.extra)

    yaml_block = yaml.safe_dump(
        frontmatter, sort_keys=False, allow_unicode=True, default_flow_style=False
    ).strip()
    return f"---\n{yaml_block}\n---\n\n{concept.text.strip()}\n"


def write_bundle(
    concepts: List[ConceptInput],
    output_dir: str,
    producer: str = "vectordb.okf_generate",
    write_log: bool = True,
) -> Dict[str, int]:
    """
    Write a list of ConceptInput as a valid OKF bundle on disk: one .md file
    per concept, index.md at every directory level (SPEC §8), and an
    optional log.md recording the generation event (SPEC §9).

    IDs are auto-derived from titles (slugified) when not given, and
    de-duplicated if two concepts would collide.
    """
    os.makedirs(output_dir, exist_ok=True)
    used_ids = set()
    written: List[Dict[str, str]] = []  # {"id", "title", "description"}

    for c in concepts:
        base_id = c.id or None
        fallback_title = _prettify_filename(base_id) if base_id else "Untitled"
        title = c.title
        consumed_first_line = False
        if title is None:
            title, consumed_first_line = _derive_title(c.text, fallback=fallback_title)
        description = c.description
        if description is None:
            description = _derive_description(c.text, strip_first_line=consumed_first_line)

        concept_id = c.id or _slugify(title)
        concept_id = _safe_concept_id(concept_id)
        base_id = concept_id
        n = 2
        while concept_id in used_ids:
            concept_id = f"{base_id}-{n}"
            n += 1
        used_ids.add(concept_id)

        resolved = ConceptInput(
            text=c.text, id=concept_id, title=title, type=c.type,
            description=description, tags=c.tags, resource=c.resource, extra=c.extra,
        )
        md = render_concept_markdown(resolved, producer=producer)

        file_path = os.path.join(output_dir, concept_id + ".md")
        # Defense in depth: even after _safe_concept_id, confirm the
        # resolved path is still actually inside output_dir before writing.
        output_dir_real = os.path.realpath(output_dir)
        file_path_real = os.path.realpath(os.path.dirname(file_path))
        if os.path.commonpath([output_dir_real, file_path_real]) != output_dir_real:
            raise ValueError(f"Refusing to write outside output_dir: {concept_id!r}")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(md)

        written.append({"id": concept_id, "title": title, "description": description})

    _write_index_files(output_dir, written)

    if write_log:
        _write_log(output_dir, len(written), producer)

    return {"generated": len(written)}


def _write_index_files(output_dir: str, written: List[Dict[str, str]]):
    """Build index.md at the bundle root and every subdirectory, in the
    bullet-list style used by Google's real reference bundles: one line per
    child, `* [Title](file.md) - description`, subdirectories listed
    separately under their own section."""
    # Group concepts by their containing directory, relative to output_dir
    by_dir: Dict[str, List[Dict[str, str]]] = {}
    for w in written:
        dirname = os.path.dirname(w["id"])
        by_dir.setdefault(dirname, []).append(w)

    all_dirs = set(by_dir.keys())
    # also register every intermediate ancestor directory, so a concept
    # nested 3 levels deep produces index.md at every level above it
    for d in list(all_dirs):
        parts = d.split("/") if d else []
        for i in range(len(parts)):
            all_dirs.add("/".join(parts[:i]))

    for dirname in all_dirs:
        entries = by_dir.get(dirname, [])
        lines = []
        if entries:
            for e in sorted(entries, key=lambda x: x["title"].lower()):
                fname = os.path.basename(e["id"]) + ".md"
                desc = f" - {e['description']}" if e["description"] else ""
                lines.append(f"* [{e['title']}]({fname}){desc}")

        # direct subdirectories of this directory
        subdirs = sorted(
            {
                d for d in all_dirs
                if d != dirname
                and os.path.dirname(d) == dirname
                and d != ""
            }
        )
        subdir_lines = [f"* [{os.path.basename(d)}]({os.path.basename(d)}/index.md)" for d in subdirs]

        body_parts = []
        if lines:
            body_parts.append("# Concepts\n\n" + "\n".join(lines))
        if subdir_lines:
            body_parts.append("# Subdirectories\n\n" + "\n".join(subdir_lines))
        content = "\n\n".join(body_parts).strip() + "\n"

        index_path = os.path.join(output_dir, dirname, "index.md") if dirname else os.path.join(output_dir, "index.md")
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        with open(index_path, "w", encoding="utf-8") as f:
            f.write(content)


def _write_log(output_dir: str, count: int, producer: str):
    today = datetime.date.today().isoformat()
    content = (
        "---\ntype: Log\ntitle: Bundle history\n---\n\n"
        "# Bundle history\n\n"
        f"## {today}\n\n"
        f"- **Bundle generated** by `{producer}` from {count} source document"
        f"{'s' if count != 1 else ''}.\n"
    )
    with open(os.path.join(output_dir, "log.md"), "w", encoding="utf-8") as f:
        f.write(content)


def documents_to_okf(
    source: Union[str, List[ConceptInput], List[Dict[str, Any]]],
    output_dir: str,
    default_type: str = "Document",
    producer: str = "vectordb.okf_generate",
    file_extensions: tuple = (".txt", ".md"),
) -> Dict[str, int]:
    """
    Convert raw documents into a valid OKF bundle.

    source: one of
      - a directory path: every .txt/.md file inside becomes one concept,
        with id = its relative path (dots/extension stripped)
      - a list of ConceptInput
      - a list of dicts with at least a "text" key (and optionally "id",
        "title", "type", "description", "tags", "resource")

    Does NOT infer cross-links between documents (unlike ingest_okf_bundle,
    which reads them) -- there's no reliable way to know which documents
    should reference each other from plain text alone. Add markdown links
    yourself in the source text if you want them, or post-process the
    output bundle's .md files.
    """
    concepts: List[ConceptInput] = []

    if isinstance(source, str):
        for dirpath, _dirnames, filenames in os.walk(source):
            for fn in sorted(filenames):
                if not fn.endswith(file_extensions):
                    continue
                full = os.path.join(dirpath, fn)
                rel = os.path.relpath(full, source).replace(os.sep, "/")
                concept_id = re.sub(r"\.(txt|md)$", "", rel)
                with open(full, "r", encoding="utf-8") as f:
                    text = f.read()
                concepts.append(ConceptInput(text=text, id=concept_id, type=default_type))
    else:
        for item in source:
            if isinstance(item, ConceptInput):
                concepts.append(item)
            elif isinstance(item, dict):
                concepts.append(ConceptInput(
                    text=item["text"],
                    id=item.get("id"),
                    title=item.get("title"),
                    type=item.get("type", default_type),
                    description=item.get("description"),
                    tags=item.get("tags"),
                    resource=item.get("resource"),
                    extra=item.get("extra", {}),
                ))
            else:
                raise TypeError(f"Unsupported source item type: {type(item)}")

    return write_bundle(concepts, output_dir, producer=producer)
