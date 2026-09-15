"""One task as a Harbor directory: written whole or not at all, read back and checked against its digest.

The layout is the one Harbor 0.23 loads (``harbor.models.task.paths.TaskPaths``)
and every example under ``recipes/`` ships::

    <name>/
      task.toml            version, [metadata], [verifier], [agent], [environment]
      instruction.md       the prompt shown to the model
      environment/         Dockerfile and whatever the image needs
      tests/test.sh        the verifier, with any helper it calls beside it
      solution/            optional reference files

``task.toml`` carries a ``[metadata.reef]`` table with the task's digest and
the agent record ids it was made from. The digest covers every file, so a
replay that writes the same task again is a no-op and a directory edited by
hand, or given an extra entry of any kind, is refused.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import stat
import time
import unicodedata
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from ipaddress import ip_address, ip_network
from pathlib import Path, PurePosixPath
from typing import Any

import tomli_w

from reef.core.errors import ReefError

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib

#: The ``version`` every task.toml under ``recipes/`` declares; Harbor reads it as ``schema_version``.
TASK_CONFIG_VERSION = "1.0"
NETWORK_MODES = ("no-network", "public", "allowlist")
#: A staging directory older than this with no writer behind it is swept before the next write.
STALE_STAGING_S = 3600.0

_NAME = re.compile(r"^[a-z0-9][a-z0-9._-]{0,99}$")
_HOST_LABEL = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")
_SIZE_KEYS = ("cpus", "memory_mb", "storage_mb", "gpus")
_KNOWN_KEYS: dict[str, tuple[str, ...]] = {
    "verifier": ("timeout_sec", "env", "user"),
    "agent": ("timeout_sec", "user"),
    "environment": (
        "build_timeout_sec",
        "docker_image",
        "network_mode",
        "allowed_hosts",
        "env",
        "workdir",
        *_SIZE_KEYS,
    ),
}
_TOP_LEVEL = ("version", "metadata", *_KNOWN_KEYS)
_REEF_KEYS = ("digest", "source_agent_record_ids")
_TREES = ("tests", "environment", "solution")
_ROOT_FILES = ("task.toml", "instruction.md")
_MAX_COMPONENT_BYTES = 255
_MAX_PATH_BYTES = 1024


class HarborTaskError(ReefError):
    """A task spec or task directory that Harbor could not run."""


class HarborTaskConflict(HarborTaskError):
    """The target directory already holds a different task under the same name."""


@dataclass(frozen=True)
class HarborTask:
    """A task before it is written: everything the directory will hold, validated at construction."""

    name: str
    instruction: str
    tests: Mapping[str, str]
    environment: Mapping[str, str]
    config: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    solution: Mapping[str, str] = field(default_factory=dict)
    source_agent_record_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not _NAME.fullmatch(self.name) or ".." in self.name:
            raise HarborTaskError(f"task name {self.name!r} must match {_NAME.pattern} and never contain '..'")
        _require_text("instruction", self.instruction)
        object.__setattr__(self, "tests", _files("tests", self.tests))
        if not self.tests.get("test.sh", "").strip():
            raise HarborTaskError("tests/test.sh must be non-empty text: it is the verifier Harbor runs")
        object.__setattr__(self, "environment", _files("environment", self.environment))
        object.__setattr__(self, "solution", _files("solution", self.solution))
        object.__setattr__(self, "config", _config(self.config))
        if "Dockerfile" not in self.environment and "docker_image" not in self.config.get("environment", {}):
            raise HarborTaskError("environment needs a Dockerfile or config environment.docker_image")
        object.__setattr__(self, "metadata", _metadata(self.metadata))
        ids = self.source_agent_record_ids
        if (
            not isinstance(ids, tuple)
            or any(not isinstance(i, str) or not i for i in ids)
            or len(set(ids)) != len(ids)
        ):
            raise HarborTaskError("source_agent_record_ids must be a tuple of distinct non-empty strings")
        # One encode of everything the writer will put on disk: a lone surrogate anywhere fails here, not mid write.
        try:
            self.task_toml().encode("utf-8")
        except UnicodeEncodeError as exc:
            raise HarborTaskError(f"task.toml is not valid Unicode text: {exc.reason}") from exc

    @property
    def digest(self) -> str:
        """sha256 over the task's content and its source record ids, the same for the same task however it was built."""
        canonical = json.dumps(
            {
                "name": self.name,
                "instruction": self.instruction,
                "tests": dict(self.tests),
                "environment": dict(self.environment),
                "solution": dict(self.solution),
                "config": {table: dict(values) for table, values in self.config.items()},
                "metadata": dict(self.metadata),
                "source_agent_record_ids": list(self.source_agent_record_ids),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def task_toml(self) -> str:
        """The task.toml text: the version, the tables in Harbor's order, reef's digest and source ids under metadata."""
        document: dict[str, Any] = {"version": TASK_CONFIG_VERSION}
        document["metadata"] = {
            **self.metadata,
            "reef": {"digest": self.digest, "source_agent_record_ids": list(self.source_agent_record_ids)},
        }
        for table in _KNOWN_KEYS:
            if table in self.config:
                document[table] = dict(self.config[table])
        return tomli_w.dumps(document)


def write_harbor_task(task: HarborTask, root: Path) -> Path:
    """Write ``task`` under ``root/<name>`` atomically; the same task again is a no-op, a different one a conflict."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    target = root / task.name
    if os.path.lexists(target):
        _require_same(task, target)
        return target
    _sweep_stale_staging(root, task.name)
    # A plain mkdir, not mkdtemp: the directory keeps the umask mode it will be published with.
    staging = root / f".{task.name}.{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        _write_files(task, staging)
        # What the filesystem kept must be what was hashed: a case folding or normalizing volume merges names.
        if _read_task(staging, task.name).digest != task.digest:
            raise HarborTaskError(
                f"{staging} does not read back as the task written to it; is the volume case folding?"
            )
        try:
            os.rename(staging, target)
        except OSError:
            if not os.path.lexists(target):
                raise
            # A concurrent writer won the rename: accept its directory only if it holds the same task.
            _require_same(task, target)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return target


def read_harbor_task(path: Path) -> HarborTask:
    """Read a directory written by :func:`write_harbor_task` back, refusing one whose content no longer matches its digest."""
    path = Path(path)
    if not path.is_dir():
        raise HarborTaskError(f"{path} is not a task directory")
    return _read_task(path, path.name)


# ----------------------------------------------------------------------------------------------- helpers


def _read_task(path: Path, name: str) -> HarborTask:
    files = _read_all(path)
    if "task.toml" not in files:
        raise HarborTaskError(f"{path / 'task.toml'} is missing")
    try:
        document = tomllib.loads(files["task.toml"])
    except tomllib.TOMLDecodeError as exc:
        raise HarborTaskError(f"{path / 'task.toml'} is not valid TOML: {exc}") from exc
    if document.get("version") != TASK_CONFIG_VERSION:
        raise HarborTaskError(f"{path / 'task.toml'} must declare version = {TASK_CONFIG_VERSION!r}")
    unknown = sorted(key for key in document if key not in _TOP_LEVEL)
    if unknown:
        raise HarborTaskError(f"{path / 'task.toml'} carries tables reef did not write: {', '.join(unknown)}")
    metadata = document.get("metadata")
    if not isinstance(metadata, dict) or not isinstance(metadata.get("reef"), dict):
        raise HarborTaskError(f"{path / 'task.toml'} carries no [metadata.reef] table; reef did not write it")
    reef = metadata["reef"]
    if sorted(reef) != sorted(_REEF_KEYS) or not isinstance(reef["source_agent_record_ids"], list):
        raise HarborTaskError(f"{path / 'task.toml'} metadata.reef must hold exactly {' and '.join(_REEF_KEYS)}")
    trees: dict[str, dict[str, str]] = {tree: {} for tree in _TREES}
    extra: list[str] = []
    for relative, text in files.items():
        parts = relative.split("/")
        if len(parts) == 1 and parts[0] in _ROOT_FILES:
            continue
        if len(parts) > 1 and parts[0] in _TREES:
            trees[parts[0]]["/".join(parts[1:])] = text
        else:
            extra.append(relative)
    if extra:
        raise HarborTaskError(f"{path} holds entries reef did not write: {', '.join(sorted(extra))}")
    if "instruction.md" not in files:
        raise HarborTaskError(f"{path / 'instruction.md'} is missing")
    for tree in ("tests", "environment"):
        if not (path / tree).is_dir():
            raise HarborTaskError(f"{path / tree} is missing")
    task = HarborTask(
        name=name,
        instruction=files["instruction.md"],
        tests=trees["tests"],
        environment=trees["environment"],
        config={table: document[table] for table in _KNOWN_KEYS if table in document},
        metadata={key: value for key, value in metadata.items() if key != "reef"},
        solution=trees["solution"],
        source_agent_record_ids=tuple(str(i) for i in reef["source_agent_record_ids"]),
    )
    if reef["digest"] != task.digest:
        raise HarborTaskError(f"{path} was edited after it was written: its content no longer matches its digest")
    return task


def _read_all(root: Path) -> dict[str, str]:
    """Every regular file under ``root`` as {relative posix path: text}; any other kind of entry is refused."""
    files: dict[str, str] = {}
    pending = [root]
    try:
        while pending:
            directory = pending.pop()
            with os.scandir(directory) as entries:
                for entry in sorted(entries, key=lambda e: e.name):
                    mode = entry.stat(follow_symlinks=False).st_mode
                    relative = Path(entry.path).relative_to(root).as_posix()
                    if stat.S_ISDIR(mode):
                        pending.append(Path(entry.path))
                    elif stat.S_ISREG(mode):
                        files[relative] = _read_text(Path(entry.path))
                    else:
                        raise HarborTaskError(
                            f"{root / relative} is not a regular file or directory; reef wrote neither"
                        )
    except OSError as exc:
        raise HarborTaskError(f"{root} cannot be read: {exc}") from exc
    return files


def _require_text(what: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HarborTaskError(f"{what} must be non-empty text")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise HarborTaskError(f"{what} is not valid Unicode text: {exc.reason}") from exc
    return value


def _files(what: str, files: Any) -> dict[str, str]:
    """A directory's files as {relative posix path: text}; the paths must stay inside the directory and never collide."""
    if not isinstance(files, Mapping):
        raise HarborTaskError(f"{what} must map relative file paths to text")
    checked: dict[str, str] = {}
    folded_files: set[str] = set()
    folded_dirs: set[str] = set()
    for name, text in files.items():
        if not isinstance(name, str) or not name or "\\" in name or any(ord(c) < 0x20 or ord(c) == 0x7F for c in name):
            raise HarborTaskError(
                f"{what} file name {name!r} must be a relative posix path without control characters"
            )
        pure = PurePosixPath(name)
        if pure.is_absolute() or not pure.parts or any(part in (".", "..") for part in pure.parts):
            raise HarborTaskError(f"{what} file name {name!r} must stay inside the {what} directory")
        if str(pure) != name:
            raise HarborTaskError(f"{what} file name {name!r} must be written plainly, as {str(pure)!r}")
        try:
            encoded = name.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise HarborTaskError(f"{what} file name {name!r} is not valid Unicode text: {exc.reason}") from exc
        if len(encoded) > _MAX_PATH_BYTES or any(
            len(part.encode("utf-8")) > _MAX_COMPONENT_BYTES for part in pure.parts
        ):
            raise HarborTaskError(f"{what} file name {name!r} is longer than a filesystem allows")
        if not isinstance(text, str):
            raise HarborTaskError(f"{what}/{name} must be text")
        try:
            text.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise HarborTaskError(f"{what}/{name} is not valid Unicode text: {exc.reason}") from exc
        # Two names one filesystem may merge (case, Unicode normalization) would leave one of them unwritten;
        # a file folding onto a directory of another file is the same accident one level up.
        fold = _fold(name)
        parents = {_fold(str(parent)) for parent in pure.parents} - {"."}
        if fold in folded_files or fold in folded_dirs or parents & folded_files:
            raise HarborTaskError(f"{what} names {name!r} twice, or under a spelling a filesystem may fold together")
        checked[name] = text
        folded_files.add(fold)
        folded_dirs |= parents
    return checked


def _fold(name: str) -> str:
    return unicodedata.normalize("NFC", name).casefold()


def _config(config: Any) -> dict[str, dict[str, Any]]:
    """The task.toml tables reef writes, checked against the keys Harbor reads."""
    if not isinstance(config, Mapping):
        raise HarborTaskError("config must map task.toml table names to their keys")
    checked: dict[str, dict[str, Any]] = {}
    for table, values in config.items():
        if table not in _KNOWN_KEYS:
            raise HarborTaskError(f"config table {table!r} is not one reef writes ({', '.join(_KNOWN_KEYS)})")
        if not isinstance(values, Mapping):
            raise HarborTaskError(f"config.{table} must be a table")
        checked[table] = {}
        for key, value in values.items():
            if key not in _KNOWN_KEYS[table]:
                raise HarborTaskError(f"config.{table}.{key} is not a key Harbor reads")
            checked[table][key] = _config_value(table, key, value)
    environment = checked.get("environment", {})
    if environment.get("allowed_hosts") and environment.get("network_mode") != "allowlist":
        raise HarborTaskError("config.environment.allowed_hosts needs network_mode = 'allowlist'")
    return checked


def _config_value(table: str, key: str, value: Any) -> Any:
    where = f"config.{table}.{key}"
    if key.endswith("timeout_sec"):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise HarborTaskError(f"{where} must be a positive number of seconds")
        return float(value)
    if key in _SIZE_KEYS:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise HarborTaskError(f"{where} must be a non-negative integer")
        return value
    if key == "network_mode":
        if value not in NETWORK_MODES:
            raise HarborTaskError(f"{where} must be one of {', '.join(NETWORK_MODES)}")
        return value
    if key == "allowed_hosts":
        if not isinstance(value, list):
            raise HarborTaskError(f"{where} must be a list of host names")
        return [_allowed_host(where, host) for host in value]
    if key == "env":
        if not isinstance(value, Mapping) or any(
            not isinstance(k, str) or not k or not isinstance(v, str) for k, v in value.items()
        ):
            raise HarborTaskError(f"{where} must map variable names to strings")
        return dict(value)
    if key == "user":
        if isinstance(value, bool) or not (
            (isinstance(value, str) and value) or (isinstance(value, int) and value >= 0)
        ):
            raise HarborTaskError(f"{where} must be a user name or a non-negative uid")
        return value
    if not isinstance(value, str) or not value:
        raise HarborTaskError(f"{where} must be a non-empty string")
    return value


def _allowed_host(where: str, host: Any) -> str:
    """One allowlist entry in the normalized form Harbor 0.23 accepts: a host name, a leading wildcard, an address or a CIDR range."""
    if not isinstance(host, str) or not host.strip():
        raise HarborTaskError(f"{where} entries must be non-empty host names")
    host = host.strip().lower().rstrip(".")
    reject = HarborTaskError(
        f"{where} entry {host!r} must be a host name, an IP address or a CIDR range, not a URL, port or path"
    )
    if "%" in host or "[" in host or "]" in host:
        raise reject
    if "/" in host:
        try:
            return ip_network(host, strict=True).compressed
        except ValueError:
            raise reject from None
    if ":" in host:
        try:
            address = ip_address(host)
        except ValueError:
            raise reject from None
        if address.version != 6:
            raise reject
        return address.compressed
    labels = host[2:] if host.startswith("*.") else host
    if not labels or "*" in labels:
        raise reject
    if host.startswith("*."):
        try:
            ip_address(labels)
        except ValueError:
            pass
        else:
            raise reject
    if not all(_HOST_LABEL.fullmatch(label) for label in labels.split(".")):
        raise reject
    return host


def _metadata(metadata: Any) -> dict[str, Any]:
    if not isinstance(metadata, Mapping) or any(not isinstance(key, str) or not key for key in metadata):
        raise HarborTaskError("metadata must be a table with string keys")
    if "reef" in metadata:
        raise HarborTaskError("metadata.reef is written by reef; put your own keys elsewhere")
    try:
        # Both writers must accept it: tomli_w for task.toml, json for the digest (so no TOML dates).
        tomli_w.dumps({"metadata": dict(metadata)})
        json.dumps(dict(metadata), sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise HarborTaskError(f"metadata must hold strings, numbers, booleans, lists and tables only: {exc}") from exc
    return dict(metadata)


def _write_files(task: HarborTask, root: Path) -> None:
    _write_text(root / "task.toml", task.task_toml())
    _write_text(root / "instruction.md", task.instruction)
    for directory, files in (("tests", task.tests), ("environment", task.environment), ("solution", task.solution)):
        # tests/ and environment/ exist even when empty: Harbor refuses a task directory without them.
        if not files and directory == "solution":
            continue
        (root / directory).mkdir()
        for name, text in files.items():
            target = root / directory / name
            target.parent.mkdir(parents=True, exist_ok=True)
            _write_text(target, text)
    (root / "tests" / "test.sh").chmod(0o755)


def _write_text(path: Path, text: str) -> None:
    # newline="" on both sides: the bytes hashed are the bytes on disk, carriage returns included.
    path.write_text(text, encoding="utf-8", newline="")


def _sweep_stale_staging(root: Path, name: str) -> None:
    """Remove staging directories an earlier writer left behind, so a root listing never shows a hidden copy."""
    now = time.time()
    for stale in root.glob(f".{name}.*"):
        _remove_if_stale(stale, now)


def _remove_if_stale(path: Path, now: float) -> None:
    try:
        if path.is_dir() and not path.is_symlink() and now - path.stat().st_mtime > STALE_STAGING_S:
            shutil.rmtree(path, ignore_errors=True)
    except OSError:
        return


def _require_same(task: HarborTask, target: Path) -> None:
    try:
        if not target.is_dir() or target.is_symlink():
            raise HarborTaskError(f"{target} is not a directory")
        existing = read_harbor_task(target)
    except HarborTaskError as exc:
        raise HarborTaskConflict(f"{target} exists and is not a task reef wrote: {exc}") from exc
    except OSError as exc:
        raise HarborTaskConflict(f"{target} exists and cannot be read: {exc}") from exc
    if existing.digest != task.digest:
        raise HarborTaskConflict(f"{target} already holds a different task with the same name")


def _read_text(path: Path) -> str:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            return handle.read()
    except UnicodeDecodeError as exc:
        raise HarborTaskError(f"{path} is not UTF-8 text; reef writes text files only") from exc
    except OSError as exc:
        raise HarborTaskError(f"{path} cannot be read: {exc}") from exc
