"""Stack dependency watch: pyproject.toml parsing and OSV vulnerability advisories (M17)."""

from __future__ import annotations

import logging
import re
import tomllib
from datetime import UTC, datetime, timedelta
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import httpx

from gleaner.db import Database
from gleaner.models import Item, Question, SourceRow
from gleaner.questions import STACK_QUESTION_ID

logger = logging.getLogger(__name__)

OSV_QUERY_URL = "https://api.osv.dev/v1/query"
STACK_SOURCE_ID = "osv"  # type "osv": run by the stack watch, not by a connector
STACK_CHECK_EVERY = timedelta(days=1)


def parse_pyproject_dependencies(pyproject_path: Path | str) -> list[str]:
    """Parse runtime and optional dependencies from a pyproject.toml file.

    Reads PEP 621 `dependencies` and `optional-dependencies`, plus legacy Poetry dependencies.
    A missing or malformed file is logged and gives an empty list.

    Args:
        pyproject_path: Path to the pyproject.toml file.

    Returns:
        Sorted, normalized package names (lowercase, `-` for `_`), without "python".
    """
    path = Path(pyproject_path)
    if not path.exists():
        logger.warning("pyproject file does not exist: %s", path)
        return []

    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:  # Safe degradation: malformed pyproject returns empty list
        logger.warning("Failed to parse %s: %s", path, exc)
        return []

    raw_specs: list[str] = []

    # PEP 621 dependencies
    project = data.get("project", {})
    if isinstance(project, dict):
        deps = project.get("dependencies", [])
        if isinstance(deps, list):
            raw_specs.extend(deps)
        opt_deps = project.get("optional-dependencies", {})
        if isinstance(opt_deps, dict):
            for group in opt_deps.values():
                if isinstance(group, list):
                    raw_specs.extend(group)

    # Legacy Poetry tool.poetry.dependencies
    tool_poetry = data.get("tool", {}).get("poetry", {})
    if isinstance(tool_poetry, dict):
        p_deps = tool_poetry.get("dependencies", {})
        if isinstance(p_deps, dict):
            raw_specs.extend(p_deps.keys())

    packages: set[str] = set()
    pkg_pattern = re.compile(r"^([a-zA-Z0-9_.-]+)")

    for spec in raw_specs:
        cleaned = spec.strip()
        match = pkg_pattern.match(cleaned)
        if match:
            pkg = match.group(1).lower().replace("_", "-")
            if pkg != "python":
                packages.add(pkg)

    return sorted(packages)


def installed_versions(packages: list[str]) -> dict[str, str]:
    """Return the installed version of each package; ones not installed are left out.

    Args:
        packages: Package names to look up.

    Returns:
        A mapping of package name to installed version.
    """
    found = {}
    for pkg in packages:
        try:
            found[pkg] = version(pkg)
        except PackageNotFoundError:
            continue
    return found


async def fetch_osv_advisories(
    packages: list[str],
    client: httpx.AsyncClient | None = None,
    versions: dict[str, str] | None = None,
) -> list[Item]:
    """Query OSV API (api.osv.dev) for advisories affecting given PyPI packages.

    With `versions`, OSV returns only advisories that affect that version. Withdrawn advisories
    are skipped; a network or JSON error on one package is logged and the rest continue.

    Args:
        packages: PyPI package names to query.
        client: HTTP client to use; when None, one is created and closed here.
        versions: Optional package name to installed version.

    Returns:
        One advisory item per vulnerability found.
    """
    if not packages:
        return []

    advisories: list[Item] = []
    own_client = False
    if client is None:
        client = httpx.AsyncClient(timeout=10.0)
        own_client = True

    try:
        for pkg in packages:
            try:
                query: dict[str, object] = {"package": {"name": pkg, "ecosystem": "PyPI"}}
                if versions and pkg in versions:
                    query["version"] = versions[pkg]
                resp = await client.post(OSV_QUERY_URL, json=query)
                if resp.status_code != 200:
                    continue
                data = resp.json()
                vulns = data.get("vulns", [])
                for v in vulns:
                    vuln_id = v.get("id")
                    if not vuln_id or v.get("withdrawn"):
                        continue
                    summary = v.get("summary") or f"Vulnerability in {pkg}"
                    details = v.get("details") or summary
                    pub_str = v.get("published")
                    pub_dt = None
                    if pub_str:
                        try:
                            pub_dt = datetime.fromisoformat(pub_str.replace("Z", "+00:00"))
                        except ValueError:
                            pub_dt = datetime.now(UTC)
                    else:
                        pub_dt = datetime.now(UTC)

                    installed = f" ({pkg} {versions[pkg]} installed)" if versions and pkg in versions else ""
                    advisory_item = Item(
                        id=f"advisory:{vuln_id}",
                        title=f"Advisory {vuln_id}: {summary}{installed}",
                        text=f"{summary}\n\n{details}",
                        kind="advisory",
                        url=f"https://osv.dev/vulnerability/{vuln_id}",
                        status="scored",
                        source_id="osv",
                        published_at=pub_dt,
                        fetched_at=datetime.now(UTC),
                    )
                    advisories.append(advisory_item)
            except Exception as exc:  # noqa: BLE001  # Safe degradation: network or JSON error on one package logs and continues
                logger.warning("Error fetching OSV advisories for %s: %s", pkg, exc)
    finally:
        if own_client:
            await client.aclose()

    return advisories


async def check_stack_dependencies(
    db: Database,
    pyproject_path: Path | str = "pyproject.toml",
    client: httpx.AsyncClient | None = None,
) -> list[Item]:
    """Fetch and store OSV advisories for a project's installed dependency versions.

    Packages that aren't installed here are skipped: without a version OSV returns every
    advisory ever filed, long-fixed ones included. Creates the OSV source and the stack question
    if they don't exist yet.

    Args:
        db: Database to store the source, question and advisories in.
        pyproject_path: Path to the project's pyproject.toml.
        client: Optional HTTP client for the OSV queries.

    Returns:
        The advisories found and stored.
    """
    versions = installed_versions(parse_pyproject_dependencies(pyproject_path))
    if not versions:
        return []

    if not db.get_source(STACK_SOURCE_ID):
        db.upsert_source(
            SourceRow(
                id=STACK_SOURCE_ID,
                type="osv",
                params={"url": OSV_QUERY_URL},
                cadence_s=int(STACK_CHECK_EVERY.total_seconds()),
                trust=0.9,
            )
        )
    if db.get_question(STACK_QUESTION_ID) is None:
        db.upsert_question(Question(id=STACK_QUESTION_ID, text="What changed in my stack?", watch=str(pyproject_path)))

    advisories = await fetch_osv_advisories(sorted(versions), client=client, versions=versions)
    for adv in advisories:
        db.upsert_item(adv)
    return advisories


async def run_stack_watch_if_due(db: Database, client: httpx.AsyncClient, now: datetime) -> list[str]:
    """Run the stack check once a day when a question watches a pyproject.toml.

    A failure is logged and retried next day; it never fails the collect tick.

    Args:
        db: Database holding questions, the OSV source and advisories.
        client: HTTP client for the OSV queries.
        now: Current time, compared with the source's next due time.

    Returns:
        The ids of the advisories found; empty when nothing is watched or the check isn't due.
    """
    watch = next((q.watch for q in db.list_questions() if q.watch), None)
    src = db.get_source(STACK_SOURCE_ID)
    if not watch or (src is not None and src.next_due_at and src.next_due_at > now):
        return []
    try:
        advisories = await check_stack_dependencies(db, watch, client=client)
    except Exception as exc:  # noqa: BLE001 - the stack watch is one more source: it must not lose the tick
        logger.warning("Stack watch failed: %s", exc)
        advisories = []
    if (src := db.get_source(STACK_SOURCE_ID)) is not None:
        src.next_due_at = now + STACK_CHECK_EVERY
        db.upsert_source(src)
    return [a.id for a in advisories]
