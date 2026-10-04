"""CODEOWNERS parsing and owner resolution.

Implements the GitHub CODEOWNERS matching rules that matter for triage:
gitignore-style patterns, last-matching-pattern-wins, user/team/email owners.
Owner recommendations are always grounded in this file — the model never
invents usernames.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class OwnerRule:
    pattern: str
    owners: tuple[str, ...]


def _pattern_to_regex(pattern: str) -> re.Pattern[str]:
    """Convert a CODEOWNERS pattern to a regex over repo-relative paths."""
    anchored = pattern.startswith("/")
    pat = pattern.lstrip("/")
    dir_only = pat.endswith("/")
    pat = pat.rstrip("/")

    out: list[str] = []
    i = 0
    while i < len(pat):
        ch = pat[i]
        if ch == "*":
            if pat[i : i + 2] == "**":
                out.append(".*")
                i += 2
                if i < len(pat) and pat[i] == "/":
                    i += 1
                continue
            out.append("[^/]*")
        elif ch == "?":
            out.append("[^/]")
        else:
            out.append(re.escape(ch))
        i += 1
    body = "".join(out)

    prefix = "^" if anchored else r"^(?:.*/)?"
    suffix = "(?:/.*)?$" if dir_only else r"(?:/.*)?$" if "." not in pat.rsplit("/", 1)[-1] else "$"
    # A pattern whose last segment has no dot (e.g. "/backend") matches both a
    # file named "backend" and everything under a "backend/" directory, which
    # mirrors GitHub behaviour closely enough for triage.
    return re.compile(prefix + body + suffix)


class CodeOwners:
    """Parsed CODEOWNERS file."""

    def __init__(self, rules: list[OwnerRule]) -> None:
        self.rules = rules
        self._compiled = [(_pattern_to_regex(rule.pattern), rule) for rule in rules]

    @classmethod
    def parse(cls, text: str | None) -> CodeOwners:
        rules: list[OwnerRule] = []
        for raw_line in (text or "").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            pattern, owners = parts[0], tuple(parts[1:])
            if owners:
                rules.append(OwnerRule(pattern=pattern, owners=owners))
        return cls(rules)

    def owners_for(self, path: str) -> list[str]:
        """Owners for a single path — the LAST matching rule wins."""
        path = path.lstrip("/")
        matched: tuple[str, ...] = ()
        for regex, rule in self._compiled:
            if regex.match(path):
                matched = rule.owners
        return list(matched)

    def owners_for_files(self, paths: list[str]) -> list[str]:
        """Owners across many files, ranked by how many files they own."""
        counts: dict[str, int] = {}
        for path in paths:
            for owner in self.owners_for(path):
                counts[owner] = counts.get(owner, 0) + 1
        return sorted(counts, key=lambda o: (-counts[o], o))

    def owners_for_terms(self, terms: list[str]) -> list[str]:
        """Best-effort owner lookup for issues (no file paths): match terms
        like 'backend' or 'docs' against the path segments of each pattern."""
        lowered = {t.lower().strip("/") for t in terms if t.strip()}
        counts: dict[str, int] = {}
        for rule in self.rules:
            segments = {
                seg.lower()
                for seg in rule.pattern.strip("/").split("/")
                if seg and "*" not in seg and "." not in seg
            }
            if segments & lowered:
                for owner in rule.owners:
                    counts[owner] = counts.get(owner, 0) + 1
        return sorted(counts, key=lambda o: (-counts[o], o))


def split_owner(owner: str) -> tuple[str, str]:
    """Classify an owner entry: ('user', name) | ('team', org/team) | ('email', addr)."""
    if owner.startswith("@") and "/" in owner:
        return "team", owner[1:]
    if owner.startswith("@"):
        return "user", owner[1:]
    return "email", owner
