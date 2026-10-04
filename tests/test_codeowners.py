from __future__ import annotations

from triage_agent.github.codeowners import CodeOwners, split_owner

SAMPLE = """
# Comment line
*           @default-team
/backend/   @backend-team @alice
/frontend/  @frontend-team
/docs/      @docs-team
*.md        @docs-team
/backend/auth/  @security-team
config.yaml jane@example.com
"""


def test_parse_skips_comments_and_blanks() -> None:
    owners = CodeOwners.parse(SAMPLE)
    assert len(owners.rules) == 7


def test_last_matching_rule_wins() -> None:
    owners = CodeOwners.parse(SAMPLE)
    assert owners.owners_for("backend/api.py") == ["@backend-team", "@alice"]
    # More specific later rule overrides the /backend/ rule.
    assert owners.owners_for("backend/auth/login.py") == ["@security-team"]


def test_unanchored_and_wildcard_patterns() -> None:
    owners = CodeOwners.parse(SAMPLE)
    assert owners.owners_for("nested/deep/notes.md") == ["@docs-team"]
    assert owners.owners_for("config.yaml") == ["jane@example.com"]
    assert owners.owners_for("src/unknown.py") == ["@default-team"]


def test_owners_for_files_ranked_by_coverage() -> None:
    owners = CodeOwners.parse(SAMPLE)
    ranked = owners.owners_for_files(
        ["backend/a.py", "backend/b.py", "frontend/app.tsx"]
    )
    # @backend-team and @alice each own 2 of 3 files; @frontend-team owns 1.
    assert set(ranked[:2]) == {"@backend-team", "@alice"}
    assert ranked[2] == "@frontend-team"


def test_owners_for_terms_maps_issue_components() -> None:
    owners = CodeOwners.parse(SAMPLE)
    assert owners.owners_for_terms(["backend", "crash"])[0] in {"@backend-team", "@alice"}
    assert owners.owners_for_terms(["docs"]) == ["@docs-team"]
    assert owners.owners_for_terms(["nonexistent-component"]) == []


def test_empty_or_missing_codeowners() -> None:
    owners = CodeOwners.parse(None)
    assert owners.owners_for("anything.py") == []
    assert owners.owners_for_files(["a", "b"]) == []


def test_split_owner_classification() -> None:
    assert split_owner("@alice") == ("user", "alice")
    assert split_owner("@acme/backend-team") == ("team", "acme/backend-team")
    assert split_owner("jane@example.com") == ("email", "jane@example.com")
