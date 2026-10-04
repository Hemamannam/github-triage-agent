# Demo repository

The agent is best demonstrated against a throwaway repository seeded with
intentionally varied issues and PRs — never against a production repo.

## 1. Create the repository

```bash
gh repo create <you>/triage-demo --private --add-readme
cd "$(mktemp -d)" && gh repo clone <you>/triage-demo && cd triage-demo
```

## 2. Add structure the agent reads

```bash
mkdir -p .github backend/auth frontend docs tests

cat > .github/CODEOWNERS <<'EOF'
/backend/       @<you>
/backend/auth/  @<you>
/frontend/      @<you>
/docs/          @<you>
EOF

cat > CONTRIBUTING.md <<'EOF'
# Contributing
Bug reports must include: package version, Python version, a complete
traceback, and minimal reproduction steps. PRs must include tests.
EOF

echo "print('demo')" > backend/app.py
git add -A && git commit -m "seed structure" && git push
```

## 3. Create labels

```bash
for l in bug enhancement documentation question security duplicate invalid needs-tests P0 P1 P2 P3; do
  gh label create "$l" --force
done
```

## 4. Seed the issues

```bash
# 1 — valid backend bug (expect: bug + P1 labels)
gh issue create -t "OAuth callback crashes when token expires" \
  -b "KeyError in refresh_token. Version 2.1.0, Python 3.12. Traceback:
\`\`\`
KeyError: 'refresh_token' at backend/auth/callback.py:42
\`\`\`
Repro: 1) login 2) wait for token expiry 3) hit /callback"

# 2 — duplicate of issue 1 (expect: duplicate detected, close held for approval)
gh issue create -t "Crash in OAuth callback after token expiry" \
  -b "Same KeyError: 'refresh_token' in the auth callback when the token has expired."

# 3 — feature request (expect: enhancement label)
gh issue create -t "Add dark mode" -b "Please support a dark theme in the UI."

# 4 — missing reproduction info (expect: missing-info comment drafted)
gh issue create -t "App crashes sometimes" -b "it just crashes, please fix"

# 5 — documentation request (expect: documentation label)
gh issue create -t "Install guide typo" -b "The pip install command in the README is wrong."

# 6 — security-sensitive report (expect: critical priority, human review)
gh issue create -t "JWT signature not verified" \
  -b "Tokens are accepted without verifying the signature in backend/auth/token.py."
```

## 5. Seed the PRs

```bash
# PR 1 — normal feature PR with tests
git checkout -b feat/rate-limiter
echo "class Limiter: pass" > backend/limiter.py
echo "def test_limiter(): pass" > tests/test_limiter.py
git add -A && git commit -m "add rate limiter" && git push -u origin HEAD
gh pr create -t "Add rate limiter" -b "Adds request throttling. Fixes #1" \
  --base main

# PR 2 — missing tests (expect: needs-tests feedback)
git checkout main && git checkout -b refactor/auth-flow
echo "# refactored" >> backend/auth/flow.py
git add -A && git commit -m "refactor auth flow" && git push -u origin HEAD
gh pr create -t "Refactor auth flow" -b "Cleanup of the auth module." --base main

# PR 3 — failing CI: add a workflow that always fails, then a PR
git checkout main && git checkout -b feat/export
mkdir -p .github/workflows
cat > .github/workflows/ci.yml <<'EOF'
name: ci
on: [pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - run: exit 1
EOF
echo "def export(): ..." > backend/export.py
git add -A && git commit -m "export endpoint" && git push -u origin HEAD
gh pr create -t "New export endpoint" -b "Adds CSV export." --base main
```

## 6. Run the agent against it

```bash
# Always start in dry-run:
triage-agent run --repo <you>/triage-demo --issue 1 --dry-run
triage-agent run --repo <you>/triage-demo --issue 2 --dry-run   # duplicate flow
triage-agent run --repo <you>/triage-demo --issue 4 --dry-run   # missing info

# Live, with interactive approval prompts:
triage-agent run --repo <you>/triage-demo --issue 2 --live
```

Add the demo repo to `config.yaml → security.allowed_repositories` so the
agent physically cannot touch anything else while you demo.
