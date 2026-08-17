#!/usr/bin/env python3
"""Install the self-dispatching, state-bound Perfect Chaos campaign loop."""

from __future__ import annotations

import sys
from pathlib import Path


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one match, found {count}")
    return source.replace(old, new)


def patch_driver(path: Path) -> None:
    source = path.read_text()
    old = '''    let inputFrontier = from === reusable.through
      ? reusable.inputFrontier
      : (() => {
        const index = preparedBoundaries.indexOf(from);
        const previous = index <= 0 ? 0 : preparedBoundaries[index - 1];
        return join(roleDirectory, `${previous}-${from}.frontier.bin`);
      })();'''
    new = '''    let inputFrontier = from === 0
      ? null
      : from === reusable.through
        ? reusable.inputFrontier
        : (() => {
          const index = preparedBoundaries.indexOf(from);
          if (index < 0) throw new Error(`Unknown prepared boundary: ${from}.`);
          const previous = index === 0 ? 0 : preparedBoundaries[index - 1];
          return join(roleDirectory, `${previous}-${from}.frontier.bin`);
        })();'''
    if old in source:
        source = source.replace(old, new, 1)
    elif new not in source:
        raise SystemExit("root rollback semantics: source shape not recognized")
    path.write_text(source)


def patch_round16(path: Path) -> None:
    source = path.read_text()
    old = '''          base64 --decode .publisher/inc-prefix-root-rollback.b64 \\
            > /tmp/perfect-chaos-prefix-root-rollback.patch
          test "$(wc -c < /tmp/perfect-chaos-prefix-root-rollback.patch)" = '1021'
          echo 'ce5b4cff8da3afb4dfd552d2299df5b24d4f18dc64aa6b0e3ebcddff50f9e51b  /tmp/perfect-chaos-prefix-root-rollback.patch' \\
            | sha256sum --check --strict -
          test "$(git hash-object scripts/perfect-chaos-prefix.mjs)" = \\
            '8ca1becbcda54b0b8d717e0a09a19da7daa6cb62'
          git apply --check --whitespace=error-all \\
            /tmp/perfect-chaos-prefix-root-rollback.patch
          git apply --whitespace=error-all \\
            /tmp/perfect-chaos-prefix-root-rollback.patch
          test "$(git hash-object scripts/perfect-chaos-prefix.mjs)" = \\
            '4716d75f92d4e161adff0f76bd686d1c9632c0ec'
          node --check scripts/perfect-chaos-prefix.mjs'''
    new = '''          node --check scripts/perfect-chaos-prefix.mjs
          node scripts/perfect-chaos-prefix.mjs verify \\
            > /tmp/perfect-chaos-prefix-verification.json
          jq -e '
            .incrementalPreparation.repairSummaries | length >= 1
            and all(.[]; .status == "safe"
              and .fallbackFullRegeneration == false
              and .repairRoots < .inputRoots)
          ' /tmp/perfect-chaos-prefix-verification.json >/dev/null'''
    if old in source:
        source = source.replace(old, new, 1)
    elif new not in source:
        raise SystemExit("16-piece production pin: source shape not recognized")
    path.write_text(source)


def patch_state16(path: Path) -> None:
    source = path.read_text()
    old_header = '''on:
  push:
    branches:
      - agent/perfect-chaos-16-recovery
    paths:
      - .campaign/perfect-chaos-16/*.json
'''
    new_header = '''on:
  push:
    branches:
      - agent/perfect-chaos-16-recovery
    paths:
      - .campaign/perfect-chaos-16/*.json
  workflow_dispatch:
    inputs:
      state_file:
        description: Exact 16-piece campaign state file to run
        required: true
        type: string
'''
    if old_header in source:
        source = source.replace(old_header, new_header, 1)
    elif new_header not in source:
        raise SystemExit("16-piece dispatch header: source shape not recognized")

    old_env = '''        env:
          BEFORE: ${{ github.event.before }}
          AFTER: ${{ github.sha }}'''
    new_env = '''        env:
          EVENT_NAME: ${{ github.event_name }}
          REQUESTED_STATE: ${{ inputs.state_file || '' }}
          BEFORE: ${{ github.event.before }}
          AFTER: ${{ github.sha }}'''
    if old_env in source:
        source = source.replace(old_env, new_env, 1)
    elif new_env not in source:
        raise SystemExit("16-piece dispatch environment: source shape not recognized")

    old_select = '''          root = Path('.campaign/perfect-chaos-16')
          before = os.environ['BEFORE']
          after = os.environ['AFTER']
          zero = '0' * 40
          if before == zero:
              command = ['git', 'diff-tree', '--no-commit-id', '--name-only', '-r', after]
          else:
              command = ['git', 'diff', '--name-only', before, after]
          changed = subprocess.check_output(command, text=True).splitlines()
          paths = sorted(
              Path(path) for path in changed
              if path.startswith(f'{root.as_posix()}/') and path.endswith('.json')
          )
          if not paths:
              raise RuntimeError('No Perfect Chaos campaign state changed in this push.')'''
    new_select = '''          root = Path('.campaign/perfect-chaos-16')
          event_name = os.environ['EVENT_NAME']
          requested_state = os.environ.get('REQUESTED_STATE', '')
          if event_name == 'workflow_dispatch':
              path = Path(requested_state)
              if path.parent != root or path.suffix != '.json':
                  raise RuntimeError(f'Unsafe dispatched 16-piece campaign state: {path}')
              paths = [path]
          else:
              before = os.environ['BEFORE']
              after = os.environ['AFTER']
              zero = '0' * 40
              if before == zero:
                  command = ['git', 'diff-tree', '--no-commit-id', '--name-only', '-r', after]
              else:
                  command = ['git', 'diff', '--name-only', before, after]
              changed = subprocess.check_output(command, text=True).splitlines()
              paths = sorted(
                  Path(path) for path in changed
                  if path.startswith(f'{root.as_posix()}/') and path.endswith('.json')
              )
              if not paths:
                  raise RuntimeError('No Perfect Chaos campaign state changed in this push.')'''
    if old_select in source:
        source = source.replace(old_select, new_select, 1)
    elif new_select not in source:
        raise SystemExit("16-piece dispatched state selection: source shape not recognized")
    path.write_text(source)


def patch_state18(path: Path) -> None:
    source = path.read_text()
    old_header = '''on:
  push:
    branches:
      - agent/perfect-chaos-16-recovery
    paths:
      - .github/workflows/continue-perfect-chaos-18-state.yml
      - .campaign/perfect-chaos-18/red.json
      - .campaign/perfect-chaos-18/yellow.json
'''
    new_header = '''on:
  push:
    branches:
      - agent/perfect-chaos-16-recovery
    paths:
      - .github/workflows/continue-perfect-chaos-18-state.yml
      - .campaign/perfect-chaos-18/red.json
      - .campaign/perfect-chaos-18/yellow.json
  workflow_dispatch:
    inputs:
      state_file:
        description: Exact 18-piece campaign state file to run
        required: true
        type: string
'''
    if old_header in source:
        source = source.replace(old_header, new_header, 1)
    elif new_header not in source:
        raise SystemExit("18-piece dispatch header: source shape not recognized")

    old_env = '''        env:
          BEFORE: ${{ github.event.before }}
          AFTER: ${{ github.sha }}'''
    new_env = '''        env:
          EVENT_NAME: ${{ github.event_name }}
          REQUESTED_STATE: ${{ inputs.state_file || '' }}
          BEFORE: ${{ github.event.before }}
          AFTER: ${{ github.sha }}'''
    if old_env in source:
        source = source.replace(old_env, new_env, 1)
    elif new_env not in source:
        raise SystemExit("18-piece dispatch environment: source shape not recognized")

    old_select = '''          root = Path('.campaign/perfect-chaos-18')
          workflow_path = '.github/workflows/continue-perfect-chaos-18-state.yml'
          before = os.environ['BEFORE']
          after = os.environ['AFTER']
          zero = '0' * 40
          if before == zero:
              command = ['git', 'diff-tree', '--no-commit-id', '--name-only', '-r', after]
          else:
              command = ['git', 'diff', '--name-only', before, after]
          changed = subprocess.check_output(command, text=True).splitlines()
          paths = sorted(
              Path(path) for path in changed
              if path in {
                  f'{root.as_posix()}/red.json',
                  f'{root.as_posix()}/yellow.json',
              }
          )
          output_path = Path(os.environ['GITHUB_OUTPUT'])
          if not paths and workflow_path in changed:
              with output_path.open('a') as output:
                  output.write('run=false\\n')
              print(json.dumps({'registrationOnly': True, 'changed': changed}, indent=2))
              raise SystemExit(0)
          if len(paths) != 1:
              raise RuntimeError(f'Exactly one 18-piece campaign state must change; got {paths}.')'''
    new_select = '''          root = Path('.campaign/perfect-chaos-18')
          workflow_path = '.github/workflows/continue-perfect-chaos-18-state.yml'
          event_name = os.environ['EVENT_NAME']
          requested_state = os.environ.get('REQUESTED_STATE', '')
          output_path = Path(os.environ['GITHUB_OUTPUT'])
          if event_name == 'workflow_dispatch':
              path = Path(requested_state)
              if path not in {root / 'red.json', root / 'yellow.json'}:
                  raise RuntimeError(f'Unsafe dispatched 18-piece campaign state: {path}')
              paths = [path]
          else:
              before = os.environ['BEFORE']
              after = os.environ['AFTER']
              zero = '0' * 40
              if before == zero:
                  command = ['git', 'diff-tree', '--no-commit-id', '--name-only', '-r', after]
              else:
                  command = ['git', 'diff', '--name-only', before, after]
              changed = subprocess.check_output(command, text=True).splitlines()
              paths = sorted(
                  Path(path) for path in changed
                  if path in {
                      f'{root.as_posix()}/red.json',
                      f'{root.as_posix()}/yellow.json',
                  }
              )
              if not paths and workflow_path in changed:
                  with output_path.open('a') as output:
                      output.write('run=false\\n')
                  print(json.dumps({'registrationOnly': True, 'changed': changed}, indent=2))
                  raise SystemExit(0)
              if len(paths) != 1:
                  raise RuntimeError(f'Exactly one 18-piece campaign state must change; got {paths}.')'''
    if old_select in source:
        source = source.replace(old_select, new_select, 1)
    elif new_select not in source:
        raise SystemExit("18-piece dispatched state selection: source shape not recognized")
    path.write_text(source)


def patch_auto(path: Path) -> None:
    source = path.read_text()
    old_inputs = '''      state_workflow:
        required: true
        type: string
'''
    new_inputs = '''      state_workflow:
        required: true
        type: string
      watcher_workflow:
        required: true
        type: string
      expected_run_id:
        required: false
        type: string
        default: ''
      expected_run_sha:
        required: false
        type: string
        default: ''
'''
    if old_inputs in source:
        source = source.replace(old_inputs, new_inputs, 1)
    elif new_inputs not in source:
        raise SystemExit("auto-advance inputs: source shape not recognized")

    source = source.replace("  actions: read\n  contents: write", "  actions: write\n  contents: write", 1)

    old_env = '''      STATE_FILE: ${{ inputs.state_file }}
      STATE_WORKFLOW: ${{ inputs.state_workflow }}'''
    new_env = '''      STATE_FILE: ${{ inputs.state_file }}
      STATE_WORKFLOW: ${{ inputs.state_workflow }}
      WATCHER_WORKFLOW: ${{ inputs.watcher_workflow }}
      EXPECTED_RUN_ID: ${{ inputs.expected_run_id }}
      EXPECTED_RUN_SHA: ${{ inputs.expected_run_sha }}'''
    if old_env in source:
        source = source.replace(old_env, new_env, 1)
    elif new_env not in source:
        raise SystemExit("auto-advance environment: source shape not recognized")

    old_validation = '''          case "$STATE_WORKFLOW" in
            continue-perfect-chaos-16-state.yml|continue-perfect-chaos-18-state.yml) ;;
            *) echo 'Unexpected state workflow.' >&2; exit 1 ;;
          esac
          test -f "$STATE_FILE" && test ! -L "$STATE_FILE"'''
    new_validation = '''          case "$STATE_WORKFLOW:$WATCHER_WORKFLOW" in
            continue-perfect-chaos-16-state.yml:auto-advance-perfect-chaos-yellow-16.yml) ;;
            continue-perfect-chaos-18-state.yml:auto-advance-perfect-chaos-red-18.yml) ;;
            *) echo 'Unexpected campaign workflow pair.' >&2; exit 1 ;;
          esac
          test -f "$STATE_FILE" && test ! -L "$STATE_FILE"'''
    if old_validation in source:
        source = source.replace(old_validation, new_validation, 1)
    elif new_validation not in source:
        raise SystemExit("auto-advance workflow pairing: source shape not recognized")

    old_lookup = '''          response=''
          count=0
          for _ in $(seq 1 120); do
            response="$(gh api --method GET \\
              "repos/$REPOSITORY/actions/workflows/$STATE_WORKFLOW/runs" \\
              -f branch="$BRANCH" -F per_page=100)"
            count="$(jq --arg sha "$state_commit" '[.workflow_runs[] | select(.head_sha == $sha)] | length' <<< "$response")"
            if [[ "$count" = 1 ]]; then
              break
            fi
            if (( count > 1 )); then
              echo "Ambiguous $STATE_WORKFLOW runs at $state_commit: $count." >&2
              exit 1
            fi
            sleep 5
          done
          if [[ "$count" != 1 ]]; then
            echo "No $STATE_WORKFLOW run appeared at $state_commit." >&2
            exit 1
          fi
          run_id="$(jq -r --arg sha "$state_commit" '.workflow_runs[] | select(.head_sha == $sha) | .id' <<< "$response")"
          run_sha="$(jq -r --arg sha "$state_commit" '.workflow_runs[] | select(.head_sha == $sha) | .head_sha' <<< "$response")"'''
    new_lookup = '''          if [[ -n "$EXPECTED_RUN_ID" || -n "$EXPECTED_RUN_SHA" ]]; then
            if ! [[ "$EXPECTED_RUN_ID" =~ ^[1-9][0-9]*$ && "$EXPECTED_RUN_SHA" =~ ^[0-9a-f]{40}$ ]]; then
              echo 'Pinned campaign run identity is incomplete or malformed.' >&2
              exit 1
            fi
            pinned="$(gh api "repos/$REPOSITORY/actions/runs/$EXPECTED_RUN_ID")"
            test "$(jq -r '.path' <<< "$pinned")" = ".github/workflows/$STATE_WORKFLOW"
            test "$(jq -r '.head_branch' <<< "$pinned")" = "$BRANCH"
            test "$(jq -r '.head_sha' <<< "$pinned")" = "$EXPECTED_RUN_SHA"
            run_id="$EXPECTED_RUN_ID"
            run_sha="$EXPECTED_RUN_SHA"
          else
            response=''
            count=0
            for _ in $(seq 1 120); do
              response="$(gh api --method GET \\
                "repos/$REPOSITORY/actions/workflows/$STATE_WORKFLOW/runs" \\
                -f branch="$BRANCH" -F per_page=100)"
              count="$(jq --arg sha "$state_commit" '[.workflow_runs[] | select(.head_sha == $sha)] | length' <<< "$response")"
              if [[ "$count" = 1 ]]; then
                break
              fi
              if (( count > 1 )); then
                echo "Ambiguous $STATE_WORKFLOW runs at $state_commit: $count." >&2
                exit 1
              fi
              sleep 5
            done
            if [[ "$count" != 1 ]]; then
              echo "No $STATE_WORKFLOW run appeared at $state_commit." >&2
              exit 1
            fi
            run_id="$(jq -r --arg sha "$state_commit" '.workflow_runs[] | select(.head_sha == $sha) | .id' <<< "$response")"
            run_sha="$(jq -r --arg sha "$state_commit" '.workflow_runs[] | select(.head_sha == $sha) | .head_sha' <<< "$response")"
          fi
          remote_state_digest="$(gh api --method GET \\
            "repos/$REPOSITORY/contents/$STATE_FILE" -f ref="$run_sha" \\
            --jq '.content' | tr -d '\\n' | base64 --decode | sha256sum | cut -d' ' -f1)"
          if [[ "$remote_state_digest" != "$state_digest" ]]; then
            echo 'Pinned exact run does not contain the current campaign-state bytes.' >&2
            exit 1
          fi'''
    if old_lookup in source:
        source = source.replace(old_lookup, new_lookup, 1)
    elif new_lookup not in source:
        raise SystemExit("auto-advance run binding: source shape not recognized")

    old_publish = '''          gh api --method PUT "repos/$REPOSITORY/contents/$STATE_FILE" \\
            -f branch="$BRANCH" \\
            -f message="Continue ${ROLE^} ${TARGET_PIECES}-piece refinement from $CUMULATIVE roots" \\
            -f content="$content" \\
            -f sha="$blob_sha" >/dev/null
          echo "Advanced $STATE_FILE to $CUMULATIVE cumulative rejected roots."'''
    new_publish = '''          gh api --method PUT "repos/$REPOSITORY/contents/$STATE_FILE" \\
            -f branch="$BRANCH" \\
            -f message="Continue ${ROLE^} ${TARGET_PIECES}-piece refinement from $CUMULATIVE roots" \\
            -f content="$content" \\
            -f sha="$blob_sha" >/dev/null
          echo "Advanced $STATE_FILE to $CUMULATIVE cumulative rejected roots."

          before_runs="$(gh api --method GET \\
            "repos/$REPOSITORY/actions/workflows/$STATE_WORKFLOW/runs" \\
            -f branch="$BRANCH" -F per_page=100 \\
            --jq '[.workflow_runs[].id]')"
          dispatch_started="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
          gh workflow run "$STATE_WORKFLOW" --repo "$REPOSITORY" --ref "$BRANCH" \\
            -f state_file="$STATE_FILE"

          next_run_id=''
          next_run_sha=''
          for _ in $(seq 1 120); do
            runs="$(gh api --method GET \\
              "repos/$REPOSITORY/actions/workflows/$STATE_WORKFLOW/runs" \\
              -f branch="$BRANCH" -F per_page=100)"
            candidates="$(jq \\
              --argjson before "$before_runs" \\
              --arg started "$dispatch_started" \\
              '[.workflow_runs[] | select(.event == "workflow_dispatch"
                and (.id as $id | ($before | index($id)) == null)
                and .created_at >= $started)]' <<< "$runs")"
            count="$(jq 'length' <<< "$candidates")"
            if [[ "$count" = 1 ]]; then
              next_run_id="$(jq -r '.[0].id' <<< "$candidates")"
              next_run_sha="$(jq -r '.[0].head_sha' <<< "$candidates")"
              break
            fi
            if (( count > 1 )); then
              echo "Ambiguous dispatched $STATE_WORKFLOW runs: $count." >&2
              exit 1
            fi
            sleep 5
          done
          if [[ -z "$next_run_id" || ! "$next_run_sha" =~ ^[0-9a-f]{40}$ ]]; then
            echo "Dispatched $STATE_WORKFLOW run did not appear." >&2
            exit 1
          fi
          gh workflow run "$WATCHER_WORKFLOW" --repo "$REPOSITORY" --ref "$BRANCH" \\
            -f expected_run_id="$next_run_id" \\
            -f expected_run_sha="$next_run_sha"
          echo "Dispatched exact run $next_run_id and pinned watcher $WATCHER_WORKFLOW."'''
    if old_publish in source:
        source = source.replace(old_publish, new_publish, 1)
    elif new_publish not in source:
        raise SystemExit("auto-advance dispatch chain: source shape not recognized")
    path.write_text(source)


def patch_watcher(path: Path, watcher_name: str) -> None:
    source = path.read_text()
    own_path = f"      - .github/workflows/{watcher_name}\n"
    source = source.replace(own_path, "", 1)
    old_dispatch = "  workflow_dispatch:\n"
    new_dispatch = '''  workflow_dispatch:
    inputs:
      expected_run_id:
        description: Optional exact workflow run id to watch
        required: false
        type: string
      expected_run_sha:
        description: Optional exact workflow run SHA to watch
        required: false
        type: string
'''
    if old_dispatch in source:
        source = source.replace(old_dispatch, new_dispatch, 1)
    elif new_dispatch not in source:
        raise SystemExit(f"{watcher_name}: dispatch inputs not recognized")
    source = source.replace("  actions: read\n  contents: write", "  actions: write\n  contents: write", 1)
    state_line = (
        "      state_workflow: continue-perfect-chaos-16-state.yml\n"
        if "yellow-16" in watcher_name
        else "      state_workflow: continue-perfect-chaos-18-state.yml\n"
    )
    addition = state_line + f"      watcher_workflow: {watcher_name}\n" + '''      expected_run_id: ${{ inputs.expected_run_id || '' }}
      expected_run_sha: ${{ inputs.expected_run_sha || '' }}
'''
    if state_line in source:
        source = source.replace(state_line, addition, 1)
    elif addition not in source:
        raise SystemExit(f"{watcher_name}: reusable inputs not recognized")
    path.write_text(source)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: patch-perfect-chaos-campaign-loop.py REPOSITORY_ROOT")
    root = Path(sys.argv[1])
    patch_driver(root / "scripts/perfect-chaos-prefix.mjs")
    patch_round16(root / ".github/workflows/reusable-perfect-chaos-16-round.yml")
    patch_state16(root / ".github/workflows/continue-perfect-chaos-16-state.yml")
    patch_state18(root / ".github/workflows/continue-perfect-chaos-18-state.yml")
    patch_auto(root / ".github/workflows/reusable-perfect-chaos-auto-advance.yml")
    patch_watcher(
        root / ".github/workflows/auto-advance-perfect-chaos-yellow-16.yml",
        "auto-advance-perfect-chaos-yellow-16.yml",
    )
    patch_watcher(
        root / ".github/workflows/auto-advance-perfect-chaos-red-18.yml",
        "auto-advance-perfect-chaos-red-18.yml",
    )


if __name__ == "__main__":
    main()
