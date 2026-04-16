from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

DEFAULT_MODE = 'auto'
DEFAULT_TIMEOUT_SEC = 120
DEFAULT_BUILD_ROOT_NAME = 'frontend_build'
BUILD_SCRIPT = Path('tools') / 'build-frontend.mjs'
ENTRYPOINTS = {
    'js/map.js': Path('frontend/js/map.js'),
    'js/recognize.js': Path('frontend/js/recognize.js'),
    'js/settings.js': Path('frontend/js/settings.js'),
}
SCHEMA_VERSION = 1


def _log(message: str) -> None:
    print(f'[frontend-build] {message}')


def _normalize_mode(raw: Optional[str]) -> str:
    value = str(raw or DEFAULT_MODE).strip().lower()
    if value in ('off', 'false', '0', 'disable', 'disabled'):
        return 'off'
    if value in ('strict', 'required', 'hard'):
        return 'strict'
    return 'auto'


def get_build_root(project_root: Path) -> Path:
    configured = os.environ.get('FRONTEND_BUILD_DIR', '').strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return (project_root / DEFAULT_BUILD_ROOT_NAME).resolve()


def get_active_dir(project_root: Path) -> Path:
    return get_build_root(project_root) / 'active'


def get_previous_dir(project_root: Path) -> Path:
    return get_build_root(project_root) / 'previous'


def get_staging_dir(project_root: Path) -> Path:
    return get_build_root(project_root) / 'staging'


def get_manifest_path(project_root: Path) -> Path:
    return get_build_root(project_root) / 'manifest.json'


def _find_npm_command() -> Optional[str]:
    return shutil.which('npm.cmd') or shutil.which('npm')


def _load_json(path: Path) -> Optional[dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _required_outputs() -> list[str]:
    return list(ENTRYPOINTS.keys())


def _validate_output_tree(root_dir: Path, manifest: Optional[dict[str, Any]] = None) -> tuple[bool, str]:
    if not root_dir.is_dir():
        return False, f'目录不存在: {root_dir}'

    entries = manifest.get('entries') if isinstance(manifest, dict) else None
    expected_outputs = []
    if isinstance(entries, list):
        for item in entries:
            if isinstance(item, dict):
                output = str(item.get('output') or '').replace('\\', '/')
                if output:
                    expected_outputs.append(output)
    if not expected_outputs:
        expected_outputs = _required_outputs()

    missing = []
    empty = []
    for relative in expected_outputs:
        candidate = root_dir / Path(relative)
        if not candidate.is_file():
            missing.append(relative)
            continue
        if candidate.stat().st_size <= 0:
            empty.append(relative)

    if missing:
        return False, '缺少构建产物: ' + ', '.join(missing)
    if empty:
        return False, '构建产物为空: ' + ', '.join(empty)
    return True, 'ok'


def _build_manifest(mode: str, duration_ms: int) -> dict[str, Any]:
    return {
        'schemaVersion': SCHEMA_VERSION,
        'createdAt': datetime.now(timezone.utc).isoformat(),
        'mode': mode,
        'durationMs': duration_ms,
        'tool': {
            'runner': 'esbuild',
            'script': str(BUILD_SCRIPT).replace('\\', '/'),
        },
        'entries': [
            {
                'requestPath': request_path,
                'source': str(source).replace('\\', '/'),
                'output': request_path,
            }
            for request_path, source in ENTRYPOINTS.items()
        ],
    }


def _run_build(project_root: Path, staging_dir: Path, timeout_sec: int) -> tuple[bool, str, int]:
    npm_cmd = _find_npm_command()
    if not npm_cmd:
        return False, '未找到 npm，请先安装 Node.js/npm。', 0

    package_json = project_root / 'package.json'
    if not package_json.is_file():
        return False, '缺少 package.json，无法执行前端预构建。', 0

    if not (project_root / 'node_modules' / 'esbuild').exists():
        return False, '缺少本地 esbuild 依赖，请先执行 npm install。', 0

    outdir = staging_dir / 'js'
    outdir.mkdir(parents=True, exist_ok=True)
    command = [
        npm_cmd,
        'run',
        'build:frontend',
        '--',
        '--outdir',
        str(outdir),
    ]
    start = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=max(1, int(timeout_sec)),
            check=False,
        )
    except subprocess.TimeoutExpired:
        duration_ms = int((time.perf_counter() - start) * 1000)
        return False, f'构建超时（>{timeout_sec}s）', duration_ms
    except Exception as exc:
        duration_ms = int((time.perf_counter() - start) * 1000)
        return False, f'启动构建失败: {exc}', duration_ms

    duration_ms = int((time.perf_counter() - start) * 1000)
    stdout = (completed.stdout or '').strip()
    stderr = (completed.stderr or '').strip()
    if stdout:
        _log(stdout)
    if stderr:
        _log(stderr)
    if completed.returncode != 0:
        return False, f'构建命令退出码 {completed.returncode}', duration_ms
    return True, 'ok', duration_ms


def _activate_staging(project_root: Path, staging_dir: Path, manifest: dict[str, Any]) -> tuple[bool, str, str]:
    build_root = get_build_root(project_root)
    active_dir = get_active_dir(project_root)
    previous_dir = get_previous_dir(project_root)
    manifest_path = get_manifest_path(project_root)
    build_root.mkdir(parents=True, exist_ok=True)

    staging_manifest_path = staging_dir / 'manifest.json'
    staging_manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')

    active_backup_made = False
    try:
        if previous_dir.exists():
            shutil.rmtree(previous_dir)
        if active_dir.exists():
            active_dir.rename(previous_dir)
            active_backup_made = True
        staging_dir.rename(active_dir)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
        return True, '已激活新的前端构建产物。', 'build'
    except Exception as exc:
        _log(f'激活新构建失败，准备回退: {exc}')
        try:
            if active_dir.exists():
                shutil.rmtree(active_dir)
        except Exception:
            pass
        if active_backup_made and previous_dir.exists():
            try:
                previous_dir.rename(active_dir)
                if manifest_path.exists():
                    previous_manifest = _load_json(active_dir / 'manifest.json') or _load_json(manifest_path)
                    if previous_manifest:
                        manifest_path.write_text(json.dumps(previous_manifest, ensure_ascii=False, indent=2), encoding='utf-8')
                return False, '激活失败，已回退到 previous 构建。', 'previous'
            except Exception as restore_exc:
                _log(f'回退 previous 失败: {restore_exc}')
        try:
            if manifest_path.exists():
                manifest_path.unlink()
        except Exception:
            pass
        return False, '激活失败，已回退到源码静态资源。', 'source'


def _validate_existing_active(project_root: Path) -> tuple[str, dict[str, Any] | None]:
    active_dir = get_active_dir(project_root)
    manifest_path = get_manifest_path(project_root)
    manifest = _load_json(manifest_path)
    valid, _ = _validate_output_tree(active_dir, manifest)
    if valid:
        return 'build', manifest

    previous_dir = get_previous_dir(project_root)
    previous_manifest = _load_json(previous_dir / 'manifest.json') or manifest
    valid_prev, _ = _validate_output_tree(previous_dir, previous_manifest)
    if valid_prev:
        return 'previous', previous_manifest
    return 'source', None


def prebuild_frontend(project_root: Path, *, mode: Optional[str] = None) -> dict[str, Any]:
    project_root = Path(project_root).resolve()
    build_mode = _normalize_mode(mode or os.environ.get('FRONTEND_PREBUILD'))
    timeout_sec = int(os.environ.get('FRONTEND_BUILD_TIMEOUT_SEC', str(DEFAULT_TIMEOUT_SEC)))

    result: dict[str, Any] = {
        'mode': build_mode,
        'attempted': False,
        'continueStartup': True,
        'activeSource': 'source',
        'message': '',
    }

    if build_mode == 'off':
        result['message'] = '已关闭前端预构建，直接使用源码静态资源。'
        _log(result['message'])
        return result

    staging_dir = get_staging_dir(project_root)
    _ensure_clean_dir(staging_dir)
    result['attempted'] = True

    ok, detail, duration_ms = _run_build(project_root, staging_dir, timeout_sec)
    if not ok:
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)
        source, _ = _validate_existing_active(project_root)
        result['activeSource'] = source
        result['message'] = f'预构建失败：{detail}'
        if build_mode == 'strict':
            result['continueStartup'] = False
            _log(result['message'])
            return result
        _log(result['message'] + f'；已回退到 {source}。')
        return result

    manifest = _build_manifest(build_mode, duration_ms)
    valid, validation_detail = _validate_output_tree(staging_dir, manifest)
    if not valid:
        shutil.rmtree(staging_dir, ignore_errors=True)
        source, _ = _validate_existing_active(project_root)
        result['activeSource'] = source
        result['message'] = f'预构建校验失败：{validation_detail}'
        if build_mode == 'strict':
            result['continueStartup'] = False
            _log(result['message'])
            return result
        _log(result['message'] + f'；已回退到 {source}。')
        return result

    activated, activate_message, active_source = _activate_staging(project_root, staging_dir, manifest)
    result['activeSource'] = active_source
    result['message'] = f'{activate_message}（耗时 {duration_ms}ms）'
    if activated:
        _log(result['message'])
        return result

    if build_mode == 'strict':
        result['continueStartup'] = False
        _log(result['message'])
        return result

    _log(result['message'])
    return result


def load_active_manifest(project_root: Path) -> Optional[dict[str, Any]]:
    project_root = Path(project_root).resolve()
    manifest = _load_json(get_manifest_path(project_root))
    valid, _ = _validate_output_tree(get_active_dir(project_root), manifest)
    if valid:
        return manifest

    previous_dir = get_previous_dir(project_root)
    previous_manifest = _load_json(previous_dir / 'manifest.json') or manifest
    valid_previous, _ = _validate_output_tree(previous_dir, previous_manifest)
    return previous_manifest if valid_previous else None


def resolve_preferred_static_root(project_root: Path) -> Optional[Path]:
    project_root = Path(project_root).resolve()
    manifest = load_active_manifest(project_root)
    if manifest is None:
        return None

    active_dir = get_active_dir(project_root)
    valid_active, _ = _validate_output_tree(active_dir, manifest)
    if valid_active:
        return active_dir

    previous_dir = get_previous_dir(project_root)
    valid_previous, _ = _validate_output_tree(previous_dir, manifest)
    if valid_previous:
        return previous_dir
    return None


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build frontend assets before starting the web app.')
    parser.add_argument('--mode', choices=['auto', 'off', 'strict'], default=None)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    project_root = Path(__file__).resolve().parent.parent
    result = prebuild_frontend(project_root, mode=args.mode)
    return 0 if result.get('continueStartup', True) else 1


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
