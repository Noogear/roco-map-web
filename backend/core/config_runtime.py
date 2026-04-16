"""运行时配置服务。

将配置元数据、校验与运行时热更新传播从 server 层拆出，
便于后续在 CLI / HTTP / WS 等入口复用。
"""

from __future__ import annotations

from collections import deque
import shlex

from backend import config


CONFIG_RUNTIME_RULES = {
    'JPEG_PAD': {
        'type': 'int', 'min': 0, 'max': 160,
        'group': '界面与渲染', 'label': 'JPEG 扩边像素',
        'description': '控制 JPEG 模式额外扩边，便于前端平滑位移。',
        'editable': True,
    },
    'SEARCH_RADIUS': {
        'type': 'int', 'min': 120, 'max': 2000,
        'group': '识别引擎', 'label': '局部搜索半径',
        'description': '影响局部匹配范围，越大越稳但也更慢。',
        'editable': True,
    },
    'NEARBY_SEARCH_RADIUS': {
        'type': 'int', 'min': 120, 'max': 2400,
        'group': '识别引擎', 'label': '附近搜索半径',
        'description': '局部失败时的附近补救搜索半径。',
        'editable': True,
    },
    'LOCAL_FAIL_LIMIT': {
        'type': 'int', 'min': 1, 'max': 20,
        'group': '识别引擎', 'label': '局部失败阈值',
        'description': '局部匹配连续失败多少次后退回全局搜索。',
        'editable': True,
    },
    'SIFT_JUMP_THRESHOLD': {
        'type': 'int', 'min': 80, 'max': 2000,
        'group': '识别引擎', 'label': '跳变阈值',
        'description': '识别结果与上一帧距离过大时，用于判定异常跳点。',
        'editable': True,
    },
    'LOCAL_REVALIDATE_INTERVAL': {
        'type': 'int', 'min': 1, 'max': 30,
        'group': '识别引擎', 'label': '局部复核周期',
        'description': '局部命中多少次后强制做一次全局复核。',
        'editable': True,
    },
    'LOCAL_REVALIDATE_MIN_QUALITY': {
        'type': 'float', 'min': 0.0, 'max': 1.0,
        'group': '识别引擎', 'label': '局部复核质量阈值',
        'description': '局部质量低于该值时提前触发全局复核。',
        'editable': True,
    },
    'LOCAL_REVALIDATE_MARGIN': {
        'type': 'float', 'min': 0.0, 'max': 0.5,
        'group': '识别引擎', 'label': '复核覆盖边际',
        'description': '全局质量至少高出该阈值才覆盖局部结果。',
        'editable': True,
    },
    'LOCAL_REVALIDATE_DIFF': {
        'type': 'int', 'min': 50, 'max': 1000,
        'group': '识别引擎', 'label': '复核冲突距离',
        'description': '局部与全局结果差异超过该值时视为冲突。',
        'editable': True,
    },
    'MAX_LOST_FRAMES': {
        'type': 'int', 'min': 0, 'max': 300,
        'group': '识别引擎', 'label': '最大惯性帧数',
        'description': '完全丢失前允许惯性维持的最大帧数。',
        'editable': True,
    },
    'LK_ENABLED': {
        'type': 'bool', 'group': '识别引擎', 'label': '启用 LK 光流',
        'description': '开启后可降低部分帧的 SIFT 压力。',
        'editable': True,
    },
    'LK_SIFT_INTERVAL': {
        'type': 'int', 'min': 1, 'max': 30,
        'group': '识别引擎', 'label': 'LK 强制 SIFT 周期',
        'description': '每隔多少帧强制做一次 SIFT 纠偏。',
        'editable': True,
    },
    'LK_MIN_CONFIDENCE': {
        'type': 'float', 'min': 0.0, 'max': 1.0,
        'group': '识别引擎', 'label': 'LK 最低置信度',
        'description': '低于该值时放弃光流结果。',
        'editable': True,
    },
    'ECC_ENABLED': {
        'type': 'bool', 'group': '识别引擎', 'label': '启用 ECC 兜底',
        'description': '低纹理附近搜索失败时允许 ECC 对齐兜底。',
        'editable': True,
    },
    'ECC_MIN_CORRELATION': {
        'type': 'float', 'min': 0.0, 'max': 1.0,
        'group': '识别引擎', 'label': 'ECC 最低相关度',
        'description': 'ECC 结果低于该值将被忽略。',
        'editable': True,
    },
    'ARROW_ANGLE_SMOOTH_ALPHA': {
        'type': 'float', 'min': 0.0, 'max': 1.0,
        'group': '箭头', 'label': '箭头平滑系数',
        'description': '数值越高，箭头越灵敏。',
        'editable': True,
    },
    'ARROW_MOVE_MIN_DISPLACEMENT': {
        'type': 'float', 'min': 0.1, 'max': 20.0,
        'group': '箭头', 'label': '最小移动判定',
        'description': '小于该位移时认为角色接近静止。',
        'editable': True,
    },
    'ARROW_POS_HISTORY_LEN': {
        'type': 'int', 'min': 2, 'max': 20,
        'group': '箭头', 'label': '箭头历史长度',
        'description': '箭头方向计算所保留的坐标历史数。',
        'editable': True,
    },
    'ARROW_STOPPED_DEBOUNCE': {
        'type': 'int', 'min': 1, 'max': 60,
        'group': '箭头', 'label': '静止防抖帧数',
        'description': '连续多少帧接近静止后才视为停下。',
        'editable': True,
    },
    'ARROW_BIG_CHANGE_THRESHOLD': {
        'type': 'float', 'min': 5.0, 'max': 180.0,
        'group': '箭头', 'label': '箭头吸附阈值',
        'description': '方向突变超过该角度时直接吸附到新方向。',
        'editable': True,
    },
    'LINEAR_FILTER_ENABLED': {
        'type': 'bool', 'group': '渲染平滑', 'label': '启用线性过滤',
        'description': '对连续异常跳点做线性过滤。',
        'editable': True,
    },
    'LINEAR_FILTER_WINDOW': {
        'type': 'int', 'min': 1, 'max': 40,
        'group': '渲染平滑', 'label': '线性过滤窗口',
        'description': '异常值过滤时参考的历史窗口大小。',
        'editable': True,
    },
    'LINEAR_FILTER_MAX_DEVIATION': {
        'type': 'int', 'min': 20, 'max': 500,
        'group': '渲染平滑', 'label': '最大偏差',
        'description': '超出该偏差的点更容易被判为异常。',
        'editable': True,
    },
    'LINEAR_FILTER_MAX_CONSECUTIVE': {
        'type': 'int', 'min': 1, 'max': 30,
        'group': '渲染平滑', 'label': '最大连续过滤数',
        'description': '避免长时间连续过滤导致无法恢复。',
        'editable': True,
    },
    'RENDER_STILL_THRESHOLD': {
        'type': 'int', 'min': 0, 'max': 50,
        'group': '渲染平滑', 'label': '静止死区',
        'description': '平滑显示时的小位移忽略阈值。',
        'editable': True,
    },
    'RENDER_EMA_ALPHA': {
        'type': 'float', 'min': 0.0, 'max': 1.0,
        'group': '渲染平滑', 'label': '最低 EMA Alpha',
        'description': '慢速移动时的最低平滑系数。',
        'editable': True,
    },
    'RENDER_EMA_ALPHA_MAX': {
        'type': 'float', 'min': 0.0, 'max': 1.0,
        'group': '渲染平滑', 'label': '最高 EMA Alpha',
        'description': '快速移动时的最高平滑系数。',
        'editable': True,
    },
    'RENDER_EMA_SLOW_DIST': {
        'type': 'int', 'min': 1, 'max': 200,
        'group': '渲染平滑', 'label': '慢速距离阈值',
        'description': '低于该距离时使用最低 EMA alpha。',
        'editable': True,
    },
    'RENDER_EMA_FAST_DIST': {
        'type': 'int', 'min': 1, 'max': 500,
        'group': '渲染平滑', 'label': '快速距离阈值',
        'description': '高于该距离时使用最高 EMA alpha。',
        'editable': True,
    },
    'RENDER_OFFSET_X': {
        'type': 'int', 'min': -200, 'max': 200,
        'group': '渲染平滑', 'label': '渲染偏移 X',
        'description': '用于显示侧的微调偏移。',
        'editable': True,
    },
    'RENDER_OFFSET_Y': {
        'type': 'int', 'min': -200, 'max': 200,
        'group': '渲染平滑', 'label': '渲染偏移 Y',
        'description': '用于显示侧的微调偏移。',
        'editable': True,
    },
    'TP_JUMP_THRESHOLD': {
        'type': 'int', 'min': 50, 'max': 2000,
        'group': '渲染平滑', 'label': '传送跳变阈值',
        'description': '超过该阈值时会进入传送候选确认。',
        'editable': True,
    },
    'TP_CONFIRM_FRAMES': {
        'type': 'int', 'min': 1, 'max': 10,
        'group': '渲染平滑', 'label': '传送确认帧数',
        'description': '连续多少帧聚类命中后确认传送。',
        'editable': True,
    },
    'TP_CLUSTER_RADIUS': {
        'type': 'int', 'min': 20, 'max': 500,
        'group': '渲染平滑', 'label': '传送聚类半径',
        'description': '传送候选聚类时允许的最大散布半径。',
        'editable': True,
    },
}

CONFIG_RESTART_REQUIRED = {
    'PORT', 'WINDOW_GEOMETRY', 'VIEW_SIZE', 'LOGIC_MAP_PATH', 'DISPLAY_MAP_PATH',
    'SIFT_CONTRAST_THRESHOLD', 'MINIMAP', 'MINIMAP_CAPTURE_MARGIN',
    'MINIMAP_CIRCLE_CALIBRATION_FRAMES', 'MINIMAP_CIRCLE_R_TOLERANCE',
    'MINIMAP_CIRCLE_CENTER_TOLERANCE', 'MINIMAP_CIRCLE_RECALIBRATE_MISS',
}


def iter_config_keys() -> list[str]:
    return sorted(
        key for key, value in vars(config).items()
        if key.isupper() and not key.startswith('_') and not callable(value)
    )


def infer_config_group(key: str) -> str:
    if key in ('PORT', 'MINIMAP', 'WINDOW_GEOMETRY', 'VIEW_SIZE', 'JPEG_PAD'):
        return '基础服务'
    if key.endswith('_PATH'):
        return '地图文件'
    if key.startswith('AUTO_DETECT') or key.startswith('MINIMAP_'):
        return '自动圆检测'
    if key.startswith('ARROW_'):
        return '箭头'
    if key.startswith('RENDER_') or key.startswith('LINEAR_FILTER_') or key.startswith('TP_'):
        return '渲染平滑'
    return '识别引擎'


def infer_config_type(value) -> str:
    if isinstance(value, bool):
        return 'bool'
    if isinstance(value, int) and not isinstance(value, bool):
        return 'int'
    if isinstance(value, float):
        return 'float'
    if isinstance(value, dict):
        return 'json'
    return 'string'


def serialize_config_value(value):
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    if isinstance(value, dict):
        return value
    if isinstance(value, (list, tuple)):
        return list(value)
    return str(value)


def build_config_meta(key: str, value):
    rule = CONFIG_RUNTIME_RULES.get(key, {})
    editable = bool(rule.get('editable', False))
    restart_required = key in CONFIG_RESTART_REQUIRED or (not editable and key not in CONFIG_RUNTIME_RULES)
    reason = rule.get('reason') or (
        '该配置需要重启相关引擎或服务后才能安全生效。' if restart_required else '当前版本仅开放只读查看。'
    )
    return {
        'key': key,
        'label': rule.get('label', key.replace('_', ' ')),
        'group': rule.get('group', infer_config_group(key)),
        'type': rule.get('type', infer_config_type(value)),
        'editable': editable,
        'restartRequired': restart_required,
        'min': rule.get('min'),
        'max': rule.get('max'),
        'step': rule.get('step'),
        'description': rule.get('description', ''),
        'reason': reason,
    }


def build_config_payload():
    values = {}
    meta = {}
    groups = set()
    for key in iter_config_keys():
        value = getattr(config, key)
        values[key] = serialize_config_value(value)
        meta[key] = build_config_meta(key, value)
        groups.add(meta[key]['group'])
    return {
        'success': True,
        'values': values,
        'meta': meta,
        'groups': sorted(groups),
        'editableKeys': sorted(key for key, item in meta.items() if item['editable']),
        'readonlyKeys': sorted(key for key, item in meta.items() if not item['editable']),
    }


def coerce_runtime_value(meta: dict, raw_value):
    value_type = meta['type']
    if value_type == 'bool':
        if isinstance(raw_value, bool):
            value = raw_value
        elif isinstance(raw_value, str):
            lowered = raw_value.strip().lower()
            if lowered in ('1', 'true', 'yes', 'on'):
                value = True
            elif lowered in ('0', 'false', 'no', 'off'):
                value = False
            else:
                raise ValueError('必须为 true / false')
        else:
            raise ValueError('必须为布尔值')
        return value

    if value_type == 'int':
        if isinstance(raw_value, bool):
            raise ValueError('不能为布尔值')
        if isinstance(raw_value, str):
            raw_value = raw_value.strip()
            if raw_value == '':
                raise ValueError('不能为空')
        try:
            value = int(float(raw_value))
        except (TypeError, ValueError):
            raise ValueError('必须为整数')
        if float(value) != float(raw_value):
            raise ValueError('必须为整数')
        return value

    if value_type == 'float':
        if isinstance(raw_value, bool):
            raise ValueError('不能为布尔值')
        try:
            return float(raw_value)
        except (TypeError, ValueError):
            raise ValueError('必须为数字')

    if value_type == 'string':
        if raw_value is None:
            raise ValueError('不能为空')
        return str(raw_value)

    raise ValueError('当前类型暂不支持在线修改')


def validate_runtime_config_updates(updates: dict):
    approved = {}
    rejected = {}

    for key, raw_value in updates.items():
        current_value = getattr(config, key, None)
        meta = build_config_meta(key, current_value)
        if current_value is None or key not in iter_config_keys():
            rejected[key] = '配置项不存在'
            continue
        if not meta['editable']:
            rejected[key] = meta['reason']
            continue
        try:
            value = coerce_runtime_value(meta, raw_value)
        except ValueError as exc:
            rejected[key] = str(exc)
            continue

        min_value = meta.get('min')
        max_value = meta.get('max')
        if min_value is not None and value < min_value:
            rejected[key] = f'不能小于 {min_value}'
            continue
        if max_value is not None and value > max_value:
            rejected[key] = f'不能大于 {max_value}'
            continue
        approved[key] = value

    return approved, rejected


_ENGINE_ATTR_UPDATERS: dict[str, tuple[str, str]] = {
    'SEARCH_RADIUS': ('SEARCH_RADIUS', 'SEARCH_RADIUS'),
    'NEARBY_SEARCH_RADIUS': ('NEARBY_SEARCH_RADIUS', 'NEARBY_SEARCH_RADIUS'),
    'LOCAL_FAIL_LIMIT': ('LOCAL_FAIL_LIMIT', 'LOCAL_FAIL_LIMIT'),
    'SIFT_JUMP_THRESHOLD': ('JUMP_THRESHOLD', 'SIFT_JUMP_THRESHOLD'),
    'LOCAL_REVALIDATE_INTERVAL': ('_local_revalidate_interval', 'LOCAL_REVALIDATE_INTERVAL'),
    'LOCAL_REVALIDATE_MIN_QUALITY': ('_local_revalidate_min_quality', 'LOCAL_REVALIDATE_MIN_QUALITY'),
    'LOCAL_REVALIDATE_MARGIN': ('_local_revalidate_margin', 'LOCAL_REVALIDATE_MARGIN'),
    'LOCAL_REVALIDATE_DIFF': ('_local_revalidate_diff', 'LOCAL_REVALIDATE_DIFF'),
}

_LK_ATTR_UPDATERS: dict[str, tuple[str, str]] = {
    'LK_ENABLED': ('enabled', 'LK_ENABLED'),
    'LK_SIFT_INTERVAL': ('sift_every', 'LK_SIFT_INTERVAL'),
    'LK_MIN_CONFIDENCE': ('min_conf', 'LK_MIN_CONFIDENCE'),
}

_ENGINE_ECC_ATTR_UPDATERS: dict[str, tuple[str, str]] = {
    'ECC_ENABLED': ('_ecc_enabled', 'ECC_ENABLED'),
    'ECC_MIN_CORRELATION': ('_ecc_min_cc', 'ECC_MIN_CORRELATION'),
}

_ARROW_ATTR_UPDATERS: dict[str, tuple[str, str]] = {
    'ARROW_ANGLE_SMOOTH_ALPHA': ('_ema_alpha', 'ARROW_ANGLE_SMOOTH_ALPHA'),
    'ARROW_MOVE_MIN_DISPLACEMENT': ('_stop_speed_px', 'ARROW_MOVE_MIN_DISPLACEMENT'),
    'ARROW_STOPPED_DEBOUNCE': ('_stop_debounce', 'ARROW_STOPPED_DEBOUNCE'),
    'ARROW_BIG_CHANGE_THRESHOLD': ('_snap_threshold', 'ARROW_BIG_CHANGE_THRESHOLD'),
}


def _apply_attr_updaters(target, updates: dict, updater_map: dict[str, tuple[str, str]]) -> None:
    for key, (target_attr, config_attr) in updater_map.items():
        if key in updates:
            setattr(target, target_attr, getattr(config, config_attr))


def _apply_engine_updates(engine, updates: dict) -> None:
    _apply_attr_updaters(engine, updates, _ENGINE_ATTR_UPDATERS)
    _apply_attr_updaters(engine, updates, _ENGINE_ECC_ATTR_UPDATERS)

    lk = getattr(engine, '_lk', None)
    if lk is not None:
        _apply_attr_updaters(lk, updates, _LK_ATTR_UPDATERS)

    arrow_dir = getattr(engine, '_arrow_dir', None)
    if arrow_dir is not None:
        _apply_attr_updaters(arrow_dir, updates, _ARROW_ATTR_UPDATERS)
        if 'ARROW_POS_HISTORY_LEN' in updates:
            history = list(arrow_dir._history)
            arrow_dir._history = deque(history[-config.ARROW_POS_HISTORY_LEN:], maxlen=config.ARROW_POS_HISTORY_LEN)


def _apply_updates_to_tracker(tracker_obj, updates: dict):
    engine = tracker_obj.sift_engine
    process_lock = getattr(tracker_obj, '_process_lock', None)
    lock_acquired = False
    if process_lock is not None:
        lock_acquired = process_lock.acquire(timeout=2.0)
        if not lock_acquired:
            print('[config] 无法获得后台处理锁（超时），跳过 tracker 配置更新')
            return

    try:
        with engine._lock:
            _apply_engine_updates(engine, updates)
    finally:
        if lock_acquired and process_lock is not None:
            process_lock.release()


def apply_runtime_config_updates(updates: dict, session_registry):
    for key, value in updates.items():
        setattr(config, key, value)

    for ctx in session_registry.snapshot_contexts():
        _apply_updates_to_tracker(ctx.tracker, updates)


def apply_runtime_config_command(command_line: str, session_registry) -> dict:
    """服务端后台配置指令入口。

        支持命令：
            - set KEY=VALUE [KEY2=VALUE2 ...]
            - show
            - help
    """
    raw = str(command_line or '').strip()
    if not raw:
        return {'success': False, 'error': '空命令。输入 help 查看示例。'}

    try:
        tokens = shlex.split(raw)
    except ValueError as exc:
        return {'success': False, 'error': f'命令解析失败: {exc}'}

    if not tokens:
        return {'success': False, 'error': '空命令。输入 help 查看示例。'}

    action = tokens[0].strip().lower()
    if action == 'help':
        return {
            'success': True,
            'message': '用法: set KEY=VALUE [KEY2=VALUE2 ...]；示例: set SEARCH_RADIUS=500 LK_ENABLED=false',
        }

    if action == 'show':
        payload = build_config_payload()
        payload['message'] = '当前后端配置快照（服务端指令模式）'
        return payload

    if action != 'set':
        return {'success': False, 'error': f'不支持的命令: {tokens[0]}（输入 help 查看用法）'}

    updates: dict[str, str] = {}
    malformed: list[str] = []
    for item in tokens[1:]:
        part = str(item).strip()
        if not part:
            continue
        if '=' not in part:
            malformed.append(part)
            continue
        key, value = part.split('=', 1)
        key = key.strip()
        value = value.strip()
        if not key:
            malformed.append(part)
            continue
        updates[key] = value

    if malformed:
        return {'success': False, 'error': f'参数格式错误（需 KEY=VALUE）: {", ".join(malformed)}'}
    if not updates:
        return {'success': False, 'error': '未提供可更新项。示例: set SEARCH_RADIUS=500'}

    approved, rejected = validate_runtime_config_updates(updates)
    if approved:
        apply_runtime_config_updates(approved, session_registry)

    payload = {
        'success': not rejected,
        'partial': bool(approved and rejected),
        'applied': {key: serialize_config_value(value) for key, value in approved.items()},
        'rejected': rejected,
    }
    if not approved and rejected:
        payload['error'] = '配置未生效，请检查 rejected 字段。'
    return payload
