#!/usr/bin/env python3
"""从 OpenRA MiniYAML 武器/单位定义生成 damage_matrix.py。

读取 OpenRA 子模块中的真实 YAML 文件并产出 Python 数据模块，
确保伤害矩阵始终与游戏内的真实数值保持一致。

用法:
    python scripts/generate_damage_matrix.py [--openra-path PATH]

默认 OpenRA 路径: ../OpenRA-RL/OpenRA
"""

import argparse
import re
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

# 需要跟踪的护甲类型（对应 RA 中的 5 类护甲）
ARMOR_TYPES = ["none", "light", "heavy", "wood", "concrete"]

# ── MiniYAML 解析器 ──────────────────────────────────────────────────────────


def parse_miniyaml(text: str) -> dict[str, Any]:
    """解析 OpenRA MiniYAML 为嵌套 dict。

    MiniYAML 使用制表符（tab）缩进表示层级，每个节点为 "key: value"，
    子节点比父节点多一层 tab 缩进。

    步骤:
      1. 逐行读取，跳过空行与 # 注释行；
      2. 用缩进深度维护栈结构，弹出深度 >= 当前行的节点作为父节点；
      3. 解析 "key: value"，创建携带 __value__ 的节点 dict 挂到父节点；
      4. 以 - 前缀标记的删除节点与普通节点同样挂载，由继承解析阶段处理。

    参数:
        text: MiniYAML 文本内容。

    返回:
        以 OrderedDict 表示的嵌套树，每个叶子节点含 __value__ 键。
    """
    root: dict[str, Any] = OrderedDict()
    stack: list[tuple[int, dict]] = [(-1, root)]

    for line in text.splitlines():
        stripped = line.lstrip("\t")
        if not stripped or stripped.startswith("#"):
            continue

        depth = len(line) - len(stripped)

        # 弹出深度 >= 当前行的节点，得到当前行的父节点
        while stack and stack[-1][0] >= depth:
            stack.pop()

        parent = stack[-1][1]

        # 解析 "key: value"，无冒号时整行作为 key、值为空
        if ":" in stripped:
            key, _, value = stripped.partition(":")
            key = key.strip()
            value = value.strip()
        else:
            key = stripped.strip()
            value = ""

        # 创建子节点 dict，__value__ 保存原始值
        child: dict[str, Any] = OrderedDict()
        child["__value__"] = value

        # 删除节点（键以 - 前缀标记），与普通节点同样挂载，继承解析时处理
        if key.startswith("-"):
            parent[key] = child
        else:
            parent[key] = child

        stack.append((depth, child))

    return root


def resolve_inherits(definitions: dict[str, dict], name: str,
                     resolved_cache: dict[str, dict] | None = None,
                     resolving: set | None = None) -> dict:
    """将定义与全部父类（Inherits）深度合并，返回继承链完全展开的结果。

    步骤:
      1. 命中缓存直接返回；定义缺失返回空 dict；正在解析（循环继承）时
         返回原始定义防止死循环；
      2. 收集该定义的全部 Inherits / Inherits@X 指令（按出现顺序作为父类列表）；
      3. 先按顺序深合并各父类的解析结果，再覆盖合并自身字段；
      4. 处理删除节点：-key 表示移除父类继承来的 key，随后删除该标记键自身；
      5. 写入缓存并返回。
    """
    if resolved_cache is None:
        resolved_cache = {}
    if resolving is None:
        resolving = set()

    if name in resolved_cache:
        return resolved_cache[name]

    if name not in definitions:
        return {}

    if name in resolving:
        return definitions.get(name, {})  # 打破循环继承

    resolving.add(name)
    node = definitions[name]
    result = OrderedDict()

    # 收集全部 Inherits 指令（含 Inherits@X 变体，按出现顺序作为父类列表）
    parents = []
    for key, val in node.items():
        if key.startswith("Inherits"):
            parent_name = val.get("__value__", "")
            if parent_name:
                parents.append(parent_name)

    # 先按顺序合并父类（父类字段作为地基）
    for parent_name in parents:
        parent_resolved = resolve_inherits(definitions, parent_name,
                                           resolved_cache, resolving)
        deep_merge(result, parent_resolved)

    # 再覆盖合并自身字段（自身优先级更高）
    deep_merge(result, node)

    # 处理删除节点：-key 表示从继承结果中移除 key
    removals = [k for k in result if k.startswith("-")]
    for removal_key in removals:
        target = removal_key[1:]  # 去掉 - 前缀，得到被删除的字段名
        result.pop(target, None)
        del result[removal_key]

    resolving.discard(name)
    resolved_cache[name] = result
    return result


def deep_copy_dict(d: dict) -> dict:
    """深拷贝嵌套 dict（保持 OrderedDict 类型）。

    步骤: 逐键递归，值为 dict 时继续递归拷贝，其余直接引用。

    参数:
        d: 待拷贝的嵌套 dict。

    返回:
        全新的一份嵌套拷贝。
    """
    result = OrderedDict()
    for k, v in d.items():
        if isinstance(v, dict):
            result[k] = deep_copy_dict(v)
        else:
            result[k] = v
    return result


def deep_merge(target: dict, source: dict) -> None:
    """将 source 深度合并进 target（原地修改）。

    注意: 源中 dict 值一律深拷贝，避免多个解析结果共享同一缓存对象。

    步骤:
      1. 跳过 Inherits 指令（继承已在 resolve_inherits 中处理）；
      2. 双方同名键均为 dict 时递归深合并；
      3. 否则直接赋值，dict 值先深拷贝再放入。
    """
    for key, val in source.items():
        if key.startswith("Inherits"):
            continue  # 跳过 Inherits 指令
        if key in target and isinstance(target[key], dict) and isinstance(val, dict):
            deep_merge(target[key], val)
        else:
            # 深拷贝 dict 值，防止后续修改污染缓存中的父条目
            target[key] = deep_copy_dict(val) if isinstance(val, dict) else val


def get_value(node: dict, *keys: str) -> str:
    """按键路径逐层下钻嵌套 dict，返回末节点的 __value__ 值。

    步骤:
      1. 依次按每个键查找：优先精确匹配，其次大小写不敏感匹配；
      2. 任一层找不到即返回空串 ""；
      3. 末节点为 dict 时返回其 __value__，否则直接转字符串返回。

    参数:
        node: 根节点 dict。
        *keys: 键路径（如 "Valued", "Cost"）。

    返回:
        末节点的值字符串；找不到时返回 ""。
    """
    current = node
    for key in keys:
        if not isinstance(current, dict):
            return ""
        # 优先精确匹配
        if key in current:
            current = current[key]
        else:
            # 兜底：大小写不敏感匹配
            found = False
            for k in current:
                if k.lower() == key.lower():
                    current = current[k]
                    found = True
                    break
            if not found:
                return ""
    if isinstance(current, dict):
        return current.get("__value__", "")
    return str(current)


def get_child(node: dict, *keys: str) -> dict:
    """按键路径逐层下钻嵌套 dict，返回末节点所在的子 dict。

    与 get_value 的区别：返回节点本身（dict），而非其 __value__。
    步骤: 逐键查找，优先精确匹配，其次大小写不敏感匹配；
    任一层找不到或命中非 dict 节点即返回空 dict。

    参数:
        node: 根节点 dict。
        *keys: 键路径。

    返回:
        末节点 dict；找不到时返回 {}。
    """
    current = node
    for key in keys:
        if not isinstance(current, dict):
            return {}
        if key in current and isinstance(current[key], dict):
            current = current[key]
        else:
            # 兜底：大小写不敏感匹配
            found = False
            for k in current:
                if k.lower() == key.lower() and isinstance(current[k], dict):
                    current = current[k]
                    found = True
                    break
            if not found:
                return {}
    return current


# ── 数据提取函数 ─────────────────────────────────────────────────────────────


def extract_versus(weapon_def: dict) -> dict[str, float]:
    """从解析完成的武器定义中提取 Versus 伤害百分比。

    步骤:
      1. 遍历全部 Warhead 节点，跳过非 dict 节点；
      2. 仅处理含 SpreadDamage 或 TargetDamage 的伤害弹头；
      3. 取该弹头的 Versus 段，逐个提取 5 类护甲的百分比并转为倍率（/100）；
      4. 一旦拿到首个（主）伤害弹头的数据即提前结束。

    参数:
        weapon_def: 解析并合并继承后的武器定义 dict。

    返回:
        {护甲类型: 倍率} 的 dict；无伤害弹头或无 Versus 数据时为空 dict。
    """
    versus = {}

    # 寻找主伤害弹头（通常是 @1Dam 或第一个 SpreadDamage）
    for key, val in weapon_def.items():
        if not isinstance(val, dict):
            continue
        if not key.lower().startswith("warhead"):
            continue
        warhead_type = val.get("__value__", "")
        if "SpreadDamage" not in warhead_type and "TargetDamage" not in warhead_type:
            continue

        vs_node = get_child(val, "Versus")
        if not vs_node:
            continue

        # 逐个提取 5 类护甲的百分比并转为倍率
        for armor in ARMOR_TYPES:
            for k, v in vs_node.items():
                if k == "__value__":
                    continue
                if k.lower() == armor:
                    pct = v.get("__value__", "") if isinstance(v, dict) else str(v)
                    try:
                        versus[armor] = int(pct) / 100.0
                    except (ValueError, TypeError):
                        pass

        # 只采用第一个（主）伤害弹头的数据
        if versus:
            break

    return versus


def extract_weapons(openra_path: Path) -> dict[str, dict]:
    """解析 mods/ra/weapons 下全部武器 YAML 并展开继承。

    步骤:
      1. 遍历武器目录下所有 .yaml，用 parse_miniyaml 解析并合入总表；
      2. 对全部定义调用 resolve_inherits 展开继承链。

    参数:
        openra_path: OpenRA 仓库根目录路径。

    返回:
        {武器名: 继承展开后的定义} 的 dict。
    """
    weapon_dir = openra_path / "mods" / "ra" / "weapons"
    all_defs: dict[str, dict] = OrderedDict()

    for yaml_file in sorted(weapon_dir.glob("*.yaml")):
        text = yaml_file.read_text(encoding="utf-8")
        parsed = parse_miniyaml(text)
        for name, node in parsed.items():
            if isinstance(node, dict):
                all_defs[name] = node

    # 展开全部武器的继承链
    resolved_cache: dict[str, dict] = {}
    resolved: dict[str, dict] = OrderedDict()
    for name in all_defs:
        resolved[name] = resolve_inherits(all_defs, name, resolved_cache)

    return resolved


def extract_units(openra_path: Path) -> dict[str, dict]:
    """解析 mods/ra/rules 下单位/建筑 YAML 并展开继承。

    步骤:
      1. 依次读取 defaults / infantry / vehicles / aircraft / ships /
         structures 六个 rules 文件（defaults.yaml 提供基础类型默认字段）；
      2. 解析并合入总表；
      3. 对全部定义展开继承链。

    参数:
        openra_path: OpenRA 仓库根目录路径。

    返回:
        {单位/建筑名: 继承展开后的定义} 的 dict。
    """
    rules_dir = openra_path / "mods" / "ra" / "rules"
    all_defs: dict[str, dict] = OrderedDict()

    # 额外解析 defaults.yaml，为基础类型提供默认字段
    for yaml_file in ["defaults.yaml", "infantry.yaml", "vehicles.yaml",
                       "aircraft.yaml", "ships.yaml", "structures.yaml"]:
        fpath = rules_dir / yaml_file
        if fpath.exists():
            text = fpath.read_text(encoding="utf-8")
            parsed = parse_miniyaml(text)
            for name, node in parsed.items():
                if isinstance(node, dict):
                    all_defs[name] = node

    # 展开全部单位的继承链
    resolved_cache: dict[str, dict] = {}
    resolved: dict[str, dict] = OrderedDict()
    for name in all_defs:
        resolved[name] = resolve_inherits(all_defs, name, resolved_cache)

    return resolved


def get_primary_weapon(unit_def: dict) -> str:
    """获取单位定义中的主武器名。

    步骤（按优先级依次尝试）:
      1. Armament@PRIMARY（显式主武器槽）；
      2. 普通 Armament；
      3. Armament@AG（对地武器槽，如高炮卡车 ftrk）；
      4. Armament@SECONDARY（作为最后兜底）。

    参数:
        unit_def: 解析后的单位定义 dict。

    返回:
        武器名；找不到时返回 ""。
    """
    # 优先检查 Armament@PRIMARY（显式主武器槽）
    arm_primary = get_child(unit_def, "Armament@PRIMARY")
    if arm_primary:
        return get_value(arm_primary, "Weapon")

    # 其次检查普通 Armament
    arm = get_child(unit_def, "Armament")
    if arm:
        return get_value(arm, "Weapon")

    # 再检查 Armament@AG（对地武器槽，如高炮卡车 ftrk）
    arm_ag = get_child(unit_def, "Armament@AG")
    if arm_ag:
        return get_value(arm_ag, "Weapon")

    # 最后检查 Armament@SECONDARY（作为地面战兜底）
    arm_sec = get_child(unit_def, "Armament@SECONDARY")
    if arm_sec:
        weapon = get_value(arm_sec, "Weapon")
        if weapon:
            return weapon

    return ""


def get_secondary_weapon(unit_def: dict) -> str:
    """获取单位副武器名（用于双武器单位，如 4TNK 的 MammothTusk）。

    参数:
        unit_def: 解析后的单位定义 dict。

    返回:
        副武器名；未配置 Armament@SECONDARY 时返回 ""。
    """
    arm = get_child(unit_def, "Armament@SECONDARY")
    if arm:
        return get_value(arm, "Weapon")
    return ""


def get_cost(unit_def: dict) -> int:
    """提取单位造价。

    步骤: 按 "Valued" → "Cost" 键路径取值并转 int；
    缺字段或非数字时返回 0。

    参数:
        unit_def: 解析后的单位定义 dict。

    返回:
        造价（金币）。
    """
    cost_str = get_value(unit_def, "Valued", "Cost")
    try:
        return int(cost_str)
    except (ValueError, TypeError):
        return 0


def get_armor_type(unit_def: dict) -> str:
    """提取单位护甲类型。

    步骤: 按 "Armor" → "Type" 键路径取值并小写化；
    缺失时默认返回 "none"（步兵护甲）。

    参数:
        unit_def: 解析后的单位定义 dict。

    返回:
        护甲类型字符串（"none" / "light" / "heavy" / "wood" / "concrete"）。
    """
    armor = get_value(unit_def, "Armor", "Type")
    return armor.lower() if armor else "none"


# ── 单位清单 ─────────────────────────────────────────────────────────────────

# 需要写入伤害矩阵的单位清单（小写 ID）
INFANTRY = ["e1", "e2", "e3", "e4", "e6", "e7", "medi", "mech", "spy", "thf", "shok", "dog"]
VEHICLES = ["1tnk", "2tnk", "3tnk", "4tnk", "v2rl", "jeep", "apc", "arty",
            "harv", "mcv", "ftrk", "mnly", "ttnk", "ctnk", "stnk", "qtnk",
            "dtrk", "mgg", "mrj", "truk"]
AIRCRAFT = ["heli", "hind", "mh60", "tran", "yak", "mig"]
SHIPS = ["ss", "dd", "ca", "pt", "lst", "msub"]

BUILDINGS = ["fact", "powr", "apwr", "barr", "tent", "proc", "weap", "dome",
             "fix", "atek", "stek", "hpad", "afld", "spen", "syrd", "silo",
             "kenn", "pbox", "hbox", "gun", "ftur", "tsla", "agun", "sam",
             "gap", "iron", "pdox", "mslo"]

DEFENSES = ["pbox", "hbox", "gun", "ftur", "tsla", "agun", "sam"]

# 需要特殊武器建模的单位（目标限制使原始 Versus 无法准确反映
# 这些单位对智能体决策的实际价值）
SPECIAL_MODELING = {
    # e7（谭雅）：Colt45 只打步兵 + C4 可摧毁建筑
    # 建模为：对步兵毁灭性、对建筑毁灭性、对装甲无效
    "e7": {"none": 10.0, "light": 0.1, "heavy": 0.1, "wood": 5.0, "concrete": 5.0},
    # spy（间谍）：SilencedPPK 只打步兵且伤害极低
    "spy": {"none": 0.1, "light": 0.01, "heavy": 0.01, "wood": 0.01, "concrete": 0.01},
    # dog（军犬）：DogJaw 秒杀步兵，无法攻击其他目标
    "dog": {"none": 5.0, "light": 0.0, "heavy": 0.0, "wood": 0.0, "concrete": 0.0},
    # ss（潜艇）：TorpTube 只能攻击水面/水下单位
    "ss": {"none": 0.0, "light": 0.75, "heavy": 1.0, "wood": 0.75, "concrete": 5.0},
}

# 双武器单位——两把武器对 5 类护甲的效能取平均
DUAL_WEAPON_UNITS = {"4tnk"}  # 120mm（打装甲）+ MammothTusk（打步兵/空中）

# 主武器为纯对空的单位——矩阵改用地面副武器，
# 因为地面战斗决定绝大多数克制关系
GROUND_WEAPON_OVERRIDE = {
    "e3": "Dragon",       # PRIMARY=RedEye（对空），地面武器为 Dragon
    "heli": "HellfireAG", # PRIMARY=HellfireAA（对空），地面武器为 HellfireAG
}

# 使用驻军武器（YAML 中无 Armament 字段）的防御建筑
DEFENSE_WEAPON_OVERRIDE = {
    "pbox": "M60mg",   # 盟军碉堡——驻军步兵使用 M60mg
    "hbox": "M60mg",   # 伪装碉堡——同为驻军武器
}


def fill_versus(vs: dict[str, float]) -> dict[str, float]:
    """补齐全部 5 类护甲，缺失的默认按 1.0（全额伤害）。

    参数:
        vs: 部分护甲类型的倍率 dict。

    返回:
        覆盖全部护甲类型的完整倍率 dict。
    """
    return {armor: vs.get(armor, 1.0) for armor in ARMOR_TYPES}


def build_versus_for_unit(unit_id: str, unit_def: dict,
                          weapons: dict[str, dict]) -> dict[str, float]:
    """构建单个单位的 Versus 效能 dict。

    步骤:
      1. 命中特殊建模表时直接返回人工修正的效能值；
      2. 否则取主武器（主武器纯对空时改用 GROUND_WEAPON_OVERRIDE 指定武器）；
      3. 无武器或无 Versus 数据视为非作战单位，返回空 dict；
      4. 双武器单位将主副武器逐护甲取平均；
      5. 补齐 5 类护甲后返回。

    参数:
        unit_id: 单位 ID（大小写均可）。
        unit_def: 解析后的单位定义 dict。
        weapons: 武器名 → 解析后武器定义的映射。

    返回:
        {护甲类型: 倍率} 的完整 dict；非作战单位返回 {}。
    """
    uid = unit_id.lower()

    # 特殊建模：直接采用人工修正的效能值
    if uid in SPECIAL_MODELING:
        return SPECIAL_MODELING[uid]

    # 主武器纯对空：改用地面武器
    if uid in GROUND_WEAPON_OVERRIDE:
        weapon_name = GROUND_WEAPON_OVERRIDE[uid]
    else:
        weapon_name = get_primary_weapon(unit_def)

    if not weapon_name:
        return {}  # 非作战单位：无武器

    weapon_def = weapons.get(weapon_name, {})
    primary_vs = extract_versus(weapon_def)

    if not primary_vs:
        return {}  # 非作战单位或武器无 Versus 数据

    # 双武器单位：主副武器逐护甲取平均
    if uid in DUAL_WEAPON_UNITS:
        sec_name = get_secondary_weapon(unit_def)
        if sec_name and sec_name in weapons:
            sec_vs = extract_versus(weapons[sec_name])
            if sec_vs:
                combined = {}
                all_armors = set(primary_vs.keys()) | set(sec_vs.keys())
                for armor in all_armors:
                    p = primary_vs.get(armor, 1.0)
                    s = sec_vs.get(armor, 1.0)
                    combined[armor] = round((p + s) / 2, 2)
                return fill_versus(combined)

    return fill_versus(primary_vs)


def build_versus_for_defense(building_id: str, unit_def: dict,
                              weapons: dict[str, dict]) -> dict[str, float]:
    """构建单个防御建筑的 Versus 效能 dict。

    步骤:
      1. 纯对空防御（agun/sam）建模为只对 light（空中）有效；
      2. 驻军建筑使用 DEFENSE_WEAPON_OVERRIDE 指定的武器；
      3. 否则取主武器，缺失时尝试 Armament@GARRISONED 驻军武器；
      4. 无武器返回空 dict，否则补齐 5 类护甲后返回。

    参数:
        building_id: 建筑 ID。
        unit_def: 解析后的建筑定义 dict。
        weapons: 武器名 → 解析后武器定义的映射。

    返回:
        {护甲类型: 倍率} 的完整 dict；无武器时返回 {}。
    """
    bid = building_id.lower()

    # 纯对空防御：对非空中目标建模为 0
    if bid in ("agun", "sam"):
        return {"none": 0.0, "light": 1.0, "heavy": 0.0, "wood": 0.0, "concrete": 0.0}

    # 驻军建筑：使用指定的驻军武器
    if bid in DEFENSE_WEAPON_OVERRIDE:
        weapon_name = DEFENSE_WEAPON_OVERRIDE[bid]
    else:
        weapon_name = get_primary_weapon(unit_def)
        if not weapon_name:
            garm = get_child(unit_def, "Armament@GARRISONED")
            if garm:
                weapon_name = get_value(garm, "Weapon")

    if not weapon_name:
        return {}

    weapon_def = weapons.get(weapon_name, {})
    vs = extract_versus(weapon_def)
    return fill_versus(vs) if vs else {}


# ── 代码生成 ─────────────────────────────────────────────────────────────────


def format_dict(d: dict[str, Any], indent: int = 1) -> str:
    """将 dict 格式化为对齐的 Python 源码字符串。

    步骤: 逐项生成 "key": value；float 与 int 直接输出，
    str 加双引号；空 dict 返回 {}。

    参数:
        d: 待格式化的 dict。
        indent: 缩进层级（当前未使用，保留为扩展用）。

    返回:
        形如 {"a": 1, "b": "x"} 的源码字符串。
    """
    if not d:
        return "{}"
    items = []
    for k, v in d.items():
        if isinstance(v, float):
            items.append(f'"{k}": {v}')
        elif isinstance(v, int):
            items.append(f'"{k}": {v}')
        elif isinstance(v, str):
            items.append(f'"{k}": "{v}"')
    return "{" + ", ".join(items) + "}"


def generate_module(unit_armor: dict, building_armor: dict,
                    unit_cost: dict, building_cost: dict,
                    unit_effectiveness: dict, defense_effectiveness: dict) -> str:
    """生成 damage_matrix.py 的完整源码文本。

    步骤:
      1. 拼接文件头 docstring（护甲类型说明）；
      2. 生成单位/建筑护甲、造价、武器效能表（含分组注释）；
      3. 生成防御建筑效能表与特殊单位角色集合；
      4. 追加查询函数源码（含中文 docstring，与旧版本保持一致）。

    参数:
        unit_armor / building_armor: 单位/建筑护甲表。
        unit_cost / building_cost: 单位/建筑造价表。
        unit_effectiveness / defense_effectiveness: 单位/防御建筑效能表。

    返回:
        可直接写入 damage_matrix.py 的完整源码字符串。
    """
    lines = []
    lines.append('"""源自 OpenRA Red Alert（红色警戒）武器定义的单位效能数据。')
    lines.append("")
    lines.append("由 scripts/generate_damage_matrix.py 根据 OpenRA 的 MiniYAML 武器与")
    lines.append("单位定义自动生成。请勿手工编辑——应重新运行生成脚本。")
    lines.append("")
    lines.append("RA（红色警戒）中的护甲类型:")
    lines.append("  - none:     步兵（无装甲）")
    lines.append("  - light:    轻型载具、直升机、飞机、潜艇")
    lines.append("  - heavy:    坦克、重型载具、驱逐舰、运输载具")
    lines.append("  - wood:     木质建筑（兵营、电厂等）")
    lines.append("  - concrete: 混凝土结构（围墙）")
    lines.append("")
    lines.append("Versus 百分比: 100 = 正常伤害, >100 = 加成, <100 = 削弱。")
    lines.append("数值以倍率表示（1.0 = 100%）。")
    lines.append('"""')
    lines.append("")
    lines.append("from typing import Optional")
    lines.append("")

    # UNIT_ARMOR（单位护甲表）
    lines.append("# " + "─" * 76)
    lines.append("# 各单位的护甲类型（取自 rules YAML 中的 Armor: Type:）")
    lines.append("")
    lines.append("UNIT_ARMOR: dict[str, str] = {")
    lines.append("    # 步兵（全部为 \"none\" 护甲）")
    inf_items = [(k, v) for k, v in unit_armor.items() if k in INFANTRY]
    lines.append("    " + ", ".join(f'"{k}": "{v}"' for k, v in inf_items) + ",")
    lines.append("    # 载具")
    veh_items = [(k, v) for k, v in unit_armor.items() if k in VEHICLES]
    for i in range(0, len(veh_items), 5):
        chunk = veh_items[i:i+5]
        lines.append("    " + ", ".join(f'"{k}": "{v}"' for k, v in chunk) + ",")
    lines.append("    # 飞机")
    air_items = [(k, v) for k, v in unit_armor.items() if k in AIRCRAFT]
    lines.append("    " + ", ".join(f'"{k}": "{v}"' for k, v in air_items) + ",")
    lines.append("    # 舰船")
    ship_items = [(k, v) for k, v in unit_armor.items() if k in SHIPS]
    lines.append("    " + ", ".join(f'"{k}": "{v}"' for k, v in ship_items) + ",")
    lines.append("}")
    lines.append("")

    # BUILDING_ARMOR（建筑护甲表）
    lines.append("# 建筑护甲")
    lines.append("BUILDING_ARMOR: dict[str, str] = {")
    regular = [(k, v) for k, v in building_armor.items() if k not in DEFENSES + ["gap", "iron", "pdox", "mslo"]]
    lines.append("    " + ", ".join(f'"{k}": "{v}"' for k, v in regular) + ",")
    lines.append("    # 防御设施")
    defense_items = [(k, v) for k, v in building_armor.items() if k in DEFENSES]
    lines.append("    " + ", ".join(f'"{k}": "{v}"' for k, v in defense_items) + ",")
    misc = [(k, v) for k, v in building_armor.items() if k in ["gap", "iron", "pdox", "mslo"]]
    if misc:
        lines.append("    # 其他")
        lines.append("    " + ", ".join(f'"{k}": "{v}"' for k, v in misc) + ",")
    lines.append("}")
    lines.append("")

    # UNIT_COST（单位造价表）
    lines.append("# " + "─" * 76)
    lines.append("# 单位造价（取自 rules YAML 中的 Valued: Cost:）")
    lines.append("")
    lines.append("UNIT_COST: dict[str, int] = {")
    lines.append("    # 步兵")
    lines.append("    " + ", ".join(f'"{k}": {unit_cost[k]}' for k in INFANTRY) + ",")
    lines.append("    # 载具")
    for i in range(0, len(VEHICLES), 5):
        chunk = VEHICLES[i:i+5]
        lines.append("    " + ", ".join(f'"{k}": {unit_cost[k]}' for k in chunk if k in unit_cost) + ",")
    lines.append("    # 飞机")
    lines.append("    " + ", ".join(f'"{k}": {unit_cost[k]}' for k in AIRCRAFT) + ",")
    lines.append("    # 舰船")
    lines.append("    " + ", ".join(f'"{k}": {unit_cost[k]}' for k in SHIPS) + ",")
    lines.append("}")
    lines.append("")

    # BUILDING_COST（建筑造价表）
    lines.append("# 建筑造价（取自 rules YAML 中的 Valued: Cost:）")
    lines.append("BUILDING_COST: dict[str, int] = {")
    bcost_items = list(building_cost.items())
    for i in range(0, len(bcost_items), 5):
        chunk = bcost_items[i:i+5]
        lines.append("    " + ", ".join(f'"{k}": {v}' for k, v in chunk) + ",")
    lines.append("}")
    lines.append("")

    # UNIT_EFFECTIVENESS（武器效能表）
    lines.append("# " + "─" * 76)
    lines.append("# 武器效能（Versus 百分比）")
    lines.append("#")
    lines.append("# 每个条目映射 护甲类型 → 倍率（1.0 = 100% 正常伤害）。")
    lines.append("# 来源于 OpenRA/mods/ra/weapons/*.yaml 的 Versus: 段。")
    lines.append("# 非作战单位为空的 dict（无法攻击）。")
    lines.append("")
    lines.append("UNIT_EFFECTIVENESS: dict[str, dict[str, float]] = {")

    def _format_vs(name: str, vs: dict, comment: str = "") -> str:
        cmt = f"  # {comment}" if comment else ""
        if not vs:
            return f'    "{name}": {{}},{cmt}'
        vs_str = ", ".join(f'"{k}": {v}' for k, v in vs.items())
        return f'    "{name}": {{{vs_str}}},{cmt}'

    for section_name, section_units in [("Infantry", INFANTRY), ("Vehicles", VEHICLES),
                                         ("Aircraft", AIRCRAFT), ("Ships", SHIPS)]:
        lines.append(f"    # ── {section_name} " + "─" * (60 - len(section_name)))
        for uid in section_units:
            vs = unit_effectiveness.get(uid, {})
            lines.append(_format_vs(uid, vs))
        lines.append("")

    lines.append("}")
    lines.append("")

    # DEFENSE_EFFECTIVENESS（防御建筑效能表）
    lines.append("# 防御建筑效能（用于奖励计算中的\"破坏\"维度）")
    lines.append("DEFENSE_EFFECTIVENESS: dict[str, dict[str, float]] = {")
    for did in DEFENSES:
        vs = defense_effectiveness.get(did, {})
        lines.append(_format_vs(did, vs))
    lines.append("}")
    lines.append("")

    # 特殊单位角色集合
    lines.append("# " + "─" * 76)
    lines.append("# 特殊单位角色（用于奖励计算中的目标分类）")
    lines.append("")
    lines.append('ECONOMIC_UNITS = {"harv", "truk"}')
    lines.append('ECONOMIC_BUILDINGS = {"proc", "silo"}')
    lines.append('PRODUCTION_BUILDINGS = {"barr", "tent", "weap", "hpad", "afld", "spen", "syrd", "kenn"}')
    lines.append('TECH_BUILDINGS = {"dome", "atek", "stek", "fix"}')
    lines.append('POWER_BUILDINGS = {"powr", "apwr"}')
    lines.append("")
    lines.append("NON_COMBAT_UNITS = {")
    lines.append("    utype for utype, vs in UNIT_EFFECTIVENESS.items() if not vs")
    lines.append("}")
    lines.append("")

    # 查询函数（与旧版本保持一致）
    lines.append("")
    lines.append("# " + "─" * 76)
    lines.append("# 查询函数")
    lines.append("")
    lines.append("")
    lines.append('def get_effectiveness(attacker_type: str, target_armor: str) -> float:')
    lines.append('    """查询攻击方对指定护甲类型目标的伤害倍率。')
    lines.append("")
    lines.append("    步骤:")
    lines.append("      1. 按攻击方类型（小写）查找 UNIT_EFFECTIVENESS 表；")
    lines.append("      2. 若查到的 dict 为空，说明该单位无法攻击，直接返回 0.0；")
    lines.append("      3. 否则返回目标护甲对应的倍率，未知护甲默认按 1.0（全额伤害）处理。")
    lines.append("")
    lines.append("    参数:")
    lines.append('        attacker_type: 单位类型（如 "e3"、"1tnk"）。')
    lines.append('        target_armor: 护甲类型（如 "none"、"heavy"）。')
    lines.append("")
    lines.append("    返回:")
    lines.append("        伤害倍率（1.0 = 正常, >1 = 有效克制, <1 = 被削弱）。")
    lines.append("        若单位无法攻击则返回 0.0。")
    lines.append('    """')
    lines.append('    vs = UNIT_EFFECTIVENESS.get(attacker_type.lower(), {})')
    lines.append("    if not vs:")
    lines.append("        return 0.0")
    lines.append('    return vs.get(target_armor.lower(), 1.0)')
    lines.append("")
    lines.append("")
    lines.append('def get_unit_vs_unit(attacker: str, target: str) -> float:')
    lines.append('    """查询攻击方对指定目标单位的伤害倍率（通过目标护甲间接求克制）。')
    lines.append("")
    lines.append("    步骤:")
    lines.append("      1. 查 UNIT_ARMOR 得到目标单位护甲（未知默认 \"none\"）；")
    lines.append("      2. 复用 get_effectiveness 按该护甲查询伤害倍率。")
    lines.append("")
    lines.append("    参数:")
    lines.append("        attacker: 攻击方单位类型。")
    lines.append("        target: 目标单位类型。")
    lines.append("")
    lines.append("    返回:")
    lines.append("        攻击方对目标单位的伤害倍率；无攻击能力时返回 0.0。")
    lines.append('    """')
    lines.append('    target_armor = UNIT_ARMOR.get(target.lower(), "none")')
    lines.append("    return get_effectiveness(attacker, target_armor)")
    lines.append("")
    lines.append("")
    lines.append('def get_unit_armor(unit_type: str) -> str:')
    lines.append('    """查询单位护甲类型，未知类型默认返回 "none"（步兵护甲）。')
    lines.append("")
    lines.append("    参数:")
    lines.append("        unit_type: 单位类型。")
    lines.append("")
    lines.append("    返回:")
    lines.append('        护甲类型字符串（"none" / "light" / "heavy" / "wood" / "concrete"）。')
    lines.append('    """')
    lines.append('    return UNIT_ARMOR.get(unit_type.lower(), "none")')
    lines.append("")
    lines.append("")
    lines.append('def get_building_armor(building_type: str) -> str:')
    lines.append('    """查询建筑护甲类型，未知类型默认返回 "wood"（木质建筑）。')
    lines.append("")
    lines.append("    参数:")
    lines.append("        building_type: 建筑类型。")
    lines.append("")
    lines.append("    返回:")
    lines.append('        护甲类型字符串（通常为 "wood" 或 "concrete"）。')
    lines.append('    """')
    lines.append('    return BUILDING_ARMOR.get(building_type.lower(), "wood")')
    lines.append("")
    lines.append("")
    lines.append('def get_unit_cost(unit_type: str) -> int:')
    lines.append('    """查询单位造价，未知类型返回 0。')
    lines.append("")
    lines.append("    参数:")
    lines.append("        unit_type: 单位类型。")
    lines.append("")
    lines.append("    返回:")
    lines.append("        造价（金币），用于按成本加权战损等计算。")
    lines.append('    """')
    lines.append("    return UNIT_COST.get(unit_type.lower(), 0)")
    lines.append("")
    lines.append("")
    lines.append('def get_building_cost(building_type: str) -> int:')
    lines.append('    """查询建筑造价，未知类型返回 0。')
    lines.append("")
    lines.append("    参数:")
    lines.append("        building_type: 建筑类型。")
    lines.append("")
    lines.append("    返回:")
    lines.append("        造价（金币）。")
    lines.append('    """')
    lines.append("    return BUILDING_COST.get(building_type.lower(), 0)")
    lines.append("")
    lines.append("")
    lines.append('def can_attack(unit_type: str) -> bool:')
    lines.append('    """判断某单位类型是否具备攻击能力。')
    lines.append("")
    lines.append("    依据: UNIT_EFFECTIVENESS 中存在该单位且其效能 dict 非空。")
    lines.append("    效能为空 dict 的单位（如矿车、基地车、运输机）视为无法攻击。")
    lines.append("")
    lines.append("    参数:")
    lines.append("        unit_type: 单位类型。")
    lines.append("")
    lines.append("    返回:")
    lines.append("        True 表示可攻击，False 表示无攻击能力。")
    lines.append('    """')
    lines.append("    return bool(UNIT_EFFECTIVENESS.get(unit_type.lower(), {}))")
    lines.append("")
    lines.append("")
    lines.append('def is_economic_target(unit_or_building: str) -> bool:')
    lines.append('    """判断摧毁该目标是否会造成经济破坏。')
    lines.append("")
    lines.append("    目标类型: 经济单位（矿车 harv、运输车 truk）或经济建筑")
    lines.append("    （精炼厂 proc、储存井 silo）。")
    lines.append("")
    lines.append("    参数:")
    lines.append("        unit_or_building: 单位或建筑类型。")
    lines.append("")
    lines.append("    返回:")
    lines.append("        True 表示属于经济目标。")
    lines.append('    """')
    lines.append("    t = unit_or_building.lower()")
    lines.append("    return t in ECONOMIC_UNITS or t in ECONOMIC_BUILDINGS")
    lines.append("")
    lines.append("")
    lines.append('def is_production_target(building_type: str) -> bool:')
    lines.append('    """判断摧毁该建筑是否会扰乱敌方生产。')
    lines.append("")
    lines.append("    覆盖兵营（barr/tent）、坦克工厂（weap）、机场（hpad/afld）、")
    lines.append("    船坞（spen/syrd）、狗窝（kenn）等生产类建筑。")
    lines.append("")
    lines.append("    参数:")
    lines.append("        building_type: 建筑类型。")
    lines.append("")
    lines.append("    返回:")
    lines.append("        True 表示属于生产目标。")
    lines.append('    """')
    lines.append("    return building_type.lower() in PRODUCTION_BUILDINGS")
    lines.append("")
    lines.append("")
    lines.append('def is_tech_target(building_type: str) -> bool:')
    lines.append('    """判断摧毁该建筑是否会造成科技倒退。')
    lines.append("")
    lines.append("    覆盖雷达站（dome）、高级科技实验室（atek/stek）、维修厂（fix）等。")
    lines.append("")
    lines.append("    参数:")
    lines.append("        building_type: 建筑类型。")
    lines.append("")
    lines.append("    返回:")
    lines.append("        True 表示属于科技目标。")
    lines.append('    """')
    lines.append("    return building_type.lower() in TECH_BUILDINGS")
    lines.append("")
    lines.append("")
    lines.append('def is_power_target(building_type: str) -> bool:')
    lines.append('    """判断摧毁该建筑是否会切断敌方电力供应。')
    lines.append("")
    lines.append("    覆盖小型电厂（powr）与高级电厂（apwr）。")
    lines.append("")
    lines.append("    参数:")
    lines.append("        building_type: 建筑类型。")
    lines.append("")
    lines.append("    返回:")
    lines.append("        True 表示属于电力目标。")
    lines.append('    """')
    lines.append("    return building_type.lower() in POWER_BUILDINGS")
    lines.append("")
    lines.append("")
    lines.append("def compute_army_counter_score(")
    lines.append("    own_units: list[dict],")
    lines.append("    enemy_units: list[dict],")
    lines.append(") -> tuple[float, float]:")
    lines.append('    """计算己方兵力对敌方兵力的克制程度与脆弱程度。')
    lines.append("")
    lines.append("    步骤:")
    lines.append("      1. 双方任一为空，或敌方护甲统计为空时返回中性分 (0.5, 0.5)；")
    lines.append("      2. 统计敌方各护甲类型的单位数量分布；")
    lines.append("      3. 过滤出己方可攻击的作战单位（调用 can_attack）；")
    lines.append("      4. 对每个己方作战单位，按敌方护甲分布加权计算平均效能：")
    lines.append("         - 平均效能 >= 1.0 计为\"克制有效\"；")
    lines.append("         - 平均效能 < 0.5 计为\"脆弱易损\"；")
    lines.append("      5. 返回 (克制单位占比, 脆弱单位占比)，均为 [0, 1]。")
    lines.append("")
    lines.append("    参数:")
    lines.append("        own_units: 己方单位列表，元素为至少含 'type' 键的 dict。")
    lines.append("        enemy_units: 敌方单位列表，元素为至少含 'type' 键的 dict。")
    lines.append("")
    lines.append("    返回:")
    lines.append("        (counter_score, vulnerability_score)，取值范围 [0, 1]。")
    lines.append("        counter_score: 己方作战单位中对敌方有效的比例。")
    lines.append("        vulnerability_score: 己方作战单位中对敌方脆弱易损的比例。")
    lines.append('    """')
    lines.append("    if not own_units or not enemy_units:")
    lines.append("        return 0.5, 0.5")
    lines.append("")
    lines.append("    enemy_armors: dict[str, int] = {}")
    lines.append("    for u in enemy_units:")
    lines.append('        armor = get_unit_armor(u.get("type", ""))')
    lines.append("        enemy_armors[armor] = enemy_armors.get(armor, 0) + 1")
    lines.append("")
    lines.append("    total_enemy = sum(enemy_armors.values())")
    lines.append("    if total_enemy == 0:")
    lines.append("        return 0.5, 0.5")
    lines.append("")
    lines.append('    own_combat = [u for u in own_units if can_attack(u.get("type", ""))]')
    lines.append("    if not own_combat:")
    lines.append("        return 0.0, 1.0")
    lines.append("")
    lines.append("    effective_count = 0")
    lines.append("    vulnerable_count = 0")
    lines.append("")
    lines.append("    for u in own_combat:")
    lines.append('        utype = u.get("type", "")')
    lines.append("        avg_eff = 0.0")
    lines.append("        for armor, count in enemy_armors.items():")
    lines.append("            avg_eff += get_effectiveness(utype, armor) * (count / total_enemy)")
    lines.append("")
    lines.append("        if avg_eff >= 1.0:")
    lines.append("            effective_count += 1")
    lines.append("        elif avg_eff < 0.5:")
    lines.append("            vulnerable_count += 1")
    lines.append("")
    lines.append("    n = len(own_combat)")
    lines.append("    return effective_count / n, vulnerable_count / n")
    lines.append("")

    return "\n".join(lines)


# ── 主流程 ───────────────────────────────────────────────────────────────────


def main():
    """命令行入口。

    流程:
      1. 解析命令行参数（OpenRA 路径 / 输出路径 / 是否校验）；
      2. 校验 OpenRA 仓库存在，读取并解析武器、单位定义；
      3. 逐单位构建护甲、造价、效能数据表，打印调试信息；
      4. 调用 generate_module 生成源码；
      5. --verify 时比对现有文件（不一致则失败退出），否则写盘。
    """
    parser = argparse.ArgumentParser(description="从 OpenRA YAML 生成 damage_matrix.py")
    parser.add_argument("--openra-path", type=Path,
                        default=Path(__file__).parent.parent.parent / "OpenRA-RL" / "OpenRA",
                        help="OpenRA 仓库根目录路径")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).parent.parent / "openra_rl_util" / "damage_matrix.py",
                        help="输出文件路径")
    parser.add_argument("--verify", action="store_true",
                        help="校验模式：对比生成结果与现有文件，不写入")
    args = parser.parse_args()

    # 校验 OpenRA 仓库存在，缺失时报错退出
    if not (args.openra_path / "mods" / "ra" / "weapons").exists():
        print(f"错误：未找到 OpenRA 路径：{args.openra_path}")
        print("请使用 --openra-path 指定 OpenRA 仓库根目录")
        sys.exit(1)

    # 解析武器定义
    print(f"正在从 {args.openra_path / 'mods' / 'ra' / 'weapons'} 读取武器定义")
    weapons = extract_weapons(args.openra_path)
    print(f"  共找到 {len(weapons)} 个武器定义")

    # 解析单位/建筑定义
    print(f"正在从 {args.openra_path / 'mods' / 'ra' / 'rules'} 读取单位定义")
    units = extract_units(args.openra_path)
    print(f"  共找到 {len(units)} 个单位/建筑定义")

    # 构建数据表：单位护甲、造价、效能
    unit_armor: dict[str, str] = OrderedDict()
    unit_cost: dict[str, int] = OrderedDict()
    unit_effectiveness: dict[str, dict[str, float]] = OrderedDict()

    all_units = INFANTRY + VEHICLES + AIRCRAFT + SHIPS
    for uid in all_units:
        udef = units.get(uid.upper(), {})
        if not udef:
            print(f"  警告：YAML 中未找到单位 {uid.upper()}")
            continue
        unit_armor[uid] = get_armor_type(udef)
        unit_cost[uid] = get_cost(udef)
        unit_effectiveness[uid] = build_versus_for_unit(uid, udef, weapons)

        # 调试输出：打印每个单位的主武器/护甲/造价/效能
        weapon = GROUND_WEAPON_OVERRIDE.get(uid, get_primary_weapon(udef))
        vs = unit_effectiveness[uid]
        if vs:
            vs_str = " ".join(f"{k}:{v}" for k, v in vs.items())
            print(f"  {uid:6s} → {weapon:20s} 护甲={unit_armor[uid]:6s} 造价={unit_cost[uid]:5d}  {vs_str}")
        else:
            print(f"  {uid:6s} → {'(无攻击)':20s} 护甲={unit_armor[uid]:6s} 造价={unit_cost[uid]:5d}")

    building_armor: dict[str, str] = OrderedDict()
    building_cost: dict[str, int] = OrderedDict()
    defense_effectiveness: dict[str, dict[str, float]] = OrderedDict()

    for bid in BUILDINGS:
        bdef = units.get(bid.upper(), {})
        if not bdef:
            print(f"  警告：YAML 中未找到建筑 {bid.upper()}")
            continue
        building_armor[bid] = get_armor_type(bdef)
        building_cost[bid] = get_cost(bdef)

    for did in DEFENSES:
        ddef = units.get(did.upper(), {})
        if ddef:
            defense_effectiveness[did] = build_versus_for_defense(did, ddef, weapons)
            weapon = DEFENSE_WEAPON_OVERRIDE.get(did, get_primary_weapon(ddef))
            if not weapon:
                garm = get_child(ddef, "Armament@GARRISONED")
                weapon = get_value(garm, "Weapon") if garm else "(garrisoned)"
            vs = defense_effectiveness[did]
            vs_str = " ".join(f"{k}:{v}" for k, v in vs.items()) if vs else "(none)"
            print(f"  {did:6s} → {weapon:20s} 护甲={building_armor.get(did, '?'):6s} 造价={building_cost.get(did, 0):5d}  {vs_str}")

    # 生成输出源码并写入或校验
    source = generate_module(unit_armor, building_armor, unit_cost, building_cost,
                             unit_effectiveness, defense_effectiveness)

    if args.verify:
        # 校验模式：对比生成结果与现有文件，不一致则以失败码退出（供 CI 使用）
        existing = args.output.read_text() if args.output.exists() else ""
        if existing.strip() == source.strip():
            print("\n✅ 生成数据与现有 damage_matrix.py 一致")
        else:
            print("\n❌ 生成数据与现有 damage_matrix.py 不一致")
            sys.exit(1)
    else:
        # 正常模式：将生成结果写入目标文件
        args.output.write_text(source)
        print(f"\n已写入 {args.output}")


if __name__ == "__main__":
    main()
