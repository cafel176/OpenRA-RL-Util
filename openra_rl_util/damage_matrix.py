"""源自 OpenRA Red Alert（红色警戒）武器定义的单位效能数据。

由 scripts/generate_damage_matrix.py 根据 OpenRA 的 MiniYAML 武器与
单位定义自动生成。请勿手工编辑——应重新运行生成脚本。

RA（红色警戒）中的护甲类型:
  - none:     步兵（无装甲）
  - light:    轻型载具、直升机、飞机、潜艇
  - heavy:    坦克、重型载具、驱逐舰、运输载具
  - wood:     木质建筑（兵营、电厂等）
  - concrete: 混凝土结构（围墙）

Versus 百分比: 100 = 正常伤害, >100 = 加成, <100 = 削弱。
数值以倍率表示（1.0 = 100%）。
"""

from typing import Optional

# ────────────────────────────────────────────────────────────────────────────
# 各单位的护甲类型（取自 rules YAML 中的 Armor: Type:）

UNIT_ARMOR: dict[str, str] = {
    # 步兵（全部为 "none" 护甲）
    "e1": "none", "e2": "none", "e3": "none", "e4": "none", "e6": "none", "e7": "none", "medi": "none", "mech": "none", "spy": "none", "thf": "none", "shok": "none", "dog": "none",
    # 载具
    "1tnk": "heavy", "2tnk": "heavy", "3tnk": "heavy", "4tnk": "heavy", "v2rl": "light",
    "jeep": "light", "apc": "heavy", "arty": "light", "harv": "heavy", "mcv": "light",
    "ftrk": "light", "mnly": "heavy", "ttnk": "light", "ctnk": "light", "stnk": "light",
    "qtnk": "heavy", "dtrk": "light", "mgg": "heavy", "mrj": "heavy", "truk": "light",
    # 飞机
    "heli": "light", "hind": "light", "mh60": "light", "tran": "light", "yak": "light", "mig": "light",
    # 舰船
    "ss": "light", "dd": "heavy", "ca": "heavy", "pt": "heavy", "lst": "heavy", "msub": "light",
}

# 建筑护甲
BUILDING_ARMOR: dict[str, str] = {
    "fact": "wood", "powr": "wood", "apwr": "wood", "barr": "wood", "tent": "wood", "proc": "wood", "weap": "wood", "dome": "wood", "fix": "wood", "atek": "wood", "stek": "wood", "hpad": "wood", "afld": "wood", "spen": "wood", "syrd": "wood", "silo": "wood", "kenn": "wood",
    # 防御设施
    "pbox": "heavy", "hbox": "heavy", "gun": "heavy", "ftur": "heavy", "tsla": "heavy", "agun": "heavy", "sam": "heavy",
    # 其他
    "gap": "heavy", "iron": "wood", "pdox": "wood", "mslo": "wood",
}

# ────────────────────────────────────────────────────────────────────────────
# 单位造价（取自 rules YAML 中的 Valued: Cost:）

UNIT_COST: dict[str, int] = {
    # 步兵
    "e1": 100, "e2": 150, "e3": 300, "e4": 300, "e6": 400, "e7": 1800, "medi": 200, "mech": 500, "spy": 500, "thf": 500, "shok": 350, "dog": 200,
    # 载具
    "1tnk": 700, "2tnk": 850, "3tnk": 1150, "4tnk": 2000, "v2rl": 900,
    "jeep": 500, "apc": 850, "arty": 850, "harv": 1100, "mcv": 2000,
    "ftrk": 600, "mnly": 800, "ttnk": 1350, "ctnk": 1350, "stnk": 1000,
    "qtnk": 2000, "dtrk": 2500, "mgg": 1000, "mrj": 1000, "truk": 500,
    # 飞机
    "heli": 2000, "hind": 1500, "mh60": 1500, "tran": 900, "yak": 1350, "mig": 2000,
    # 舰船
    "ss": 950, "dd": 1000, "ca": 2400, "pt": 500, "lst": 500, "msub": 2000,
}

# 建筑造价（取自 rules YAML 中的 Valued: Cost:）
BUILDING_COST: dict[str, int] = {
    "fact": 2000, "powr": 300, "apwr": 500, "barr": 500, "tent": 500,
    "proc": 1400, "weap": 2000, "dome": 1500, "fix": 1200, "atek": 1500,
    "stek": 1500, "hpad": 500, "afld": 500, "spen": 800, "syrd": 1000,
    "silo": 150, "kenn": 200, "pbox": 600, "hbox": 750, "gun": 800,
    "ftur": 600, "tsla": 1200, "agun": 800, "sam": 700, "gap": 800,
    "iron": 2000, "pdox": 1500, "mslo": 2500,
}

# ────────────────────────────────────────────────────────────────────────────
# 武器效能（Versus 百分比）
#
# 每个条目映射 护甲类型 → 倍率（1.0 = 100% 正常伤害）。
# 来源于 OpenRA/mods/ra/weapons/*.yaml 的 Versus: 段。
# 非作战单位为空的 dict（无法攻击）。

UNIT_EFFECTIVENESS: dict[str, dict[str, float]] = {
    # ── Infantry ────────────────────────────────────────────────────
    "e1": {"none": 1.5, "light": 0.4, "heavy": 0.1, "wood": 0.3, "concrete": 0.1},
    "e2": {"none": 0.6, "light": 0.25, "heavy": 0.25, "wood": 1.0, "concrete": 1.0},
    "e3": {"none": 0.1, "light": 0.34, "heavy": 1.0, "wood": 0.74, "concrete": 0.5},
    "e4": {"none": 0.7, "light": 0.4, "heavy": 0.2, "wood": 0.8, "concrete": 0.1},
    "e6": {},
    "e7": {"none": 10.0, "light": 0.1, "heavy": 0.1, "wood": 5.0, "concrete": 5.0},
    "medi": {},
    "mech": {},
    "spy": {"none": 0.1, "light": 0.01, "heavy": 0.01, "wood": 0.01, "concrete": 0.01},
    "thf": {},
    "shok": {"none": 10.0, "light": 1.0, "heavy": 0.6, "wood": 0.73, "concrete": 1.0},
    "dog": {"none": 5.0, "light": 0.0, "heavy": 0.0, "wood": 0.0, "concrete": 0.0},

    # ── Vehicles ────────────────────────────────────────────────────
    "1tnk": {"none": 0.32, "light": 1.16, "heavy": 0.48, "wood": 0.52, "concrete": 0.32},
    "2tnk": {"none": 0.3, "light": 0.75, "heavy": 1.15, "wood": 0.75, "concrete": 0.5},
    "3tnk": {"none": 0.3, "light": 0.75, "heavy": 1.15, "wood": 0.75, "concrete": 0.5},
    "4tnk": {"none": 0.65, "light": 0.68, "heavy": 0.69, "wood": 0.74, "concrete": 0.5},
    "v2rl": {"none": 0.9, "light": 0.7, "heavy": 0.4, "wood": 0.75, "concrete": 1.0},
    "jeep": {"none": 1.5, "light": 0.3, "heavy": 0.1, "wood": 0.1, "concrete": 0.1},
    "apc": {"none": 1.5, "light": 0.3, "heavy": 0.1, "wood": 0.1, "concrete": 0.1},
    "arty": {"none": 0.6, "light": 0.6, "heavy": 0.25, "wood": 0.4, "concrete": 0.5},
    "harv": {},
    "mcv": {},
    "ftrk": {"none": 0.4, "light": 0.6, "heavy": 0.1, "wood": 0.1, "concrete": 0.2},
    "mnly": {},
    "ttnk": {"none": 10.0, "light": 1.0, "heavy": 1.0, "wood": 1.0, "concrete": 1.0},
    "ctnk": {"none": 0.1, "light": 0.34, "heavy": 1.0, "wood": 0.74, "concrete": 0.5},
    "stnk": {"none": 0.1, "light": 0.34, "heavy": 1.0, "wood": 0.74, "concrete": 0.5},
    "qtnk": {},
    "dtrk": {},
    "mgg": {},
    "mrj": {},
    "truk": {},

    # ── Aircraft ────────────────────────────────────────────────────
    "heli": {"none": 0.3, "light": 0.9, "heavy": 1.0, "wood": 0.9, "concrete": 1.0},
    "hind": {"none": 1.44, "light": 0.72, "heavy": 0.28, "wood": 0.6, "concrete": 0.28},
    "mh60": {"none": 1.44, "light": 0.72, "heavy": 0.28, "wood": 0.6, "concrete": 0.28},
    "tran": {},
    "yak": {"none": 1.0, "light": 0.6, "heavy": 0.25, "wood": 0.5, "concrete": 0.25},
    "mig": {"none": 0.3, "light": 0.9, "heavy": 1.15, "wood": 0.9, "concrete": 1.0},

    # ── Ships ───────────────────────────────────────────────────────
    "ss": {"none": 0.0, "light": 0.75, "heavy": 1.0, "wood": 0.75, "concrete": 5.0},
    "dd": {"none": 0.36, "light": 0.66, "heavy": 1.2, "wood": 0.88, "concrete": 0.6},
    "ca": {"none": 0.6, "light": 0.6, "heavy": 0.25, "wood": 0.35, "concrete": 1.0},
    "pt": {"none": 0.28, "light": 0.72, "heavy": 1.0, "wood": 0.72, "concrete": 0.48},
    "lst": {},
    "msub": {"none": 0.8, "light": 0.48, "heavy": 0.3, "wood": 0.5, "concrete": 1.0},

}

# 防御建筑效能（用于奖励计算中的"破坏"维度）
DEFENSE_EFFECTIVENESS: dict[str, dict[str, float]] = {
    "pbox": {"none": 1.5, "light": 0.3, "heavy": 0.1, "wood": 0.1, "concrete": 0.1},
    "hbox": {"none": 1.5, "light": 0.3, "heavy": 0.1, "wood": 0.1, "concrete": 0.1},
    "gun": {"none": 0.2, "light": 0.75, "heavy": 1.0, "wood": 0.5, "concrete": 0.5},
    "ftur": {"none": 0.9, "light": 0.5, "heavy": 0.25, "wood": 0.5, "concrete": 0.2},
    "tsla": {"none": 10.0, "light": 1.0, "heavy": 1.0, "wood": 0.6, "concrete": 1.0},
    "agun": {"none": 0.0, "light": 1.0, "heavy": 0.0, "wood": 0.0, "concrete": 0.0},
    "sam": {"none": 0.0, "light": 1.0, "heavy": 0.0, "wood": 0.0, "concrete": 0.0},
}

# ────────────────────────────────────────────────────────────────────────────
# 特殊单位角色（用于奖励计算中的目标分类）

ECONOMIC_UNITS = {"harv", "truk"}
ECONOMIC_BUILDINGS = {"proc", "silo"}
PRODUCTION_BUILDINGS = {"barr", "tent", "weap", "hpad", "afld", "spen", "syrd", "kenn"}
TECH_BUILDINGS = {"dome", "atek", "stek", "fix"}
POWER_BUILDINGS = {"powr", "apwr"}

NON_COMBAT_UNITS = {
    utype for utype, vs in UNIT_EFFECTIVENESS.items() if not vs
}


# ────────────────────────────────────────────────────────────────────────────
# 查询函数


def get_effectiveness(attacker_type: str, target_armor: str) -> float:
    """查询攻击方对指定护甲类型目标的伤害倍率。

    步骤:
      1. 按攻击方类型（小写）查找 UNIT_EFFECTIVENESS 表；
      2. 若查到的 dict 为空，说明该单位无法攻击，直接返回 0.0；
      3. 否则返回目标护甲对应的倍率，未知护甲默认按 1.0（全额伤害）处理。

    参数:
        attacker_type: 单位类型（如 "e3"、"1tnk"）。
        target_armor: 护甲类型（如 "none"、"heavy"）。

    返回:
        伤害倍率（1.0 = 正常, >1 = 有效克制, <1 = 被削弱）。
        若单位无法攻击则返回 0.0。
    """
    vs = UNIT_EFFECTIVENESS.get(attacker_type.lower(), {})
    if not vs:
        return 0.0
    return vs.get(target_armor.lower(), 1.0)


def get_unit_vs_unit(attacker: str, target: str) -> float:
    """查询攻击方对指定目标单位的伤害倍率（通过目标护甲间接求克制）。

    步骤:
      1. 查 UNIT_ARMOR 得到目标单位护甲（未知默认 "none"）；
      2. 复用 get_effectiveness 按该护甲查询伤害倍率。

    参数:
        attacker: 攻击方单位类型。
        target: 目标单位类型。

    返回:
        攻击方对目标单位的伤害倍率；无攻击能力时返回 0.0。
    """
    target_armor = UNIT_ARMOR.get(target.lower(), "none")
    return get_effectiveness(attacker, target_armor)


def get_unit_armor(unit_type: str) -> str:
    """查询单位护甲类型，未知类型默认返回 "none"（步兵护甲）。

    参数:
        unit_type: 单位类型。

    返回:
        护甲类型字符串（"none" / "light" / "heavy" / "wood" / "concrete"）。
    """
    return UNIT_ARMOR.get(unit_type.lower(), "none")


def get_building_armor(building_type: str) -> str:
    """查询建筑护甲类型，未知类型默认返回 "wood"（木质建筑）。

    参数:
        building_type: 建筑类型。

    返回:
        护甲类型字符串（通常为 "wood" 或 "concrete"）。
    """
    return BUILDING_ARMOR.get(building_type.lower(), "wood")


def get_unit_cost(unit_type: str) -> int:
    """查询单位造价，未知类型返回 0。

    参数:
        unit_type: 单位类型。

    返回:
        造价（金币），用于按成本加权战损等计算。
    """
    return UNIT_COST.get(unit_type.lower(), 0)


def get_building_cost(building_type: str) -> int:
    """查询建筑造价，未知类型返回 0。

    参数:
        building_type: 建筑类型。

    返回:
        造价（金币）。
    """
    return BUILDING_COST.get(building_type.lower(), 0)


def can_attack(unit_type: str) -> bool:
    """判断某单位类型是否具备攻击能力。

    依据: UNIT_EFFECTIVENESS 中存在该单位且其效能 dict 非空。
    效能为空 dict 的单位（如矿车、基地车、运输机）视为无法攻击。

    参数:
        unit_type: 单位类型。

    返回:
        True 表示可攻击，False 表示无攻击能力。
    """
    return bool(UNIT_EFFECTIVENESS.get(unit_type.lower(), {}))


def is_economic_target(unit_or_building: str) -> bool:
    """判断摧毁该目标是否会造成经济破坏。

    目标类型: 经济单位（矿车 harv、运输车 truk）或经济建筑
    （精炼厂 proc、储存井 silo）。

    参数:
        unit_or_building: 单位或建筑类型。

    返回:
        True 表示属于经济目标。
    """
    t = unit_or_building.lower()
    return t in ECONOMIC_UNITS or t in ECONOMIC_BUILDINGS


def is_production_target(building_type: str) -> bool:
    """判断摧毁该建筑是否会扰乱敌方生产。

    覆盖兵营（barr/tent）、坦克工厂（weap）、机场（hpad/afld）、
    船坞（spen/syrd）、狗窝（kenn）等生产类建筑。

    参数:
        building_type: 建筑类型。

    返回:
        True 表示属于生产目标。
    """
    return building_type.lower() in PRODUCTION_BUILDINGS


def is_tech_target(building_type: str) -> bool:
    """判断摧毁该建筑是否会造成科技倒退。

    覆盖雷达站（dome）、高级科技实验室（atek/stek）、维修厂（fix）等。

    参数:
        building_type: 建筑类型。

    返回:
        True 表示属于科技目标。
    """
    return building_type.lower() in TECH_BUILDINGS


def is_power_target(building_type: str) -> bool:
    """判断摧毁该建筑是否会切断敌方电力供应。

    覆盖小型电厂（powr）与高级电厂（apwr）。

    参数:
        building_type: 建筑类型。

    返回:
        True 表示属于电力目标。
    """
    return building_type.lower() in POWER_BUILDINGS


def compute_army_counter_score(
    own_units: list[dict],
    enemy_units: list[dict],
) -> tuple[float, float]:
    """计算己方兵力对敌方兵力的克制程度与脆弱程度。

    步骤:
      1. 双方任一为空，或敌方护甲统计为空时返回中性分 (0.5, 0.5)；
      2. 统计敌方各护甲类型的单位数量分布；
      3. 过滤出己方可攻击的作战单位（调用 can_attack）；
      4. 对每个己方作战单位，按敌方护甲分布加权计算平均效能：
         - 平均效能 >= 1.0 计为"克制有效"；
         - 平均效能 < 0.5 计为"脆弱易损"；
      5. 返回 (克制单位占比, 脆弱单位占比)，均为 [0, 1]。

    参数:
        own_units: 己方单位列表，元素为至少含 'type' 键的 dict。
        enemy_units: 敌方单位列表，元素为至少含 'type' 键的 dict。

    返回:
        (counter_score, vulnerability_score)，取值范围 [0, 1]。
        counter_score: 己方作战单位中对敌方有效的比例。
        vulnerability_score: 己方作战单位中对敌方脆弱易损的比例。
    """
    if not own_units or not enemy_units:
        return 0.5, 0.5

    enemy_armors: dict[str, int] = {}
    for u in enemy_units:
        armor = get_unit_armor(u.get("type", ""))
        enemy_armors[armor] = enemy_armors.get(armor, 0) + 1

    total_enemy = sum(enemy_armors.values())
    if total_enemy == 0:
        return 0.5, 0.5

    own_combat = [u for u in own_units if can_attack(u.get("type", ""))]
    if not own_combat:
        return 0.0, 1.0

    effective_count = 0
    vulnerable_count = 0

    for u in own_combat:
        utype = u.get("type", "")
        avg_eff = 0.0
        for armor, count in enemy_armors.items():
            avg_eff += get_effectiveness(utype, armor) * (count / total_enemy)

        if avg_eff >= 1.0:
            effective_count += 1
        elif avg_eff < 0.5:
            vulnerable_count += 1

    n = len(own_combat)
    return effective_count / n, vulnerable_count / n
