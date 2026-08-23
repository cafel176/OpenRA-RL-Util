"""OpenRA-RL 的多维奖励向量。

不同于单个标量奖励，这里提供 7+1 维向量，每一维代表一项独立的战略技能:

  1. combat:         按成本加权的战损交换（高效打赢战斗）
  2. economy:        经济发展与战争（收入、资产、矿车）
  3. infrastructure: 基地建设、科技进程、生产流程
  4. intelligence:   侦察、开雾、威胁发现
  5. composition:    兵力结构对敌方军队的质量
  6. tempo:          行动的时效性（避免闲置单位）
  7. disruption:     战略破坏（电厂、生产、科技倒退）
  8. outcome:        终局胜负信号（+1/-1，仅游戏结束时产生）

奖励向量（逐 tick 训练信号）有意与 Benchmark 综合评分
（rubrics.py 中的整局排行榜分数）分开:
向量训练智能体技能；Benchmark 衡量整体表现。

用法:
    computer = RewardVectorComputer()
    computer.reset()

    for tick in game:
        obs = get_observation()
        vector = computer.compute(obs)
        # vector.combat, vector.economy, etc.
        scalar = vector.weighted_scalar(weights)  # 供单值 head 的 RL 算法使用
"""

from dataclasses import dataclass, field
from typing import Optional

from openra_rl_util.damage_matrix import (
    BUILDING_COST,
    ECONOMIC_BUILDINGS,
    ECONOMIC_UNITS,
    POWER_BUILDINGS,
    PRODUCTION_BUILDINGS,
    TECH_BUILDINGS,
    UNIT_COST,
    can_attack,
    compute_army_counter_score,
    get_building_armor,
    get_effectiveness,
    get_unit_armor,
    get_unit_cost,
)

# ── 将向量折叠为标量时使用的默认权重 ──────────────────────────────────────────

DEFAULT_WEIGHTS: dict[str, float] = {
    "combat": 0.30,           # 战斗：权重最高，强调打赢战斗
    "economy": 0.15,          # 经济
    "infrastructure": 0.10,   # 基础设施
    "intelligence": 0.10,     # 情报
    "composition": 0.10,      # 兵力构成
    "tempo": 0.10,            # 节奏
    "disruption": 0.15,       # 破坏
    "outcome": 1.00,          # 终局结果（单值 head 时以胜负为绝对主导）
}

# ── 归一化与奖励常量 ──────────────────────────────────────────────────────────

COMBAT_NORMALIZER = 5000.0      # 约等于一次中型坦克交战的成本，用于归一化战损
ECONOMY_NORMALIZER = 10000.0    # 约等于一座精炼厂的造价，用于归一化经济增量
HARVESTER_BONUS = 0.3           # 每击毁敌方一辆矿车的额外奖励
REFINERY_BONUS = 0.5            # 每击毁敌方一座精炼厂的额外奖励
POWER_PLANT_BONUS = 0.2         # 每击毁敌方一座电厂的额外奖励
PRODUCTION_BONUS = 0.2          # 每击毁敌方一座生产建筑的额外奖励
TECH_BONUS = 0.3                # 每击毁敌方一座科技建筑的额外奖励
NEW_BUILDING_TYPE_REWARD = 0.2  # 每完成一种新建筑类型的奖励
ENEMY_BASE_DISCOVERY_BONUS = 0.5  # 首次发现敌方基地位置的奖励
ENEMY_PRODUCTION_DISCOVERY_BONUS = 0.2  # 首次发现敌方生产建筑的奖励
ENEMY_UNIT_SIGHTING_BONUS = 0.05 # 每新发现一个敌方单位的奖励


@dataclass
class RewardVector:
    """7+1 维奖励信号——每一维对应一项智能体技能。

    各维度取值范围 [-1, 1]，作为强化学习逐 tick 的奖励向量。
    """

    combat: float = 0.0          # 战斗：按成本加权的战损交换
    economy: float = 0.0         # 经济：收入/资产增长与矿车战
    infrastructure: float = 0.0  # 基础设施：基地建设、科技进程、生产利用率
    intelligence: float = 0.0    # 情报：侦察、开雾、威胁发现
    composition: float = 0.0     # 兵力构成：对敌方军队的克制质量
    tempo: float = 0.0           # 节奏：行动时效性（闲置惩罚/指令速率）
    disruption: float = 0.0      # 破坏：战略破坏（电厂/生产/科技）
    outcome: float = 0.0         # 终局：胜负信号（仅游戏结束时非零）

    def as_dict(self) -> dict[str, float]:
        """将向量转为以维度名为键的 dict（便于日志记录与逐维访问）。"""
        return {
            "combat": self.combat,
            "economy": self.economy,
            "infrastructure": self.infrastructure,
            "intelligence": self.intelligence,
            "composition": self.composition,
            "tempo": self.tempo,
            "disruption": self.disruption,
            "outcome": self.outcome,
        }

    def as_array(self) -> list[float]:
        """将向量转为固定顺序的数值列表（便于存入数组/张量）。"""
        return [
            self.combat, self.economy, self.infrastructure,
            self.intelligence, self.composition, self.tempo,
            self.disruption, self.outcome,
        ]

    def weighted_scalar(self, weights: Optional[dict[str, float]] = None) -> float:
        """按权重将各维度折叠为单个标量（供需要单一奖励值的算法使用）。

        步骤:
          1. 未指定权重时使用 DEFAULT_WEIGHTS；
          2. 逐维度将数值乘以对应权重并累加（未覆盖的维度权重按 0 处理）。

        参数:
            weights: 各维度的权重 dict；缺省时使用 DEFAULT_WEIGHTS。

        返回:
            加权求和后的标量奖励。
        """
        w = weights or DEFAULT_WEIGHTS
        total = 0.0
        for dim, val in self.as_dict().items():
            total += val * w.get(dim, 0.0)
        return total


@dataclass
class RewardVectorState:
    """跨 tick 跟踪增量奖励计算所需的状态。

    增量奖励基于"本帧值 - 上一帧值"，该状态保存上一帧的各类
    计数器、HP 快照与已发现目标集合。
    """

    # 军事增量（战损统计的上一帧快照）
    prev_kills_cost: int = 0       # 上一帧累计击杀成本（用于求击杀增量）
    prev_deaths_cost: int = 0      # 上一帧累计阵亡成本
    prev_units_killed: int = 0     # 上一帧累计击杀单位数
    prev_buildings_killed: int = 0 # 上一帧累计击毁建筑数
    prev_units_lost: int = 0       # 上一帧累计损失单位数
    prev_buildings_lost: int = 0   # 上一帧累计损失建筑数

    # 经济增量
    prev_cash: int = 0             # 上一帧现金
    prev_ore: int = 0              # 上一帧矿石储量
    prev_assets_value: int = 0     # 上一帧资产总值（军事口径）
    prev_harvester_count: int = 0  # 上一帧矿车数量

    # HP 快照（用于部分伤害奖励：掉血但未击杀也给予/扣除奖励）
    prev_own_unit_hp: dict[int, float] = field(default_factory=dict)      # 己方单位 actor_id → 血量百分比
    prev_own_building_hp: dict[int, float] = field(default_factory=dict)  # 己方建筑 actor_id → 血量百分比
    prev_enemy_unit_hp: dict[int, float] = field(default_factory=dict)    # 敌方单位 actor_id → 血量百分比
    prev_enemy_building_hp: dict[int, float] = field(default_factory=dict)# 敌方建筑 actor_id → 血量百分比

    # 情报追踪
    prev_visible_cell_count: int = 0                                        # 上一帧已探明格数（当前未使用，预留）
    discovered_enemy_building_types: set[str] = field(default_factory=set)  # 已发现过的敌方建筑类型（每类只奖励一次）
    discovered_enemy_base: bool = False                                     # 是否已发现敌方基地位置
    prev_visible_enemy_ids: set[int] = field(default_factory=set)           # 上一帧可见敌方单位 ID 集合

    # 基础设施追踪
    own_building_types_built: set[str] = field(default_factory=set)  # 己方已建成过的建筑类型（用于新建筑奖励）
    prev_order_count: int = 0                                        # 上一帧累计指令数

    # 敌方建筑追踪（用于破坏维度：数量减少即判定被摧毁）
    prev_enemy_building_count: int = 0       # 上一帧可见敌方建筑总数
    prev_enemy_power_buildings: int = 0      # 上一帧可见敌方电厂数
    prev_enemy_production_buildings: int = 0 # 上一帧可见敌方生产建筑数
    prev_enemy_tech_buildings: int = 0       # 上一帧可见敌方科技建筑数


class RewardVectorComputer:
    """根据游戏观测计算多维奖励向量的主计算器。

    compute() 每次调用接收完整观测 dict，返回 7+1 维全部填好的
    RewardVector。内部用 RewardVectorState 记录上一帧状态，实现
    基于增量的稠密奖励。
    """

    def __init__(self, weights: Optional[dict[str, float]] = None):
        """初始化计算器。

        参数:
            weights: 折叠标量时的权重；缺省使用 DEFAULT_WEIGHTS。
        """
        self.weights = weights or DEFAULT_WEIGHTS
        self._state = RewardVectorState()

    def reset(self) -> None:
        """重置全部跟踪状态，开始新 episode。"""
        self._state = RewardVectorState()

    def compute(self, obs: dict) -> RewardVector:
        """从观测字典计算奖励向量。

        步骤:
          1. 从 obs 中取出各字段（military/economy/units/buildings/...）；
          2. 依次计算 7 个维度（combat 至 disruption）；
          3. done 时按 result 设置 outcome（win=+1, lose=-1）；
          4. 更新跟踪状态供下一 tick 增量使用。

        参数:
            obs: 观测字典，包含键 military、economy、units、buildings、
                 visible_enemies、visible_enemy_buildings、
                 production_queues、spatial_map_meta、done、result、tick。

        返回:
            全部维度填好的 RewardVector。
        """
        military = obs.get("military", {})
        economy = obs.get("economy", {})
        units = obs.get("units", [])
        buildings = obs.get("buildings", [])
        visible_enemies = obs.get("visible_enemies", [])
        visible_enemy_buildings = obs.get("visible_enemy_buildings", [])
        production_queues = obs.get("production_queues", [])
        done = obs.get("done", False)
        result = obs.get("result", "")

        vector = RewardVector()

        vector.combat = self._compute_combat(military, units, buildings,
                                             visible_enemies, visible_enemy_buildings)
        vector.economy = self._compute_economy(economy, military,
                                               visible_enemies, visible_enemy_buildings)
        vector.infrastructure = self._compute_infrastructure(buildings, production_queues, economy)
        vector.intelligence = self._compute_intelligence(visible_enemies, visible_enemy_buildings)
        vector.composition = self._compute_composition(units, visible_enemies)
        vector.tempo = self._compute_tempo(units, military, production_queues)
        vector.disruption = self._compute_disruption(visible_enemy_buildings)

        if done:
            if result == "win":
                vector.outcome = 1.0
            elif result == "lose":
                vector.outcome = -1.0

        self._update_state(military, economy, units, buildings,
                           visible_enemies, visible_enemy_buildings)

        return vector

    # ── 各维度计算函数 ─────────────────────────────────────────────────────────

    def _compute_combat(
        self,
        military: dict,
        own_units: list,
        own_buildings: list,
        enemies: list,
        enemy_buildings: list,
    ) -> float:
        """计算战斗维度：按成本加权的战损交换，含部分伤害跟踪。

        步骤:
          1. 用累计击杀成本与阵亡成本的增量差计算净击杀收益（归一化）；
          2. 敌方单位/建筑血量下降（未死）时按成本与掉血比例给正奖励；
          3. 己方单位/建筑掉血时给等量负奖励；
          4. 结果夹取到 [-1, 1]。

        参数:
            military: 军事统计 dict（含 kills_cost/deaths_cost）。
            own_units: 己方单位列表。
            own_buildings: 己方建筑列表。
            enemies: 可见敌方单位列表。
            enemy_buildings: 可见敌方建筑列表。

        返回:
            战斗维度奖励，范围 [-1, 1]。
        """
        reward = 0.0

        # 1. 按成本加权的净击杀（来自聚合统计的本帧与上一帧差值）
        kills_cost = military.get("kills_cost", 0)
        deaths_cost = military.get("deaths_cost", 0)
        kills_delta = kills_cost - self._state.prev_kills_cost
        deaths_delta = deaths_cost - self._state.prev_deaths_cost
        reward += (kills_delta - deaths_delta) / COMBAT_NORMALIZER

        # 2. 对敌方造成的部分伤害（HP 下降但未击杀，按掉血量与成本给正奖励）
        for enemy in enemies:
            eid = enemy.get("actor_id", 0)
            hp = enemy.get("hp_percent", 1.0)
            etype = enemy.get("type", "")
            prev_hp = self._state.prev_enemy_unit_hp.get(eid)

            if prev_hp is not None and hp < prev_hp:
                cost = get_unit_cost(etype)
                damage_value = cost * (prev_hp - hp)
                reward += damage_value / COMBAT_NORMALIZER

        # 敌方建筑掉血：按建筑成本（未知默认 500）与掉血量给正奖励
        for eb in enemy_buildings:
            eid = eb.get("actor_id", 0)
            hp = eb.get("hp_percent", 1.0)
            etype = eb.get("type", "")
            prev_hp = self._state.prev_enemy_building_hp.get(eid)

            if prev_hp is not None and hp < prev_hp:
                cost = BUILDING_COST.get(etype.lower(), 500)
                damage_value = cost * (prev_hp - hp)
                reward += damage_value / COMBAT_NORMALIZER

        # 3. 己方受到的部分伤害（己方单位掉血，扣等量奖励）
        for unit in own_units:
            uid = unit.get("actor_id", 0)
            hp = unit.get("hp_percent", 1.0)
            utype = unit.get("type", "")
            prev_hp = self._state.prev_own_unit_hp.get(uid)

            if prev_hp is not None and hp < prev_hp:
                cost = get_unit_cost(utype)
                damage_value = cost * (prev_hp - hp)
                reward -= damage_value / COMBAT_NORMALIZER

        # 己方建筑掉血：同样按成本与掉血量扣奖励
        for bldg in own_buildings:
            bid = bldg.get("actor_id", 0)
            hp = bldg.get("hp_percent", 1.0)
            btype = bldg.get("type", "")
            prev_hp = self._state.prev_own_building_hp.get(bid)

            if prev_hp is not None and hp < prev_hp:
                cost = BUILDING_COST.get(btype.lower(), 500)
                damage_value = cost * (prev_hp - hp)
                reward -= damage_value / COMBAT_NORMALIZER

        return max(-1.0, min(1.0, reward))

    def _compute_economy(
        self,
        economy: dict,
        military: dict,
        enemies: list,
        enemy_buildings: list,
    ) -> float:
        """计算经济维度：经济发展与战争（收入、资产、矿车战）。

        步骤:
          1. 现金+矿石+资产总值相比上一帧的增量（归一化后奖励）；
          2. 敌方经济建筑（精炼厂）被摧毁时奖励 REFINERY_BONUS；
          3. 己方矿车损失时按损失数量扣 HARVESTER_BONUS；
          4. 结果夹取到 [-1, 1]。

        参数:
            economy: 经济统计 dict（含 cash/ore/harvester_count）。
            military: 军事统计 dict（含 assets_value/buildings_killed）。
            enemies: 可见敌方单位列表。
            enemy_buildings: 可见敌方建筑列表。

        返回:
            经济维度奖励，范围 [-1, 1]。
        """
        reward = 0.0

        # 1. 己方经济增量（现金 + 矿石 + 资产总值的本帧与上一帧差值）
        cash = economy.get("cash", 0)
        ore = economy.get("ore", 0)
        assets = military.get("assets_value", 0)

        prev_total = self._state.prev_cash + self._state.prev_ore + self._state.prev_assets_value
        curr_total = cash + ore + assets
        econ_delta = curr_total - prev_total
        reward += econ_delta / ECONOMY_NORMALIZER

        # 敌方矿车击杀检测（通过统计可见敌方矿车数量；当前为近似实现，
        # 依赖 kills_cost 增量判定，详见经济建筑摧毁检测）
        enemy_harv_count = sum(
            1 for e in enemies if e.get("type", "").lower() == "harv"
        )
        # 理论上可通过 kills_cost 增量 + 单位类型检测击杀，但实现更简单的方式是：
        # 通过可见敌方建筑的 HP 快照判断经济建筑是否被摧毁
        buildings_killed = military.get("buildings_killed", 0)
        prev_bk = self._state.prev_buildings_killed

        # 2. 敌方经济建筑被摧毁检测（上一帧存活、本帧血量归零即判定）
        for eb in enemy_buildings:
            eid = eb.get("actor_id", 0)
            hp = eb.get("hp_percent", 1.0)
            etype = eb.get("type", "").lower()
            prev_hp = self._state.prev_enemy_building_hp.get(eid)

            # 建筑刚被摧毁（上一帧可见且血量 > 0，本帧归零）
            if prev_hp is not None and prev_hp > 0 and hp <= 0:
                if etype in ECONOMIC_BUILDINGS:
                    reward += REFINERY_BONUS

        # 3. 己方矿车损失惩罚
        harv_count = economy.get("harvester_count", 0)
        if harv_count < self._state.prev_harvester_count:
            lost = self._state.prev_harvester_count - harv_count
            reward -= HARVESTER_BONUS * lost

        return max(-1.0, min(1.0, reward))

    def _compute_infrastructure(
        self,
        buildings: list,
        production_queues: list,
        economy: dict,
    ) -> float:
        """计算基础设施维度：基地建设、科技进程、生产利用率。

        步骤:
          1. 新完成建筑类型：每出现一种新类型奖励 NEW_BUILDING_TYPE_REWARD；
          2. 生产利用率：正在生产的建筑占生产类建筑总数的比例；
          3. 电力健康度：电力盈余（供应-消耗）归一化到 [-1, 1]；
          4. 三个子信号取平均后夹取到 [-1, 1]。

        参数:
            buildings: 己方建筑列表。
            production_queues: 生产队列列表（当前未直接使用，预留）。
            economy: 经济统计 dict（含 power_provided/power_drained）。

        返回:
            基础设施维度奖励，范围 [-1, 1]。
        """
        # 1. 新建筑类型奖励（推进科技/基地发展）
        tech_reward = 0.0
        current_types = set()
        for b in buildings:
            btype = b.get("type", "").lower()
            if btype:
                current_types.add(btype)

        # 与已建成类型集合做差，得到本帧新出现的建筑类型
        new_types = current_types - self._state.own_building_types_built
        tech_reward = len(new_types) * NEW_BUILDING_TYPE_REWARD

        # 2. 生产利用率：正在生产的建筑占比
        producing_count = 0
        total_production_buildings = 0
        for b in buildings:
            btype = b.get("type", "").lower()
            if btype in {"barr", "tent", "weap", "hpad", "afld", "spen", "syrd", "kenn"}:
                total_production_buildings += 1
                if b.get("is_producing", False):
                    producing_count += 1

        production_util = (
            producing_count / total_production_buildings
            if total_production_buildings > 0
            else 0.0
        )

        # 3. 电力健康度：盈余（供应-消耗）除以 100 并夹取到 [-1, 1]
        power_provided = economy.get("power_provided", 0)
        power_drained = economy.get("power_drained", 0)
        surplus = power_provided - power_drained
        power_health = max(-1.0, min(1.0, surplus / 100.0))

        # 组合：三个子信号取平均
        infra = (tech_reward + production_util + power_health) / 3.0

        return max(-1.0, min(1.0, infra))

    def _compute_intelligence(
        self,
        enemies: list,
        enemy_buildings: list,
    ) -> float:
        """计算情报维度：侦察、开雾、威胁发现。

        步骤:
          1. 新发现敌方单位：与上一帧可见 ID 集合做差，每个奖励
             ENEMY_UNIT_SIGHTING_BONUS；
          2. 新发现敌方建筑类型（每种类型一次性奖励）：
             - 生产建筑 → ENEMY_PRODUCTION_DISCOVERY_BONUS；
             - 任意建筑（即基地位置）首次发现 → ENEMY_BASE_DISCOVERY_BONUS；
          3. 结果夹取到 [-1, 1]。

        参数:
            enemies: 可见敌方单位列表。
            enemy_buildings: 可见敌方建筑列表。

        返回:
            情报维度奖励，范围 [-1, 1]。
        """
        reward = 0.0

        # 1. 新发现的敌方单位（本帧与上一帧 ID 集合做差）
        current_enemy_ids = set()
        for e in enemies:
            eid = e.get("actor_id", 0)
            if eid:
                current_enemy_ids.add(eid)

        new_sightings = current_enemy_ids - self._state.prev_visible_enemy_ids
        reward += len(new_sightings) * ENEMY_UNIT_SIGHTING_BONUS

        # 2. 敌方建筑类型发现（每种类型只奖励一次）
        for eb in enemy_buildings:
            btype = eb.get("type", "").lower()
            if btype and btype not in self._state.discovered_enemy_building_types:
                self._state.discovered_enemy_building_types.add(btype)
                # 首次发现敌方生产建筑 = 重要情报
                if btype in PRODUCTION_BUILDINGS:
                    reward += ENEMY_PRODUCTION_DISCOVERY_BONUS
                # 首次发现任意建筑 = 定位到敌方基地
                if not self._state.discovered_enemy_base:
                    self._state.discovered_enemy_base = True
                    reward += ENEMY_BASE_DISCOVERY_BONUS

        return max(-1.0, min(1.0, reward))

    def _compute_composition(
        self,
        own_units: list,
        enemies: list,
    ) -> float:
        """计算兵力构成维度：己方军队对敌方的克制质量。

        步骤:
          1. 双方任意一方为空时返回 0（无可比较对象）；
          2. 调用 compute_army_counter_score 得到克制分与脆弱分；
          3. 净构成优势 = 克制分 - 脆弱分，夹取到 [-1, 1]。

        参数:
            own_units: 己方单位列表。
            enemies: 敌方单位列表。

        返回:
            兵力构成维度奖励，范围 [-1, 1]。
        """
        if not own_units or not enemies:
            return 0.0

        counter_score, vulnerability_score = compute_army_counter_score(
            own_units, enemies
        )

        # 净构成优势：克制有效的占比减去脆弱易损的占比
        return max(-1.0, min(1.0, counter_score - vulnerability_score))

    def _compute_tempo(
        self,
        units: list,
        military: dict,
        production_queues: list,
    ) -> float:
        """计算节奏维度：闲置惩罚与指令速率。

        步骤:
          1. 无单位时返回 0；
          2. 闲置作战单位比例 * 0.1 作为惩罚；
          3. 本 tick 指令增量 / 5 作为指令速率（每 tick 5 条指令即满速）；
          4. 节奏 = 指令速率 * 0.05 - 闲置惩罚，夹取到 [-1, 1]。

        参数:
            units: 己方单位列表。
            military: 军事统计 dict（含 order_count）。
            production_queues: 生产队列列表（当前未直接使用，预留）。

        返回:
            节奏维度奖励，范围 [-1, 1]。
        """
        if not units:
            return 0.0

        # 1. 闲置作战单位惩罚（鼓励军队保持活跃，不做无谓等待）
        combat_units = [u for u in units if can_attack(u.get("type", ""))]
        if combat_units:
            idle_combat = sum(1 for u in combat_units if u.get("is_idle", True))
            idle_ratio = idle_combat / len(combat_units)
            idle_penalty = idle_ratio * 0.1
        else:
            idle_penalty = 0.0

        # 2. 指令速率（本 tick 是否在下达指令，衡量行动的积极性）
        order_count = military.get("order_count", 0)
        order_delta = order_count - self._state.prev_order_count
        order_rate = min(1.0, order_delta / 5.0)  # 每 tick 5 条指令即视为满速

        # 组合：奖励活跃行动，惩罚闲置作战单位
        tempo = (order_rate * 0.05) - idle_penalty

        return max(-1.0, min(1.0, tempo))

    def _compute_disruption(
        self,
        enemy_buildings: list,
    ) -> float:
        """计算破坏维度：战略破坏——摧毁敌方关键建筑。

        步骤:
          1. 统计可见敌方建筑中电厂/生产/科技建筑的数量；
          2. 与上一帧数量对比，数量减少即判定被摧毁：
             - 电厂减少 → POWER_PLANT_BONUS/座；
             - 生产建筑减少 → PRODUCTION_BONUS/座；
             - 科技建筑减少 → TECH_BONUS/座；
          3. 结果夹取到 [-1, 1]。

        参数:
            enemy_buildings: 可见敌方建筑列表。

        返回:
            破坏维度奖励，范围 [-1, 1]。
        """
        reward = 0.0

        # 统计当前帧各类敌方关键建筑数量
        power_count = 0
        production_count = 0
        tech_count = 0
        total_count = 0

        for eb in enemy_buildings:
            btype = eb.get("type", "").lower()
            if not btype:
                continue
            total_count += 1
            if btype in POWER_BUILDINGS:
                power_count += 1
            if btype in PRODUCTION_BUILDINGS:
                production_count += 1
            if btype in TECH_BUILDINGS:
                tech_count += 1

        # 检测被摧毁的敌方建筑（数量比上一帧减少，且上一帧确实有建筑）
        if self._state.prev_enemy_building_count > 0:
            # 电力破坏：敌方电厂数量下降
            if power_count < self._state.prev_enemy_power_buildings:
                lost = self._state.prev_enemy_power_buildings - power_count
                reward += lost * POWER_PLANT_BONUS

            # 生产破坏：敌方生产建筑数量下降
            if production_count < self._state.prev_enemy_production_buildings:
                lost = self._state.prev_enemy_production_buildings - production_count
                reward += lost * PRODUCTION_BONUS

            # 科技倒退：敌方科技建筑数量下降
            if tech_count < self._state.prev_enemy_tech_buildings:
                lost = self._state.prev_enemy_tech_buildings - tech_count
                reward += lost * TECH_BONUS

        return max(-1.0, min(1.0, reward))

    # ── 状态更新 ───────────────────────────────────────────────────────────────

    def _update_state(
        self,
        military: dict,
        economy: dict,
        own_units: list,
        own_buildings: list,
        enemies: list,
        enemy_buildings: list,
    ) -> None:
        """计算完奖励后更新跟踪状态（将本帧快照保存为下一帧的"上一帧"）。

        步骤:
          1. 保存军事统计与指令数（击杀/阵亡/击毁/损失/指令）；
          2. 保存经济统计与矿车数（现金/矿石/资产/矿车数）；
          3. 重建双方单位与建筑的 HP 快照；
          4. 更新可见敌方 ID 集合；
          5. 累积己方已建成建筑类型；
          6. 统计敌方关键建筑（电厂/生产/科技）数量，供破坏维度使用。

        参数:
            military: 军事统计 dict。
            economy: 经济统计 dict。
            own_units: 己方单位列表。
            own_buildings: 己方建筑列表。
            enemies: 可见敌方单位列表。
            enemy_buildings: 可见敌方建筑列表。
        """
        s = self._state

        # 1. 军事统计快照
        s.prev_kills_cost = military.get("kills_cost", 0)
        s.prev_deaths_cost = military.get("deaths_cost", 0)
        s.prev_units_killed = military.get("units_killed", 0)
        s.prev_buildings_killed = military.get("buildings_killed", 0)
        s.prev_units_lost = military.get("units_lost", 0)
        s.prev_buildings_lost = military.get("buildings_lost", 0)
        s.prev_order_count = military.get("order_count", 0)

        # 2. 经济统计快照
        s.prev_cash = economy.get("cash", 0)
        s.prev_ore = economy.get("ore", 0)
        s.prev_assets_value = military.get("assets_value", 0)
        s.prev_harvester_count = economy.get("harvester_count", 0)

        # 3. HP 快照（actor_id → 血量百分比，缺失 actor_id 的实体跳过）
        s.prev_own_unit_hp = {
            u.get("actor_id", 0): u.get("hp_percent", 1.0)
            for u in own_units if u.get("actor_id")
        }
        s.prev_own_building_hp = {
            b.get("actor_id", 0): b.get("hp_percent", 1.0)
            for b in own_buildings if b.get("actor_id")
        }
        s.prev_enemy_unit_hp = {
            e.get("actor_id", 0): e.get("hp_percent", 1.0)
            for e in enemies if e.get("actor_id")
        }
        s.prev_enemy_building_hp = {
            eb.get("actor_id", 0): eb.get("hp_percent", 1.0)
            for eb in enemy_buildings if eb.get("actor_id")
        }

        # 4. 情报：更新可见敌方 ID 集合
        s.prev_visible_enemy_ids = {
            e.get("actor_id", 0) for e in enemies if e.get("actor_id")
        }

        # 5. 基础设施：累积己方已建成建筑类型
        for b in own_buildings:
            btype = b.get("type", "").lower()
            if btype:
                s.own_building_types_built.add(btype)

        # 6. 敌方建筑追踪（供破坏维度判定摧毁）
        s.prev_enemy_building_count = len(enemy_buildings)
        s.prev_enemy_power_buildings = sum(
            1 for eb in enemy_buildings
            if eb.get("type", "").lower() in POWER_BUILDINGS
        )
        s.prev_enemy_production_buildings = sum(
            1 for eb in enemy_buildings
            if eb.get("type", "").lower() in PRODUCTION_BUILDINGS
        )
        s.prev_enemy_tech_buildings = sum(
            1 for eb in enemy_buildings
            if eb.get("type", "").lower() in TECH_BUILDINGS
        )
