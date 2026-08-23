"""OpenRA-Bench 智能体评估评分规则。

遵循 OpenEnv rubric 模式（见 openenv.core.rubrics）。
这些规则根据胜负、军事实效与经济发展对整局游戏进行打分。

OpenRA-RL 生态中的两套评分体系
===============================

生态中刻意保持两套相互独立的评分系统:

1. **奖励向量**（reward_vector.py）:
   - 逐 tick、多维度的 RL 训练信号
   - 7 个技能维度 + 终局胜负结果
   - 稠密、基于增量的信号，适合策略梯度
   - 用于智能体训练与游戏过程

2. **Benchmark 综合评分**（本文件）:
   - 整局终了后的综合分数，用于排行榜排名
   - 组成: 胜率 (50%)、军事实效 (25%)、经济 (25%)
   - 归一化到 0-100 分
   - 由 OpenRA-Bench 评估框架使用

两者互补: 奖励向量训练智能体技能，Benchmark 衡量整体表现。
一个情报与节奏出色的智能体可能通过走位而非硬碰硬赢得对局——
Benchmark 捕捉胜负结果，而奖励向量解释获胜原因。

用法:
    rubric = OpenRABenchRubric()
    rubric.reset()
    for action, obs in episode:
        reward = rubric(action, obs)  # 终局前为 0.0
    step_rewards = rubric.win_loss.compute_step_rewards()
"""

from typing import Any, Dict, List, Tuple

from openra_rl_util.rubric_base import (
    ExponentialDiscountingTrajectoryRubric,
    TrajectoryRubric,
    WeightedSum,
)


class OpenRAWinLossRubric(ExponentialDiscountingTrajectoryRubric):
    """根据胜负/平局结果打分，并带时间折扣。

    终局奖励:
    - 胜利:  +1.0
    - 失败:  -1.0
    - 平局:   0.0
    """

    def score_trajectory(self, trajectory: List[Tuple[Any, Any]]) -> float:
        """从终局观测的 result 字段映射胜负分数。

        步骤:
          1. 轨迹为空返回 0.0；
          2. 取最后一步观测的 result 字段；
          3. "win" → +1.0，"lose" → -1.0，其余（平局等）→ 0.0。
        """
        if not trajectory:
            return 0.0
        _, final_obs = trajectory[-1]
        result = getattr(final_obs, "result", "")
        if result == "win":
            return 1.0
        elif result == "lose":
            return -1.0
        return 0.0


class MilitaryEfficiencyRubric(TrajectoryRubric):
    """根据终局观测的击杀/阵亡成本比打分。

    分数 = kills_cost / (kills_cost + deaths_cost)
    归一化到 [0, 1] 区间；完全未交战时给中性分 0.5。
    """

    def score_trajectory(self, trajectory: List[Tuple[Any, Any]]) -> float:
        """计算整局军事实效分。

        步骤:
          1. 轨迹为空或终局观测缺 military 时返回 0.0；
          2. 取 kills_cost 与 deaths_cost；
          3. 两者和为 0（未交战）返回中性分 0.5；
          4. 否则返回击杀成本占比。
        """
        if not trajectory:
            return 0.0
        _, final_obs = trajectory[-1]
        military = getattr(final_obs, "military", None)
        if military is None:
            return 0.0
        kills = getattr(military, "kills_cost", 0)
        deaths = getattr(military, "deaths_cost", 0)
        total = kills + deaths
        if total == 0:
            return 0.5  # 未发生交战
        return kills / total

    def compute_step_rewards(self) -> List[float]:
        """将整局军事实效分平均分配给每一步（逐帧等值返回）。"""
        if not self._trajectory:
            return []
        score = self.score_trajectory(self._trajectory)
        return [score] * len(self._trajectory)


class EconomyRubric(TrajectoryRubric):
    """根据终局经济状态打分。

    分数 = assets_value / (assets_value + 10000)
    类 Sigmoid 归一化，把 [0, +inf) 映射到 [0, 1)。
    """

    def score_trajectory(self, trajectory: List[Tuple[Any, Any]]) -> float:
        """计算整局经济分。

        步骤:
          1. 轨迹为空或终局观测缺 military 时返回 0.0；
          2. 取 assets_value；
          3. 资产为负时返回 0.0，否则按 Sigmoid 公式归一化。
        """
        if not trajectory:
            return 0.0
        _, final_obs = trajectory[-1]
        military = getattr(final_obs, "military", None)
        if military is None:
            return 0.0
        assets = getattr(military, "assets_value", 0)
        # Sigmoid 归一化: 将 [0, +inf) 映射到 [0, 1)
        return assets / (assets + 10000) if assets >= 0 else 0.0

    def compute_step_rewards(self) -> List[float]:
        """将整局经济分平均分配给每一步（逐帧等值返回）。"""
        if not self._trajectory:
            return []
        score = self.score_trajectory(self._trajectory)
        return [score] * len(self._trajectory)


class OpenRABenchRubric(WeightedSum):
    """胜负、军事、经济三项的综合 Benchmark 评分。

    权重: 50% 胜负 + 25% 军事实效 + 25% 经济。
    """

    def __init__(self, gamma: float = 0.99):
        win_loss = OpenRAWinLossRubric(gamma=gamma)
        military = MilitaryEfficiencyRubric()
        economy = EconomyRubric()
        super().__init__(
            rubrics=[win_loss, military, economy],
            weights=[0.5, 0.25, 0.25],
        )
        # 保留命名引用，方便直接访问各子评分
        self.win_loss = win_loss
        self.military = military
        self.economy = economy

    def reset(self) -> None:
        """同时重置三个子 Rubric 的轨迹状态。"""
        self.win_loss.reset()
        self.military.reset()
        self.economy.reset()


def compute_game_metrics(final_obs: Any) -> Dict[str, Any]:
    """从终局观测中提取 Benchmark 指标。

    步骤:
      1. 从 final_obs 属性式读取 military / economy 统计；
      2. 缺省字段按 0 处理（缺失对象整体按 0）；
      3. kd_ratio 用 max(deaths, 1) 防除零；
      4. 汇总为指标 dict 返回。

    参数:
        final_obs: 终局 GameObservation（done=True 的那一帧）。

    返回:
        含 result、ticks、kills_cost、deaths_cost、kd_ratio、
        assets_value、cash、win 键的指标 dict。
    """
    military = getattr(final_obs, "military", None)
    economy = getattr(final_obs, "economy", None)

    kills = getattr(military, "kills_cost", 0) if military else 0
    deaths = getattr(military, "deaths_cost", 0) if military else 0
    assets = getattr(military, "assets_value", 0) if military else 0
    cash = getattr(economy, "cash", 0) if economy else 0
    result = getattr(final_obs, "result", "")
    tick = getattr(final_obs, "tick", 0)

    return {
        "result": result,
        "win": result == "win",
        "ticks": tick,
        "kills_cost": kills,
        "deaths_cost": deaths,
        "kd_ratio": kills / max(deaths, 1),
        "assets_value": assets,
        "cash": cash,
    }


def compute_composite_score_from_games(game_results: List[Dict[str, Any]]) -> float:
    """根据多局游戏结果计算 OpenRA-Bench 综合评分。

    这是 Benchmark 评分的唯一事实来源（single source of truth）。
    公式与 OpenRABenchRubric 一致: 50% 胜率 + 25% 军事实效 + 25% 经济。

    关键设计: 先按单局公式算出每局的军事/经济子分再取平均，
    而不是先合并总量再归一化——以避免多局聚合时 Jensen 不等式带来的失真。

    步骤:
      1. 无结果时返回 0.0；
      2. 统计胜率；
      3. 逐局计算军事实效分（未交战计 0.5）后取平均；
      4. 逐局计算经济分（资产为负计 0.0）后取平均；
      5. 按 0.5/0.25/0.25 加权并放大到 0-100 返回。

    参数:
        game_results: 由 compute_game_metrics() 生成的指标 dict 列表。

    返回:
        0-100 区间的综合评分。
    """
    total = len(game_results)
    if total == 0:
        return 0.0

    # 胜率
    win_rate = sum(1 for g in game_results if g["win"]) / total

    # 逐局军事实效分再取平均（与 MilitaryEfficiencyRubric 公式一致）
    mil_scores = []
    for g in game_results:
        kills, deaths = g["kills_cost"], g["deaths_cost"]
        total_cost = kills + deaths
        mil_scores.append(kills / total_cost if total_cost > 0 else 0.5)
    avg_mil = sum(mil_scores) / total

    # 逐局经济分再取平均（与 EconomyRubric 公式一致）
    econ_scores = []
    for g in game_results:
        assets = g["assets_value"]
        econ_scores.append(assets / (assets + 10000) if assets >= 0 else 0.0)
    avg_econ = sum(econ_scores) / total

    return 100.0 * (0.5 * win_rate + 0.25 * avg_mil + 0.25 * avg_econ)
