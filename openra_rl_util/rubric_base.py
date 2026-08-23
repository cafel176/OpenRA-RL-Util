"""从 OpenEnv 引入的最小 Rubric 基类。

这些类实现核心 Rubric API（forward/call 模式、轨迹累积、指数折扣、
加权组合）。当 openenv-core 在 PyPI 发布 openenv.core.rubrics 后，
本文件可替换为 re-export 形式:

    from openenv.core.rubrics import (  # noqa: F401
        Rubric, TrajectoryRubric,
        ExponentialDiscountingTrajectoryRubric, WeightedSum,
    )

来源: https://github.com/OpenEnvs/OpenEnv  (BSD license)
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple


class Rubric(ABC):
    """奖励计算的抽象基类。

    子类通过实现 forward() 定义具体奖励逻辑；外部统一使用
    rubric(action, observation) 调用（内部转发到 forward 并记录分数）。

    每次调用都会把结果写入 last_score，供外部随时查看最近一次打分。
    """

    def __init__(self):
        self.last_score: float | None = None

    def __call__(self, action: Any, observation: Any) -> float:
        """对外统一入口：调用 forward 计算分数并记录到 last_score。

        参数:
            action: 智能体动作。
            observation: 环境观测。

        返回:
            本步奖励分数。
        """
        result = self.forward(action, observation)
        self.last_score = result
        return result

    @abstractmethod
    def forward(self, action: Any, observation: Any) -> float:
        """核心打分逻辑（由子类实现）。

        参数:
            action: 智能体动作。
            observation: 环境观测。

        返回:
            本步奖励分数。
        """
        raise NotImplementedError

    def reset(self) -> None:
        """重置内部状态（基类无状态，子类按需覆写）。"""
        pass

    def state_dict(self) -> Dict[str, Any]:
        """导出可序列化状态（基类无状态，返回空 dict）。"""
        return {}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """从 state_dict() 恢复状态（基类无状态，直接忽略）。"""
        pass


class TrajectoryRubric(Rubric):
    """累积整局轨迹、在终局统一评分的 Rubric。

    步骤:
      - 每步调用 forward() 时把 (action, observation) 追加进轨迹；
      - 非终局返回中间奖励 intermediate_reward（默认 0.0）；
      - done=True 时调用 score_trajectory() 对整局轨迹打分。

    子类需实现 score_trajectory() 与 compute_step_rewards()。
    """

    def __init__(self, intermediate_reward: float = 0.0):
        super().__init__()
        self.intermediate_reward = intermediate_reward  # 非终局帧的中间奖励
        self._trajectory: List[Tuple[Any, Any]] = []

    def forward(self, action: Any, observation: Any) -> float:
        """累积轨迹并返回本步分数。

        步骤:
          1. 将 (action, observation) 追加到轨迹；
          2. 观测含 done=True 时调用 score_trajectory() 返回终局分数；
          3. 否则返回中间奖励。
        """
        self._trajectory.append((action, observation))
        if getattr(observation, "done", False):
            return self.score_trajectory(self._trajectory)
        return self.intermediate_reward

    @abstractmethod
    def score_trajectory(self, trajectory: List[Tuple[Any, Any]]) -> float:
        """对整局轨迹打分（由子类实现）。

        参数:
            trajectory: 整局 (action, observation) 序列。

        返回:
            终局分数。
        """
        raise NotImplementedError

    @abstractmethod
    def compute_step_rewards(self) -> List[float]:
        """将终局分数分配回每一步（由子类实现）。

        返回:
            与轨迹等长的逐帧奖励列表。
        """
        raise NotImplementedError

    def reset(self) -> None:
        """清空累积的轨迹，开始新 episode。"""
        self._trajectory = []

    @property
    def trajectory(self) -> List[Tuple[Any, Any]]:
        """返回累积轨迹的只读副本。"""
        return list(self._trajectory)


class ExponentialDiscountingTrajectoryRubric(TrajectoryRubric):
    """带指数折扣信用分配的 TrajectoryRubric。

    逐帧奖励: r_t = gamma^(T-1-t) * R_final
    即离终局越远的帧，折扣越重，用于把终局分数按时间衰减分配给每步。
    """

    def __init__(self, gamma: float = 0.99, intermediate_reward: float = 0.0):
        super().__init__(intermediate_reward=intermediate_reward)
        # 校验折扣因子必须在 [0, 1] 区间
        if not 0.0 <= gamma <= 1.0:
            raise ValueError(f"gamma must be in [0, 1], got {gamma}")
        self.gamma = gamma

    def compute_step_rewards(self) -> List[float]:
        """按指数折扣把终局分数分配回每一步。

        步骤:
          1. 轨迹为空时返回空列表；
          2. 对整局轨迹调用 score_trajectory() 得到终局分数；
          3. 逐帧计算 r_t = 终局分数 * gamma^(T-1-t)。
        """
        if not self._trajectory:
            return []
        final_score = self.score_trajectory(self._trajectory)
        T = len(self._trajectory)
        return [final_score * (self.gamma ** (T - 1 - t)) for t in range(T)]


class WeightedSum(Rubric):
    """多个子 Rubric 的加权组合。

    构造时校验子 Rubric 数量与权重数量一致，且权重和必须为 1.0。
    forward() 时依次调用每个子 Rubric 打分，再按权重加权求和。
    """

    def __init__(self, rubrics: List[Rubric], weights: List[float]):
        super().__init__()
        if len(rubrics) != len(weights):
            raise ValueError(
                f"Number of rubrics ({len(rubrics)}) must match "
                f"number of weights ({len(weights)})"
            )
        if abs(sum(weights) - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {sum(weights)}")
        self._rubric_list = list(rubrics)  # 子 Rubric 列表
        self._weights = list(weights)      # 对应权重列表

    def forward(self, action: Any, observation: Any) -> float:
        """依次调用各子 Rubric 打分并加权求和。

        参数:
            action: 智能体动作。
            observation: 环境观测。

        返回:
            加权后的综合分数。
        """
        total = 0.0
        for rubric, weight in zip(self._rubric_list, self._weights):
            score = rubric(action, observation)
            total += score * weight
        return total

    @property
    def weights(self) -> List[float]:
        """返回权重列表的只读副本。"""
        return list(self._weights)
