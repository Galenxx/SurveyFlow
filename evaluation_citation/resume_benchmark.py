"""幂等续跑驱动：只补缺失产物，绝不重跑已完成的 cell。

背景：run_benchmark 的各 phase 本身无幂等跳过（generate 无条件重生成、
verify 每次都打网络）。直接 `--phase all` 会覆盖已完成的 58+ 个 generate、
白烧大量 API。本脚本按 (system, topic, seed, phase) 粒度检查产物是否已落盘，
只对缺失的 cell 调用对应 phase，因此可以反复运行直到全绿——任何一次中断后
再跑一遍即可自动补齐，已完成的瞬间跳过。

用法：
    python -u -m evaluation_citation.resume_benchmark
"""
from __future__ import annotations

import sys
import time

from evaluation_citation.configs import SYSTEMS, TOPICS, SEEDS
from evaluation_citation import run_benchmark as rb

# 6 phase 顺序，与 run_benchmark.main 的 "all" 序列一致
_PHASES = ("generate", "parse", "verify", "attribute", "classify", "judge")


def _all_cells() -> list[tuple[str, str, int]]:
    return [
        (s["id"], t["id"], seed)
        for s in SYSTEMS
        for t in TOPICS
        for seed in SEEDS
    ]


def _done(system_id: str, topic_id: str, seed: int, phase: str) -> bool:
    """产物已落盘则视为该 cell 的该 phase 完成（幂等跳过依据）。"""
    return rb._result_path(system_id, topic_id, seed, phase).exists()


def main() -> int:
    cells = _all_cells()
    print(f"== resume: {len(cells)} cells × {len(_PHASES)} phases ==", flush=True)

    for phase in _PHASES:
        pending = [c for c in cells if not _done(*c, phase)]
        done_n = len(cells) - len(pending)
        print(
            f"\n== phase={phase}: {done_n}/{len(cells)} 已完成，待跑 {len(pending)} ==",
            flush=True,
        )
        for system_id, topic_id, seed in pending:
            try:
                t0 = time.time()
                rb._dispatch(phase, system_id, topic_id, seed)
                dt = time.time() - t0
                print(f"  [OK] {system_id} / {topic_id} / seed{seed}  ({dt:.1f}s)", flush=True)
            except Exception as e:  # noqa: BLE001 — 单 cell 失败不中断其余
                print(f"  [FAIL] {system_id} / {topic_id} / seed{seed}: {e}", flush=True)

    # 收尾：报告每个 phase 的最终完成度
    print("\n== 最终完成度 ==", flush=True)
    for phase in _PHASES:
        n_ok = sum(1 for c in cells if _done(*c, phase))
        print(f"  {phase}: {n_ok}/{len(cells)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
