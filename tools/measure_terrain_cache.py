"""terrain_cache 久跑記憶體成長實測（T03-1 情境 A）。

目的：用真實的 TerrainManager（非 mock），模擬機器人在 INFINITE 模式下
持續前進，量測 terrain_cache 的地塊數與 Python 記憶體成長，佐證
「快取無上限成長 → 需要 LRU 上界」的重構論點。

不需 policy、不需渲染、不開視窗：只建 MuJoCo model/data 餵給 TerrainManager，
沿直線路徑推進並週期性取樣。輸出 CSV + JSON 供後續產圖。

用法：
    PYTHONUTF8=1 python tools/measure_terrain_cache.py [--distance 8000] [--step 0.5]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import tracemalloc

import numpy as np

# 讓腳本可從 repo 根目錄直接執行
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.config import load_config
from src.core.logger import log


def main() -> None:
    parser = argparse.ArgumentParser(description="terrain_cache 成長實測")
    parser.add_argument("--distance", type=float, default=8000.0,
                        help="機器人沿 +x 直線前進的總距離（公尺）")
    parser.add_argument("--step", type=float, default=0.5,
                        help="每次推進的步長（公尺）")
    parser.add_argument("--sample-every", type=float, default=25.0,
                        help="每前進多少公尺取樣一次")
    parser.add_argument("--out", default="docs/reports/data",
                        help="輸出目錄")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    csv_path = os.path.join(args.out, "terrain_cache_growth.csv")
    json_path = os.path.join(args.out, "terrain_cache_growth.json")

    log.info("建立 headless Simulation + TerrainManager ...")
    config = load_config()
    from src.simulation.simulation import Simulation
    from src.simulation.terrain_manager import TerrainManager

    sim = Simulation(config)
    terrain = TerrainManager(config, sim.model, sim.data)
    if not terrain.is_functional:
        sys.exit("TerrainManager 非功能態（XML 缺 hfield？），無法量測。")

    # 估算單一 TerrainTile 物件大小（含字串），用於把地塊數換算成記憶體量級
    sample_tile = next(iter(terrain.terrain_cache.values()), None)
    tile_bytes = 0
    if sample_tile is not None:
        tile_bytes = (sys.getsizeof(sample_tile)
                      + sys.getsizeof(sample_tile.__dict__)
                      + sys.getsizeof(sample_tile.terrain_type))

    tracemalloc.start()
    base_current, _ = tracemalloc.get_traced_memory()

    rows = []
    n_steps = int(args.distance / args.step)
    sample_interval_steps = max(1, int(args.sample_every / args.step))

    log.info(f"開始推進：總距離 {args.distance} m、步長 {args.step} m、"
             f"共 {n_steps} 步、每 {args.sample_every} m 取樣。")
    t0 = time.perf_counter()

    def sample(dist):
        cur, peak = tracemalloc.get_traced_memory()
        rows.append({
            "distance_m": round(dist, 2),
            "cache_tiles": len(terrain.terrain_cache),
            "world_center_x": int(terrain.world_center_x),
            "traced_current_mb": round((cur - base_current) / 1e6, 4),
            "traced_peak_mb": round(peak / 1e6, 4),
            "cache_bytes_est": len(terrain.terrain_cache) * tile_bytes,
            "elapsed_s": round(time.perf_counter() - t0, 2),
        })

    sample(0.0)
    for i in range(1, n_steps + 1):
        dist = i * args.step
        robot_pos = np.array([dist, 0.0, 0.3])
        terrain.update(robot_pos, "INFINITE")
        if i % sample_interval_steps == 0:
            sample(dist)

    # 收尾再取一次
    sample(n_steps * args.step)
    tracemalloc.stop()

    # 寫 CSV
    fields = ["distance_m", "cache_tiles", "world_center_x",
              "traced_current_mb", "traced_peak_mb", "cache_bytes_est", "elapsed_s"]
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(fields) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in fields) + "\n")

    first, last = rows[0], rows[-1]
    summary = {
        "scenario": "terrain_cache_growth (INFINITE, straight +x)",
        "distance_m": args.distance,
        "step_m": args.step,
        "tile_bytes_est": tile_bytes,
        "tiles_start": first["cache_tiles"],
        "tiles_end": last["cache_tiles"],
        "traced_mb_end": last["traced_current_mb"],
        "elapsed_s": last["elapsed_s"],
        "samples": len(rows),
        "rows": rows,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    log.info(f"✅ 完成：地塊 {first['cache_tiles']} → {last['cache_tiles']}，"
             f"Python heap +{last['traced_current_mb']:.2f} MB，"
             f"耗時 {last['elapsed_s']:.1f}s。")
    log.info(f"   CSV: {csv_path}")
    log.info(f"   JSON: {json_path}")


if __name__ == "__main__":
    main()
