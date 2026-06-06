"""把 terrain_cache 量測 JSON 產成 HTML + 內嵌 SVG 報告（T03-1）。

用法：
    PYTHONUTF8=1 python tools/gen_terrain_cache_report.py
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSON_PATH = os.path.join(HERE, "docs/reports/data/terrain_cache_growth.json")
OUT_PATH = os.path.join(HERE, "docs/reports/TERRAIN_CACHE_2026-06-07.html")

W, H = 760, 320          # SVG 畫布
PAD_L, PAD_R, PAD_T, PAD_B = 64, 20, 20, 46


def _scale(rows, xkey, ykey):
    xs = [r[xkey] for r in rows]
    ys = [r[ykey] for r in rows]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = 0, max(ys) if max(ys) > 0 else 1
    plot_w = W - PAD_L - PAD_R
    plot_h = H - PAD_T - PAD_B

    def px(x):
        return PAD_L + (x - xmin) / (xmax - xmin) * plot_w if xmax > xmin else PAD_L

    def py(y):
        return PAD_T + plot_h - (y - ymin) / (ymax - ymin) * plot_h if ymax > ymin else PAD_T + plot_h

    return px, py, xmin, xmax, ymin, ymax, plot_h


def svg_chart(rows, xkey, ykey, color, y_label, x_label, fmt_y=lambda v: f"{v:g}"):
    px, py, xmin, xmax, ymin, ymax, plot_h = _scale(rows, xkey, ykey)
    pts = " ".join(f"{px(r[xkey]):.1f},{py(r[ykey]):.1f}" for r in rows)

    # y 軸格線 + 標籤（5 等分）
    grid = []
    for i in range(6):
        yval = ymin + (ymax - ymin) * i / 5
        y = py(yval)
        grid.append(f'<line x1="{PAD_L}" y1="{y:.1f}" x2="{W-PAD_R}" y2="{y:.1f}" '
                    f'stroke="#2a3039" stroke-width="1"/>')
        grid.append(f'<text x="{PAD_L-8}" y="{y+4:.1f}" text-anchor="end" '
                    f'fill="#9aa4b2" font-size="11">{fmt_y(yval)}</text>')
    # x 軸標籤（5 等分）
    for i in range(6):
        xval = xmin + (xmax - xmin) * i / 5
        x = px(xval)
        grid.append(f'<text x="{x:.1f}" y="{H-PAD_B+18}" text-anchor="middle" '
                    f'fill="#9aa4b2" font-size="11">{xval:g}</text>')

    # 面積填充（折線下方）
    area = f"{PAD_L},{H-PAD_B} {pts} {W-PAD_R},{H-PAD_B}"

    return f'''<svg viewBox="0 0 {W} {H}" width="100%" style="max-width:{W}px">
  <rect x="0" y="0" width="{W}" height="{H}" fill="#12151b" rx="8"/>
  {''.join(grid)}
  <polygon points="{area}" fill="{color}" opacity="0.10"/>
  <polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2"/>
  <text x="{PAD_L}" y="14" fill="#e6e9ef" font-size="12" font-weight="600">{y_label}</text>
  <text x="{W/2:.0f}" y="{H-6}" text-anchor="middle" fill="#9aa4b2" font-size="11">{x_label}</text>
</svg>'''


def main():
    if not os.path.exists(JSON_PATH):
        sys.exit(f"找不到資料：{JSON_PATH}（請先跑 measure_terrain_cache.py）")
    with open(JSON_PATH, encoding="utf-8") as f:
        data = json.load(f)
    rows = data["rows"]

    chart_tiles = svg_chart(rows, "distance_m", "cache_tiles", "#5b9dff",
                            "快取地塊數 (cache_tiles)", "行進距離 (m)")
    chart_mb = svg_chart(rows, "distance_m", "traced_current_mb", "#f4a949",
                         "Python heap 增量 (MB)", "行進距離 (m)",
                         fmt_y=lambda v: f"{v:.1f}")

    tiles_per_km = (data["tiles_end"] - data["tiles_start"]) / (data["distance_m"] / 1000)
    mb_per_km = data["traced_mb_end"] / (data["distance_m"] / 1000)

    html = f'''<!DOCTYPE html>
<html lang="zh-Hant"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>terrain_cache 久跑實測（T03-1）</title>
<style>
  body{{margin:0;background:#0f1115;color:#e6e9ef;font-family:"Segoe UI","Microsoft JhengHei",system-ui,sans-serif;line-height:1.65}}
  .wrap{{max-width:880px;margin:0 auto;padding:30px 22px 80px}}
  h1{{font-size:24px;margin:0 0 4px}} .sub{{color:#9aa4b2;font-size:14px}}
  h2{{font-size:19px;margin:34px 0 8px;padding-left:10px;border-left:4px solid #5b9dff}}
  code{{background:#0b0d11;padding:2px 6px;border-radius:5px;font-size:12.5px;color:#c9d4e3}}
  .mini{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:18px 0}}
  @media(max-width:640px){{.mini{{grid-template-columns:repeat(2,1fr)}}}}
  .b{{background:#171a21;border:1px solid #2a3039;border-radius:10px;padding:12px;text-align:center}}
  .n{{font-size:24px;font-weight:700}} .l{{font-size:12px;color:#9aa4b2;margin-top:2px}}
  .card{{background:#171a21;border:1px solid #2a3039;border-radius:11px;padding:16px;margin:12px 0}}
  .callout{{border:1px solid #2c3a52;background:linear-gradient(180deg,#1a2230,#171a21);border-radius:11px;padding:14px 16px;margin:14px 0;font-size:13.5px}}
  .callout.warn{{border-color:#5a4a1e;background:linear-gradient(180deg,#221d10,#1a1710)}}
  .green{{color:#36d399}} .amber{{color:#f4a949}} .blue{{color:#5b9dff}}
  ul{{padding-left:20px}} li{{margin:5px 0}}
</style></head><body><div class="wrap">

<h1>🗺️ terrain_cache 久跑記憶體成長實測</h1>
<div class="sub">2026-06-07 · T03-1 情境 A · headless 真實 TerrainManager（INFINITE，沿 +x 直線前進 {data["distance_m"]:.0f} m）</div>

<div class="mini">
  <div class="b"><div class="n blue">{data["tiles_start"]} → {data["tiles_end"]:,}</div><div class="l">快取地塊數</div></div>
  <div class="b"><div class="n amber">+{data["traced_mb_end"]:.2f} MB</div><div class="l">Python heap 增量</div></div>
  <div class="b"><div class="n">{tiles_per_km:.0f}</div><div class="l">地塊 / 公里</div></div>
  <div class="b"><div class="n">{data["tile_bytes_est"]} B</div><div class="l">單塊估計大小</div></div>
</div>

<div class="callout">
  <b>一句話：</b>快取地塊數隨行進距離<b>線性無上限成長</b>（永不淘汰），證實該上 <b>LRU 上界</b>。
  但同時也發現——<b>記憶體量其實很小</b>（2 萬塊僅約 4.5 MB），所以
  <b class="amber">「久跑超卡」的主因應該不是 terrain_cache 的記憶體壓力</b>，卡頓根因需另查。
</div>

<h2>① 快取地塊數 vs 行進距離</h2>
<div class="card">{chart_tiles}
<p class="sub" style="margin:8px 0 0">完美直線 → 每前進 5 m（一次網格滑動）就新增約 5 個邊界地塊，舊地塊永不釋放。走越遠、數量無上限累積。</p></div>

<h2>② Python heap 增量 vs 行進距離</h2>
<div class="card">{chart_mb}
<p class="sub" style="margin:8px 0 0">記憶體與地塊數同步線性爬升，但斜率小（約 {mb_per_km:.2f} MB/km）。<code>TerrainTile</code> 僅存兩個 int 與型別字串，故單塊極輕。</p></div>

<h2>③ 結論與重構意涵</h2>
<div class="card">
  <ul>
    <li><b class="green">確認</b>：<code>terrain_cache</code> 在 INFINITE 模式下隨移動<b>無上限成長</b>（<code>get_or_generate_tile</code> 只增不刪）。架構上應加 <b>LRU 有界淘汰</b>。</li>
    <li><b class="amber">修正既有假設</b>：成長雖無上限，但記憶體量級很小（{data["distance_m"]:.0f} m → +{data["traced_mb_end"]:.2f} MB）。<b>單純記憶體不是「久跑超卡」的合理主因</b>。</li>
    <li><b class="blue">下一步查卡頓真因</b>（另立調查）：每次網格滑動會重繪整張 hfield 並<b>重新呼叫 25 個地形生成器</b>（<code>update_hfield</code> 對 5×5 視窗每塊都重跑 generator）；再加上把 ~501×501 高度場寫回 <code>model.hfield_data</code> 並觸發物理/渲染同步——這些每次滑動的固定開銷，比快取大小更可能是卡頓來源。</li>
  </ul>
</div>

<div class="callout warn">
  <b>方法說明：</b>本實測不跑 policy、不渲染，純驅動 <code>TerrainManager.update()</code> 沿直線推進，以 <code>tracemalloc</code> 量 Python heap。
  量到的是「快取資料結構本身」的成長，已足以證明上界需求；卡頓的端到端效能需另用真實物理 + 渲染情境量測。
  原始資料：<code>docs/reports/data/terrain_cache_growth.csv</code> / <code>.json</code>。腳本：<code>tools/measure_terrain_cache.py</code>。
</div>

</div></body></html>'''

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"✅ 已產生：{OUT_PATH}")


if __name__ == "__main__":
    main()
