"""Fix the CuEq calculation in run_pipeline.py."""
path = r'D:\project\mymine\run_pipeline.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

old = """# === 7.1 CuEq 计算与矿化段圈定 ===
# CuEq = Cu% + 8.33 * Mo%
df_ana['CuEq'] = df_ana['Cu %'] + 8.33 * df_ana['Mo %']"""

new = """# === 7.1 CuEq 计算与矿化段圈定 ===
# 确保数值类型（原始数据可能含字符串如 "<0.001" 或空值）
for col in ['Cu %', 'Mo %', 'Au ppm', 'Ag ppm', 'Pb %', 'Zn %']:
    df_ana[col] = pd.to_numeric(df_ana[col], errors='coerce')
print(f'Cu % 缺失: {df_ana["Cu %"].isna().sum()}, Mo % 缺失: {df_ana["Mo %"].isna().sum()}')
# CuEq = Cu% + 8.33 * Mo%
df_ana['CuEq'] = df_ana['Cu %'] + 8.33 * df_ana['Mo %']"""

if old in content:
    content = content.replace(old, new)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print("FIXED: CuEq calculation updated with numeric conversion")
else:
    print("ERROR: old string not found")
    # Debug: show what's around line 338-341
    lines = content.split('\n')
    for i in range(335, 345):
        if i < len(lines):
            print(f"  L{i+1}: {repr(lines[i])}")
