#!/usr/bin/env python3
import json
import os
import subprocess
import sys

# 调用 conda 获取所有环境列表（JSON 格式）
try:
    info = subprocess.run(["conda", "env", "list", "--json"], capture_output=True, text=True, check=True).stdout
except subprocess.CalledProcessError as e:
    print("Error 调用 conda 时出错：", e, file=sys.stderr)
    sys.exit(1)

data = json.loads(info)
envs = data.get("envs", [])

print("开始检测各环境中 usdcat 可执行性…")

for path in envs:
    name = os.path.basename(path)
    usdcat = "usdcat"  # os.path.join(path, "bin", "usdcat")
    if os.path.isfile(usdcat) and os.access(usdcat, os.X_OK):
        # 试运行 usdcat --help
        try:
            subprocess.run([usdcat, "--help"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            print(f"找到：环境 '{name}' （路径：{path}）中 usdcat 可正常执行")
        except subprocess.CalledProcessError:
            print(f"未找到：环境 '{name}' 中的 usdcat 不能正常执行（执行失败）")
    else:
        print(f"未找到：环境 '{name}' 中不存在 usdcat 可执行文件")
