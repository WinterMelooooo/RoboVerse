import os

try:
    cwd = os.getcwd()
    print("当前工作目录：", cwd)
except FileNotFoundError as e:
    # e.filename 就是那个“被删掉”或“丢失”的目录
    print("不存在的目录是：", e.filename)
