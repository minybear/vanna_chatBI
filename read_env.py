#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("=" * 60)
print(".env 文件完整内容:")
print("=" * 60)

try:
    with open('.env', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for i, line in enumerate(lines, 1):
            # 显示行号和内容
            print(f"{i:3d} | {line}", end='')
            if not line.endswith('\n'):
                print()  # 如果最后一行没有换行符，手动添加
    
    print("\n" + "=" * 60)
    print(f"总共 {len(lines)} 行")
    print("=" * 60)
    
    # 查找LARK相关行
    lark_lines = [(i, line) for i, line in enumerate(lines, 1) if 'LARK' in line.upper()]
    if lark_lines:
        print(f"\n找到 {len(lark_lines)} 行包含 'LARK':")
        for line_num, content in lark_lines:
            print(f"  第 {line_num} 行: {content.strip()}")
    else:
        print("\n❌ 未找到包含 'LARK' 的配置行")
        print("   请在 .env 文件末尾添加:")
        print("   LARK_WEBHOOK_URL=https://open.feishu.cn/open-apis/bot/v2/hook/8ff07b7a-f947-4724-a73e-0d9cac275dc1")
        
except FileNotFoundError:
    print("❌ .env 文件不存在")
except Exception as e:
    print(f"❌ 读取文件出错: {e}")
