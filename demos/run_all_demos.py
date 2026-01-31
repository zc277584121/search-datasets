#!/usr/bin/env python3
"""
Run all Milvus search demos in non-interactive mode.
批量运行所有 Milvus 搜索 demo（非交互模式）。
"""

import sys
import importlib.util
from pathlib import Path


def run_demo_module(module_path: Path, demo_name: str):
    """Run a demo module's main function with limited interaction."""
    print(f"\n{'='*70}")
    print(f"运行 Demo: {demo_name}")
    print(f"{'='*70}")

    try:
        # Load module
        spec = importlib.util.spec_from_file_location(demo_name, module_path)
        module = importlib.util.module_from_spec(spec)

        # Monkey-patch input to skip interactive mode
        original_input = __builtins__["input"] if isinstance(__builtins__, dict) else __builtins__.input

        def mock_input(prompt=""):
            print(f"{prompt}[跳过交互模式]")
            return "quit"

        if isinstance(__builtins__, dict):
            __builtins__["input"] = mock_input
        else:
            __builtins__.input = mock_input

        try:
            spec.loader.exec_module(module)
            if hasattr(module, "main"):
                module.main()
        finally:
            # Restore input
            if isinstance(__builtins__, dict):
                __builtins__["input"] = original_input
            else:
                __builtins__.input = original_input

        print(f"\n✓ {demo_name} 完成!")
        return True

    except Exception as e:
        print(f"\n✗ {demo_name} 失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 70)
    print("Milvus 向量搜索 Demo 批量运行")
    print("=" * 70)

    demos_dir = Path(__file__).parent

    # List of demos to run
    demo_files = [
        "demo_squad2.py",
        "demo_cmrc2018.py",
        "demo_multihop_rag.py",
        "demo_eli5_reddit.py",
        "demo_wildchat.py",
        "demo_github_issues.py",
        "demo_enron_email.py",
        "demo_coco_captions.py",
        "demo_msvd_video.py",
        "demo_audiocaps.py",
        "demo_discord_chat.py",
        "demo_spider_sql.py",
    ]

    results = {}

    for demo_file in demo_files:
        demo_path = demos_dir / demo_file
        if demo_path.exists():
            demo_name = demo_file.replace(".py", "")
            success = run_demo_module(demo_path, demo_name)
            results[demo_name] = success
        else:
            print(f"跳过 {demo_file}: 文件不存在")
            results[demo_file] = False

    # Summary
    print("\n" + "=" * 70)
    print("运行结果汇总")
    print("=" * 70)

    success_count = sum(1 for v in results.values() if v)
    total_count = len(results)

    for name, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {name}")

    print(f"\n总计: {success_count}/{total_count} 成功")


if __name__ == "__main__":
    main()
