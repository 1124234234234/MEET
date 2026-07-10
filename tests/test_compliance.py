"""
合规检查模块测试
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from modules.compliance_checker import (
    calculate_compliance_score, get_score_level,
    realtime_compliance_check
)


def test_compliance_score_calculation():
    """测试合规评分计算"""
    print("=" * 60)
    print("测试: 合规评分计算")
    print("=" * 60)
    
    # 模拟知识库条目 (keywords 和 required_points 需为 JSON 格式)
    knowledge_items = [
        type('KB', (), {
            'id': 1,
            'title': '风险告知',
            'content': '投资有风险，入市需谨慎',
            'required_points': '["投资风险提示", "风险等级说明"]',
            'keywords': '["风险", "投资", "谨慎"]',
            'status': 'active',
            'item_type': 'required'
        })(),
        type('KB', (), {
            'id': 2,
            'title': '禁止承诺',
            'content': '不得承诺保本保收益',
            'required_points': '["禁止保本", "禁止保收益"]',
            'keywords': '["保本", "保收益", "承诺"]',
            'status': 'active',
            'item_type': 'forbidden'
        })()
    ]
    
    # 测试文本 - 包含部分要点和风险内容
    test_text = "各位投资者，投资有风险入市需谨慎。今天我们讨论产品的风险等级。请注意保本保收益是不可能的。"
    
    try:
        result = calculate_compliance_score(test_text, knowledge_items)
        
        print(f"✅ 合规评分计算成功")
        print(f"   总分: {result['total_score']}")
        print(f"   等级: {get_score_level(result['total_score'])}")
        print(f"   语义相似度: {result['components']['semantic_similarity']:.1f}")
        print(f"   要点覆盖: {result['components']['point_coverage']:.1f}")
        print(f"   风险检测: {result['components']['risk_detection']:.1f}")
        print(f"   关键词命中: {result['components']['keyword_matching']:.1f}")
        print(f"   遗漏要点: {result['missing_points']}")
        print(f"   风险关键词: {result['risk_keywords_found']}")
        print(f"   建议: {result['suggestions']}")
        
        # 验证基本逻辑
        assert 0 <= result['total_score'] <= 100, "评分应在0-100之间"
        assert isinstance(result['components'], dict), "components应为字典"
        assert isinstance(result['missing_points'], list), "missing_points应为列表"
        
        return True
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_realtime_compliance():
    """测试实时合规检查"""
    print("\n" + "=" * 60)
    print("测试: 实时合规检查")
    print("=" * 60)
    
    knowledge_items = [
        type('KB', (), {
            'id': 1,
            'title': '禁止承诺',
            'content': '不得承诺保本保收益',
            'required_points': '["禁止保本"]',
            'keywords': '["保本", "保收益", "承诺"]',
            'status': 'active',
            'item_type': 'forbidden'
        })()
    ]
    
    # 包含风险内容的文本
    test_segment = "这个产品绝对保本保收益，没有任何风险。"
    
    try:
        result = realtime_compliance_check(
            test_segment, knowledge_items, 0.0, 5.0
        )
        
        print(f"✅ 实时合规检查成功")
        print(f"   是否有风险: {result['has_risk']}")
        print(f"   风险项: {len(result['risk_items'])} 个")
        for item in result['risk_items']:
            print(f"      - 关键词: {item['keyword']}, 严重级别: {item['severity']}")
        print(f"   告警: {len(result['alerts'])} 个")
        for alert in result['alerts']:
            print(f"      - {alert['message']}")
        print(f"   要点覆盖: {len(result['covered_points'])} 个")
        
        # 验证风险检测
        assert result['has_risk'], "应检测到风险内容"
        
        return True
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_score_level():
    """测试评分等级"""
    print("\n" + "=" * 60)
    print("测试: 评分等级判断")
    print("=" * 60)
    
    test_cases = [
        (95, "优秀"),
        (85, "良好"),
        (70, "合格"),
        (50, "不合格"),
        (0, "不合格"),
        (100, "优秀")
    ]
    
    all_pass = True
    for score, expected in test_cases:
        actual = get_score_level(score)
        status = "✅" if actual == expected else "❌"
        print(f"   {status} 分数{score} -> {actual} (期望: {expected})")
        if actual != expected:
            all_pass = False
    
    return all_pass


def test_edge_cases():
    """测试边界情况"""
    print("\n" + "=" * 60)
    print("测试: 边界情况")
    print("=" * 60)
    
    results = []
    
    # 空文本
    try:
        result = calculate_compliance_score("", [])
        print(f"✅ 空文本处理成功: 分数={result['total_score']}")
        results.append(True)
    except Exception as e:
        print(f"❌ 空文本处理失败: {e}")
        results.append(False)
    
    # 无知识库
    try:
        result = calculate_compliance_score("测试文本", [])
        print(f"✅ 无知识库处理成功: 分数={result['total_score']}")
        results.append(True)
    except Exception as e:
        print(f"❌ 无知识库处理失败: {e}")
        results.append(False)
    
    # 长文本
    try:
        long_text = "会议记录。" * 100
        result = calculate_compliance_score(long_text, [])
        print(f"✅ 长文本处理成功: 分数={result['total_score']}")
        results.append(True)
    except Exception as e:
        print(f"❌ 长文本处理失败: {e}")
        results.append(False)
    
    return all(results)


if __name__ == '__main__':
    print("\n" + "📋" * 30)
    print("合规检查模块测试套件")
    print("📋" * 30 + "\n")
    
    results = []
    
    results.append(("合规评分计算", test_compliance_score_calculation()))
    results.append(("实时合规检查", test_realtime_compliance()))
    results.append(("评分等级判断", test_score_level()))
    results.append(("边界情况", test_edge_cases()))
    
    print("\n" + "=" * 60)
    print("测试汇总")
    print("=" * 60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{status:8} - {name}")
    
    print(f"\n总计: {passed}/{total} 通过 ({passed/total*100:.1f}%)")
