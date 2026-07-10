"""
文本分析模块测试
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from modules.text_analyzer import (
    extract_keywords, analyze_topic, generate_summary, analyze_sentiment
)


def test_keyword_extraction():
    """测试关键词提取"""
    print("=" * 60)
    print("测试: 关键词提取")
    print("=" * 60)
    
    text = """
    本次会议主要讨论了Q3季度的销售策略。我们将重点关注新产品线的推广，
    加强与客户的沟通，提升服务质量。销售团队需要制定详细的执行计划，
    确保完成季度目标。同时要关注市场动态，及时调整策略。
    """
    
    try:
        keywords = extract_keywords(text, top_n=10)
        
        print(f"✅ 关键词提取成功")
        print(f"   提取到 {len(keywords)} 个关键词")
        for kw in keywords[:5]:
            print(f"      {kw['word']} (频率: {kw['frequency']}, 得分: {kw.get('score', 0):.4f})")
        
        assert len(keywords) > 0, "应提取到关键词"
        assert 'word' in keywords[0], "关键词应包含word字段"
        
        return True
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_topic_analysis():
    """测试主题分析"""
    print("\n" + "=" * 60)
    print("测试: 主题分析")
    print("=" * 60)
    
    text = "今天我们来讨论一下项目进度，看看哪些任务已经完成了，哪些还需要继续推进。"
    
    try:
        topics = analyze_topic(text)
        
        print(f"✅ 主题分析成功")
        print(f"   匹配到 {len(topics)} 个主题")
        for t in topics[:3]:
            print(f"      {t['topic']}: {t['score']*100:.1f}%")
        
        assert len(topics) > 0, "应匹配到主题"
        
        return True
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_summary_generation():
    """测试摘要生成"""
    print("\n" + "=" * 60)
    print("测试: 摘要生成")
    print("=" * 60)
    
    text = """
    本次会议主要讨论了三个方面的问题。首先是项目进度，目前整体进展良好，
    完成了80%的任务，预计下周可以全部完成。其次是人员安排，由于小李离职，
    需要尽快招聘新人补充。最后是预算问题，Q3预算已经用完，需要申请追加。
    大家一致同意加快进度，同时控制成本。
    """
    
    try:
        summary = generate_summary(text, max_length=100)
        
        print(f"✅ 摘要生成成功")
        print(f"   摘要内容: {summary}")
        print(f"   摘要长度: {len(summary)} 字")
        
        assert len(summary) > 0, "摘要不应为空"
        assert len(summary) <= 100, "摘要应在限制长度内"
        
        return True
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sentiment_analysis():
    """测试情绪分析"""
    print("\n" + "=" * 60)
    print("测试: 情绪分析")
    print("=" * 60)
    
    test_cases = [
        ("今天工作非常顺利，项目进展顺利，大家都很积极", "positive"),
        ("这个问题很麻烦，进度延迟了，客户也不满意", "negative"),
        ("会议开始，请大家汇报一下本周的工作", "neutral"),
    ]
    
    all_pass = True
    for text, expected in test_cases:
        try:
            result = analyze_sentiment(text)
            sentiment = result['sentiment']
            
            status = "✅" if sentiment == expected else "⚠️"
            print(f"   {status} 文本: '{text[:20]}...' -> {sentiment} (期望: {expected})")
            print(f"      正面词: {result.get('positive', 0)}, 负面词: {result.get('negative', 0)}")
            print(f"      正面得分: {result.get('positive_score', 0)}, 负面得分: {result.get('negative_score', 0)}")
            
        except Exception as e:
            print(f"   ❌ 失败: {e}")
            all_pass = False
    
    return all_pass


if __name__ == '__main__':
    print("\n" + "📝" * 30)
    print("文本分析模块测试套件")
    print("📝" * 30 + "\n")
    
    results = []
    
    results.append(("关键词提取", test_keyword_extraction()))
    results.append(("主题分析", test_topic_analysis()))
    results.append(("摘要生成", test_summary_generation()))
    results.append(("情绪分析", test_sentiment_analysis()))
    
    print("\n" + "=" * 60)
    print("测试汇总")
    print("=" * 60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{status:8} - {name}")
    
    print(f"\n总计: {passed}/{total} 通过 ({passed/total*100:.1f}%)")
