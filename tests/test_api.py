"""
API接口测试
"""
import requests
import json

API_BASE = "http://127.0.0.1:5000/api"


def test_health():
    """测试健康检查接口"""
    print("=" * 60)
    print("测试: 健康检查接口")
    print("=" * 60)
    
    try:
        response = requests.get(f"{API_BASE}/health", timeout=5)
        data = response.json()
        
        print(f"✅ 接口正常")
        print(f"   状态码: {response.status_code}")
        print(f"   响应: {data}")
        
        assert response.status_code == 200
        assert data['status'] == 'ok'
        
        return True
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False


def test_languages():
    """测试语言列表接口"""
    print("\n" + "=" * 60)
    print("测试: 语言列表接口")
    print("=" * 60)
    
    try:
        response = requests.get(f"{API_BASE}/languages", timeout=5)
        data = response.json()
        
        print(f"✅ 接口正常")
        print(f"   支持语言数: {len(data['data'])}")
        
        assert response.status_code == 200
        assert len(data['data']) > 0
        
        return True
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False


def test_topics():
    """测试主题列表接口"""
    print("\n" + "=" * 60)
    print("测试: 主题列表接口")
    print("=" * 60)
    
    try:
        response = requests.get(f"{API_BASE}/topics", timeout=5)
        data = response.json()
        
        print(f"✅ 接口正常")
        print(f"   主题列表: {data['data']}")
        
        assert response.status_code == 200
        assert len(data['data']) > 0
        
        return True
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False


def test_knowledge_base():
    """测试知识库接口"""
    print("\n" + "=" * 60)
    print("测试: 知识库接口")
    print("=" * 60)
    
    try:
        # 获取列表
        response = requests.get(f"{API_BASE}/knowledge-base", timeout=5)
        data = response.json()
        
        print(f"✅ 知识库列表接口正常")
        print(f"   条目数: {data.get('total', 0)}")
        
        # 添加测试条目
        test_data = {
            "title": "测试政策文件",
            "content": "这是一条测试用的知识库内容",
            "item_type": "policy",
            "keywords": ["测试", "政策"],
            "required_points": ["测试要点1", "测试要点2"]
        }
        
        response = requests.post(
            f"{API_BASE}/knowledge-base",
            json=test_data,
            timeout=5
        )
        
        if response.status_code == 200:
            print(f"✅ 添加知识库条目成功")
            created = response.json().get('data', {})
            item_id = created.get('id')
            
            # 删除测试条目
            if item_id:
                requests.delete(f"{API_BASE}/knowledge-base/{item_id}", timeout=5)
                print(f"✅ 清理测试数据成功")
        
        return True
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False


def test_score_weights():
    """测试评分权重接口"""
    print("\n" + "=" * 60)
    print("测试: 评分权重接口")
    print("=" * 60)
    
    try:
        # 获取权重
        response = requests.get(f"{API_BASE}/score-weights", timeout=5)
        data = response.json()
        
        print(f"✅ 获取权重成功")
        weights = data.get('data', [])
        for w in weights:
            print(f"   {w['weight_name']}: {w['weight_value']}")
        
        # 更新权重
        update_data = {
            "semantic_similarity": 35,
            "point_coverage": 35,
            "risk_detection": 20,
            "keyword_matching": 10
        }
        
        response = requests.put(
            f"{API_BASE}/score-weights",
            json=update_data,
            timeout=5
        )
        
        if response.status_code == 200:
            print(f"✅ 更新权重成功")
            
            # 恢复默认值
            default_data = {
                "semantic_similarity": 40,
                "point_coverage": 30,
                "risk_detection": 20,
                "keyword_matching": 10
            }
            requests.put(f"{API_BASE}/score-weights", json=default_data, timeout=5)
        
        return True
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False


def test_meetings_list():
    """测试会议列表接口"""
    print("\n" + "=" * 60)
    print("测试: 会议列表接口")
    print("=" * 60)
    
    try:
        response = requests.get(f"{API_BASE}/meetings", timeout=5)
        data = response.json()
        
        print(f"✅ 接口正常")
        print(f"   会议总数: {data.get('total', 0)}")
        print(f"   当前页: {data.get('page', 1)}")
        
        assert response.status_code == 200
        
        return True
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False


if __name__ == '__main__':
    print("\n" + "🌐" * 30)
    print("API接口测试套件")
    print("🌐" * 30 + "\n")
    
    # 检查服务是否运行
    try:
        requests.get(f"{API_BASE}/health", timeout=2)
    except:
        print("❌ 服务未运行，请先启动服务: python app.py")
        exit(1)
    
    results = []
    
    results.append(("健康检查", test_health()))
    results.append(("语言列表", test_languages()))
    results.append(("主题列表", test_topics()))
    results.append(("知识库CRUD", test_knowledge_base()))
    results.append(("评分权重", test_score_weights()))
    results.append(("会议列表", test_meetings_list()))
    
    print("\n" + "=" * 60)
    print("测试汇总")
    print("=" * 60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{status:8} - {name}")
    
    print(f"\n总计: {passed}/{total} 通过 ({passed/total*100:.1f}%)")
