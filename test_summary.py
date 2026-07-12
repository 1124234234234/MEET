import sys
sys.path.insert(0, '.')
from modules.text_analyzer import generate_summary

text = """各位同事大家好，今天我们来讨论一下新推出的理财产品销售方案。请客户经理小王来介绍一下产品情况。
好的，谢谢大家。这款理财产品年化收益率可以达到8%以上。而且我们公司承诺保本保收益，绝对安全，大家可以放心推荐给客户。
不过我有个问题，如果客户问起来，我可以说这款产品基本不会亏钱的吧？
不行，绝对不可以。我们必须向客户说明产品的风险等级是中高风险。
还有，必须提醒客户，过往业绩不代表未来表现。
对，我们必须如实告知客户投资风险，不得使用任何可能误导客户的表述。
好的，我明白了。我们会严格遵守合规要求，做好风险揭示工作。
嗯，大家还有什么问题吗？
没有了。
好，那今天的会议就到这里，散会。"""

print('Input text length:', len(text))
print('---')
result = generate_summary(text, max_length=300)
print('Result:')
print(result)
print('---')
print('Result length:', len(result))
