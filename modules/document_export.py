"""
会议文档导出：把 Markdown 纪要在同一目录导出成多种格式（md / txt / docx / pdf）。

为什么放在这里：用户要下载的不只是 Markdown —— 给领导看的要 Word 或 PDF，
复制到聊天工具里的要纯文本。导出文件与 .md 放在同一会议目录，文件名相同，
扩展名不同，按日期归档的目录结构不变。

只解析本系统自己生成的 Markdown 子集（一级/二级/三级标题、段落、有序/无序列表、
表格、引用、分隔线、行内加粗），不引入 Markdown 渲染引擎，避免离线环境缺依赖。
"""
import os
import re

# 支持的导出格式（接口与前端按这个顺序展示）
EXPORT_FORMATS = (
    ('md', 'Markdown', 'md'),
    ('txt', '纯文本', 'txt'),
    ('docx', 'Word 文档', 'docx'),
    ('pdf', 'PDF', 'pdf'),
)
FORMAT_LABELS = {key: label for key, label, _ in EXPORT_FORMATS}
FORMAT_EXTENSIONS = {key: ext for key, _, ext in EXPORT_FORMATS}

_INLINE_BOLD = re.compile(r'\*\*(.+?)\*\*')
_INLINE_CODE = re.compile(r'`(.+?)`')


# ---------------------------------------------------------------- Markdown 解析


def _strip_inline(text):
    """去掉行内标记（加粗、行内代码），用于纯文本导出。"""
    text = _INLINE_BOLD.sub(r'\1', text or '')
    return _INLINE_CODE.sub(r'\1', text)


def parse_blocks(markdown):
    """
    把 Markdown 切成块：('h1'|'h2'|'h3'|'p'|'bullet'|'number'|'table'|'quote'|'hr', payload)。

    表格 payload 是二维列表；其它是字符串。有序列表的 payload 是 (序号, 文本)：
    序号取自 Markdown 本身，保证每个章节各自从 1 开始（Word 的 List Number
    样式会跨章节连续编号，和原文对不上）。
    """
    blocks = []
    table = []
    paragraph = []

    def flush_paragraph():
        if paragraph:
            # 段落内的换行要保留：公文的信息项（会议名称/时间/地点…）是逐行写的
            blocks.append(('p', '\n'.join(paragraph)))
            paragraph.clear()

    def flush_table():
        if table:
            rows = [row for row in table
                    if not all(re.fullmatch(r':?-{2,}:?', cell or '') for cell in row)]
            blocks.append(('table', rows))
            table.clear()

    for raw in (markdown or '').replace('\r\n', '\n').split('\n'):
        line = raw.strip()
        if not line:
            flush_paragraph()
            flush_table()
            continue
        if line.startswith('|'):
            flush_paragraph()
            table.append([cell.strip() for cell in line.strip('|').split('|')])
            continue
        flush_table()
        if re.match(r'^#{1,6}\s', line):
            flush_paragraph()
            level = len(line) - len(line.lstrip('#'))
            blocks.append((f'h{min(level, 3)}', _strip_inline(line.lstrip('# ').strip())))
            continue
        if re.fullmatch(r'-{3,}|\*{3,}', line):
            flush_paragraph()
            blocks.append(('hr', ''))
            continue
        if line.startswith('>'):
            flush_paragraph()
            blocks.append(('quote', _strip_inline(line.lstrip('> ').strip())))
            continue
        match = re.match(r'^(\d+)\.\s+(.*)$', line)
        if match:
            flush_paragraph()
            blocks.append(('number', (int(match.group(1)), _strip_inline(match.group(2)))))
            continue
        if line.startswith('- '):
            flush_paragraph()
            blocks.append(('bullet', _strip_inline(line[2:])))
            continue
        paragraph.append(line)
    flush_paragraph()
    flush_table()
    return blocks


# ---------------------------------------------------------------- 纯文本


def to_plain_text(markdown):
    """Markdown → 纯文本（标题加序号感，表格用「列：值」）。"""
    lines = []
    for kind, payload in parse_blocks(markdown):
        if kind == 'h1':
            lines.extend(['', payload, '=' * min(60, max(8, len(payload) * 2)), ''])
        elif kind == 'h2':
            lines.extend(['', payload, '-' * min(60, max(8, len(payload) * 2))])
        elif kind == 'h3':
            lines.append(payload)
        elif kind == 'p':
            lines.append(payload)
        elif kind == 'bullet':
            lines.append(f'· {payload}')
        elif kind == 'number':
            number, text = payload
            lines.append(f'  {number}. {text}')
        elif kind == 'quote':
            lines.append(f'  {payload}')
        elif kind == 'hr':
            lines.append('')
        elif kind == 'table':
            for index, row in enumerate(payload):
                if index == 0:
                    lines.append('  '.join(row))
                    lines.append('  ' + '  '.join('-' * max(2, len(cell)) for cell in row))
                else:
                    lines.append('  '.join(row))
    return '\n'.join(line.rstrip() for line in lines).strip() + '\n'


# ---------------------------------------------------------------- Word


_CJK_FONT = '微软雅黑'


def _set_docx_font(run, size=None, bold=None, color=None):
    from docx.oxml.ns import qn
    from docx.shared import Pt, RGBColor

    run.font.name = _CJK_FONT
    run._element.rPr.rFonts.set(qn('w:eastAsia'), _CJK_FONT)
    if size:
        run.font.size = Pt(size)
    if bold is not None:
        run.font.bold = bold
    if color is not None:
        run.font.color.rgb = RGBColor(*color)


def to_docx(markdown, output_path):
    """Markdown → Word 文档（带标题层级、表格与中文样式）。"""
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Cm, Pt

    document = Document()
    style = document.styles['Normal']
    style.font.name = _CJK_FONT
    style.font.size = Pt(10.5)
    document.styles['Normal']._element.rPr.rFonts.set(
        __import__('docx').oxml.ns.qn('w:eastAsia'), _CJK_FONT)

    for section in document.sections:
        section.top_margin = Cm(2.4)
        section.bottom_margin = Cm(2.2)
        section.left_margin = Cm(2.6)
        section.right_margin = Cm(2.4)

    for kind, payload in parse_blocks(markdown):
        if kind == 'h1':
            paragraph = document.add_paragraph()
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            paragraph.paragraph_format.space_after = Pt(14)
            _set_docx_font(paragraph.add_run(payload), size=20, bold=True)
        elif kind == 'h2':
            paragraph = document.add_paragraph()
            paragraph.paragraph_format.space_before = Pt(12)
            paragraph.paragraph_format.space_after = Pt(6)
            _set_docx_font(paragraph.add_run(payload), size=13, bold=True)
        elif kind == 'h3':
            paragraph = document.add_paragraph()
            _set_docx_font(paragraph.add_run(payload), size=11.5, bold=True)
        elif kind == 'p':
            paragraph = document.add_paragraph()
            paragraph.paragraph_format.line_spacing = 1.5
            paragraph.paragraph_format.space_after = Pt(4)
            for line_index, line in enumerate(payload.split('\n')):
                if line_index:
                    paragraph.add_run().add_break()
                _append_inline(paragraph, line, _set_docx_font)
        elif kind == 'bullet':
            paragraph = document.add_paragraph(style='List Bullet')
            paragraph.paragraph_format.line_spacing = 1.4
            _append_inline(paragraph, payload, _set_docx_font)
        elif kind == 'number':
            number_value, text = payload
            paragraph = document.add_paragraph()
            paragraph.paragraph_format.left_indent = Cm(0.75)
            paragraph.paragraph_format.first_line_indent = Cm(-0.5)
            paragraph.paragraph_format.line_spacing = 1.4
            _set_docx_font(paragraph.add_run(f'{number_value}. '))
            _append_inline(paragraph, text, _set_docx_font)
        elif kind == 'quote':
            paragraph = document.add_paragraph()
            paragraph.paragraph_format.left_indent = Cm(0.6)
            _set_docx_font(paragraph.add_run(payload), size=9, color=(0x88, 0x88, 0x88))
        elif kind == 'hr':
            document.add_paragraph('')
        elif kind == 'table' and payload:
            columns = max(len(row) for row in payload)
            table = document.add_table(rows=len(payload), cols=columns)
            table.style = 'Light Grid Accent 1'
            for row_index, row in enumerate(payload):
                for column_index in range(columns):
                    cell = table.cell(row_index, column_index)
                    cell.text = ''
                    text = row[column_index] if column_index < len(row) else ''
                    _append_inline(cell.paragraphs[0], text, _set_docx_font,
                                   size=9.5, bold=(row_index == 0))
            document.add_paragraph('')

    document.save(output_path)
    return output_path


def _append_inline(paragraph, text, setter, size=None, bold=None):
    """把含 **加粗** 的文本写进段落（保留部分加粗）。"""
    position = 0
    for match in _INLINE_BOLD.finditer(text or ''):
        if match.start() > position:
            setter(paragraph.add_run(text[position:match.start()]), size=size, bold=bold)
        setter(paragraph.add_run(match.group(1)), size=size, bold=True)
        position = match.end()
    if position < len(text or ''):
        setter(paragraph.add_run(text[position:]), size=size, bold=bold)


# ---------------------------------------------------------------- PDF


_PDF_FONT = None


def _register_pdf_font():
    """注册一个能显示中文的字体（优先嵌入系统字体，保证换机器也能看）。"""
    global _PDF_FONT
    if _PDF_FONT:
        return _PDF_FONT
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    candidates = [
        (r'C:\Windows\Fonts\msyh.ttc', 'MSYaHei', 0),
        (r'C:\Windows\Fonts\simsun.ttc', 'SimSun', 0),
        (r'C:\Windows\Fonts\simhei.ttf', 'SimHei', None),
        (r'C:\Windows\Fonts\Deng.ttf', 'DengXian', None),
    ]
    for path, name, index in candidates:
        if not os.path.exists(path):
            continue
        try:
            if index is None:
                pdfmetrics.registerFont(TTFont(name, path))
            else:
                pdfmetrics.registerFont(TTFont(name, path, subfontIndex=index))
            _PDF_FONT = name
            return name
        except Exception:
            continue
    # 兜底：reportlab 自带的中文 CID 字体（不嵌入，依赖阅读器字体）
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont

    pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'))
    _PDF_FONT = 'STSong-Light'
    return _PDF_FONT


def to_pdf(markdown, output_path, title=None):
    """Markdown → PDF（A4，中文标题居中，表格自动换列宽）。"""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (HRFlowable, Paragraph, SimpleDocTemplate,
                                    Spacer, Table, TableStyle)

    font = _register_pdf_font()
    styles = {
        'h1': ParagraphStyle('h1', fontName=font, fontSize=19, leading=26,
                             alignment=1, spaceAfter=16),
        'h2': ParagraphStyle('h2', fontName=font, fontSize=13, leading=20,
                             spaceBefore=12, spaceAfter=6),
        'h3': ParagraphStyle('h3', fontName=font, fontSize=11.5, leading=18,
                             spaceBefore=8, spaceAfter=4),
        'p': ParagraphStyle('p', fontName=font, fontSize=10.5, leading=17, spaceAfter=4),
        'li': ParagraphStyle('li', fontName=font, fontSize=10.5, leading=17,
                             leftIndent=14, spaceAfter=2),
        'quote': ParagraphStyle('quote', fontName=font, fontSize=9, leading=14,
                                leftIndent=12, textColor=colors.HexColor('#777777')),
    }

    def escape(text):
        return (text or '').replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

    def inline(text):
        escaped = escape(text)
        return _INLINE_BOLD.sub(r'<b>\1</b>', escaped)

    story = []
    for kind, payload in parse_blocks(markdown):
        if kind == 'h1':
            story.append(Paragraph(inline(payload), styles['h1']))
        elif kind == 'h2':
            story.append(Paragraph(inline(payload), styles['h2']))
        elif kind == 'h3':
            story.append(Paragraph(inline(payload), styles['h3']))
        elif kind == 'p':
            story.append(Paragraph(inline(payload).replace('\n', '<br/>'), styles['p']))
        elif kind == 'bullet':
            story.append(Paragraph('• ' + inline(payload), styles['li']))
        elif kind == 'number':
            number_value, text = payload
            story.append(Paragraph(f'{number_value}. ' + inline(text), styles['li']))
        elif kind == 'quote':
            story.append(Paragraph(inline(payload), styles['quote']))
        elif kind == 'hr':
            story.append(Spacer(1, 6))
            story.append(HRFlowable(width='100%', color=colors.HexColor('#dddddd')))
            story.append(Spacer(1, 6))
        elif kind == 'table' and payload:
            columns = max(len(row) for row in payload)
            data = []
            for row in payload:
                cells = [inline(row[i]) if i < len(row) else '' for i in range(columns)]
                data.append([Paragraph(cell, styles['p']) for cell in cells])
            table = Table(data, colWidths=[(17 * cm) / columns] * columns, repeatRows=1)
            table.setStyle(TableStyle([
                ('FONTNAME', (0, 0), (-1, -1), font),
                ('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor('#dddddd')),
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f5f6f8')),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ]))
            story.append(table)
            story.append(Spacer(1, 8))

    if not story:
        story = [Paragraph('（空文档）', styles['p'])]

    document = SimpleDocTemplate(
        output_path, pagesize=A4,
        leftMargin=2.4 * cm, rightMargin=2.4 * cm,
        topMargin=2.2 * cm, bottomMargin=2.0 * cm,
        title=title or '',
    )
    document.build(story)
    return output_path


# ---------------------------------------------------------------- 统一入口


def export_document(markdown_path, fmt, output_path=None, title=None):
    """
    把 Markdown 文档导出为指定格式，返回输出文件路径。

    fmt 支持 md / txt / docx / pdf；未知格式抛 ValueError。
    """
    fmt = (fmt or 'md').lower().lstrip('.')
    if fmt not in FORMAT_EXTENSIONS:
        raise ValueError(f'不支持的导出格式: {fmt}')
    source = open(markdown_path, encoding='utf-8-sig').read()

    if fmt == 'md':
        return markdown_path

    if output_path is None:
        base, _ = os.path.splitext(markdown_path)
        output_path = f'{base}.{FORMAT_EXTENSIONS[fmt]}'

    if fmt == 'txt':
        with open(output_path, 'w', encoding='utf-8-sig') as handle:
            handle.write(to_plain_text(source))
        return output_path
    if fmt == 'docx':
        return to_docx(source, output_path)
    return to_pdf(source, output_path, title=title)
