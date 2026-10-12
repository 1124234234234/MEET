"""
推送本地提交到 GitHub（走 api.github.com 的 Git Data API）。

为什么需要这个脚本：本机到 github.com 的 git 端口不稳定（实测经常连不上，
DNS 也可能被污染），但 api.github.com 往往可达。于是按 Git 数据模型逐层构造
blob → tree → commit → 更新 ref，让远端提交哈希与本地完全一致；之后在网络
正常的地方直接 `git push` / `git pull` 都不会分叉。

两个踩过的坑：
  1. 路径：git 默认把非 ASCII 路径加引号并转义（core.quotepath），必须用
     `ls-tree -r -z` 取原始字节，否则会把转义后的假文件名写进树里；
  2. 时间：GitHub API 要求 ISO 8601 带冒号的时区（+08:00），不接受 git 的 +0800。

凭据从 `git credential fill` 读取（就是 git 已保存的那份），只放在请求头里，
不打印、不落盘。若远端 HEAD 与本地父提交不一致（别人推过东西）会直接停下，
不会强推覆盖。

用法：
    python push_to_github.py                # 推送当前分支
    python push_to_github.py --branch main  # 指定分支
    python push_to_github.py --retry 4      # 网络不通时最多重试 4 次（每次间隔 45 秒）
"""
import argparse
import base64
import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
OWNER, REPO = '1124234234234', 'MEET'


def sh(*args, binary=False):
    out = subprocess.run(['git', *args], cwd=REPO_DIR, capture_output=True)
    if out.returncode != 0:
        raise RuntimeError(f'git {" ".join(args)}: {out.stderr.decode("utf-8", "replace").strip()}')
    return out.stdout if binary else out.stdout.decode('utf-8', 'replace')


def get_token():
    env = dict(os.environ, GCM_INTERACTIVE='never', GIT_TERMINAL_PROMPT='0')
    proc = subprocess.run(['git', 'credential', 'fill'],
                          input=b'protocol=https\nhost=github.com\n\n',
                          capture_output=True, cwd=REPO_DIR, env=env, timeout=30)
    for line in proc.stdout.decode('utf-8', 'replace').split('\n'):
        if line.startswith('password='):
            return line.split('=', 1)[1].strip()
    raise RuntimeError('未能从凭据管理器取到 GitHub token（先配置一次 git 凭据）')


HDR = {'Accept': 'application/vnd.github+json', 'User-Agent': 'voice-reco-push'}


def api(method, path, body=None, token=None):
    data = json.dumps(body).encode('utf-8') if body is not None else None
    request = urllib.request.Request(f'https://api.github.com{path}', method=method,
                                     data=data, headers=dict(HDR))
    if token:
        request.add_header('Authorization', f'Bearer {token}')
    if data:
        request.add_header('Content-Type', 'application/json')
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.status, json.loads(response.read().decode('utf-8') or '{}')
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode('utf-8', 'replace')


def api_reachable():
    try:
        sock = socket.create_connection(('api.github.com', 443), timeout=12)
        sock.close()
        return True
    except Exception:
        return False


def parse_ident(line):
    match = re.match(r'^(.*) <(.*)> (\d+) ([+-])(\d{2})(\d{2})$', line)
    if not match:
        raise RuntimeError(f'身份行解析失败: {line}')
    name, email, stamp, sign, hours, minutes = match.groups()
    offset = int(hours) * 3600 + int(minutes) * 60
    if sign == '-':
        offset = -offset
    iso = time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime(int(stamp) + offset))
    return {'name': name, 'email': email, 'date': f'{iso}{sign}{hours}:{minutes}'}


def push(branch, token):
    head_sha = sh('rev-parse', 'HEAD').strip()
    status, ref = api('GET', f'/repos/{OWNER}/{REPO}/git/ref/heads/{branch}')
    if status == 404:
        print(f'远端分支 {branch} 不存在：请先在 GitHub 上建好（或改用 --branch main）')
        return 2
    remote_head = ref['object']['sha']
    if remote_head == head_sha:
        print('远端已是最新，无需推送')
        return 0

    # 待推送的提交（按老到新）。远端 HEAD 必须是本地的祖先，否则说明远端有
    # 本地没有的提交，直接停下，绝不强推覆盖别人的内容。
    pending = [line.strip() for line in
               sh('rev-list', '--reverse', f'{remote_head}..HEAD').splitlines() if line.strip()]
    if not pending:
        print('本地没有落后于远端的新提交（远端可能已包含本地历史）')
        return 3

    blob_cache = {}
    parent_sha = remote_head
    for index, commit_sha in enumerate(pending, 1):
        header = sh('cat-file', 'commit', commit_sha).split('\n\n', 1)[0]
        tree_sha = author_line = committer_line = None
        for line in header.split('\n'):
            if line.startswith('tree '):
                tree_sha = line.split()[1]
            elif line.startswith('author '):
                author_line = line[len('author '):]
            elif line.startswith('committer '):
                committer_line = line[len('committer '):]

        raw = sh('-c', 'core.quotepath=false', 'ls-tree', '-r', '-z', commit_sha)
        items = []
        uploaded = 0
        for record in raw.split('\0'):
            if not record.strip():
                continue
            meta, path = record.split('\t', 1)
            mode, otype, sha = meta.split()
            if otype != 'blob':
                continue
            if sha not in blob_cache:
                content = sh('cat-file', 'blob', sha, binary=True)
                status, blob = api('POST', f'/repos/{OWNER}/{REPO}/git/blobs',
                                   {'content': base64.b64encode(content).decode('ascii'),
                                    'encoding': 'base64'}, token=token)
                if status not in (200, 201):
                    print(f'  上传 {path} 失败: HTTP {status} {str(blob)[:200]}')
                    return 1
                blob_cache[sha] = blob['sha']
                uploaded += 1
            items.append({'path': path, 'mode': mode, 'type': 'blob',
                          'sha': blob_cache[sha]})

        status, tree = api('POST', f'/repos/{OWNER}/{REPO}/git/trees', {'tree': items}, token=token)
        if status not in (200, 201):
            print(f'  建树失败: HTTP {status} {str(tree)[:200]}')
            return 1
        if tree['sha'] != tree_sha:
            print(f'  树哈希不一致（本地 {tree_sha[:10]} / 远端 {tree["sha"][:10]}），停止')
            return 1

        message = sh('cat-file', 'commit', commit_sha)
        message = message.split('\n\n', 1)[1] if '\n\n' in message else ''
        status, commit = api('POST', f'/repos/{OWNER}/{REPO}/git/commits', {
            'message': message.rstrip() + '\n',
            'tree': tree['sha'],
            'parents': [parent_sha],
            'author': parse_ident(author_line),
            'committer': parse_ident(committer_line),
        }, token=token)
        if status not in (200, 201):
            print(f'  建提交失败: HTTP {status} {str(commit)[:200]}')
            return 1
        if commit['sha'] != commit_sha:
            print(f'  提交哈希不一致（本地 {commit_sha[:10]} / 远端 {commit["sha"][:10]}），停止')
            return 1
        parent_sha = commit['sha']
        print(f'  [{index}/{len(pending)}] {commit_sha[:10]} 已推送（新上传 {uploaded} 个文件）')

    status, body = api('PATCH', f'/repos/{OWNER}/{REPO}/git/refs/heads/{branch}',
                       {'sha': parent_sha, 'force': False}, token=token)
    if status not in (200, 201):
        print(f'  更新分支失败: HTTP {status} {str(body)[:200]}')
        return 1
    print(f'  远端 {branch} 已更新为 {parent_sha[:10]}（与本地一致）')
    return 0


def main():
    parser = argparse.ArgumentParser(description='通过 GitHub API 推送本地提交')
    parser.add_argument('--branch', default=None, help='目标分支（默认当前分支）')
    parser.add_argument('--retry', type=int, default=1, help='网络不通时的尝试次数')
    parser.add_argument('--gap', type=int, default=45, help='重试间隔（秒）')
    args = parser.parse_args()

    branch = args.branch or sh('branch', '--show-current').strip() or 'main'

    for attempt in range(1, max(1, args.retry) + 1):
        print(f'=== 尝试 {attempt}/{max(1, args.retry)}（分支 {branch}）===')
        if not api_reachable():
            print('  api.github.com 不可达')
        else:
            token = get_token()
            code = push(branch, token)
            if code == 0:
                return 0
            if code == 3:
                return 3
        if attempt < max(1, args.retry):
            time.sleep(max(1, args.gap))

    print('推送失败：网络无法访问 api.github.com（在能访问的网络下直接 git push 即可）')
    return 1


if __name__ == '__main__':
    sys.exit(main())
