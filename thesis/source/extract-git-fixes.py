#!/usr/bin/env python3

import re
import json
import ast
import textwrap
from git import Repo
import sys

ONE_FIELD = False
CONVERT_AST = False

class FunctionCroppingVisitor(ast.NodeVisitor):
    def __init__(self, target_line):
        self.target_line = target_line
        self.start_lineno = None
        self.end_lineno = None
        self.found_target_line = False

    def visit_FunctionDef(self, node):
        if self.found_target_line:
            return  # Stop visiting if the target line has already been found

        if node.lineno <= self.target_line <= node.end_lineno:
            self.start_lineno = node.lineno
            self.end_lineno = node.end_lineno
            self.found_target_line = True

        self.generic_visit(node)

def crop_to_function_at_line(source_code, target_line):
    tree = ast.parse(source_code)
    visitor = FunctionCroppingVisitor(target_line)
    visitor.visit(tree)

    if visitor.start_lineno is not None and visitor.end_lineno is not None:
        # Extract the relevant part of the source code using line numbers
        cropped_source = "\n".join(source_code.splitlines()[visitor.start_lineno-1:visitor.end_lineno])
        cropped_source_dedented = textwrap.dedent(cropped_source)
        return cropped_source_dedented

    return None

def get_file_content(repo, commit, file_path):
    # Get the content of the file after the commit
    parent_commit = commit.parents[0]
    previous_content = parent_commit.tree[file_path].data_stream.read().decode("utf-8")
    current_content = commit.tree[file_path].data_stream.read().decode("utf-8")
    return previous_content, current_content

def find_func_def(n, content):
    for i in range(n, 0, -1):
        if re.search("^[ \t]*def [a-zA-Z][a-zA-z0-9]*: *$", content[i]):
            return i

def get_changed_line_number(prev, post):
    shorter_length = min(len(prev), len(post))

    for i in range(shorter_length):
        if prev[i] != post[i]:
            return i

    if len(prev) != len(post):
        return shorter_length

    return None

def search_branch_history(repo_path, branch_names, patterns):
    data = []
    repo = Repo(repo_path)
    branch = None
    for name in branch_names:
        if name in repo.branches:
            branch = repo.branches[name]
    if branch is None:
        print(f"No branch found in {repo_path}")
        return []

    # Iterate over commits in the branch
    for commit in repo.iter_commits(branch):
        found_pattern = False
        for pattern in patterns:
            if re.search(keyword, commit.message):
                found_pattern = True
                break

        if found_pattern:
            print(f"searching {commit.hexsha}")

            # Get the changes (diff) for each commit

            diffs = commit.diff(commit.parents[0]) if commit.parents else []
            print(f"{type(diffs)=}")
            print(diffs)

            if len(diffs) != 1:
                print(f"{len(diffs)=}")
                continue
            change = diffs[0]

            # Skip Commit if it changed more than five lines
            if (len(change.diff.splitlines()) > 2):
                print(f"{len(change.diff.splitlines())=}")
                continue

            # Skip if the file didn't exist before
            if not change.a_blob:
                continue

            # Path of the file in the before-diff
            file_path = change.a_blob.path

            # If the diff doesn't concern a python file skip it
            if not re.search(".*\\.py$" ,file_path):
                print("not python")
                continue

            try:
                previous_content, post_content = get_file_content(repo, commit, file_path)
            except KeyError:
                continue

            prev = previous_content.replace('\r', '').split('\n')
            post = post_content.replace('\r', '').split('\n')

            n = get_changed_line_number(prev, post)
            if n is None:
                print("N is none wow")
                continue

            try:
                result_prev = crop_to_function_at_line(previous_content, n)
                result_post = crop_to_function_at_line(post_content, n)
            except SyntaxError:
                print("Syntax error")
                continue

            if result_prev is None or result_post is None: 
                print("Diffs empty")
                continue

            if result_prev == result_post:
                print("No changes??")
                continue

            if len(result_prev.split('\n')) > 100:
                print("Too long")
                continue

            if CONVERT_AST:
                result_prev = ast.dump(ast.parse(result_prev), indent=1)
                result_post = ast.dump(ast.parse(result_post), indent=1)
            
            obj = {}
            if ONE_FIELD:
                obj['text'] = f'{result_prev}[SEP]{result_post}'
            else:
                obj['input'] = result_prev
                obj['output'] = result_post
            obj['path'] = file_path
            obj['description'] = commit.message
            obj['hash'] = commit.hexsha
            data.append(obj)
            print("appended")
            print(result_prev)

    return data



if __name__ == "__main__":
    branch_names = ["devel", "master", "main", "stable"]
    repos_paths = sys.argv[1:]
    keyword = "[^a-zA-Z][Ff][Ii][Xx]([Ee][Dd])?([Ee][Ss])?[^a-zA-Z]"
    for repo_path in repos_paths:
        data = search_branch_history(repo_path, branch_names, keyword)
        print("dumping jsonl...")

        if (len(data) == 0):
            exit(0)

        with open(f"data/{repo_path.split('/')[-1]}-diffdata.jsonl", mode='w', encoding="utf-8") as file:
            file.write("")

        with open(f"data/{repo_path.split('/')[-1]}-diffdata.jsonl", mode='a', encoding="utf-8") as file:
            for entry in data:
                entry["repo"] = repo_path.split('/')[-1]
                file.write(json.dumps(entry)+"\n")
