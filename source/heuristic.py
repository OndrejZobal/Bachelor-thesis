import ast
import tokenize
import textwrap
from io import BytesIO

class CommentSuggester(ast.NodeVisitor):
    def __init__(self, source_code):
        self.source_code = source_code
        self.source_tree = ast.parse(source_code)
        self.lines = source_code.splitlines()
        self.suggestions = []
        self.has_comment = [False] * len(self.lines)  # Track commented lines

    def unique_lexemes(self, line, limit):
        # No string interface, annoying
        tokens = tokenize.tokenize(BytesIO(line.encode("utf-8")).readline)
        unique_tokens = set()
        try:
            for toknum, tokval, _, _, _ in tokens:
                if toknum in {tokenize.NAME, tokenize.NUMBER, tokenize.OP}:
                    unique_tokens.add(tokval)
        except tokenize.TokenError:
            return False
        return len(unique_tokens) > limit

    def visit(self, node):
        if hasattr(node, "lineno") and node.lineno not in [s[0] for s in self.suggestions]:

            line_index = node.lineno - 1
            line = self.lines[line_index].strip()

            # Check if this line or the previous line already has a comment
            if not (self.has_comment[line_index] or (line_index > 0 and self.has_comment[line_index - 1])):
                # Determine if this line should have a comment based on its complexity or length
                if self.unique_lexemes(line, 12):
                    self.suggestions.append((node.lineno, line, "complicated"))

                # Check for control structures
                elif isinstance(node, (ast.If, ast.For, ast.While, ast.Try, ast.ExceptHandler)):
                    self.suggestions.append((node.lineno, line, f"branching"))

                elif line_index > 0 and not self.lines[line_index - 1].strip():
                    self.suggestions.append((node.lineno, line, "blank before"))

        # Continue traversal to child nodes
        self.generic_visit(node)

    def report(self):
        # debug and testing, prints lines and reasons for their inclusion
        for lineno, line, reason in self.suggestions:
            print(f"Line {lineno}: '{line}' might need a comment. Reason: {reason}")

    def output(self):
        return [s[0] for s in self.suggestions]

    def susggest(self):
        """Mark lines that already have comments, properly handling '#' within strings."""
        tokens = tokenize.tokenize(BytesIO(self.source_code.encode('utf-8')).readline)
        for token in tokens:
            if token.type == tokenize.COMMENT:
                line_index = token.start[0] - 1
                self.has_comment[line_index] = True
                if line_index > 0:
                    self.has_comment[line_index - 1] = True
        self.visit(self.source_tree)
        return self.output()

def lines_for_comment(source_code):
    source_code = textwrap.dedent(source_code)
    suggester = CommentSuggester(source_code)
    result = suggester.susggest()
    return result

