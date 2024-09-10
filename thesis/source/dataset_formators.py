#!/usr/bin/env python3
import tokenize
from io import StringIO
import random
import anotate
import re

random.seed(42)

def select_mode(mode):
    return {
        'docstring_generation': docstring_generation,
        'comment_generation': comment_generation,
        'comment_detection': comment_detection,
        'wizard': wizard_patch,
        'error_detection': error_detection,
        'error_generation': error_generation,
        'rename': token_renaming,
    }[mode]


rest_pattern = re.compile(r':(param|return|rtype|type)(\s[a-zA-Z0-9\-_ ]+)?:')
google_pattern = re.compile(r'\n\s+(Args:|Returns:|Attributes:|Raises:)\n\s', re.MULTILINE)
numpy_pattern = re.compile(r'[A-Z][a-z]+:?\n\s*-+\n', re.MULTILINE)
epytext_pattern = re.compile(r'\s@(param|return|raise|type)\s+\w+', re.MULTILINE)

def comment_detection(source, target, tokenizer, max_source_tokens, max_target_tokens):
    source = '\n'.join(source.split('NEW_LINE'))
    modified_source = []  # To store source code lines without comments
    removed_comment_lines = set()  # To track line numbers where comments were removed
    source_io = StringIO(source)
    skip = False
    removed_lines = 0
    pending_indent = False
    last_line_tokens = 0

    try:
        for (
            toktype,
            tokstr,
            (srow, scol),
            (erow, ecol),
            line,
        ) in tokenize.generate_tokens(source_io.readline):
            if srow == skip:
                continue

            line_no = srow - removed_lines

            if last_line_tokens > max_source_tokens:
                break

            if toktype == tokenize.COMMENT:
                # Check if the comment is at the end of a line of code
                if line.strip() != tokstr:  # Inline comment
                    # Remove the comment part from the line
                    line_without_comment = [
                        '$',
                        f'{line_no}',
                        '$',
                    ] + tokenizer.tokenize(
                        (' INDENT' if pending_indent else '') + line[:scol] + 'NEW_LINE'
                    )
                    pending_indent = False
                    if not pylint_filter.search(line[scol:]):
                        removed_comment_lines.add(line_no)
                    if line[:scol].strip() == 'INDENT':
                        pending_indent = True
                        removed_lines += 1
                    else:
                        modified_source += line_without_comment
                    skip = srow
                # Full-line comment
                else:
                    if not pylint_filter.search(line[scol:]):
                        removed_comment_lines.add(line_no)
                    removed_lines += 1
                    skip = srow

            elif toktype in [tokenize.NL, tokenize.NEWLINE]:
                modified_source += ['$', f'{line_no}', '$'] + tokenizer.tokenize(
                    (' INDENT' if pending_indent else '') + line + 'NEW_LINE'
                )
                pending_indent = False

            last_line_tokens = len(modified_source)
        if modified_source[-1] != 'NEW_LINE':
            modified_source += ['$', f'{line_no}', '$', tokenizer.tokenize('NEW_LINE')]
        # Join the modified source code lines

    except tokenize.TokenError:
        raise ValueError
    if (
        len(removed_comment_lines) == 0
        or float(line_no) / (float(len(removed_comment_lines))) < 0.5
    ):
        raise ValueError(f"{len(removed_comment_lines)}")

    removed_comment_lines = list(removed_comment_lines)
    removed_comment_lines.sort()
    removed_comment_lines = tokenizer.tokenize(
        ' '.join([f'{element} ' for element in removed_comment_lines])
    )

    modified_source = modified_source[:last_line_tokens][:max_source_tokens - 2]
    modified_source += [tokenizer.eos_token] + ['__python__']

    # Global attention
    global_attention_mask = [True] + [False] * (len(modified_source) - 2) + [True]
    removed_comment_lines = ['__en_XX__'] + (removed_comment_lines[:max_target_tokens - 1])

    ret = (
        [(modified_source, removed_comment_lines, global_attention_mask, len(removed_comment_lines))]
    )

    return ret

def identify_docstring(comment):
    if rest_pattern.search(comment):
        return "RE"
    if numpy_pattern.search(comment):
        return "NP"
    if google_pattern.search(comment):
        return "GO"
    if epytext_pattern.search(comment):
        return "EP"
    return "NA"

pylint_filter = re.compile(r'pylint:')
type_filter = re.compile(r'type:')
op_filter = re.compile(r"[&\|/<>+\*()\[\]^\"'=]")
todo_filter = re.compile(r'TODO')
pragma_filter = re.compile(r'pragma:')
def ok_comment(comment):
    if re.search(pylint_filter, comment): return False
    if re.search(type_filter, comment): return False
    if re.search(op_filter, comment): return False
    if re.search(todo_filter, comment): return False
    if re.search(pragma_filter, comment): return False
    if len(comment) < 10 : return False
    if len(comment) > 120 : return False
    return True

def comment_generation(
    source, target, tokenizer, max_source_tokens, max_target_tokens
):
    """Requires code with comments and docstring in the input
    :source: contains code with comments but no docstring.
    :target: contains code summarization (like one that would be found in a docstring.)
    """
    source = '\n'.join(source.split('NEW_LINE'))
    modified_source = []  # To store source code lines without comments
    extracted_comments = {}  # To track line numbers where comments were removed
    source_io = StringIO(source)
    skip = False
    removed_lines = 0
    pending_indent = False
    last_line_tokens = 0
    previously_saw_comment = False

    try:
        for (
            toktype,
            tokstr,
            (srow, scol),
            (erow, ecol),
            line,
        ) in tokenize.generate_tokens(source_io.readline):
            if srow == skip:
                continue

            line_no = srow - removed_lines

            if last_line_tokens > max_source_tokens:
                break

            if toktype == tokenize.COMMENT and not (
                random.randint(1, 5) == 5 and not previously_saw_comment
            ):
                # Check if the comment is at the end of a line of code
                if line.strip() != tokstr:  # Inline comment
                    # Remove the comment part from the line
                    line_without_comment = tokenizer.tokenize(
                        (' INDENT' if pending_indent else '') + line[:scol] + 'NEW_LINE'
                    )
                    comment = line[scol + 1 : ecol] + 'NEW_LINE'
                    pending_indent = False
                    if ok_comment(line[scol:]):
                        extracted_comments.update({line_no: comment.lstrip().lstrip('#')})
                    if line[:scol].strip() == 'INDENT':
                        pending_indent = True
                        removed_lines += 1
                    else:
                        modified_source.append(line_without_comment)
                    skip = srow
                    previously_saw_comment = False
                # Full-line comment
                else:
                    if previously_saw_comment:
                        n = max(list(extracted_comments.keys()))
                    else:
                        if ok_comment(line[scol:]):
                            extracted_comments.update(
                                {line_no: line[scol + 1 :].lstrip().lstrip('#')}
                            )
                    previously_saw_comment = True
                    removed_lines += 1
                    skip = srow

            elif toktype in [tokenize.NL, tokenize.NEWLINE]:
                modified_source.append(
                    tokenizer.tokenize(
                        (' INDENT' if pending_indent else '') + line + 'NEW_LINE'
                    )
                )
                pending_indent = False

            else:
                previously_saw_comment = False

            last_line_tokens = len(modified_source)
        # Join the modified source code lines

    except tokenize.TokenError:
        raise ValueError

    if (
        len(extracted_comments) == 0
        or float(line_no) / (float(len(extracted_comments))) < 0.5
    ):
        raise ValueError

    target = target.replace("\n", "NEW_LINE")
    tokenized_target = tokenizer.tokenize(target.split(">>>")[0])
    result = []
    docstr = identify_docstring(target)
    if len(tokenized_target) < max_target_tokens-6:
        docstr = ['$'] + tokenizer.tokenize('DOCSTR ') + tokenizer.tokenize(docstr) + ['$']
        cat = docstr[:]
        for line in modified_source[:line_no]:
            cat += line
        cat = cat[:max_source_tokens - 2]

        cat += [tokenizer.eos_token] + ['__python__']
        global_attention_mask =\
            [True] * (len(docstr)) +\
            [True] * len(modified_source[0]) +\
            [False] * (len(cat) - len(docstr) - len(modified_source[0]) -1) +\
            [True]
    result = []
    keys = list(extracted_comments.keys())
    keys.sort()
    for line in keys:
        tokens_before_flag = 0
        comment = extracted_comments[line].replace('NEW_LINE', ' ')
        flag = ['$'] + tokenizer.tokenize(f'{line}') + ['$']
        comment = (
            flag + tokenizer.tokenize(comment)
        )
        if len(comment) < 2+3:
            continue
        source = modified_source[:line_no]
        if line - 1 >= len(source):
            continue
        source[line - 1] = flag + source[line - 1]
        cat = []

        for i, l in enumerate(source):
            cat += l
            if i < line-1:
                tokens_before_flag += len(l)

        if tokens_before_flag >= max_source_tokens: 
            continue

        cat = cat[:max_source_tokens - 2]
        cat += [tokenizer.eos_token] + ['__python__']
        comment = ['__en_XX__'] + (comment[:max_target_tokens-2]) + [tokenizer.eos_token]


        global_attention_mask =\
            [True] * len(source[0]) +\
            [False] * (tokens_before_flag - len(source[0])) +\
            [True] * len(flag) +\
            [False] * (len(cat) - len(source[0]) - max(tokens_before_flag - len(source[0]), 0) - len(flag) - 1) +\
            [True]

        result.append(
            (
                cat, comment, global_attention_mask, len(cat)
            )
        )

    return [random.choice(result)] if len(result) else []


def docstring_generation(
    source, target, tokenizer, max_source_tokens, max_target_tokens
):
    """Requires code with comments and docstring in the input
    :source: contains code with comments but no docstring.
    :target: contains code summarization (like one that would be found in a docstring.)
    """
    source = '\n'.join(source.split('NEW_LINE'))
    modified_source = []  # To store source code lines without comments
    extracted_comments = {}  # To track line numbers where comments were removed
    source_io = StringIO(source)
    skip = False
    removed_lines = 0
    pending_indent = False
    last_line_tokens = 0
    previously_saw_comment = False

    try:
        for (
            toktype,
            tokstr,
            (srow, scol),
            (erow, ecol),
            line,
        ) in tokenize.generate_tokens(source_io.readline):
            if srow == skip:
                continue

            line_no = srow - removed_lines

            if last_line_tokens > max_source_tokens:
                break

            if toktype == tokenize.COMMENT and not (
                random.randint(1, 5) == 5 and not previously_saw_comment
            ):
                # Check if the comment is at the end of a line of code
                if line.strip() != tokstr:  # Inline comment
                    # Remove the comment part from the line
                    line_without_comment = tokenizer.tokenize(
                        (' INDENT' if pending_indent else '') + line[:scol] + 'NEW_LINE'
                    )
                    comment = line[scol + 1 : ecol] + 'NEW_LINE'
                    pending_indent = False
                    if not pylint_filter.search(line[scol:]):
                        extracted_comments.update({line_no: comment})
                    if line[:scol].strip() == 'INDENT':
                        pending_indent = True
                        removed_lines += 1
                    else:
                        modified_source.append(line_without_comment)
                    skip = srow
                    previously_saw_comment = False
                # Full-line comment
                else:
                    if previously_saw_comment:
                        n = max(list(extracted_comments.keys()))

                        if not pylint_filter.search(line[scol:]):
                            extracted_comments[n] += line[scol + 1 :] + 'NEW_LINE'
                    else:
                        if not pylint_filter.search(line[scol:]):
                            extracted_comments.update(
                                {line_no: line[scol + 1 :] + 'NEW_LINE'}
                            )
                    previously_saw_comment = True
                    removed_lines += 1
                    skip = srow

            elif toktype in [tokenize.NL, tokenize.NEWLINE]:
                modified_source.append(
                    tokenizer.tokenize(
                        (' INDENT' if pending_indent else '') + line + 'NEW_LINE'
                    )
                )
                pending_indent = False

            else:
                previously_saw_comment = False

            last_line_tokens = len(modified_source)
        # Join the modified source code lines

    except tokenize.TokenError:
        raise ValueError

    if (
        len(extracted_comments) == 0
        or float(line_no) / (float(len(extracted_comments))) < 0.5
    ):
        raise ValueError

    target = target.replace("\n", "NEW_LINE")
    tokenized_target = tokenizer.tokenize(target.split(">>>")[0])
    result = []
    docstr = identify_docstring(target)
    if len(tokenized_target) < max_target_tokens-6:
        docstr = ['$'] + tokenizer.tokenize('DOCSTR ') + tokenizer.tokenize(docstr) + ['$']
        cat = docstr[:]
        for line in modified_source[:line_no]:
            cat += line
        cat = cat[:max_source_tokens - 2]

        cat += [tokenizer.eos_token] + ['__python__']
        global_attention_mask =\
            [True] * (len(docstr)) +\
            [True] * len(modified_source[0]) +\
            [False] * (len(cat) - len(docstr) - len(modified_source[0]) -1) +\
            [True]
        result = [
            (
                cat,
                (['__en_XX__'] + docstr + tokenizer.tokenize(target))[:max_target_tokens-1] + [tokenizer.eos_token],
                global_attention_mask,
                len(cat[:]),
            )
        ]
    return result


def error_detection(source, target, tokenizer, max_source_tokens, max_target_tokens):
    # source
    nl, target = target.split('<sep>', maxsplit=1)
    nl_tokens = tokenizer.tokenize(nl + ' NEW_LINE')[:128]
    source_tokens = source.split('NEW_LINE')
    source_tokens_numbered = []

    for index, line in enumerate(source_tokens):
        source_tokens_numbered += ['$', f'{index}', '$'] + tokenizer.tokenize(
            line + 'NEW_LINE'
        )

    target_tokens = []
    print(source)
    print(target)
    diff = anotate.token_diff(source, target, new_line='NEW_LINE')
    print(diff)
    keys = list(diff.keys())
    keys.sort()
    target_numbers = ''
    for key in keys:
        target_numbers += f'{key} '

    if len(keys) > 3:
        raise ValueError()
    # TARGET
    target_tokens = tokenizer.tokenize('__en_XX__') + nl_tokens + tokenizer.tokenize(target_numbers)
    target_tokens = target_tokens[:max_target_tokens-1] + [tokenizer.eos_token]

    # SOURCE
    source_tokens = source_tokens_numbered[: max_source_tokens - 2] + [tokenizer.eos_token, '__python__']

    # Global attention
    global_attention_mask = [True] + [False] * (len(source_tokens) - 2) + [True]

    return [(source_tokens, target_tokens, global_attention_mask, len(target_tokens))]


def edpen(source, target, tokenizer, max_source_tokens, max_target_tokens):
    # source
    nl, target = target.split('<sep>', maxsplit=1)
    nl_tokens = tokenizer.tokenize(nl)[:128]


    source_tokens = tokenizer.tokenize(source)
    target_tokens = []

    # TARGET
    target_tokens = nl_tokens 
    target_tokens = target_tokens[:max_target_tokens]

    # SOURCE
    source_tokens = source_tokens[: max_source_tokens - 1] + ['__python__']

    # Global attention
    global_attention_mask = [True] + [False] * (len(source_tokens) - 2) + [True]

    return [(source_tokens, target_tokens, global_attention_mask)], '__en_XX__'


def straight_to_patch(source, target, tokenizer, max_source_tokens, max_target_tokens):
    # source
    nl, target = target.split('<sep>', maxsplit=1)
    nl_tokens = tokenizer.tokenize(nl + ' NEW_LINE')[:128]
    source_tokens = source.split('NEW_LINE')

    # Limit to shosrter fuctions for testing
    if len(source_tokens) > 25:
        raise ValueError()

    source_tokens_numbered = []

    for index, line in enumerate(source_tokens):
        source_tokens_numbered += (
            ['$']
            + tokenizer.tokenize(f'{index}')
            + ['$']
            + tokenizer.tokenize(line + 'NEW_LINE')
        )

    target_tokens = []
    diff = anotate.token_diff(source, target, new_line='NEW_LINE')
    keys = list(diff.keys())
    if len(keys) > 3:
        raise ValueError()
    keys.sort()
    target_numbers = []
    for key in keys:
        target_numbers += (
            ['$'] + tokenizer.tokenize(f'{key}') + ['$'] + tokenizer.tokenize(diff[key])
        )

    # TARGET
    target_tokens = (nl_tokens + target_numbers)[:max_target_tokens]

    # SOURCE
    source_tokens = source_tokens_numbered[: max_source_tokens - 1] + ['__python__']

    # Global attention
    global_attention_mask = [True, True] + [False] * (len(source_tokens) - 3) + [True]

    return [(source_tokens, target_tokens, global_attention_mask)], '__python__'


def error_generation(source, target, tokenizer, max_source_tokens, max_target_tokens):
    # source
    nl, target = target.split('<sep>', maxsplit=1)
    nl_tokens = tokenizer.tokenize(nl + ' NEW_LINE')[:128]
    source_tokens = source.split('NEW_LINE')
    source_tokens_numbered = nl_tokens[:]

    for index, line in enumerate(source_tokens):
        source_tokens_numbered += (
            ['$']
            + tokenizer.tokenize(f'{index}')
            + ['$']
            + tokenizer.tokenize(line + 'NEW_LINE')
        )

    target_tokens = []
    diff = anotate.token_diff(source, target, new_line='NEW_LINE')
    keys = list(diff.keys())
    keys.sort()
    target_numbers = []
    for key in keys:
        target_numbers += (
            ['$'] + tokenizer.tokenize(f'{key}') + ['$'] + tokenizer.tokenize(diff[key])
        )

    # TARGET
    target_tokens = ['__python__'] + target_numbers[:max_target_tokens -2] + [tokenizer.eos_token]

    # SOURCE
    source_tokens = source_tokens_numbered[: max_source_tokens - 3] + [tokenizer.eos_token, '__python__']

    # Global attention
    global_attention_mask = [True] * len(nl_tokens) + [False] * (len(source_tokens) - 1 - len(nl_tokens)) + [True]

    return [(source_tokens, target_tokens, global_attention_mask, len(source_tokens))]


def straight_to_code(source, target, tokenizer, max_source_tokens, max_target_tokens):
    # source
    nl, target = target.split('<sep>', maxsplit=1)
    nl_tokens = tokenizer.tokenize(nl + ' NEW_LINE')[:128]
    source_tokens = source.split('NEW_LINE')

    # Limit to shosrter fuctions for testing
    if len(source_tokens) > 25:
        raise ValueError()

    source_tokens_numbered = []

    for index, line in enumerate(source_tokens):
        source_tokens_numbered += (
            ['$']
            + tokenizer.tokenize(f'{index}')
            + ['$']
            + tokenizer.tokenize(line + 'NEW_LINE')
        )

    target_tokens = tokenizer.tokenize(target)
    diff = anotate.token_diff(source, target, new_line='NEW_LINE')
    keys = list(diff.keys())
    if len(keys) > 3:
        raise ValueError()

    # TARGET
    target_tokens = (nl_tokens + target_tokens)[:max_target_tokens]

    # SOURCE
    source_tokens = source_tokens_numbered[: max_source_tokens - 1] + ['__python__']

    # Global attention
    global_attention_mask = [True, True] + [False] * (len(source_tokens) - 3) + [True]
    if source_tokens == target_tokens:
        raise ValueError()

    return [(source_tokens, target_tokens, global_attention_mask)], '__python__'

def wizard_patch(source, target, tokenizer, max_source_tokens, max_target_tokens):
    nl, target = target.split('<sep>', maxsplit=1)
    nl_tokens = tokenizer.tokenize(nl + '\n')[:128]
    source_tokens = source.split('\n')
    source_tokens_numbered = []

    for index, line in enumerate(source_tokens):
        source_tokens_numbered += ['$'] + tokenizer.tokenize(f'{index}') + ['$'] + tokenizer.tokenize(line + '\n')

    source_tokens = tokenizer.tokenize("Try to see if the following code has any errors. If not say \"No errors\" if it does explain the error and write a patch with the help of the line numbers.\n### Code:\n") + source_tokens_numbered + tokenizer.tokenize("\n###Response:\n")
    target_tokens = []
    diff = anotate.token_diff(source, target, new_line='\n')
    keys = list(diff.keys())
    keys.sort()
    target_numbers = []
    for key in keys:
        target_numbers += (
            ['$'] + tokenizer.tokenize(f'{key}') + ['$'] + tokenizer.tokenize(diff[key])
        )

    if len(keys) == 0:
        target_numbers += tokenizer.tokenize('No errors\n')

    if len(keys) > 3:
        raise ValueError()
    # TARGET
    target_tokens = nl_tokens + target_numbers
    target_tokens = target_tokens[:max_target_tokens]

    # SOURCE
    source_tokens = source_tokens[: max_source_tokens - 1]

    target_tokens = source_tokens + target_tokens
    hidden_index = len(source_tokens)
    #source_tokens = target_tokens
    target_tokens = target_tokens[:]

    # Global attention
    global_attention_mask = [True] + [False] * (len(source_tokens) - 2) + [True]

    return [(source_tokens, target_tokens, global_attention_mask, hidden_index)]

def wizard_code(source, target, tokenizer, max_source_tokens, max_target_tokens):
    nl, target = target.split('<sep>', maxsplit=1)
    nl_tokens = tokenizer.tokenize(nl + '\n')[:128]
    source_tokens = source.split('\n')
    source_tokens_numbered = []

    for index, line in enumerate(source_tokens):
        source_tokens_numbered += tokenizer.tokenize(line + '\n')

    source_tokens = tokenizer.tokenize("Try to see if the following code has any errors. Explain the error and write a fix it.\n### Code:\n") + source_tokens_numbered + tokenizer.tokenize("\n### Fixed Code:\n")
    target_tokens = []

    target_tokens = target.split('\n')
    target_tokens_numbered = []
    for index, line in enumerate(target_tokens):
        target_tokens_numbered += tokenizer.tokenize(line + '\n')

    if source == target:
        nl_tokens = tokenizer.tokenize('No errors\n')

    diff = anotate.token_diff(source, target, new_line='\n')
    keys = list(diff.keys())
    if len(keys) > 3:
        raise ValueError()

    # TARGET
    target_tokens = nl_tokens + (target_tokens_numbered if source != target else [])
    if len(target_tokens) > max_target_tokens:
        raise  ValueError()
    target_tokens = target_tokens[:max_target_tokens]

    # SOURCE
    source_tokens = source_tokens[: max_source_tokens - 1]

    target_tokens = source_tokens + target_tokens
    hidden_index = len(source_tokens)
    if len(target_tokens) > max_target_tokens:
        raise ValueError()
    target_tokens = target_tokens[: max_target_tokens]

    # Global attention
    global_attention_mask = [True] + [False] * (len(source_tokens) - 2) + [True]

    return [(source_tokens, target_tokens, global_attention_mask, hidden_index)]

def token_renaming(source, target, tokenizer, max_source_tokens, max_target_tokens):
    tokenized_masked_source = tokenizer.tokenize(source.replace("< mask >", "<mask>")) + [tokenizer.eos_token, "__python__"]
    tokenized_label = tokenizer.tokenize(target)
    tokenized_label = ["__python__"] + tokenized_label + [tokenizer.eos_token]

    if len(tokenized_masked_source) > max_source_tokens:
        raise ValueError
    if len(tokenized_label) > max_target_tokens:
        raise ValueError

    global_attention = [entry == "<mask>" for entry in tokenized_masked_source]
    global_attention[-1] = True

    return [(tokenized_masked_source, tokenized_label, global_attention, len(tokenized_masked_source))]

