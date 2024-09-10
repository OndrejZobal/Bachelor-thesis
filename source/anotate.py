import difflib

def token_diff(inp, out, new_line='\n'):
    inp_split = inp.split(new_line)
    text_diff = list(difflib.context_diff(inp_split, out.split(new_line), n=0))

    diff = {}
    state = None

    lines_removed = 0
    lines_added = 0

    for line in text_diff:
        if line.startswith('*** '):
            if len(line) <= len('*** ')+1: continue
            state = 'remove'
            p2 = len('*** ')
            p3_a = line[len('*** '):].find(' ')
            p3_b = line[len('*** '):].find(',')
            p3 = min(p3_a if p3_a != -1 else 10000, p3_b if p3_b != -1 else 10000)
            line_number = int( line[len('*** ') : p2 + p3] )
            initial_line = line_number
            lines_removed = 0
            lines_added = 0

        if line.startswith('--- '):
            if len(line) <= len('--- ')+1: continue
            line_number = initial_line
            state = 'add'

        if line.startswith('! ') or line.startswith('+ ') or line.startswith('- '):
            if state == 'add':
                lines_added += 1
            elif state == 'remove':
                lines_removed += 1

            string = f'{line[len("! "):]}' if state == 'add' else ''
            if line_number in diff:
                if diff[line_number] != '' and string != '':
                    string = f'{new_line}{string}'
                diff[line_number] += string
            else:
                if lines_removed == 0:
                    string = f'{inp_split[line_number-1]}{new_line}{string}'
                diff[line_number] = string
            if lines_added < lines_removed:
                line_number += 1

    keys = list(diff.keys())
    keys.sort()
    string = ''

    for k in keys:
        diff[k] = f'{diff[k]}{new_line}'
    
    for k in keys:
        string += f'<mask{k}>{diff[k]}'

    return diff

def indents(code):
    tablen = 0
    prev_len = 0
    newcode = []
    for line in code.replace('\t', '  ').split('\n'):
        stripped = line.lstrip(' ')
        spaces = len(line) - len(stripped)
        if tablen==0 and spaces != 0:
            tablen = spaces
        if spaces > prev_len:
            # Add indent
            stripped = ('\f' * ((spaces - prev_len)//tablen)) + stripped
        if spaces < prev_len:
            # Add dedent
            stripped = ('\a' * ((prev_len - spaces)//tablen)) + stripped
        prev_len = spaces
        newcode.append(stripped)
    return '\n'.join(newcode)

def base_90(num):
    if num == 0:
        return [0]

    digits = []
    while num > 0:
        remainder = num % 90
        digits.insert(0, remainder)
        num = num // 90

    return digits
