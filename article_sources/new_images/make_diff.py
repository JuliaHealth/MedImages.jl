import difflib
import sys

def main():
    with open('o.tex', 'r', encoding='utf-8') as f:
        old_lines = f.readlines()
        
    with open('n.tex', 'r', encoding='utf-8') as f:
        new_lines = f.readlines()
        
    preamble = []
    doc_start_idx = 0
    for i, line in enumerate(new_lines):
        preamble.append(line)
        if '\\begin{document}' in line:
            doc_start_idx = i + 1
            break
            
    old_doc_start_idx = 0
    for i, line in enumerate(old_lines):
        if '\\begin{document}' in line:
            old_doc_start_idx = i + 1
            break
            
    new_body = new_lines[doc_start_idx:]
    old_body = old_lines[old_doc_start_idx:]
    
    new_doc_end_idx = len(new_body)
    for i, line in enumerate(new_body):
        if '\\end{document}' in line:
            new_doc_end_idx = i
            break
            
    new_body_content = new_body[:new_doc_end_idx]
    new_tail = new_body[new_doc_end_idx:]
    
    old_doc_end_idx = len(old_body)
    for i, line in enumerate(old_body):
        if '\\end{document}' in line:
            old_doc_end_idx = i
            break
            
    old_body_content = old_body[:old_doc_end_idx]
    
    sm = difflib.SequenceMatcher(None, old_body_content, new_body_content)
    
    output = []
    output.extend(preamble)
    
    for opcode, i1, i2, j1, j2 in sm.get_opcodes():
        if opcode == 'equal':
            output.extend(new_body_content[j1:j2])
        elif opcode == 'insert':
            output.append('\n\\color{green}\n')
            output.extend(new_body_content[j1:j2])
            output.append('\n\\color{black}\n')
        elif opcode == 'delete':
            output.append('\n\\color{red}\n')
            output.extend(old_body_content[i1:i2])
            output.append('\n\\color{black}\n')
        elif opcode == 'replace':
            output.append('\n\\color{red}\n')
            output.extend(old_body_content[i1:i2])
            output.append('\n\\color{green}\n')
            output.extend(new_body_content[j1:j2])
            output.append('\n\\color{black}\n')
            
    output.extend(new_tail)
    
    with open('diff.tex', 'w', encoding='utf-8') as f:
        f.writelines(output)

if __name__ == '__main__':
    main()
