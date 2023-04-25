import json

file1 = open('data/plain-text.txt', 'r')    
file_content = file1.read()
print('File read')
file_content = file_content.replace( '\n\n\n', '\n' )
file_content = file_content.replace( '\n\n', '\n' )
file_content = file_content.replace( '\n', ' ' )    
file_content = file_content.replace( '\t', ' ' )    
print('File cleaned of newlines')

file_content = file_content.replace( '...', '…' )    
file_content = file_content.replace( '.', '.#$#' )    
file_content = file_content.replace( '!', '!#$#' )    
file_content = file_content.replace( '?', '?#$#' )    
file_content = file_content.replace( ';', ';#$#' )    
file_content = file_content.replace( '…', ';#$#' )    

Lines0 = file_content.split('#$#')
Lines = []

def create_prompts(set_of_lines, max_chars):
    max_line_count = len(set_of_lines)
    i = 0
    to_return = []
    while i <  max_line_count: 
        print(f'\r{i} / {max_line_count}', sep=' ', end="", flush=True)
        use_line = ""
        use_counter = 0
        while len(use_line) < max_chars and (i + use_counter) < max_line_count:
            use_line += Lines0[i+use_counter].strip() + ' '
            use_counter += 1
        if len(use_line) > max_chars + 100:
            use_line = use_line[0:(max_chars+100)]

        use_line = use_line.replace( '  ', ' ' )
        one = {}
        one["instruction"] = set_of_lines[i].strip()
        one["output"] = use_line
        if (len(one["instruction"])>1):
            to_return.append(one)
        i = i + use_counter
    print('')
    return to_return

# This works well, use 1200, 600 and 300 max_chars to get really good results
Lines.extend( create_prompts( Lines0, 1200 ) )
Lines.extend( create_prompts( Lines0, 600 ) )
Lines.extend( create_prompts( Lines0, 300 ) )

print(f'\nTotal prompts: {len(Lines)}')

with open("data/fake-prompts.json", "w") as write_file:
    json.dump(Lines, write_file)