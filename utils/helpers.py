from typing_extensions import List
import io
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_message(message, max_char=1800) -> List:
    '''
    Heuristic way to split message while complying with code format
    '''
    messages = []
    lines = io.StringIO(message).readlines()

    cur_char_count = 0
    message = ""

    in_code_snippet = False

    for line in lines:
        cur_char_count += len(line)
        if cur_char_count < max_char:
            message += line
            if len(re.findall(r"```", line)) == 1:
                if line[0:3] == "```" and not in_code_snippet:
                    in_code_snippet = True
                    lang = line[3:-1]
                elif line == "```" and in_code_snippet:
                    in_code_snippet = False
            if line == lines[-1]:
                messages.append(message)
        else:
            if in_code_snippet:
                message += "```"
                messages.append(message)
                message = f"```{lang}\n"
            else:
                messages.append(message)
                message = ""
            message += line
            cur_char_count = len(line)

    return messages