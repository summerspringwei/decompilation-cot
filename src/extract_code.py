import re

def extract_function_names_from_llvm_ir(llvm_ir_code):
    # Regular expression to match function definitions in LLVM IR and capture function names
    # function_pattern = re.compile(r'define\s+\w+\s+@(\w+)\s*\([^)]*\)\s*{')
    # function_pattern = re.compile(r'define\s+\S+\s+@(\w+)\s*\(')
    function_pattern = re.compile(r'define\s+(?:\S+\s+)*@\s*([\w\d_]+)\s*\(')
    # Find all function names
    function_names = function_pattern.findall(llvm_ir_code)
    if function_names and len(function_names) > 0:
        return function_names[0]
    else:
        return None


def extract_function_name_from_C(declaration):
    """
    Extracts the function name from a C++ or C function declaration.

    Args:
    declaration (str): A string containing the function declaration.

    Returns:
    str: The name of the function, or None if no function name is found.
    """
    # Regular expression pattern to match C++ and C function declarations
    pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)\s*(const)?\s*(;)?\s*$'
    
    # Search for the pattern in the declaration
    match = re.search(pattern, declaration)
    
    if match:
        return match.group(1)
    else:
        return None


def extract_llmcompiler_code_blocks(text):
    pattern = re.compile(r'<code>(.*?)</code>', re.DOTALL)
    matches = pattern.findall(text)
    return matches
