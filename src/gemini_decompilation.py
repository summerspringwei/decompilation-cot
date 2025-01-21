import os
import json
import re
from google import genai
from datasets import load_from_disk, Dataset


def extract_llvm_code(markdown_content: str):
    llvm_code_blocks = []
    # Use a non-greedy regex to match multiple code blocks
    pattern = r"```llvm\n(.*?)\n```"  # The \n is crucial to prevent matching across blocks
    matches = re.findall(pattern, markdown_content, re.DOTALL) # re.DOTALL to match across multiple lines

    if matches:
        llvm_code_blocks = matches

    return llvm_code_blocks


def gemini_predict(dataset_path: str, output_path: str):

    api_key = os.getenv('GEMINI_API_KEY').strip()
    client = genai.Client(api_key=api_key, http_options={'api_version':'v1alpha'})
    dataset = load_from_disk(dataset_path)

    try:
        with open(output_path, 'a') as f:
            for p in dataset:
                asm_code = p["asm"]["code"][-1]
                input_str = "decompile the x86 assembly to llvm ir: \n" + asm_code

                response = client.models.generate_content(
                    model='gemini-2.0-flash-thinking-exp', contents=input_str, config={
                        "response_logprobs": True, "response_lengt": 10
                    }
                )
                parts = [part.text for part in response.candidates[0].content.parts]
                # We need to extract the LLVM code from the response
                llvm_code = [extract_llvm_code(part) for part in parts]
                llvm_code = [code[0] for code in llvm_code if len(code) > 0]
                predict = llvm_code[0] if len(llvm_code) > 0 else ""
                out = {
                                "instruction": input_str,
                                "input": asm_code,
                                "predict": predict,
                                "raw_response": parts,
                                "file": p["path"],
                                "output": p["llvm_ir"]["code"],
                                "func_head_types": p["func_head_types"]
                }
                json.dump(out, f)
                f.write('\n')
                f.flush()
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    dir_path = "data/train_synth_rich_io_filtered_llvm_ir/train_synth_rich_io_filtered_0_llvm_extract_func_ir_assembly_O2_llvm_diff_sample_100"
    output_path = 'data/gemini-2.0-flash-thinking-exp-train_synth_rich_io_filtered_0_llvm_extract_func_ir_assembly_O2_llvm_diff_sample_100.json'
    gemini_predict(dir_path, output_path)
