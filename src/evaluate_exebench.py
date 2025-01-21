
import os
import json
import subprocess
import logging
import pathlib
import shutil
from typing import Dict
from multiprocessing import Pool

import tqdm
import fire
from datasets import load_from_disk
from exebench import Wrapper, diff_io, exebench_dict_to_dict, LLVMAssembler

from extract_code import extract_llmcompiler_code_blocks

logging.basicConfig(format='%(asctime)s - %(filename)s:%(lineno)s - %(message)s ', level=logging.INFO)

# validation_dir = "/home/xiachunwei/Projects/alpaca-lora-decompilation/tmp_validate_exebench"

def compile_target_ir(target_llvm_ir: str, full_path: str)->bool:
    target_llvm_ir_path = os.path.join(full_path, "target.ll")
    target_assembly_path = os.path.join(full_path, "target.s")
    predict_error_path = os.path.join(full_path, "error_predict.error")
    with open(target_llvm_ir_path, 'w') as f:
        f.write(target_llvm_ir[0] if isinstance(target_llvm_ir, list) else target_llvm_ir)
    target_success = False
    try:
        # 3. Compile the ground truth llvm ir to assembly
        cmd = ["llc", target_llvm_ir_path, "-o", target_assembly_path]
        ret = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if ret.returncode == 0:
            target_success = True
        else:
            # Save the stderr output to the specified file
            with open(predict_error_path, 'a') as f:
                f.write(ret.stderr.decode())
            target_success = False
    except Exception as e:
        logging.error(e)
        target_success = False
    return target_success, target_assembly_path


def compile_predicted_record(
        predict_llvm_ir: str,
        full_path: str)->bool:
    """Compile the llvm ir to assembly and save the results to the validation directory, return true if success compile"""
    # 1. First save LLVM IR and assembly to file
    predict_llvm_ir_path = os.path.join(full_path, "predict.ll")
    predict_assembly_path = os.path.join(full_path, "predict.s")
    predict_error_path = os.path.join(full_path, "error_predict.error")

    with open(predict_llvm_ir_path, 'w') as f:
        f.write(predict_llvm_ir)
    predict_success = True
    try:
        # 2. Compile predicted llvm ir to assembly
        cmd = ["llc", predict_llvm_ir_path, "-o", predict_assembly_path]
        ret = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if ret.returncode != 0:
            # Save the stderr output to the specified file
            with open(predict_error_path, 'w') as f:
                f.write(ret.stderr.decode())
            predict_success = False
    except Exception as e:
        logging.error(e)
        predict_success = False
    
    return predict_success, predict_assembly_path


def eval_assembly(row: Dict, assembly: str) -> bool:
    success = True
    synth_wrapper = None
    try:
        c_deps=(row['synth_deps'] + '\n' +
                    row['synth_io_pairs']['dummy_funcs'][0] + '\n').replace(
                        'typedef int bool;', '')
        synth_wrapper = Wrapper(
            c_deps=c_deps + '\n',
            func_c_signature=row['func_head_types'].replace('extern', ''),
            func_assembly=assembly,
            cpp_wrapper=row['synth_exe_wrapper'],
            assembler_backend=LLVMAssembler())
        count, total = 0, len(row['synth_io_pairs']['input'])
        for i, o in zip(row['synth_io_pairs']['input'],
                        row['synth_io_pairs']['output']):
            observed_output = synth_wrapper(
                exebench_dict_to_dict(i))  # Run synthetic
            if observed_output is None:
                logging.error('Error: The code could not be compiled')
                success = False
                return success
            # print(observed_output, exebench_dict_to_dict(o))
            count += 1 if diff_io(
                observed_output=observed_output,
                expected_output=exebench_dict_to_dict(o)) else 0
        success = (count == total)
        if not success:
            logging.info(
                f"Error for {row['path']} total cases {total}, success cases {count}"
            )
    except Exception as e:
        logging.error(f"Error for {row['path']}")
        logging.error(e)
        success = False
    finally:
        return success


def validate_by_execution(record: Dict, row: Dict, validation_dir:str)->Dict:
    # 1. First validate the target assembly
    file_path = record['file']
    full_path = os.path.join(validation_dir, file_path, row['fname'])
    pathlib.Path(full_path).mkdir(parents=True, exist_ok=True)
    if isinstance(record["output"], list):
        record["output"] = record["output"][0]
    target_success, target_assembly_path = compile_target_ir(record["output"], full_path)
    target_execution_success = False
    # Validate the target assembly
    if target_success:
        try:
            with open(target_assembly_path, 'r') as f:
                target_execution_success = eval_assembly(row, f.read())
        except Exception as e:
            logging.error(e)
    record["target_compile_success"] = target_success
    record["target_execution_success"] = target_execution_success
    # 2. Validate the predicted assembly
    if isinstance(record["predict"], str):
        record['predict'] = [record['predict'], ]
    if isinstance(record["predict"], list):        
        record["predict_compile_success"] = []
        record["predict_execution_success"] = []
        for predict in record["predict"]:
            predict_success, predict_assembly_path = compile_predicted_record(predict, full_path)
            predict_execution_success = False
            # Validate the predict assembly
            if predict_success:
                try:
                    with open(predict_assembly_path, 'r') as f:
                        predict_execution_success = eval_assembly(row, f.read())
                except Exception as e:
                    logging.error(e)
            record["predict_compile_success"].append(predict_success)
            record["predict_execution_success"].append(predict_execution_success)
        print((record["predict_compile_success"], record["predict_execution_success"], target_success, target_execution_success))
    else:
        logging.error(f"Invalid format of record['predict']: {record['predict']}")
    return record


def wrapper(args):
    if len(args) != 3 or not isinstance(args[0], dict) or not isinstance(args[1], dict):
        logging.error(f"Invalid input: {args}")
        return None
    return validate_by_execution(*args)


def format_path_and_func_def(path: str, func_def: str)->str:
    return str(path)+":"+str(func_def)


def preprocess_records(all_records: list[Dict])->Dict:
    """Preprocess the records to make sure the format is correct.
    Note the record here means the output of the LLM model with instruction and assembly code.

    Parameters:
    all_records: list[Dict], the output of the LLM model with instruction and assembly code.

    Returns:
    path_to_record_mapping: Dict, a mapping from the (file_path:func_def) to the record.
    """
    path_to_record_mapping = {}
    for record in tqdm.tqdm(all_records):
        # Preprocessing the LLM output here:
        if isinstance(record["predict"], str):
            record["predict"] = [record["predict"], ]
        if isinstance(record["predict"], list):
            new_predict_list = []
            for predict in record["predict"]:
                # For llmcompiler, the output is wrapped in code block
                if predict.find("code") >= 0:
                    matched_predict_llvm_ir = extract_llmcompiler_code_blocks(predict)
                    if matched_predict_llvm_ir and len(matched_predict_llvm_ir) > 0:
                        new_predict_list.append(matched_predict_llvm_ir[0])
                    else:
                        # logging.error(f"Cannot find code block in {predict}")
                        logging.error(f"Cannot find code block in {record['file']}")
                        new_predict_list.append(predict)
                else:
                    new_predict_list.append(predict)
                if predict.find("aarch64") >= 0:
                    logging.error(f"Find aarch64 in {record['file']}")
            record["predict"] = new_predict_list
        
        path_to_record_mapping[format_path_and_func_def(record['file'], record["func_head_types"])] = record

    return path_to_record_mapping


def match_record_with_row(path_to_record_mapping: Dict, path_to_row_mapping: Dict):
    # We need also to make sure the function name is the same
    path_to_record_row_mapping = {}
    for path_func_def, record in path_to_record_mapping.items():
        if path_func_def in path_to_row_mapping:
            path_to_record_row_mapping[path_func_def] = (record, path_to_row_mapping[path_func_def])
        else:
            logging.error(f"Cannot find record for {path_func_def}")
    return path_to_record_row_mapping

        
def validate_exebench(path_to_json: str = "fexebench_train_synth_rich_io_filtered_llvm_ir_0_llm-compiler-13b-ftd-rl-ppo-step-80-bs-32-beams-1.json", 
                      path_to_dataset: str = "/home/xiachunwei/Datasets/filtered_exebench/train_synth_rich_io_filtered_llvm_ir/train_synth_rich_io_filtered_0_llvm_extract_func_ir_assembly_O2", 
                      path_to_result: str = "exebench_train_synth_rich_io_filtered_llvm_ir_0_llm-compiler-13b-ftd-rl-ppo-step-80-bs-32-beams-1_validate_exebench.json",
                      validation_dir: str = "/home/xiachunwei/Projects/alpaca-lora-decompilation/tmp_validate_exebench"):
    dataset = load_from_disk(
        path_to_dataset
    )
    path_to_row_mapping = {}
    for row in dataset:
        path_to_row_mapping[format_path_and_func_def(row['path'], row["func_head_types"])] = row
    if os.path.exists(validation_dir):
        shutil.rmtree(validation_dir)
    pathlib.Path(validation_dir).mkdir(parents=True, exist_ok=True)
    all_records = json.load(open(path_to_json, 'r'))
    path_to_record_mapping = preprocess_records(all_records)
    path_to_record_row_mapping = match_record_with_row(path_to_record_mapping, path_to_row_mapping)

    # Run in parallel
    args = [value + (validation_dir,) for _, value in path_to_record_row_mapping.items()]
    with Pool(processes=80) as pool:
        results = pool.map(wrapper, args)
    
    predict_compile_results = [any(r["predict_compile_success"]) if isinstance(r, dict) else False for r in results]
    predict_execution_results = [any(r["predict_execution_success"]) if isinstance(r, dict) else False for r in results]
    target_compile_results = [r["target_compile_success"] if isinstance(r, dict) else False for r in results]
    target_execution_results = [r["target_execution_success"] if isinstance(r, dict) else False for r in results]
    logging.info(f"""Total records: {len(all_records)}, 
                 predict_compile_success:{sum(predict_compile_results)}, 
                 predict_execution_success: {sum(predict_execution_results)},
                 target_compile_success: {sum(target_compile_results)},
                 target_execution_success: {sum(target_execution_results)}""")
    json.dump(results, open(path_to_result, 'w'), indent=4, sort_keys=False, separators=(',', ':'))


if __name__ == "__main__":    
    fire.Fire(validate_exebench)
