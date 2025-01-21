

## Decompilation using Chain-of-Thought

### Install dependencies

- Install [exebench](https://github.com/summerspringwei/exebench): You don't need to download the dataset, just clone this repo to your local dir, set the `PYTHONPATH="path/to/exebench/exebench:$PYTHONPATH"`

- Install python packages: 
```shell
pip install -r requirements.txt
```


### Run the code:

1. Use Gemini to do the decompilation task:
```shell
python3 src/gemini_decompilation.py 
```
This is an example on how to load json and use Gemini to do the decompilation.
**Note: This will consume around 10 miniutes to finish and consume aroun 1 pound money**

2. Validate whether the prediction can pass the compilation and IO test.
```bash
python3 src/evaluate_exebench.py \
--path_to_json "data/gemini-2.0-flash-thinking-exp-train_synth_rich_io_filtered_0_llvm_extract_func_ir_assembly_O2_llvm_diff_sample_100_new.json", 
--path_to_dataset "data/train_synth_rich_io_filtered_0_llvm_extract_func_ir_assembly_O2_llvm_diff_sample_100", 
--path_to_result "gemini-2.0-flash-thinking-exp-train_synth_rich_io_filtered_0_llvm_extract_func_ir_assembly_O2_llvm_diff_sample_100_new_validate_exebench.json",
--validation_dir "tmp_validate_exebench"
```

### How to compile LLVM IR to assembly

Save the llvm ir to a file, let's say `tmp.ll`, then:
```shell
llc tmp.ll -o tmp.s
```
will get the assembly file `tmp.s`.


### TODO
1. First read through the `gemini_decompilation.py` to understand how the gemini api works.

2. Use the [Gemini](https://gemini.google.com) to try to decompile the assembly code to llvm ir.
You can open an premire account with 1 month free.

3. Use the feed back from `llc` compiler to guide the Gemini to produce the correct answer. This may require you to write some python code;

4. Read through the `evaluate_exebench.py` to write the code the verify the correctness of the predicted llvm ir.
