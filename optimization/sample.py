import os
import re
import ast
import json
import random
import argparse
import time
from datetime import datetime
from jinja2 import Template
from termcolor import cprint
import multiprocessing as mp
import numpy as np
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

import optimization_config





os.environ["TOKENIZERS_PARALLELISM"] = "false" 





####### vllm inference #######

def worker_fn(worker_id, pretrained_model, gpu_ids, task_queue, result_queue, max_model_len, max_generation_token):
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"  # vLLM needs spawn for its internal GPU workers
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
    base_port = int(os.environ.get("VLLM_BASE_PORT", os.environ.get("VLLM_PORT", "29510")))
    os.environ["VLLM_PORT"] = str(base_port + worker_id * 10)

    print(f"Loading model on GPUs {gpu_ids} with VLLM_PORT={os.environ['VLLM_PORT']}...")
    llm = LLM(
        model=pretrained_model,
        dtype="bfloat16",
        tensor_parallel_size=len(gpu_ids),
        gpu_memory_utilization=0.85,
        max_model_len=max_model_len,
        disable_custom_all_reduce=True,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        async_scheduling=False,  # async scheduler causes generate() deadlock under concurrent GPU load
    )
    print(f"LLM done!")
    # === MUST_MIRROR_BEGIN(case_sampling_params): evaluation/eval.py:SamplingParams ===
    # Stop tokens MUST match between train and eval. In particular, "</answer>" MUST
    # NOT be a stop token: the parser regex requires the closing tag to be in the
    # captured output. (logprobs=1 is train-only; sampling distribution
    # may legitimately differ at eval time — temperature/top_p are eval-policy choices.)
    sampling_params = SamplingParams(
        temperature=temp,
        top_p=1.0,
        top_k=-1,
        min_p=0.0,
        max_tokens=max_generation_token,
        logprobs=1,
        stop=["User:", "Human:", "Assistant:", "<|im_end|>", "<|endoftext|>"]
    )
    # === MUST_MIRROR_END(case_sampling_params) ===

    while True:
        task = task_queue.get()
        if task == "STOP":
            print("Stopping worker...")
            break
        task_id, prompts = task
        t0 = time.time()
        print(f"[{datetime.now().strftime('%H:%M:%S')}] GPU {gpu_ids}: generating {len(prompts)} prompts...", flush=True)
        outputs = llm.generate(prompts, sampling_params)
        elapsed = time.time() - t0
        result_texts = [out.outputs[0].text for out in outputs]
        # Extract chosen-token logprobs as list of (decoded_token, logprob) tuples
        # Serialized as plain Python types to avoid pickling vLLM objects across processes
        result_logprobs = []
        for out in outputs:
            lp_list = out.outputs[0].logprobs
            token_ids = out.outputs[0].token_ids
            if lp_list is None:
                result_logprobs.append(None)
            else:
                pairs = []
                for j, lp_dict in enumerate(lp_list):
                    tid = token_ids[j] if j < len(token_ids) else None
                    entry = lp_dict.get(tid) if (tid is not None and lp_dict) else None
                    decoded = entry.decoded_token if entry else ""
                    lp = entry.logprob if entry else 0.0
                    pairs.append((decoded, lp))
                result_logprobs.append(pairs)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] GPU {gpu_ids}: generation done in {elapsed:.1f}s ({len(prompts)} prompts)", flush=True)
        result_queue.put((task_id, result_texts, result_logprobs))

# To run the worker setup:
def start_workers(pretrained_model, gpu_configs, max_model_len, max_generation_token):
    task_queues = []
    result_queues = []
    processes = []

    for i, gpu_ids in enumerate(gpu_configs):
        task_q = mp.Queue()
        result_q = mp.Queue()
        p = mp.Process(
            target=worker_fn,
            args=(i, pretrained_model, gpu_ids, task_q, result_q, max_model_len, max_generation_token)
        )
        p.start()
        task_queues.append(task_q)
        result_queues.append(result_q)
        processes.append(p)
    
    return task_queues, result_queues, processes

# Submit tasks
def submit_prompt_set(task_queues, prompt_sets):
    for i, prompts in enumerate(prompt_sets):
        task_queues[i].put((i, prompts))

# Collect results — poll with timeout so a dead worker process raises immediately
# instead of blocking forever on queue.get().
def collect_results(result_queues, num_sets, processes=None):
    import queue as _queue
    results = [None] * num_sets
    logprobs_sets = [None] * num_sets
    for idx, q in enumerate(result_queues):
        while True:
            try:
                item = q.get(timeout=5)
                task_id, result, lp_set = item
                results[task_id] = result
                logprobs_sets[task_id] = lp_set
                break
            except _queue.Empty:
                # Check if the worker is still alive; if not, bail out.
                if processes is not None and not processes[idx].is_alive():
                    exitcode = processes[idx].exitcode
                    raise RuntimeError(
                        f"vLLM worker process {idx} died (exitcode={exitcode}) "
                        f"without returning results."
                    )
    return results, logprobs_sets

# Stop workers
def stop_workers(task_queues, processes):
    for q in task_queues:
        q.put("STOP")
    for p in processes:
        p.join()

# Split prompts into N chunks
def split_prompts(prompts, n):
    k, m = divmod(len(prompts), n)
    return [prompts[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

def get_token_lengths(strings, tokenizer):
    return [len(tokenizer.encode(s, add_special_tokens=False)) for s in strings]

# vllm inference — returns (result_list, logprobs_list) aligned by prompt index
def generate_results(all_prompts, gpu_groups, task_queues, result_queues, processes=None):
    prompt_sets = split_prompts(all_prompts, len(gpu_groups))
    submit_prompt_set(task_queues, prompt_sets)
    results, logprobs_sets = collect_results(result_queues, len(prompt_sets), processes=processes)
    result_list = []
    logprobs_list = []
    for result_set, lp_set in zip(results, logprobs_sets):
        for r in result_set:
            result_list.append(r)
        for lp in (lp_set or [None] * len(result_set)):
            logprobs_list.append(lp)
    return result_list, logprobs_list

def extract_code(full_output):
    matches = re.findall(r"```python(.*?)```", full_output, re.DOTALL)
    if matches:
        code_output = matches[-1].strip()
    else:
        code_output = "We can not extract the code in the output. "
    return code_output

import random 
def random_select(data_list, random_k):
    data_list = random.sample(data_list, random_k)
    return data_list





def str2bool(x):
    return x.lower() in ("1", "true", "yes")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model", type=str, default=optimization_config.pretrained_model)
    parser.add_argument("--dataset", type=str, default=optimization_config.train_dataset)
    parser.add_argument("--k_code", type=int, default=optimization_config.k_code)
    parser.add_argument("--k_case", type=int, default=optimization_config.k_case)
    parser.add_argument("--max_model_len", type=int, default=optimization_config.max_model_len)
    parser.add_argument("--max_generation_token", type=int, default=optimization_config.max_generation_token)
    parser.add_argument("--temp", type=float, default=optimization_config.temp)
    parser.add_argument("--p_give_example", type=float, default=optimization_config.p_give_example)
    parser.add_argument("--max_input_examples", type=int, default=optimization_config.max_input_examples)
    parser.add_argument("--max_ground_truth_test", type=int, default=optimization_config.max_ground_truth_test)
    parser.add_argument("--random_select_num", type=int, default=optimization_config.n_sample_per_step)
    parser.add_argument("--gpu_groups", type=ast.literal_eval, default=optimization_config.gpu_groups)
    parser.add_argument("--system_prompts", type=str, default=optimization_config.system_prompts)
    parser.add_argument("--system_case_prompts", type=str, default=optimization_config.system_case_prompts)
    parser.add_argument("--special_requirements", type=str, default=optimization_config.special_requirements)
    parser.add_argument("--post_stage", type=str2bool, default=optimization_config.post_stage)
    return parser.parse_args()


if __name__ == "__main__":
    mp.freeze_support()
    args = parse_args()
    globals().update(vars(args))

    # post_stage=True would set p_give_example=0.0 (no example in prompt)
    # Keep False — we always send the example (p_give_example=1.0)
    if post_stage == True:
        p_give_example = 0.0


    # read dataset
    with open("../data/" + dataset + ".json", 'r') as f:
        data = json.load(f)
    #data = [data[i] for i in range(10)]
    random_select_num = min(random_select_num, len(data))
    data = random_select(data, random_select_num)
    num = len(data)


    # load model, tokenizer, build vllm engines...
    task_queues, result_queues, processes = start_workers(pretrained_model, gpu_groups, max_model_len, max_generation_token)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    outputs_name = "rl-" + pretrained_model.replace("/", ".") + "-" + dataset









def bernoulli(p):
    return 1 if random.random() < p else 0

# obtain prompt
def get_scaling_prompt(data_i, method):
    problem = data_i["question"]
    if method == "sample":
        return Template(system_prompts).render(language = "python", special_requirements = special_requirements, problem = problem)
    if method == "case":
        # === MUST_MIRROR_BEGIN(case_example_intro): evaluation/eval.py:get_scaling_prompt ===
        # Few-shot example rendering. Note: eval uses min(k_case, len(...)) for n_example
        # which is functionally equivalent (k_case >> typical example count) but is a
        # known cosmetic difference flagged by the drift checker.
        n_example = len(data_i["example_input"])
        example_input = ", ".join([repr(item) for item in data_i['example_input']])
        example_output = ", ".join([repr(item) for item in data_i['example_output']])
        if n_example == 0:
            example_intro = """ """
        if n_example == 1:
            example_intro = """We already have one test sample:\n Its input is {{example_input}}. Its output is {{example_output}}.\n"""
            example_intro = Template(example_intro).render(example_input = example_input, example_output = example_output)
        if n_example > 1:
            example_intro = """We already have {{n_sample}} test samples:\n The inputs are, respectively, {{example_input}}. The corresponding outputs are {{example_output}}.\n"""
            example_intro = Template(example_intro).render(n_sample = n_example, example_input = example_input, example_output = example_output)
        return Template(system_case_prompts).render(problem = problem, example_intro = example_intro)
        # === MUST_MIRROR_END(case_example_intro) ===

# === MUST_MIRROR_BEGIN(case_modify): evaluation/eval.py:modify ===
# Train and eval must strip the same leaked first-line tokens or otherwise the eval
# parser will accept content the training parser rejects (or vice versa).
_LANG_SPECIFIERS = {"plaintext", "input", "output", "text", "txt", "markdown", "python", "bash"}

def modify(c):
    # Strip leading fenced-code-block language specifier (e.g. "input\n", "output\n", "plaintext\n")
    first_newline = c.find("\n")
    if first_newline != -1 and c[:first_newline].strip().lower() in _LANG_SPECIFIERS:
        c = c[first_newline + 1:]

    # Convert literal "\n" to actual newlines
    c = c.replace("\\n", "\n")

    # Ensure there's a trailing newline
    if not c.endswith("\n"):
        c += "\n"

    return c
# === MUST_MIRROR_END(case_modify) ===

def compute_oracle_logprob(full_text, oracle_text, token_pairs):
    """Geometric mean token logprob over the output section. Returns 0.5 on failure."""
    if not token_pairs or not oracle_text or not oracle_text.strip():
        return 0.5
    try:
        reconstructed = "".join(t for t, _ in token_pairs)
        # Try new <output> tag format first, then legacy **Test Output:** marker
        marker_pos = -1
        for marker in ("<output>", "**Test Output:**"):
            marker_pos = reconstructed.rfind(marker)
            if marker_pos != -1:
                marker_pos += len(marker)
                break
        if marker_pos == -1:
            return 0.5
        # Locate oracle_text directly (skips ```plaintext fence boilerplate)
        oracle_text_clean = oracle_text.strip()
        if not oracle_text_clean:
            return 0.5
        oracle_start = reconstructed.find(oracle_text_clean, marker_pos)
        if oracle_start == -1:
            return 0.5
        oracle_end = oracle_start + len(oracle_text_clean)
        cumpos = 0
        oracle_lps = []
        for t_str, lp in token_pairs:
            t_end = cumpos + len(t_str)
            if t_end > oracle_start and cumpos < oracle_end:
                oracle_lps.append(lp)
            cumpos = t_end
            if cumpos > oracle_end + 50:
                break
        if not oracle_lps:
            return 0.5
        return float(np.exp(np.mean(oracle_lps)))
    except Exception:
        return 0.5


# extract the unit tests from responses
# === MUST_MIRROR_BEGIN(case_extract): evaluation/eval.py:extract_test_cases ===
# Parser priority order (XML primary, then markdown fallbacks) MUST match between
# train and eval. If train accepts a format eval rejects (or vice versa), training
# rewards behavior that eval scores as a parse failure → silent quality regression.
def extract_test_cases(full_output):
    fail_case = [""]

    # Normalize Unicode angle brackets to ASCII so tag patterns work uniformly
    normalized = full_output.replace('〈', '<').replace('〉', '>') \
                            .replace('⟨', '<').replace('⟩', '>')

    # Primary: <answer><input>...</input><output>...</output></answer> format
    # </answer> is in the output (not a stop token) so this match is reliable
    m_answer = re.search(r'<answer>(.*?)</answer>', normalized, re.DOTALL)
    if m_answer:
        answer_block = m_answer.group(1)
        m_in  = re.search(r'<input>(.*?)</input>',   answer_block, re.DOTALL)
        m_out = re.search(r'<output>(.*?)</output>', answer_block, re.DOTALL)
        if m_in and m_out:
            test_input  = [modify(m_in.group(1).strip())]
            test_output = [modify(m_out.group(1).strip())]
            example_text = [normalized[normalized.rfind('<answer>'):]]
            return test_input, test_output, example_text

    # Fallback: legacy **Test Input:** / **Test Output:** with backtick blocks
    pattern_input_backticks  = r'\*\*Test Input:\*\*\s*```(.*?)```'
    pattern_output_backticks = r'\*\*Test Output:\*\*\s*```(.*?)```'
    matches_input  = re.findall(pattern_input_backticks,  normalized, re.DOTALL)
    matches_output = re.findall(pattern_output_backticks, normalized, re.DOTALL)
    if matches_input and matches_output:
        test_input  = [modify(matches_input[-1].lstrip('\n'))]
        test_output = [modify(matches_output[-1].lstrip('\n'))]
        index = normalized.rfind("**Test Input:**")
        example_text = [normalized[index:]]
        return test_input, test_output, example_text

    # Fallback: plain-text **Test Input:** / **Test Output:** without backticks
    m_in_plain  = re.search(r'\*\*Test Input:\*\*\s*([\s\S]*?)(?=\*\*Test Output:\*\*)', normalized, re.DOTALL)
    m_out_plain = re.search(r'\*\*Test Output:\*\*\s*([\s\S]*?)(?=\*\*Explanation:|\*\*Test Input:|</answer>|$)', normalized, re.DOTALL)
    if m_in_plain and m_out_plain:
        test_input  = [modify(m_in_plain.group(1).strip())]
        test_output = [modify(m_out_plain.group(1).strip())]
        index = normalized.rfind("**Test Input:**")
        example_text = [normalized[index:]]
        return test_input, test_output, example_text

    return fail_case, fail_case, fail_case
# === MUST_MIRROR_END(case_extract) ===


if __name__ == "__main__":



    # initialization
    code_generation_prompts = []
    code_index = []
    case_generation_prompts = []
    case_index = []
    for i in range(num):
        # preprocess
        data[i]["full_code_generation"] = []
        data[i]["code_response_length"] = []
        data[i]["full_case_generation"] = []
        data[i]["case_response_length"] = []
        data[i]["generated_code"] = []
        max_k = min(max_ground_truth_test, len(data[i]["test_input"]))
        data[i]["num_ground_truth_test"] = max_k 
        data[i]["all_case_input"] = (data[i]["test_input"][:max_k]).copy()
        data[i]["all_case_output"] = (data[i]["test_output"][:max_k]).copy()
        data[i]["case_input"] = []
        data[i]["case_output"] = []
        data[i]["case_text"] = []
        data[i]["case_parse_ok"] = [True] * max_k
        data[i]["oracle_status"] = ["ground_truth"] * max_k

        data_i = data[i].copy()
        # get code generation prompts
        prompt_i = get_scaling_prompt(data_i, "sample")
        data[i]["code_generation_prompt"] = prompt_i
        code_generation_prompts = code_generation_prompts + [prompt_i] * k_code
        code_index = code_index + [i] * k_code
        # get case generation prompts
        #k_case_generate = k_case - min(k_case, len(data_i["example_input"]))
        k_case_generate = k_case
        if_give_example = bernoulli(p_give_example)
        if if_give_example == 0:
            data_i["example_input"] = []
            data_i["example_output"] = []
            data[i]["no_example"] = True
        else:
            max_input_examples_n = min(max_input_examples, len(data_i["example_input"]))
            data_i["example_input"] = data_i["example_input"][:max_input_examples_n]
            data_i["example_output"] = data_i["example_output"][:max_input_examples_n]
            data[i]["no_example"] = False
        prompt_i = get_scaling_prompt(data_i, "case")
        data[i]["case_generation_prompt"] = prompt_i

        if k_case_generate > 0:
            case_generation_prompts = case_generation_prompts + [prompt_i] * k_case_generate
            case_index = case_index + [i] * k_case_generate








    # sampling process

    cprint("start generation...", "green")

    # shuffle first, to achieve efficiency
    all_prompts = code_generation_prompts + case_generation_prompts
    N = len(all_prompts)
    indices = list(range(N))
    shuffled_idx = indices[:]
    random.shuffle(shuffled_idx)
    shuffled_prompts = [all_prompts[i] for i in shuffled_idx]
    # generate — returns (texts, logprobs) aligned by prompt index
    shuffled_outputs, shuffled_logprobs = generate_results(shuffled_prompts, gpu_groups, task_queues, result_queues, processes=processes)
    restored_outputs = [None] * N
    restored_logprobs = [None] * N
    for out, lp, idx in zip(shuffled_outputs, shuffled_logprobs, shuffled_idx):
        restored_outputs[idx] = out
        restored_logprobs[idx] = lp
    code_generation_result = restored_outputs[:len(code_generation_prompts)]
    case_generation_result = restored_outputs[len(code_generation_prompts):]
    case_logprobs = restored_logprobs[len(code_generation_prompts):]

    cprint("generation job done!", "green")








    # calculate the response length
    code_response_length = get_token_lengths(code_generation_result, tokenizer)
    case_response_length = get_token_lengths(case_generation_result, tokenizer)
    mean_code = sum(code_response_length)/len(code_response_length)
    mean_case = sum(case_response_length)/len(case_response_length)

    os.makedirs(os.path.dirname("./results/results-" + outputs_name + ".txt"), exist_ok=True)
    with open("./results/results-" + outputs_name + ".txt", "a") as f:
        def save_and_print(text):
            cprint(text, color="green")
            f.write(text + "\n")
        save_and_print(f"code response length: {mean_code}, case response length: {mean_case}")

    # process generated codes
    i = 0
    for full_output in code_generation_result:
        code_output = extract_code(full_output)
        index_i = code_index[i]
        data[index_i]["full_code_generation"] = data[index_i]["full_code_generation"] + [full_output]
        data[index_i]["generated_code"] = data[index_i]["generated_code"] + [code_output]
        data[index_i]["code_response_length"].append(code_response_length[i])
        i += 1

    # process generated unit tests
    i = 0
    for full_output in case_generation_result:
        test_input, test_output, example_text = extract_test_cases(full_output)
        index_i = case_index[i]
        parse_ok = bool(test_input and test_output and test_input[0] and test_output[0])
        # compute p_oracle: geometric mean logprob over the oracle output tokens
        oracle_str = test_output[0] if test_output and test_output[0] else ""
        token_pairs = case_logprobs[i] if i < len(case_logprobs) else None
        p_oracle = compute_oracle_logprob(full_output, oracle_str, token_pairs)
        data[index_i]["full_case_generation"] = data[index_i]["full_case_generation"] + [full_output]
        data[index_i]["case_input"] = data[index_i]["case_input"] + test_input
        data[index_i]["case_output"] = data[index_i]["case_output"] + test_output
        data[index_i]["case_text"] = data[index_i]["case_text"] + example_text
        data[index_i]["all_case_input"] = data[index_i]["all_case_input"] + test_input
        data[index_i]["all_case_output"] = data[index_i]["all_case_output"] + test_output
        data[index_i]["case_response_length"].append(case_response_length[i])
        data[index_i]["oracle_logprob"] = data[index_i].get("oracle_logprob", []) + [p_oracle]
        data[index_i]["case_parse_ok"].append(parse_ok)
        data[index_i]["oracle_status"].append("pending" if parse_ok else "parse_fail")
        i += 1

    # output the data
    os.makedirs(os.path.dirname("./temp_data/outputs-" + outputs_name + ".json"), exist_ok=True)
    with open("./temp_data/outputs-" + outputs_name + ".json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)



    stop_workers(task_queues, processes)







