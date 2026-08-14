import json
import os
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from collections import defaultdict
from tqdm import tqdm
import argparse

from verifier import labeling_responses

THOUGHT_DELIMITER_START = "<think>"
THOUGHT_DELIMITER_END = "</think>"





def generate_vllm(messages, model_path, template='own', temperature=0.7, top_p=0.95, max_tokens=8192, n=1, gpu_memory_utilization=0.8, args=None):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    stop_token_ids = [151645, 151643]
    sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens, n=n, repetition_penalty=args.repetition_penalty, presence_penalty=args.presence_penalty, stop_token_ids=stop_token_ids)
    llm = LLM(model=model_path, tensor_parallel_size=args.tensor_parallel_size, gpu_memory_utilization=gpu_memory_utilization)

    gen_prompts = []
    for cur_message in messages:
        if template == 'own':
            gen_prompt = tokenizer.apply_chat_template(cur_message, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        elif template == 'simplerl':
            gen_prompt = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n' + cur_message[0]['content'] + '\nPlease reason step by step, and put your final answer within\\boxed{{}}.<|im_end|>\n<|im_start|>assistant\n'
        else:
            gen_prompt = cur_message[0]['content']
        gen_prompts.append(gen_prompt)
    outputs = llm.generate(gen_prompts, sampling_params)
    return outputs

def main(args):
    df = pd.read_parquet(args.input_file)

    messages = df['prompt'].tolist()
    answers = [answer['ground_truth'] for answer in df['reward_model'].tolist()]
    data_sources = df['data_source'].tolist()

    outputs = generate_vllm(messages, args.model_path, template=args.template, temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens, n=args.n, gpu_memory_utilization=args.gpu_memory_utilization, args=args)

    save_data = []
    rets = defaultdict(list)
    avg = 0

    for i in tqdm(range(len(outputs))):
        prompt = outputs[i].prompt
        answer = answers[i]
        ds = data_sources[i]

        # 存放当前 prompt 的所有生成文本和正确性
        generated_texts = []
        correctness_list = []

        for j in range(args.n):
            generated_text = outputs[i].outputs[j].text

            try:
                labels = labeling_responses([generated_text], answer)
            except:
                labels = [False]

            correctness_list.append(labels[0])
            avg += int(labels[0])
            generated_texts.append(generated_text)

        # 把 n 个 rollout 合并成一个 item
        item = {
            'prompt': prompt,
            'answer': answer,
            'correctness': correctness_list,  # list of bools
            'data_source': ds
        }
        if args.save_generated_text:
            item['generated_texts'] = generated_texts  # list of generated texts
        save_data.append(item)

        # 更新每个 data_source 的统计
        rets[ds].extend(correctness_list)

    # 输出 accuracy
    print('accuracy:', avg / (len(outputs) * args.n))
    for ds, labels in rets.items():
        print(f'{ds}: {np.mean(labels)}')

    print(f"Saving results to {args.output_file}")
    # 保存到 JSONL，每行一个 prompt，包含 n 个 rollout
    with open(args.output_file, 'w') as f:
        for item in save_data:
            f.write(json.dumps(item) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default="./evaluation_data/valid_with_ood.parquet")
    parser.add_argument('--model_beautiful_name', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--n', type=int, default=8, help="number of outputs per prompt")
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--presence_penalty", type=float, default=0.0)
    parser.add_argument('--max_tokens', type=int, default=8192)
    parser.add_argument('--template', type=str, default='own')
    parser.add_argument('--save_generated_text', default=True)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    parser.add_argument("--tensor_parallel_size", type=int, default=4)
    args = parser.parse_args()

    os.makedirs("./llm_outputs", exist_ok=True)
    if args.input_file == "./evaluation_data/valid_without_mmlu_pro.parquet":
        args.output_file = f"./llm_output/{args.model_beautiful_name}_{args.template}_{args.temperature}_{args.top_p}_{args.repetition_penalty}_{args.presence_penalty}_{args.max_tokens}.jsonl"
    elif args.input_file == "./evaluation_data/valid.mmlu_pro.parquet":
        args.output_file = f"./llm_output/mmlu_pro_{args.model_beautiful_name}_{args.template}_{args.temperature}_{args.top_p}_{args.repetition_penalty}_{args.presence_penalty}_{args.max_tokens}.jsonl"
    elif args.input_file == "../training_data/valid_math500.parquet":
        args.output_file = f"./llm_output/math500_{args.model_beautiful_name}_{args.template}_{args.temperature}_{args.top_p}_{args.repetition_penalty}_{args.presence_penalty}_{args.max_tokens}.jsonl"
    else:
        raise NotImplementedError

    main(args)