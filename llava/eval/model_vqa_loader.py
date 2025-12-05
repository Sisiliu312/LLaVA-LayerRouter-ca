import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids, image_tensor, image.size

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    input_ids, image_tensors, image_sizes = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    return input_ids, image_tensors, image_sizes


# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config)

    for (input_ids, image_tensor, image_sizes), line in tqdm(zip(data_loader, questions), total=len(questions)):
        idx = line["question_id"]
        cur_prompt = line["text"]

        input_ids = input_ids.to(device='cuda', non_blocking=True)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                image_sizes=image_sizes,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True)

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        # ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    eval_model(args)



# 测试代码
# import argparse
# import torch
# import os
# import json
# from tqdm import tqdm
# import shortuuid

# from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
# from llava.conversation import conv_templates, SeparatorStyle
# from llava.model.builder import load_pretrained_model
# from llava.utils import disable_torch_init
# from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

# from PIL import Image


# # ============ 诊断Hook：捕获中间值 ============
# class DiagnosticHook:
#     def __init__(self):
#         self.router_outputs = []
#         self.ca_outputs = []
#         self.combined_features_stats = []
#         self.nan_detected = False
        
#     def reset(self):
#         self.router_outputs.clear()
#         self.ca_outputs.clear()
#         self.combined_features_stats.clear()
#         self.nan_detected = False

# diagnostic = DiagnosticHook()


# def to_json_serializable(obj):
#     """将torch tensor或numpy array转换为JSON可序列化类型"""
#     if isinstance(obj, torch.Tensor):
#         return obj.detach().cpu().numpy().tolist()
#     elif hasattr(obj, 'tolist'):  # numpy array
#         return obj.tolist()
#     elif isinstance(obj, (list, tuple)):
#         return [to_json_serializable(item) for item in obj]
#     elif isinstance(obj, dict):
#         return {key: to_json_serializable(value) for key, value in obj.items()}
#     else:
#         return obj


# def hook_router_output(module, input, output):
#     """捕获Router的输出"""
#     if len(output) >= 3:
#         top_indices, top_weights, layer_probs = output[:3]
#         diagnostic.router_outputs.append({
#             'indices': to_json_serializable(top_indices),
#             'weights': to_json_serializable(top_weights),
#             'probs': to_json_serializable(layer_probs),
#             'entropy': -(layer_probs * torch.log(layer_probs + 1e-10)).sum().item()
#         })


# def hook_ca_output(module, input, output):
#     """捕获CrossAttention的输出"""
#     if output is not None:
#         diagnostic.ca_outputs.append({
#             'mean': output.mean().item(),
#             'std': output.std().item(),
#             'min': output.min().item(),
#             'max': output.max().item(),
#             'has_nan': torch.isnan(output).any().item(),
#             'has_inf': torch.isinf(output).any().item()
#         })
#         if torch.isnan(output).any():
#             diagnostic.nan_detected = True

# def check_model_weights(model):
#     """检查模型权重是否有NaN"""
#     print("\n" + "="*60)
#     print("🔍 Checking model weights for NaN/Inf:")
    
#     nan_params = []
#     for name, param in model.named_parameters():
#         if torch.isnan(param).any():
#             nan_params.append(name)
#             print(f"  ⚠️ NaN in {name}")
#         elif torch.isinf(param).any():
#             nan_params.append(name)
#             print(f"  ⚠️ Inf in {name}")
    
#     if not nan_params:
#         print("  ✅ All weights are clean")
#     else:
#         print(f"\n  ⚠️ Total {len(nan_params)} parameters have NaN/Inf")
    
#     print("="*60 + "\n")
#     return nan_params


# def hook_combined_features(module, input, output):
#     """捕获encode_images的最终输出"""
#     if output is not None:
#         diagnostic.combined_features_stats.append({
#             'mean': output.mean().item(),
#             'std': output.std().item(),
#             'min': output.min().item(),
#             'max': output.max().item(),
#             'has_nan': torch.isnan(output).any().item(),
#             'has_inf': torch.isinf(output).any().item()
#         })


# def eval_model_with_diagnosis(args):
#     disable_torch_init()
#     model_path = os.path.expanduser(args.model_path)
#     model_name = get_model_name_from_path(model_path)
#     tokenizer, model, image_processor, context_len = load_pretrained_model(
#         model_path, args.model_base, model_name
#     )
#     nan_params = check_model_weights(model)
    
#     # ✅ 注册Hook
#     hooks = []
#     if hasattr(model.model, 'layer_router'):
#         print("✅ Registering Router hook")
#         h = model.model.layer_router.register_forward_hook(hook_router_output)
#         hooks.append(h)
#     else:
#         print("⚠️ No layer_router found in model")
    
#     if hasattr(model.model, 'ca'):
#         print("✅ Registering CA hook")
#         h = model.model.ca.register_forward_hook(hook_ca_output)
#         hooks.append(h)
#     else:
#         print("⚠️ No CA found in model")
    
#     # Hook到encode_images
#     original_encode_images = model.encode_images
#     def wrapped_encode_images(images, text_token):
#         result = original_encode_images(images, text_token)
#         hook_combined_features(None, None, result)
#         return result
#     model.encode_images = wrapped_encode_images
    
#     # ============ 测试数据 ============
#     questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
#     questions = questions[:args.num_samples]
    
#     answers_file = os.path.expanduser(args.answers_file)
#     os.makedirs(os.path.dirname(answers_file), exist_ok=True)
#     ans_file = open(answers_file, "w")
    
#     diagnosis_file = open(answers_file.replace('.jsonl', '_diagnosis.jsonl'), "w")
    
#     # ============ 统计信息 ============
#     empty_count = 0
#     nan_count = 0
    
#     # ============ 逐样本测试 ============
#     for idx, line in enumerate(tqdm(questions)):
#         diagnostic.reset()
        
#         image_file = line["image"]
#         qs = line["text"]
        
#         if model.config.mm_use_im_start_end:
#             qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
#         else:
#             qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        
#         conv = conv_templates[args.conv_mode].copy()
#         conv.append_message(conv.roles[0], qs)
#         conv.append_message(conv.roles[1], None)
#         prompt = conv.get_prompt()
        
#         # 加载图像
#         image = Image.open(os.path.join(args.image_folder, image_file)).convert('RGB')
#         image_tensor = process_images([image], image_processor, model.config)[0]
        
#         input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
#         input_ids = input_ids.unsqueeze(0).to(device='cuda', non_blocking=True)
        
#         # ============ 生成 ============
#         with torch.inference_mode():
#             try:
#                 output_ids = model.generate(
#                     input_ids,
#                     images=image_tensor.unsqueeze(0).to(dtype=torch.float16, device='cuda'),
#                     image_sizes=[image.size],
#                     do_sample=True if args.temperature > 0 else False,
#                     temperature=args.temperature,
#                     top_p=args.top_p,
#                     num_beams=args.num_beams,
#                     max_new_tokens=args.max_new_tokens,
#                     use_cache=True
#                 )
                
#                 outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
                
#             except Exception as e:
#                 outputs = f"[ERROR] {str(e)}"
#                 print(f"⚠️ Sample {idx} failed: {e}")
        
#         # 统计
#         if len(outputs) == 0:
#             empty_count += 1
#         if diagnostic.nan_detected:
#             nan_count += 1
        
#         # ============ 保存结果 ============
#         ans_id = shortuuid.uuid()
#         result = {
#             "question_id": line.get("question_id", idx),
#             "prompt": line["text"],
#             "text": outputs,
#             "answer_id": ans_id,
#             "model_id": model_name,
#             "metadata": {}
#         }
#         ans_file.write(json.dumps(result) + "\n")
        
#         # ============ 保存诊断信息 ============
#         diagnosis = {
#             "question_id": line.get("question_id", idx),
#             "output_length": len(outputs),
#             "is_empty": len(outputs) == 0,
#             "nan_detected": diagnostic.nan_detected,
#             "router_outputs": diagnostic.router_outputs,
#             "ca_stats": diagnostic.ca_outputs,
#             "combined_features": diagnostic.combined_features_stats
#         }
#         diagnosis_file.write(json.dumps(diagnosis) + "\n")
        
#         # ============ 实时输出诊断（前5个 或 空输出 或 有NaN） ============
#         if idx < 5 or len(outputs) == 0 or diagnostic.nan_detected:
#             print(f"\n{'='*60}")
#             print(f"Sample {idx}: {line.get('question_id', idx)}")
#             print(f"Output length: {len(outputs)}")
#             print(f"Output preview: {outputs[:100] if outputs else '[EMPTY]'}")
            
#             if diagnostic.router_outputs:
#                 r = diagnostic.router_outputs[-1]
#                 print(f"\n📊 Router:")
#                 print(f"  Selected layers: {r['indices']}")
#                 print(f"  Weights: {[f'{w:.4f}' for w in r['weights']]}")
#                 print(f"  Entropy: {r['entropy']:.4f}")
#                 if r['probs']:
#                     max_prob = max(r['probs'])
#                     print(f"  Max prob: {max_prob:.4f}")
#                     # 检查是否塌陷
#                     if max_prob > 0.95:
#                         print(f"  ⚠️ COLLAPSED! One layer dominates")
            
#             if diagnostic.ca_outputs:
#                 ca = diagnostic.ca_outputs[-1]
#                 print(f"\n🔍 CrossAttention output:")
#                 print(f"  Mean: {ca['mean']:.6f}, Std: {ca['std']:.6f}")
#                 print(f"  Range: [{ca['min']:.6f}, {ca['max']:.6f}]")
#                 if ca['has_nan'] or ca['has_inf']:
#                     print(f"  ⚠️ Has NaN: {ca['has_nan']}, Has Inf: {ca['has_inf']}")
#                 if ca['std'] < 0.01:
#                     print(f"  ⚠️ STD too small - features collapsed!")
            
#             if diagnostic.combined_features_stats:
#                 cf = diagnostic.combined_features_stats[-1]
#                 print(f"\n✅ Combined features:")
#                 print(f"  Mean: {cf['mean']:.6f}, Std: {cf['std']:.6f}")
#                 print(f"  Range: [{cf['min']:.6f}, {cf['max']:.6f}]")
#                 if cf['has_nan'] or cf['has_inf']:
#                     print(f"  ⚠️ Has NaN: {cf['has_nan']}, Has Inf: {cf['has_inf']}")
#                 if abs(cf['mean']) < 0.001 and cf['std'] < 0.01:
#                     print(f"  ⚠️ Features near zero - LLM input corrupted!")
            
#             print(f"{'='*60}\n")
    
#     ans_file.close()
#     diagnosis_file.close()
    
#     # ============ 移除hooks ============
#     for h in hooks:
#         h.remove()
    
#     # ============ 最终统计 ============
#     print(f"\n{'='*80}")
#     print(f"🎯 FINAL STATISTICS:")
#     print(f"  Total samples: {len(questions)}")
#     print(f"  Empty outputs: {empty_count} ({empty_count/len(questions)*100:.1f}%)")
#     print(f"  NaN detected: {nan_count} ({nan_count/len(questions)*100:.1f}%)")
#     print(f"\n✅ Results saved to: {answers_file}")
#     print(f"✅ Diagnosis saved to: {answers_file.replace('.jsonl', '_diagnosis.jsonl')}")
#     print(f"{'='*80}\n")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model-path", type=str, required=True)
#     parser.add_argument("--model-base", type=str, default=None)
#     parser.add_argument("--image-folder", type=str, required=True)
#     parser.add_argument("--question-file", type=str, required=True)
#     parser.add_argument("--answers-file", type=str, required=True)
#     parser.add_argument("--conv-mode", type=str, default="llava_v1")
#     parser.add_argument("--num-samples", type=int, default=100)
#     parser.add_argument("--temperature", type=float, default=0.0)
#     parser.add_argument("--top_p", type=float, default=None)
#     parser.add_argument("--num_beams", type=int, default=1)
#     parser.add_argument("--max_new_tokens", type=int, default=128)
#     args = parser.parse_args()
    
#     eval_model_with_diagnosis(args)