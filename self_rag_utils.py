
from vllm import LLM, SamplingParams
import jsonlines, json, copy, re
from tqdm import tqdm
import numpy as np
import string
import sys

sys.path.append("./self-rag")
sys.path.append("./self-rag/retrieval_lm")
from passage_retrieval import Retriever

from transformers import AutoTokenizer

from utils import (
    rel_tokens_names,
    retrieval_tokens_names,
    utility_tokens_names,
    ground_tokens_names,
    other_special_tokens,
    control_tokens,
    PROMPT_DICT,
    TASK_INST,
)

def load_special_tokens(tokenizer, use_grounding=False, use_utility=False):
    ret_tokens = {token: tokenizer.convert_tokens_to_ids(
        token) for token in retrieval_tokens_names}
    rel_tokens = {}
    for token in ["[Irrelevant]", "[Relevant]"]:
        rel_tokens[token] = tokenizer.convert_tokens_to_ids(token)

    grd_tokens = None
    if use_grounding is True:
        grd_tokens = {}
        for token in ground_tokens_names:
            grd_tokens[token] = tokenizer.convert_tokens_to_ids(token)

    ut_tokens = None
    if use_utility is True:
        ut_tokens = {}
        for token in utility_tokens_names:
            ut_tokens[token] = tokenizer.convert_tokens_to_ids(token)

    return ret_tokens, rel_tokens, grd_tokens, ut_tokens

def postprocess(pred):
    special_tokens = ["[Fully supported]", "[Partially supported]", "[No support / Contradictory]", "[No Retrieval]", "[Retrieval]",
                      "[Irrelevant]", "[Relevant]", "<paragraph>", "</paragraph>", "[Utility:1]", "[Utility:2]", "[Utility:3]", "[Utility:4]", "[Utility:5]"]
    for item in special_tokens:
        pred = pred.replace(item, "")
    pred = pred.replace("</s>", "")

    if len(pred) == 0:
        return ""
    if pred[0] == " ":
        pred = pred[1:]
    return pred


def fix_spacing(input_text):
    # Add a space after periods that lack whitespace
    output_text = re.sub(r'(?<=\w)([.!?])(?=\w)', r'\1 ', input_text)
    return output_text

def run_self_rag(args, model, input_data, retriever):

    def generate(prompt, ctxs, max_new_tokens):
        processed_prompt = PROMPT_DICT["prompt_no_input"].format_map(
            {"instruction": prompt})
        return call_model_beam_batch(processed_prompt, model=model, max_new_tokens=max_new_tokens, ctxs=ctxs, query=prompt,
                                     rel_tokens=rel_tokens, ret_tokens=ret_tokens, grd_tokens=grd_tokens, ut_tokens=ut_tokens,
                                     use_seqscore=args.use_seqscore, threshold=args.threshold, 
                                     beam_width=args.beam_width, max_depth=args.max_depth,
                                     w_rel=1.0, w_sup=1.0, w_use=0.5, mode=args.mode, ignore_cont=args.ignore_cont, )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, padding_side="left")

    # Get token ids for reflection tokens.
    ret_tokens, rel_tokens, grd_tokens, ut_tokens = load_special_tokens(
        tokenizer, use_grounding=args.use_grounding, use_utility=args.use_utility)

    prompt = input_data[0]
    ctxs = retriever.search_document_demo(input_data[0], args.ndocs)
    instructions = TASK_INST[args.task]
    prev_gen = []
    prompt = instructions + "## Input:\n\n" + prompt
    final_pred, intermediate = generate(
        prompt, ctxs, args.max_new_tokens,)
    final_output = ""
    docs = []
    prev_gen = []

    item = {"question": input_data[0], "docs": ctxs}
    new_results = {"data": [], "args": [],
                    "total_cost": 0.0, "azure_filter_fail": ""}

    if "splitted_sentences" not in intermediate:
        item["output"] = postprocess(final_pred)
    else:
        if len(postprocess(final_pred[0])) == 0:
            intermediate["splitted_sentences"][0], intermediate["ctxs"][
                0] = intermediate["splitted_sentences"][1], intermediate["ctxs"][1]
        for idx, (sent, doc) in enumerate(zip(intermediate["splitted_sentences"][0], intermediate["ctxs"][0])):
            if len(sent) == 0:
                continue
            postprocessed_result = postprocess(sent)
            if postprocessed_result in prev_gen:
                continue
            else:
                prev_gen.append(postprocessed_result)
            final_output += postprocessed_result[:-1] + " [{}]".format(idx) + ". "
            docs.append(doc)
        if len(final_output) == 0:
            item["output"] = fix_spacing(final_output)
        if len(final_output) > 0 and final_output[-1] == " ":
            final_output = final_output[:-1]
        item["output"] = fix_spacing(final_output)
        item["output"] = item["output"].replace(
            ".[Continue to Use Evidence]", " [1]. ")
        item["output"] = item["output"].replace(". [1] ", " [1]. ")
    item["docs"] = docs
    if "original_splitted_sentences" in intermediate:
        item["intermediate"] = intermediate["original_splitted_sentences"][0]
    new_results["data"].append(item)
    print(new_results)

    return new_results


def run_step_generation_batch(model, prompt, paragraphs,  max_new_tokens,
                              rel_tokens=None, grd_tokens=None, ret_tokens=None, ut_tokens=None,
                              threshold=None, w_rel=1.0, w_sup=1.0, w_use=0.5, use_seqscore=False):

    if paragraphs is not None:
        aug_prompts = [prompt + "[Retrieval]" + "<paragraph>{}</paragraph>".format(
            paragraph["title"] + "\n" + paragraph["text"]) for paragraph in paragraphs]
    else:
        aug_prompts = [prompt]
    print(aug_prompts)
    sampling_params = SamplingParams(
        temperature=0.0, top_p=1.0, max_tokens=max_new_tokens, logprobs=20, )
    preds = model.generate(aug_prompts, sampling_params)

    # compute the scores for each generation
    relevance_score_dict = {}
    grd_score_dict = {}
    ut_score_dict = {}
    overall_scores = {}
    final_preds = []
    for p_idx, pred in enumerate(preds):
        pred_token_ids = pred.outputs[0].token_ids
        pred_text = pred.outputs[0].text
        pred_log_probs = pred.outputs[0].logprobs
        seq_score = pred.outputs[0].cumulative_logprob / \
            max(len(pred.outputs[0].token_ids), 1)
        assert len(pred_log_probs) == len(pred_token_ids)

        relevance_score_dict.setdefault(p_idx, {})
        grd_score_dict.setdefault(p_idx, {})
        ut_score_dict.setdefault(p_idx, {})
        # Compute reward scores
        for tok, id in rel_tokens.items():
            if id not in pred_log_probs[0]:
                prob = -100
            else:
                prob = np.exp(pred_log_probs[0][id])
            relevance_score_dict[p_idx][tok] = prob

        if grd_tokens is not None:
            groundness_token_appear_indices = []
            for tok_idx, tok in enumerate(pred_token_ids):
                if tok in list(grd_tokens.values()):
                    groundness_token_appear_indices.append(tok_idx)
                    break
            if len(groundness_token_appear_indices) > 0:
                idx = groundness_token_appear_indices[0]
                for token, token_id in grd_tokens.items():
                    prob = pred_log_probs[idx][token_id] if token_id in pred_log_probs[idx] else -100
                    grd_score_dict[p_idx][token] = np.exp(prob)

        utility_token_appear_indices = []
        if ut_tokens is not None:
            for tok_idx, tok in enumerate(pred_token_ids):
                if tok in list(ut_tokens.values()):
                    utility_token_appear_indices.append(tok_idx)
            if len(utility_token_appear_indices) > 0:
                idx = utility_token_appear_indices[0]
                for token, token_id in grd_tokens.items():
                    prob = pred_log_probs[idx][token_id] if token_id in pred_log_probs[idx] else -100
                    ut_score_dict[p_idx][token] = np.exp(prob)

        relevance_score = relevance_score_dict[p_idx]["[Relevant]"] / (
            np.sum(list(relevance_score_dict[p_idx].values())))

        if len(grd_score_dict[p_idx]) == 3:
            gt_sum = np.sum(list(grd_score_dict[p_idx].values()))
            ground_score = (grd_score_dict[p_idx]["[Fully supported]"] / gt_sum) + 0.5 * (
                grd_score_dict[p_idx]["[Partially supported]"] / gt_sum)
        else:
            ground_score = 0.0

        if len(ut_score_dict[p_idx]) == 5:
            ut_sum = np.sum(list(ut_score_dict[p_idx].values()))
            ut_scores = [-1, -0.5, 0, 0.5, 1]
            utility_score = np.sum([ut_scores[i] * (ut_score_dict[p_idx]["[Utility:{}]".format(i+1)] / ut_sum)
                                   if "[Utility:{}]".format(i+1) in ut_score_dict[p_idx] else 0.0 for i in range(0, 5)])
        else:
            utility_score = 0.0

        if use_seqscore is True:
            final_score =np.exp(seq_score) + w_rel * relevance_score + \
                w_sup * ground_score + w_use * utility_score
        else:
            final_score = w_rel * relevance_score + \
                w_sup * ground_score + w_use * utility_score
            
        overall_scores[p_idx] = {"final_score": final_score,
                                 "relevance_score": relevance_score,
                                 "ground_score": ground_score,
                                 "utility_score": utility_score,
                                 "relevance_score_dict": relevance_score_dict,
                                 "grd_score_dict": grd_score_dict,
                                 "ut_score_dict": utility_score}

        # modify and add do retrieve tokens
        if "[No Retrieval]" in pred_text:
            ret_token_appear_indices = []
            substrings = pred_text.split("[No Retrieval]")

            for tok_idx, tok in enumerate(pred_token_ids):
                if tok == ret_tokens["[No Retrieval]"]:
                    ret_token_appear_indices.append(tok_idx)
                    substrings
                    print("retrieval_tokens")

            ret_token_score_dict = {}
            retrieval_remap = {}
            for order, idx in enumerate(ret_token_appear_indices):
                ret_token_score_dict.setdefault(order, {})
                for tok, tok_id in ret_tokens.items():
                    prob = pred_log_probs[idx][tok_id] if tok_id in pred_log_probs[idx] else -100
                    ret_token_score_dict[order][tok] = np.exp(prob)
                if ret_token_score_dict[order]["[Retrieval]"] + ret_token_score_dict[order]["[No Retrieval]"] != 0.0:
                    do_retrieve = (ret_token_score_dict[order]["[Retrieval]"] + ret_token_score_dict[order]["[Continue to Use Evidence]"]) / (
                        ret_token_score_dict[order]["[Retrieval]"] + ret_token_score_dict[order]["[No Retrieval]"]) > threshold
                else:
                    do_retrieve = 0.0
                if do_retrieve > threshold:
                    retrieval_remap[order] = True
                else:
                    retrieval_remap[order] = False
            processed_pred = ""
            for substr_i, substring in enumerate(substrings):
                if substr_i in retrieval_remap and retrieval_remap[substr_i] is True:
                    processed_pred += substring + "[Retrieval]"
                else:
                    processed_pred += substring + "[No Retrieval]"
            pred_text = processed_pred
            final_preds.append(pred_text)
        else:
            final_preds.append(pred_text)

    preds = final_preds
    scores = [overall_scores[p_idx]["final_score"] for p_idx in overall_scores]

    return preds, scores, overall_scores

def call_model_beam_batch(prompt, model, max_new_tokens=15, ctxs=None, query=None, max_depth=5, rel_tokens=None,
                          grd_tokens=None, ret_tokens=None, threshold=None, beam_width=2, ut_tokens=None, use_seqscore=False,
                          w_rel=1.0, w_sup=1.0, w_use=0.5, ignore_cont=False, mode="adaptive_retrieval"):
    special_tokens = []
    if "## Input:\n\n" in query:
        query = query.split("## Input:\n\n")[1]
    if rel_tokens is not None:
        special_tokens = list(rel_tokens.keys())
    if ret_tokens is not None:
        special_tokens += list(ret_tokens.keys())

    if mode == "no_retrieval":
        sampling_params = SamplingParams(
            temperature=0.0, top_p=1, max_tokens=max_new_tokens)
        prompt += "[No Retrieval]"
        preds = model.generate([prompt], sampling_params)
        preds = [pred.outputs[0].text.split("\n\n")[0] for pred in preds]
        return preds[0], prediction_tree

    do_retrieve = False
    if mode == "always_retrieve":
        do_retrieve = True

    else:
        sampling_params = SamplingParams(
            temperature=0.0, top_p=1, max_tokens=25, logprobs=20)
        preds = model.generate([prompt], sampling_params)
        pred_log_probs = preds[0].outputs[0].logprobs
        preds = [pred.outputs[0].text.split("\n\n")[0] for pred in preds]
        if "[Retrieval]" not in preds[0]:
            do_retrieve = False
        else:
            if threshold is None:
                do_retrieve = False
            else:
                ret_token_score_dict = {}
                for tok, tok_id in ret_tokens.items():
                    prob = pred_log_probs[0][tok_id]
                    ret_token_score_dict[tok] = np.exp(prob)
                retrieve_prob = ret_token_score_dict["[Retrieval]"] / (
                    ret_token_score_dict["[Retrieval]"] + ret_token_score_dict["[No Retrieval]"])
                do_retrieve = True if retrieve_prob > threshold else False

    if do_retrieve is False:
        sampling_params = SamplingParams(
            temperature=0.0, top_p=1, max_tokens=max_new_tokens)
        prompt += "[No Retrieval]"
        preds = model.generate([prompt], sampling_params)
        preds = [pred.outputs[0].text.split("\n\n")[0] for pred in preds]
        prediction_tree = {}
        return preds[0], prediction_tree
    elif do_retrieve is True:
        curr_depth = 1
        terminated = False
        node_id = 0
        prediction_tree = {}
        levels = {}
        prediction_tree[node_id] = {"prompt": prompt, "pred": "[Retrieval]",
                                    "processed_pred": "", "score": None, "ctx": None, "parent": None}
        levels[0] = [0]
        while curr_depth < max_depth:
            levels[curr_depth] = []
            if curr_depth-1 in levels and terminated is False:
                for node in levels[curr_depth-1]:
                    pred = prediction_tree[node]["pred"]
                    if pred == "</s>":
                        terminated = True
                        continue
                    prompt = prediction_tree[node]["prompt"]
                    prev_generation = prediction_tree[node]["processed_pred"]
                    score = prediction_tree[node]["score"]
                    if "[Retrieval]" in pred:
                        retrieval_results = {}
                        preds, scores, overall_score_dict = run_step_generation_batch(
                            model, prompt + prev_generation, ctxs, max_new_tokens,
                            rel_tokens, ret_tokens=ret_tokens, grd_tokens=grd_tokens, ut_tokens=ut_tokens,
                            threshold=threshold, w_rel=w_rel, w_sup=w_sup, w_use=w_use)
                        for i, (pred, p_score) in enumerate(zip(preds, scores)):
                            retrieval_results[i] = {
                                "pred": pred, "score": p_score}

                        for i, result in retrieval_results.items():
                            node_id += 1
                            node_score = result["score"] * \
                                score if score is not None else result["score"]
                            pred = result["pred"]
                            prediction_tree[node_id] = {"prompt": prompt + prev_generation, "pred": pred,
                                                        "score": node_score, "ctx": ctxs[i], "parent": node,
                                                        "overall_score_dict": overall_score_dict}

                            if "[Retrieval]" in pred:
                                gen_result_index = pred.index("[Retrieval]")
                                prev_generation = pred[:gen_result_index]
                            else:
                                prev_generation = pred
                            prediction_tree[node_id]["processed_pred"] = prev_generation
                            levels[curr_depth].append(node_id)

                current_rank = levels[curr_depth]
                node2score = {
                    node_id: prediction_tree[node_id]["score"] for node_id in current_rank}
                top_nodes = sorted(node2score.items(), key=lambda x: x[1], reverse=True)[
                    :beam_width]
                levels[curr_depth] = [node[0] for node in top_nodes]
                curr_depth += 1
            else:
                break

    final_prediction = ""
    parent = 0
    best_selections = {}

    # Traverse from the bottom
    levels = {k: v for k, v in levels.items() if len(v) > 0 and k != 0}
    for path_i, node in enumerate(levels[len(levels)]):
        if node == 0:
            break
        best_selections[path_i] = [node]
        current_node = node
        current_level = curr_depth
        if current_node is None:
            continue
        while current_level > 0 and current_node is not None:
            parent = prediction_tree[current_node]["parent"]
            best_selections[path_i] = [parent] + best_selections[path_i]
            current_node = parent
            current_level += 1

    final_prediction = {}
    splitted_sentences = {}
    original_splitted_sentences = {}
    ctxs = {}
    for path_i, nodes in best_selections.items():
        final_prediction[path_i] = " ".join([prediction_tree[node]["processed_pred"] for node in nodes if node is not None and (
            ignore_cont is False or (ignore_cont is True and "[No support / Contradictory]" not in prediction_tree[node]["processed_pred"]))])
        splitted_sentences[path_i] = [prediction_tree[node]["processed_pred"] for node in nodes if node is not None and (
            ignore_cont is False or (ignore_cont is True and "[No support / Contradictory]" not in prediction_tree[node]["processed_pred"]))]
        original_splitted_sentences[path_i] = [prediction_tree[node]["pred"] for node in nodes if node is not None and (
            ignore_cont is False or (ignore_cont is True and "[No support / Contradictory]" not in prediction_tree[node]["processed_pred"]))]
        ctxs[path_i] = [prediction_tree[node]["ctx"] for node in nodes if node is not None and (ignore_cont is False or (
            ignore_cont is True and "[No support / Contradictory]" not in prediction_tree[node]["processed_pred"]))]

    result = {"final_prediction": final_prediction,
              "splitted_sentences": splitted_sentences,
              "original_splitted_sentences": original_splitted_sentences,
              "best_selections": best_selections,
              "ctxs": ctxs,
              "prediction_tree": prediction_tree}

    return final_prediction, result

