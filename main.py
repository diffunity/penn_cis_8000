import sys

from utils import (
    run_baseline_without_retrieval,
    eval_baseline_without_retrieval,
)
from self_rag_utils import run_self_rag

from vllm import LLM, SamplingParams

import sys
sys.path.append("./self-rag")
sys.path.append("./self-rag/retrieval_lm")
from passage_retrieval import Retriever

class Args:
    def __init__(self, **kwargs):
        for k, i in kwargs.items():
            setattr(self, k, i)

def input_reformat(question, answer, id_=0):
    return {"instruction": question,
        "output": answer,
        "input": "",
        "topic": "",
        "id": id_,
        "dataset_name": "dummy"
    }

def init_model(args):

    if args.model == "no_retrieve" or args.model == "retrieve":
        args.model_name = "meta-llama/Llama-2-7b-hf"
        download_dir = "./"
        world_size = 1

        model = LLM(model=args.model_name,
                    download_dir=download_dir,
                    tensor_parallel_size=world_size)

    elif args.model == "self_rag":
        args.model_name = "selfrag/selfrag_llama2_7b"
        model = model = LLM(args.model_name, download_dir="./", dtype="half")

    return model


def main(args):
    
    model = init_model(args)
    if args.model == "retrieve":
        has_eval = hasattr(args, "answer")
        single_input = [input_reformat(
            question=args.question,
            answer=args.answer if has_eval else "",
        )]

        retriever = Retriever({})
        retriever.setup_retriever_demo(
            "facebook/contriever-msmarco", 
            "enwiki_2020_intro_only/enwiki_2020_dec_intro_only.jsonl", 
            "enwiki_2020_intro_only/enwiki_dec_2020_contriever_intro/*",  
            n_docs=5, save_or_load_index=False
        )
        final_results, input_data = run_baseline_without_retrieval(args, model, single_input, retriever)

        # variable name here is misleading. both variables usually store the same data but make sure you use input_data variable for evaluation
        print("Results:", final_results)
        print("Input Data:", input_data)

        print("="*32)
        print("Response:", final_results[0]["output"])

    elif args.model == "no_retrieve":
        has_eval = hasattr(args, "answer")
        single_input = [input_reformat(
            question=args.question,
            answer=args.answer if has_eval else "",
        )]
        final_results, input_data = run_baseline_without_retrieval(args, model, single_input)
        # variable name here is misleading. both variables usually store the same data but make sure you use input_data variable for evaluation
        print("Results:", final_results)
        print("="*32)
        print("Response:", final_results[0]["output"])
        if has_eval:
            eval_baseline_without_retrieval(input_data, args)

    elif args.model == "self_rag":

        retriever = Retriever({})
        retriever.setup_retriever_demo(
            "facebook/contriever-msmarco", 
            "enwiki_2020_intro_only/enwiki_2020_dec_intro_only.jsonl", 
            "enwiki_2020_intro_only/enwiki_dec_2020_contriever_intro/*",  
            n_docs=5, save_or_load_index=False
        )
        pred = run_self_rag(args, model, [args.question], retriever)
        print("="*32)
        print("Response:", pred['data'][0]['output'])


if __name__ == "__main__":
    assert len(sys.argv) >= 3, "Must input mode from (retrieve | no_retrieve | self_rag) followed by your question wrapped in quotation"
    print(sys.argv[1:])
    args = Args(
        mode = "vanilla",
        max_new_tokens = 100,
        metric = "match",
        result_fp = "./results",
        task = "asqa",
        prompt_name = "prompt_no_input",
        batch_size = 1,
        instruction=None,
        choices=None,
        use_grounding=True,
        use_utility=True,
        ndocs = 5,
        use_seqscore = True,
        threshold = 0.2,
        beam_width = 2,
        max_depth = 7,
        ignore_cont = False,
    )

    args.model = sys.argv[1]
    args.question = sys.argv[2]
    if args.model == "self_rag":
        args.task = "asqa"
    elif args.model == "no_retrieve":
        args.task = "qa"
        args.prompt_name = "prompt_no_input"
    elif args.model == "retrieve":
        args.task = "qa"
        args.prompt_name = "prompt_no_input_retrieval"

    args.mode = sys.argv[3] if len(sys.argv) > 3 else "no_retrieve"

    main(args)

# python3 main.py no_retrieve "What is Zimbabwe?"
# python3 main.py self_rag "What is Zimbabwe?" always_retrieve
