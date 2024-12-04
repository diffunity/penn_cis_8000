# Usage

## Clone self-rag repo

```
git clone git@github.com:AkariAsai/self-rag.git
```

## Install dependencies

```
bash self-rag/setup.sh
```

## Run

* Baseline

```
python3 main.py no_retrieve "What is Zimbabwe?"
```

> Output

```
Results: [{'instruction': 'What is Zimbabwe?', 'output': '\n2022-01-22 00:56:27 UTC\n', 'input': '', 'topic': '', 'id': 0, 'dataset_name': 'dummy', 'golds': ''}]
```


* Baseline w/ RAG

> WIP

* Self-RAG

```
python3 main.py self_rag "What is Zimbabwe?" always_retrieve
```

> Output

```
{'data': [{'question': 'What is Zimbabwe?', 'docs': [{'id': '24852125', 'title': 'Zimbabwe', 'section': '', 'text': ' Zimbabwe, officially the Republic of Zimbabwe, is a landlocked country located in Southeast Africa, between the Zambezi and Limpopo Rivers, bordered by South Africa to the south, Botswana to the south-west, Zambia to the north, and Mozambique to the east. The capital and largest city is Harare. The second largest city is Bulawayo. A country of roughly 15 million people, Zimbabwe has 16 official languages, with English, Shona, and Ndebele the most common. Since the 11th century, the region that is now Zimbabwe has been the site of several organised states and kingdoms such as the Rozvi, Mutapa and Mthwakazi kingdoms, as '}], 'output': 'Zimbabwe is a landlocked country located in Southeast Africa, between the Zambezi and Limpopo Rivers, bordered by South Africa to the south, Botswana to the south-west, Zambia to the north, and Mozambique to the east. The capital and largest city is Harare. The second largest city is Bulawayo. A country of roughly 15 million people, Zimbabwe ha [1].', 'intermediate': ['[Retrieval]', 'Zimbabwe is a landlocked country located in Southeast Africa, between the Zambezi and Limpopo Rivers, bordered by South Africa to the south, Botswana to the south-west, Zambia to the north, and Mozambique to the east.The capital and largest city is Harare.The second largest city is Bulawayo.A country of roughly 15 million people, Zimbabwe has']}], 'args': [], 'total_cost': 0.0, 'azure_filter_fail': ''}
```

> look for 'output' key ('output': 'Zimbabwe is a landlocked country located in Southeast Africa, between the Zambezi and Limpopo Rivers, bordered by South Africa to the south, Botswana to the south-west, Zambia to the north, and Mozambique to the east. The capital and largest city is Harare. The second largest city is Bulawayo. A country of roughly 15 million people, Zimbabwe ha [1].')