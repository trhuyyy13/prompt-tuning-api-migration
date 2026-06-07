from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets


def truncate_back_no_signature(completion):
    lines = completion.split("\n")
    code = []
    for line in lines:
        if len(line.strip()) == 0:
            code.append(line)
            continue
        indent = len(line) - len(line.lstrip())
        if indent == 0:
            break
        code.append(line)
    return "\n".join(code)


def load_dataset_split(dataset_name):
    if dataset_name == "HumanEval":
        dataset = load_dataset("openai/openai_humaneval")
        return concatenate_datasets([dataset[key] for key in dataset.keys()])
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def load_dataset_map(dataset_name):
    dataset = load_dataset_split(dataset_name)
    dataset_map = {}
    for index in tqdm(range(len(dataset))):
        dataset_map[dataset[index]["task_id"]] = dataset[index]
    return dataset_map
