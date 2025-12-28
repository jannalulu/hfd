from dataclasses import dataclass
from typing import Dict, List, Union, Optional
import torch
from transformers import AutoTokenizer
import datasets
import glob
import os
from functools import partial

def detect_format(directory) -> str:
    """
    Detect the format of the dataset in the given directory.
    Args:
    directory: The directory containing the dataset files.
    Returns:
    The format of the dataset.['jsonl', 'parquet', 'huggingface','unknown']
    """
    #if dataset_info.json exists, it is a huggingface dataset
    if os.path.exists(os.path.join(directory, 'dataset_info.json')):
        return 'huggingface'
    #if parquet files exist, it is a parquet dataset
    if glob.glob(os.path.join(directory, '*.parquet')):
        return 'parquet'
    #if jsonl files exist, it is a jsonl dataset
    if glob.glob(os.path.join(directory, '*.jsonl')):
        return 'jsonl'
    return 'unknown'
def load_jsonl_dataset(file_path):
    jsonl_files = glob.glob(file_path+"/*.jsonl")
    # print(f'load jsonl files: {jsonl_files}')
    dataset = datasets.load_dataset('json', data_files=jsonl_files)['train']
    return dataset

def load_parquet_dataset(file_path):
    parquet_files = glob.glob(file_path+"/*.parquet")
    # print(f'load parquet files: {parquet_files}')
    dataset = datasets.load_dataset('parquet', data_files=parquet_files)['train']
    return dataset

def load_huggingface_dataset(file_path):
    dataset = datasets.load_from_disk(file_path)
    return dataset

def check_feature(dataset):
    """
    check the features of dataset
    if the feature contains messages and the value is a list, return "conversation"
    if the feature contains text and the value is a string, return "text"
    """
    for feature, value in dataset[0].items():
        if (feature == 'messages' and isinstance(value, list)) or \
            ("conversations" in feature and isinstance(value, list)):
            return 'conversation'
        if feature == 'text' and isinstance(value, str):
            return 'text'
    return 'unknown'

load_functions = {
    'jsonl': load_jsonl_dataset,
    'parquet': load_parquet_dataset,
    'huggingface': load_huggingface_dataset
}

def convert_conversation_to_text(example: Dict,tokenizer) -> Dict:
    """
    Convert a single conversation to ChatML format.
    
    Args:
        example: A dictionary containing a 'messages' list of conversations
        
    Returns:
        A dictionary with a single 'text' key containing the ChatML formatted conversation
    """
    result = []
    if "messages" in example:
        messages = example["messages"]
    elif "conversations" in example:
        messages = example["conversations"]
    new_message = []
    count = 0
    for message in messages:
        if count == 0 and message['role'] != "system":
            new_message.append({'role': 'system', 'content': ''})
        count = count + 1
            



        if 'role' in message:
            role = message['role']
        elif 'from' in message:
            role = message['from']
        role = role.lower()
        
        if 'content' in message:
            content = message['content']
        elif 'value' in message:
            content = message['value']
        
        if role == 'human':
            role = 'user'
        elif role == 'gpt':
            role = 'assistant'
        new_message.append({'role': role, 'content': content})
        # result.extend([
        #     f"<|im_start|>{role}\n",
        #     f"{content}",
        #     "<|im_end|>"
        # ])
    if len(new_message) == 0:
        return {'text': ""}
    #print(new_message)
    try:
        return {'text': tokenizer.apply_chat_template(new_message, tokenize=False, enable_thinking=False)}
    except:
        print(f'except:{new_message}')
        return {'text': ""}
    #return {'text': tokenizer.apply_chat_template(new_message, tokenize=False)}
    # return {'text': "".join(result)}

def convert_conversational_ds_to_text(ds: datasets.Dataset,tokenizer) -> datasets.Dataset:
    """
    Convert a conversational dataset to ChatML format.
    
    Args:
        ds: A dataset containing 'messages' lists of conversations
        
    Returns:
        A dataset with a single 'text' key containing the ChatML formatted conversation
    """
    
    return ds.map(partial(convert_conversation_to_text,tokenizer=tokenizer),  # 使用convert_conversation_to_text函数处理 
                  num_proc=8,  # 使用8个进程并行处理
        remove_columns=ds.column_names,  # 移除所有原始列
        desc="Converting conversations"  # 显示进度条描述
    )



def load_datasets_from_directories(directories,tokenizer):
    """
    Load datasets from directories.
    Args:
    directories: A list of directories containing the dataset files.
    Returns:
    A list of datasets.
    """
    feature_types = []
    all_ds = []
    for directory in directories:
        print(f'checking {directory}')
        dataset_type = detect_format(directory)
        print(f"Detected dataset type: {dataset_type}")
        if dataset_type == 'unknown':
            print(f"Unknown dataset type for directory: {directory}")
            continue
        ds = load_functions[dataset_type](directory)
        feature_type = check_feature(ds)
        print(f'feature_type = {feature_type}')
        feature_types.append(feature_type)
        if feature_type == 'conversation':
            ds = convert_conversational_ds_to_text(ds,tokenizer)
        else:
            ds = ds.select_columns(['text'])
        print(f"Loaded dataset from directory: {directory}")
        all_ds.append(ds)
    return all_ds,feature_types

import itertools
import random
import time
from typing import Optional, Dict, List, Union, Tuple
from torch.utils.data import Dataset
class TypedDataset(Dataset):
    def __init__(self, all_ds: List[datasets.Dataset], feature_types: List[str]):
        super().__init__()
        
        self.conversation_datasets = []
        self.text_datasets = []
        
        for ds, ds_type in zip(all_ds, feature_types):
            if ds_type == 'conversation':
                self.conversation_datasets.append(ds)
            else:
                self.text_datasets.append(ds)
        
        self.conversation_lengths = [len(ds) for ds in self.conversation_datasets]
        self.text_lengths = [len(ds) for ds in self.text_datasets]
        
        self.total_conversation = sum(self.conversation_lengths)
        self.total_text = sum(self.text_lengths)
        self.total_length = self.total_conversation + self.total_text
        
        self.conversation_offsets = [0] + list(itertools.accumulate(self.conversation_lengths))
        self.text_offsets = [0] + list(itertools.accumulate(self.text_lengths))

        # 現在時刻をシードとして使用
        random.seed(int(time.time()))
    
    def __len__(self) -> int:
        return self.total_length
    
    def _get_dataset_and_local_idx(self, idx: int) -> Tuple[datasets.Dataset, int, bool]:
        if idx < self.total_conversation:
            for ds_idx, offset in enumerate(self.conversation_offsets[1:]):
                if idx < offset:
                    local_idx = idx - self.conversation_offsets[ds_idx]
                    return self.conversation_datasets[ds_idx], local_idx, True
        else:
            idx = idx - self.total_conversation
            for ds_idx, offset in enumerate(self.text_offsets[1:]):
                if idx < offset:
                    local_idx = idx - self.text_offsets[ds_idx]
                    return self.text_datasets[ds_idx], local_idx, False
        
        raise IndexError(f"Failed to map index {idx}")

    def __getitem__(self, idx: int) -> Dict:
        dataset, local_idx, is_conversation = self._get_dataset_and_local_idx(idx)
        item = dataset[local_idx]
        return {
            'text': item['text'],
            'is_conversation': is_conversation,
            'dataset_index': idx
        }
    
    def get_random_sample(self, is_conversation: bool) -> Tuple[str, bool]:
        """Get a random sample and its type"""
        try:
            if is_conversation and self.conversation_datasets:
                ds_idx = random.randrange(len(self.conversation_datasets))
                local_idx = random.randrange(len(self.conversation_datasets[ds_idx]))
                return self.conversation_datasets[ds_idx][local_idx]['text'], True
            elif not is_conversation and self.text_datasets:
                ds_idx = random.randrange(len(self.text_datasets))
                local_idx = random.randrange(len(self.text_datasets[ds_idx]))
                return self.text_datasets[ds_idx][local_idx]['text'], False
        except (ValueError, IndexError):
            # Fallback to any available type
            if self.conversation_datasets:
                ds_idx = random.randrange(len(self.conversation_datasets))
                local_idx = random.randrange(len(self.conversation_datasets[ds_idx]))
                return self.conversation_datasets[ds_idx][local_idx]['text'], True
            elif self.text_datasets:
                ds_idx = random.randrange(len(self.text_datasets))
                local_idx = random.randrange(len(self.text_datasets[ds_idx]))
                return self.text_datasets[ds_idx][local_idx]['text'], False
            
        raise ValueError("No datasets available for sampling")
    
@dataclass
class TypedStreamingCLMDataCollator:
    tokenizer: AutoTokenizer
    max_length: int
    min_length: int
    typed_dataset: TypedDataset
    pad_to_multiple_of: Optional[int] = None
    need_to_pack: bool = True
    padding_side :str = "right"
    
    def concatenate_if_needed(self, text: str, is_conversation: bool) -> str:
        """Concatenate text with random samples of the same type if it's too short"""
        tokens = self.tokenizer(text, truncation=False)
        current_length = len(tokens['input_ids'])
        
        max_attempts = 1000000000
        attempts = 0
        
        while current_length < self.min_length and attempts < max_attempts:
            #print(f'attempts = {attempts}')
            sample_text, sample_is_conversation = self.typed_dataset.get_random_sample(is_conversation)
            
            if is_conversation:
                text += sample_text
            else:
                text += "\n\n" + sample_text
            
            tokens = self.tokenizer(text, truncation=False)
            current_length = len(tokens['input_ids'])
            attempts += 1
                
        return text
    
    def __call__(self, examples: List[Dict[str, Union[str, List[str]]]]) -> Dict[str, torch.Tensor]:
        texts = []
        for example in examples:
            text = example['text'] if isinstance(example['text'], str) else example['text'][0]
            processed_text = self.concatenate_if_needed(text, example['is_conversation']) if self.need_to_pack else text
            texts.append(processed_text)

        #print(texts)
        
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
            padding_side=self.padding_side
        )
        
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        labels = input_ids.clone()
        
        # Shift labels for next token prediction
        if self.padding_side == "right":
            labels[:, :-1] = input_ids[:, 1:]
            last_token = self.tokenizer.pad_token_id
            labels[:, -1] = last_token
        else:
            labels[:, 1:] = input_ids[:, :-1]
            first_token = self.tokenizer.pad_token_id 
            labels[:, 0] = first_token
        
        # Handle padding in labels
        labels[attention_mask == 0] = -100
        padding_start = attention_mask.sum(dim=1) - 1
        for i in range(len(padding_start)):
            if padding_start[i] > 0:
                labels[i, padding_start[i]] = -100
            
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            
        }
