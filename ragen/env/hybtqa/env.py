import gym
from gym import spaces
import numpy as np
from datasets import load_dataset
import json
import re
import random
from ragen.env.base import BaseLanguageBasedEnv
from ragen.utils import all_seed
from .config import HybTQAEnvConfig
from .retriever import Retriever
from .evaluate import *
class HybTQAEnv(BaseLanguageBasedEnv):
    def __init__(self, config: HybTQAEnvConfig):
        super(HybTQAEnv, self).__init__()
        
        self.config = config
        with open(self.config.dataset_path, "r") as file:
            self.dataset = json.loads(file.read())
        # self.dataset = load_dataset(path=self.config.dataset_path, cache_dir=self.config.cache_dir)
        self.retriever = Retriever(self.config.retriever_path)
        self.retrieve_memory = {
            "text_inds": [],
            "table_inds": []
        }  # (index, content) list
        self.current_question_idx = None
        self.current_question = None
        self.correct_answer = None
        self.step_num = None
        self.render_cache = None
        
    def _split_action(self, action):
        if len(action) == 0:
            return None, None
        if "[R]" in action:
            if "[A]" in action:
                return None, None
            action_type = "R"
            action_content = action.split("[R]")[-1]
        elif "[A]" in action:
            action_type = "A"
            action_content = action.split("[A]")[-1]
        else:
            return None, None
        if len(action_content) == 0:
            return None, None
        return action_type, action_content
        
    def reset(self,seed=None, mode=None):
        dataset = self.dataset #[self.config.split]
        with all_seed(seed):
            self.current_question_idx = random.randint(0, len(dataset) - 1)
        sample_data = dataset[self.current_question_idx]
        self.current_question = sample_data['qa']['question']
        self.correct_answer = str(sample_data['qa']['answer'])

        # process table description
        table_keys, table_values = [], []
        for key, value in sample_data['table_description'].items():
            table_keys.append(key)
            table_index = key.split('-')[0]
            table_values.append(value.split(f"Table {table_index} shows ")[-1])

        self.doc_kwargs = {
            "text_docs": sample_data['paragraphs'],
            "table_docs": [table_keys, table_values],
            "table_description": sample_data['table_description'],
            "text_evidence": sample_data['qa']['text_evidence'],
            "table_evidence": sample_data['qa']['table_evidence']
        }
        self.step_num = 0
        self.render_cache = "[Q]" + self.current_question + "Your next action should be either retrieve [R] or answer [A]. "
        return self.render_cache
        
    def step(self, action):
        action_type, action_content = self._split_action(action)
        if action_type == "R":
            retrieved_texts_inds = self.retriever.retrieve(action_content, self.doc_kwargs["text_docs"], top_k=3)  # [1, 2, 3]
            retrieved_tables = self.retriever.retrieve(action_content, self.doc_kwargs["table_docs"][1], top_k=3)
            retrieved_tables_inds = [self.doc_kwargs["table_docs"][0][ind] for ind in retrieved_tables]  # ["0-1-2", "0-1-3", "2-3-4"]
            
            reward = 0.0

            # evaluate retrieval
            text_gth = list(dict.fromkeys(self.doc_kwargs["text_evidence"]).keys())
            text_inds = list(dict.fromkeys(retrieved_texts_inds).keys())
            text_memory = self.retrieve_memory["text_inds"]
            text_reward = self.retriever.memorize_eval(text_inds, text_gth, text_memory)
            
            table_gth = list(dict.fromkeys(self.doc_kwargs["table_evidence"]).keys())
            table_inds = list(dict.fromkeys(retrieved_tables_inds).keys())
            table_memory = self.retrieve_memory["table_inds"]
            table_reward = self.retriever.memorize_eval(table_inds, table_gth, table_memory)
            
            if len(text_gth) and len(table_gth):
                reward += 0.5 * text_reward + 0.5 * table_reward
            elif len(text_gth) == 0:
                reward += table_reward
            elif len(table_gth) == 0:
                reward += text_reward

            self.retrieve_memory["text_inds"].extend(text_inds)
            self.retrieve_memory["table_inds"].extend(table_inds)

            retrieved_text_contents = '; '.join([self.doc_kwargs["text_docs"][ind] for ind in text_inds])
            retrieved_table_contents = '; '.join([
                self.doc_kwargs["table_description"][ind].split(f'Table {ind.split('-')[0]} shows ')[-1] for ind in table_inds])

            observation = f"[Retrieved Information] {retrieved_text_contents}; {retrieved_table_contents}"
            info = {"valid": True}
            done = False

        # elif action_type == "analyze":
        #     reward = 0.0
        #     if self.step_num > 5:
        #         reward -= (self.step_num - 5) / 2
        #     observation = f"Continue Reasoning. "
        #     info = {"valid": True}
        #     done = False

        elif action_type == "A":
            exact_acc, f1_acc = get_span_selection_metrics(action_content, self.correct_answer)
            retrieve_text_reward = self.retriever.eval(
                self.retrieve_memory["text_inds"], 
                self.doc_kwargs["text_evidence"]
            )[2]
            retrieve_table_reward = self.retriever.eval(
                self.retrieve_memory["table_inds"], 
                self.doc_kwargs["table_evidence"]
            )[2]
            answer_reward = 2 * (exact_acc + f1_acc)
            reward = retrieve_text_reward + retrieve_table_reward + answer_reward
            observation = "END"
            info = {"valid": True}
            done = True

        else:
            reward = -3
            observation = "Illegal or lengthy generation, try again. "
            info = {"valid": False}
            done = False

        self.step_num += 1
        self.render_cache = observation + "Your next action should be either retrieve [R] or answer [A]. "
        
        return self.render_cache, reward, done, info

    def render(self):
        return self.render_cache


if __name__ == "__main__":
    # Create the environment configuration
    config = HybTQAEnvConfig(
        dataset_path="./data/hybtqa/train.json",
    )
    
    # Initialize the environment
    env = HybTQAEnv(config)
    
    # Reset the environment to get the first question
    print("Question:")
    question = env.reset(seed=42)
    print(question)
    print("\nCorrect answer (for testing purposes):")
    print(env.correct_answer)
    
    # Interactive loop for testing
    while True:
        user_answer = input("\nEnter your answer (or 'q' to quit): ")
        if user_answer.lower() == 'q':
            break
        
        # Take a step in the environment with the user's answer
        #breakpoint()
        obs, reward, done, info = env.step(user_answer)
        
        
        # Print the results
        print("\nFeedback:", obs)
        print("Reward:", reward)
        print("Done:", done)
        print("Info:", info)
        
        # If the episode is done, reset the environment for a new question
        if done:
            print("\n--- New Question ---")
            question = env.reset()
            print(question)
            print("\nCorrect answer (for testing purposes):")
            print(env.correct_answer)