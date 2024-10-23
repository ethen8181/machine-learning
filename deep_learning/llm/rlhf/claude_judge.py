"""
LLM as pairwise judge. This script takes in two dataframe with prompt/responses,
uses AWS bedrock's claude as LLM judge,
prints out a win/tie/lose table

https://aws.amazon.com/blogs/aws/anthropics-claude-3-sonnet-foundation-model-is-now-available-in-amazon-bedrock/
"""
import json
import boto3
import numpy as np
import pandas as pd
from typing import Optional
from botocore.exceptions import ClientError


class PairwiseBedRockLLMJudgeModule:

    default_system_prompt = '''
    I require a leaderboard for various large language models. I'll provide you with prompts given to these models and their corresponding outputs. Your task is to assess these responses, and select the model that produces the best output from a human perspective.

    Instruction: {prompt}

    Model Outputs: Here are the unordered outputs from the models. Each output is associated with a specific model, identified by a unique model identifier.

    "model_identifier": "1", "output": """{response1}""" "model_identifier": "2", "output": """{response2}"""

    Task Evaluate the models on the basis of the quality and relevance of their results, and select the model that generated the best result. Reply with the identifier of the best model. Our evaluation will only take into account the first character of your answer, so make sure it contains only one of the identifiers and nothing else (no quotation marks, no spaces, no new lines, ...).
    '''

    def __init__(
        self,
        prompt_col_name: str = "prompts",
        response1_col_name: str = "responses1",
        response2_col_name: str = "responses2",
        system_prompt: Optional[str] = None,
        model_id: str = "anthropic.claude-3-sonnet-20240229-v1:0",
        max_tokens: int = 512,
        temperature: float = 0.1,
    ):
        self.prompt_col_name = prompt_col_name
        self.response1_col_name = response1_col_name
        self.response2_col_name = response2_col_name
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.system_prompt = system_prompt if system_prompt is not None else self.default_system_prompt

    def __call__(self, features):
        df_responses = self.generate_responses(features)
        df_responses = self.calculate_result(df_responses)
        return df_responses

    def generate_responses(self, features):
        """
        prompt/response1/response2 are basically pass through column saved for interpretability
        our main goal is to obtain judge's response (swapped position to account for position bias)
        """
        prompts = []
        responses1 = []
        responses2 = []
        judge_responses = []
        judge_responses_swapped_position = []
        for feature in features:
            prompt = feature[self.prompt_col_name]
            response1 = feature[self.response1_col_name]
            response2 = feature[self.response2_col_name]
            
            judge_prompt = self.system_prompt.format(
                prompt=prompt, response1=response1, response2=response2
            )
            judge_swapped_position_prompt = self.system_prompt.format(
                prompt=prompt, response1=response2, response2=response1
            )
            judge_response = self.call_bedrock(judge_prompt, self.model_id)
            judge_responses.append(judge_response)

            judge_response_swapped_position = self.call_bedrock(judge_swapped_position_prompt, self.model_id)
            judge_responses_swapped_position.append(judge_response_swapped_position)

            prompts.append(prompt)
            responses1.append(response1)
            responses2.append(response2)

        responses = {
            "prompts": prompts,
            "responses1": responses1,
            "responses2": responses2,
            "judge_responses": judge_responses,
            "judge_responses_swapped_position": judge_responses_swapped_position
        }
        df_responses = pd.DataFrame(responses)
        return df_responses

    @staticmethod
    def calculate_result(df_responses):
        """calculate win/tie/loss result from LLM judge's response"""
        conditions = [
            (df_responses['judge_responses'] > df_responses['judge_responses_swapped_position']),
            (df_responses['judge_responses'] == df_responses['judge_responses_swapped_position']),
            (df_responses['judge_responses'] < df_responses['judge_responses_swapped_position'])
        ]
        choices = ['win', 'tie', 'lose']
        df_responses['result'] = np.select(conditions, choices, default='error')
        return df_responses

    def call_bedrock(
        self,
        prompt,
        model_id
    ):
        """
        References
        ----------
        https://docs.aws.amazon.com/code-library/latest/ug/python_3_bedrock-runtime_code_examples.html#anthropic_claude
        """
        client = boto3.client("bedrock-runtime", region_name="us-east-1")
    
        # Format the request payload using the model's native structure,
        # note different model variants may have different native structure,
        # this is for claude
        native_request = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}],
                }
            ],
        }
        
        request = json.dumps(native_request)
        try:
            response = client.invoke_model(modelId=model_id, body=request)
        except (ClientError, Exception) as e:
            print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
            exit(1)
    
        model_response = json.loads(response["body"].read())
        response_text = model_response["content"][0]["text"]
        return response_text



if __name__ == "__main__":
    # model completion/answer, we treat prediction1's as our baseline/reference model
    prediction1_path = "prediction_instruction_3B_model"
    prediction2_path = "prediction_dpo_model_v7"
    llm_judge_response_path = "llm_judge_responses_v7.parquet"
    
    df_prediction1 = pd.read_parquet(prediction1_path).rename(columns={"responses": "responses1"})
    df_prediction2 = pd.read_parquet(prediction2_path).rename(columns={"responses": "responses2"})
    df_prediction = df_prediction1.merge(df_prediction2, on=["prompts"])
    examples = df_prediction.to_dict("records")
    pairwise_judge = PairwiseBedRockLLMJudgeModule()
    df_responses = pairwise_judge(examples)
    df_responses.to_parquet(llm_judge_response_path, index=False)
    print(df_responses["result"].value_counts())
