import openai
import time
# from dotenv import load_dotenv  # Importing python-dotenv to manage environment variables securely

# Load environment variables from the .env file
# load_dotenv()

# Set the OpenAI API key from environment variables instead of hardcoding it
# api_key = os.getenv('OPENAI_API_KEY')
# base_url = os.getenv('OPENAI_API_BASE')

class Demo:
    def __init__(self, api_key, engine, temperature=0, max_tokens=300, top_p=1, frequency_penalty=0, presence_penalty=0, logprobs=False):
        self.api_key = api_key
        self.engine = engine
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.logprobs = logprobs  # Renamed from logprobs to logit_bias, adjusting for new API

        # if self.engine.lower().startswith('gpt'):
        #     self.client = openai.OpenAI(
        #         api_key=api_key,
        #         base_url='https://api.ai.it.ufl.edu'
        #     )
        # else:
        self.client = openai.OpenAI(
            api_key=api_key
        )

    def process_sample(self, prompt_list):
        response = self.client.chat.completions.create(model=self.engine,
                                                  messages=prompt_list,
                                                  max_tokens=self.max_tokens,
                                                  top_p=self.top_p)
        result = [choice.message.content for choice in response.choices][0]
        return result

    def process_batch(self, input_file_path):
        # Upload the input file
        with open(input_file_path, "rb") as file:
            uploaded_file = self.client.files.create(
                file=file,
                purpose="batch"
            )

        # Create the batch job
        batch_job = self.client.batches.create(
            input_file_id=uploaded_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )

        # Monitor the batch job status
        while batch_job.status not in ["completed", "failed", "cancelled"]:
            time.sleep(30)  # Wait for 3 seconds before checking the status again
            print(f"Batch job status: {batch_job.status}...trying again in 3 seconds...")
            batch_job = self.client.batches.retrieve(batch_job.id)

        # Download and save the results
        if batch_job.status == "completed":
            result_file_id = batch_job.output_file_id
            if result_file_id:
                result = self.client.files.content(result_file_id)
                # result = client.files.retrieve(result_file_id)
                return result.text
            else:
                print(f"Batch job failed with status: {batch_job.status}")
                return None
        else:
            print(f"Batch job failed with status: {batch_job.status}")
            return None

    def get_multiple_sample(self, prompt_list, no_prob=False):
        messages = [{'role': 'user', 'content': prompt_list}]

        if no_prob:
            response = self.client.chat.completions.create(
                model=self.engine,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty,
            )
            probs = None
        else:
            # Modified API call method and parameters for newer API
            response = self.client.chat.completions.create(
                model=self.engine,  # Changed from 'engine' to 'model' to align with new API
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty,
                # Removed 'best_of' and 'logprobs', as these are no longer valid in the new API
                logprobs=self.logprobs
            )
            probs = [choice.logprobs for choice in response.choices]
        # Correctly accessing the results according to the newer API response structure
        results = [choice.message.content for choice in response.choices]

        return results, probs

def run(prompt_list):
    demo = Demo(
        engine="gpt-3.5-turbo",  # Adjust as needed
        temperature=0,  # Control randomness: lowering results in less random completions (0 ~ 1.0)
        max_tokens=100,  # Max number of tokens to generate (1 ~ 4,000)
        top_p=1,  # Control diversity (0 ~ 1.0)
        frequency_penalty=0,  # Penalty for new tokens based on their existing frequency (0 ~ 2.0)
        presence_penalty=0,  # Penalty based on whether tokens have appeared before (0 ~ 2.0)
        logprobs=True
        # logit_bias={}  # Using an empty dictionary for simplicity, as the new API uses logit_bias instead of logprobs
    )
    results = demo.get_multiple_sample(prompt_list)
    print(results[0])

if __name__ == '__main__':
    prompt_list = ["I am very happy,"]
    run(prompt_list)