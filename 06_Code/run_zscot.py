# -*- coding: utf-8 -*-

# Representative audit code for PROMPTING = Zero-Shot Chain-of-Thought (ZSCOT)

# This notebook demonstrates the audit procedure using one LLM (GPT-4o) as example.
# The procedure includes (1) forced A/B decision parsing, (2) repeated sampling (100 runs), (3) retry/timeout handling, and (4) CSV logging.
# Other LLMs were evaluated using analogous execution logic with API-specific adjustments.
# Input prompts for this regime are provided as CSV files in data/inputs/.
# max_tokens and max_attempts were raised to accommodate chain-of-thought control logic.
# Running requires API access and keys; notebook is provided to document procedure and output schema.

# Install dependencies (example)
# pip install openai pandas

import pandas as pd
import openai
from openai import OpenAI
import time
import os

# Configuration
MODEL_NAME = "gpt-4o-2024-11-20"
DATA_DIR = "data/inputs"

# List of expected files for ZSCOT scenarios (YO for this run)
expected_file_names = [
    'YO_ZSCOT_EN.csv',
    'YO_ZSCOT_CN.csv',
    'YO_ZSCOT_JP.csv',
    'YO_ZSCOT_ES.csv'
]

# Map expected filenames to full paths on disk
file_mapping = {name: os.path.join(DATA_DIR, name) for name in expected_file_names}

# Verify all files are present
missing_files = [name for name, path in file_mapping.items() if not os.path.exists(path)]
if missing_files:
    raise Exception(f"Missing files in {DATA_DIR}: {', '.join(missing_files)}")

# Set up API
api_key = "YOUR_API_KEY_HERE"  # Replace with your LLM API key
client = OpenAI(api_key=api_key) # This example uses OpenAI

# Function to parse response in strict output format
def parse_response(response):
    # Look for an explicit decision
    if "Decision: A" in response:
        decision = "A"
        explanation = response.split("Decision: A")[1].strip()
    elif "Decision: B" in response:
        decision = "B"
        explanation = response.split("Decision: B")[1].strip()
    else:
        # If the decision format is not found, return None to trigger a retry
        return None, None

    # Validate that the explanation is present
    if not explanation or "no explanation" in explanation.lower():
        return None, None
    return decision, explanation

# Function to log invalid responses to a file
def log_invalid_response(prompt, response, attempt, scenario_id, language, run):
    with open("invalid_responses_gpt4o.txt", "a") as f:
        f.write(f"Scenario: {scenario_id}, Language: {language}, Run: {run}, Attempt: {attempt}\n")
        f.write(f"Prompt: {prompt}\n")
        f.write(f"Invalid Response: {response}\n")
        f.write("-" * 50 + "\n")

# Retry logic (retries until valid response with timeout)
def query_gpt4o(prompt, scenario_id, language, run, max_attempts=10, timeout_seconds=60):
    start_time = time.time()
    attempt = 0
    failed_attempts = 0
    while attempt < max_attempts:
        # Check if exceeding the timeout
        if time.time() - start_time > timeout_seconds:
            raise Exception(f"Timeout after {timeout_seconds} seconds for scenario {scenario_id}, language {language}, run {run}.")

        attempt += 1
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=300  # Increased to 300
            )
            result = response.choices[0].message.content.strip()
            decision, explanation = parse_response(result)
            if decision in ["A", "B"] and explanation != "No valid explanation provided.":
                return result, failed_attempts
            failed_attempts += 1
            log_invalid_response(prompt, result, attempt, scenario_id, language, run)
            print(f"  Attempt {attempt} failed: Invalid response. Retrying...")
            time.sleep(0.5)
        except Exception as e:
            print(f"GPT-4o error on attempt {attempt}: {e}")
            failed_attempts += 1
            time.sleep(0.5)
    raise Exception(f"Failed to get a valid response after {max_attempts} attempts for scenario {scenario_id}, language {language}, run {run}.")

# Main testing loop: Process only YO dilemma
# with specific language order and keep tracking failed attempts
dilemma = 'YO'
languages = ['EN', 'CN', 'JP', 'ES']
results = []
total_failed_attempts = 0

# Process each language file for the YO dilemma in the specified order
for language in languages:
    file_name = f"{dilemma}_ZSCOT_{language}.csv"
    if file_name not in file_mapping:
        print(f"Skipping {file_name} - not found in uploaded files.")
        continue

    print(f"Processing {file_name}...")
    df = pd.read_csv(file_mapping[file_name], quotechar='"')

    for index, row in df.iterrows():
        scenario_id = row['scenario_id']
        # Extract scenario text starting from "Context:"
        scenario_text = row['scenario_text']
        if "Context:" not in scenario_text:
            print(f"Error: 'Context:' not found in scenario_text for {scenario_id}")
            continue
        prompt = scenario_text[scenario_text.find("Context:"):]

        print(f"Testing {scenario_id} (Language: {language})")
        for run in range(1, 101):
            print(f"  Run {run} of 100")
            response, failed_attempts = query_gpt4o(prompt, scenario_id, language, run)
            total_failed_attempts += failed_attempts
            decision, explanation = parse_response(response)
            results.append([scenario_id, language, run, decision, explanation])
            # Save intermediate results after every run to avoid data loss
            temp_df = pd.DataFrame(results, columns=['scenario_id', 'language', 'run_number', 'decision', 'explanation'])
            temp_df.to_csv(f"{dilemma}_ZSCOT_results_gpt4o_temp.csv", index=False)
            # Add a 1-second delay between requests to avoid rate limits
            time.sleep(1)
            # Add a 5-second delay after every 10 requests to avoid burst limits
            if run % 10 == 0:
                print(f"  Adding a 5-second delay after run {run} to avoid burst rate limits...")
                time.sleep(5)

# Save final results for the YO dilemma
results_df = pd.DataFrame(results, columns=['scenario_id', 'language', 'run_number', 'decision', 'explanation'])
results_df.to_csv(f"{dilemma}_ZSCOT_results_gpt4o.csv", index=False)