# -*- coding: utf-8 -*-

# Representative audit code for PROMPTING = Cultural Calibration (CC)

# This notebook demonstrates the audit procedure using one LLM (GPT-4o) as example.
# The procedure includes (1) forced A/B decision parsing, (2) repeated sampling (100 runs), (3) retry/timeout handling, and (4) CSV logging.
# Other LLMs were evaluated using analogous execution logic with API-specific adjustments.
# Input prompts for this regime are provided as CSV files in data/inputs/.
# Running requires API access and keys; notebook is provided to document procedure and output schema.

# Install dependencies (example)
# pip install openai pandas

import pandas as pd
import openai
from openai import OpenAI
import time
import os

# Configuration
MODEL_NAME = "gpt-4o-2024-11-20" # as example
DATA_DIR = "data/inputs"  # folder that contains CSV files

# List of expected files for CC scenarios (only MF for this run)
expected_file_names = [
    'MF_CC_EN.csv',
    'MF_CC_CN.csv',
    'MF_CC_JP.csv',
    'MF_CC_ES.csv'
]

# Map expected filenames to full paths on disk
file_mapping = {name: os.path.join(DATA_DIR, name) for name in expected_file_names}

# Verify all expected files
missing_files = [name for name, path in file_mapping.items() if not os.path.exists(path)]
if missing_files:
    raise Exception(f"Missing files in {DATA_DIR}: {', '.join(missing_files)}")

# Set up API
api_key = "YOUR_API_KEY_HERE"  # Replace with your LLM API key
client = OpenAI(api_key=api_key) # This example uses OpenAI

# Strictly defined, forced choice output format
def parse_response(response):
    if "Decision: A" in response:
        decision = "A"
        explanation = response.split("Decision: A")[1].strip()
    elif "Decision: B" in response:
        decision = "B"
        explanation = response.split("Decision: B")[1].strip()
    else:
        decision = "N/A"
        explanation = "N/A"
    # Validate that both decision and explanation are present
    if decision == "N/A" or not explanation or "no explanation" in explanation.lower():
        return "N/A", "N/A"
    return decision, explanation

# Retry logic
def query_gpt4o(prompt, retries=5):
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=100
            )
            result = response.choices[0].message.content.strip()
            decision, explanation = parse_response(result)
            if decision in ["A", "B"] and explanation != "N/A":
                return result
            print(f"  Attempt {attempt + 1} failed: Invalid response - {result}")
            time.sleep(1)
        except Exception as e:
            print(f"GPT-4o error on attempt {attempt + 1}: {e}")
            time.sleep(1)

    return "Decision: N/A\nNo valid response after retries."

# Main testing loop with explicit language order
dilemma = 'MF'
languages = ['EN', 'CN', 'JP', 'ES']
results = []
failed_runs = 0

# Process each language file for the MF dilemma in the specified order
for language in languages:
    file_name = f"{dilemma}_CC_{language}.csv"
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
            response = query_gpt4o(prompt)
            decision, explanation = parse_response(response)
            # Retry if decision or explanation is missing
            if decision == "N/A" or explanation == "N/A":
                print(f"  Run {run} failed: Missing decision or explanation. Retrying...")
                response = query_gpt4o(prompt)
                decision, explanation = parse_response(response)
                if decision == "N/A" or explanation == "N/A":
                    failed_runs += 1
            results.append([scenario_id, language, run, decision, explanation])
            # Save intermediate results after every run to avoid data loss
            temp_df = pd.DataFrame(results, columns=['scenario_id', 'language', 'run_number', 'decision', 'explanation'])
            temp_df.to_csv(f"{dilemma}_CC_results_gpt4o_temp.csv", index=False)
            # Add a 1-second delay in between requests to avoid rate limits
            time.sleep(1)
            # Add a 5-second delay after every 10 requests to avoid burst limits
            if run % 10 == 0:
                print(f"  Adding a 5-second delay after run {run} to avoid burst rate limits...")
                time.sleep(5)

# Save final results for the MF dilemma
results_df = pd.DataFrame(results, columns=['scenario_id', 'language', 'run_number', 'decision', 'explanation'])
results_df.to_csv(f"{dilemma}_CC_results_gpt4o.csv", index=False)