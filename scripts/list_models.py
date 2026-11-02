import os
import json
import requests
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("DIAL_API_KEY")


all_models = requests.get("https://ai-proxy.lab.epam.com/openai/models", headers={"Api-Key": api_key}).json()["data"]

def get_model_limits(model):
    try:
        limits = requests.get(f"https://ai-proxy.lab.epam.com/v1/deployments/{model['id']}/limits", headers={"Api-Key": api_key}).json()
        minute_limit = limits.get("minuteTokenStats", {}).get("total", 0)
        day_limit = limits.get("dayTokenStats", {}).get("total", 0)
        if minute_limit > 0 or day_limit > 0:
            return {model['id']: {"limits": {"minute": minute_limit, "day": day_limit}}}
    except:
        pass

limits_per_available_model = {}
with ThreadPoolExecutor(max_workers=128) as executor:
    for model_limits in executor.map(get_model_limits, all_models):
        if model_limits:  # Only include models with nonzero limits (available to you)
            limits_per_available_model.update(model_limits)

print(json.dumps(limits_per_available_model, indent=4))