import openai
from openai import OpenAI

client = OpenAI(
  organization='org-fXh5scCwsB7yQSlPWKTjcZG7',
  project='proj_d0xwMSEMY1BHSDTXPTAZWwCB',
)




openai.api_key = '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'

response = openai.ChatCompletion.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "Say hello!"}
    ]
)

try:
    response = openai.ChatCompletion.create(...)
except Exception as e:
    print(f"Error: {e}")

print(response.choices[0].message['content'])