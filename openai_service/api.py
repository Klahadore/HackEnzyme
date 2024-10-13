from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI
client = OpenAI()

def generate(prompt):
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Generate the STRING sequences associated with the reaction."},
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    
    return completion.choices[0].message
print(generate("breaks down hydrogen peroxide"))
