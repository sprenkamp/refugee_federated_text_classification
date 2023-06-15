import pandas as pd
import openai
import os
from tqdm import tqdm

openai.organization = os.environ["OPENAI_ORGANIZATION"]
openai.api_key = os.getenv("OPENAI_API_KEY") 

df_in = pd.read_csv('data/df_gpt_input.csv')

df_out = pd.DataFrame(columns=['federation_level', 'x', 'x_translation', 'y_gpt_pred_str'])

def prompt_gen(input_text):
    prompt_instruction = f"""You are a Text Classifier identifying 5 types of refugee needs in a telegram message into one of the following categories: 'Medical', 'Accommodation', 'Government Services', 'Banking', 'Transport'. Cou you only classify one label per message.
    Further, the message are in Ukrainian or Russian language, please translate them into English. A sample output is:
    Доброго дня. Шукаю хорошого офтальмолога в Граці якій розуміє українську чи російську (чи англійську у крайньому випадку). Дякую.| Good day. I am looking for a good ophthalmologist in Graz who understands Ukrainian or Russian (or English as a last resort). Thank you.|Medical
    Here is the telegram_message:
    """
    prompt = f'{prompt_instruction} <{input_text}>'
    return prompt

def inference(prompt, model_name):
    try:
        if model_name in ['gpt-3.5-turbo', 'gpt-4']:
            completion = openai.ChatCompletion.create(
            model = model_name,
            messages = [
                {'role': 'assistant', 'content': f"""{prompt}"""},
            ],
            n = 1,
            stop = None,
            temperature=0.0, # set to 0 to get deterministic results
            timeout=100
            )
            return completion['choices'][0]['message']['content']
        else:
            completion = openai.Completion.create(
            engine=model_name,
            prompt=prompt,
            temperature=0.0,  # set to 0 to get deterministic results
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            )
            return completion['choices'][0]['text']
    except openai.error.OpenAIError as e:
        print(f"An error occurred while running inference: {str(e)}")
        return None
    
for index in tqdm(range(df_in.shape[0])):
    row = df_in.iloc[index]
    prompt = prompt_gen(row['x'])
    # print(prompt)
    category = inference(prompt, 'gpt-3.5-turbo')
    if category is not None:
        try:
            category_str = category.split('|')
            if category_str[1].strip() not in  ['Medical', 'Accommodation', 'Government Services', 'Banking', 'Transport']:
                print("Not correct category", category_str[1])
                continue
            df_out = df_out._append({'federation_level':row["federation_level"] ,'x': row['x'], 'x_translation': category_str[0], 'y_gpt_pred_str': category_str[1]}, ignore_index=True)
        except IndexError:
            print("The returned category string does not have the correct format.")

df_out.to_csv('data/df_gpt_output.csv', index=False)