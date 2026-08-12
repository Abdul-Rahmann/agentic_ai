import ollama
response = ollama.chat(model='llama3.1:latest', messages=[
    {
        'role': 'user',
        'content': 'What is 4 * 4 *4 ',
    },
])
print(response['message']['content'])