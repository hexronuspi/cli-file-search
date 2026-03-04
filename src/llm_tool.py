from litellm import completion

def call_llm(api_key, model, messages):
    response = completion(
        model=model,
        api_key=api_key,
        messages=[
         {"role": "user", "content": f"{messages}"}
     ],
    )
    return response.choices[0].message.content

# response = completion(
#     model="mistral/mistral-small-latest",
#     api_key="v0Jditz2urXNkkMidyBMeUWn3mDifO9y",
#     messages=[
#         {"role": "user", "content": "hello from litellm"}
#     ],
# )

# print(response.choices[0].message.content)