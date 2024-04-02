# pip3 install openai
import openai
import os
import time
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()
# client = OpenAI(api_key="local-generated-key", base_url="http://devnuc.lan:5000/v1")


def query_llm(messages, temperature=0.1):
    # Retry forever
    while True:
        try:
            response = client.chat.completions.create(
                model=os.environ["MODEL"],
                messages=messages,
                temperature=temperature,
                n=1,
            )

            content = response.choices[0].message.content.strip()

            return content
        except Exception as e:
            print("Failure querying the AI. Retrying...")
            time.sleep(1)

def query_openai(prompt):
    messages = [
        { "role": "user", "content": prompt }
    ]
    return query_llm(messages)

# STAGE 1

def select_reasoning_modules(task_description, reasoning_modules):
    """
    Step 1: SELECT relevant reasoning modules for the task.
    """
    prompt = f"Given the task: {task_description}, which of the following reasoning modules are relevant? Do not elaborate on why.\n\n" + "\n".join(reasoning_modules)
    selected_modules = query_openai(prompt)
    return selected_modules

def adapt_reasoning_modules(selected_modules, task_example):
    """
    Step 2: ADAPT the selected reasoning modules to be more specific to the task.
    """
    prompt = f"Without working out the full solution, adapt the following reasoning modules to be specific to our task:\n{selected_modules}\n\nOur task:\n{task_example}"
    adapted_modules = query_openai(prompt)
    return adapted_modules

def implement_reasoning_structure(adapted_modules, task_description):
    """
    Step 3: IMPLEMENT the adapted reasoning modules into an actionable reasoning structure.
    """
    prompt = f"Without working out the full solution, create an actionable reasoning structure for the task using these adapted reasoning modules:\n{adapted_modules}\n\nTask Description:\n{task_description}"
    reasoning_structure = query_openai(prompt)
    return reasoning_structure

# STAGE 2

def execute_reasoning_structure(reasoning_structure, task_instance):
    """
    Execute the reasoning structure to solve a specific task instance.
    """
    prompt = f"Using the following reasoning structure: {reasoning_structure}\n\nSolve this task, providing your final answer: {task_instance}"
    solution = query_openai(prompt)
    return solution

# Example usage
if __name__ == "__main__":
    reasoning_modules = [
        "1. How could I devise an experiment to help solve that problem?",
        "2. Make a list of ideas for solving this problem, and apply them one by one to the problem to see if any progress can be made.",
        "3. How could I measure progress on this problem?",
        "4. How can I simplify the problem so that it is easier to solve?",
        "5. What are the key assumptions underlying this problem?",
        "7. What are the alternative perspectives or viewpoints on this problem?",
        "8. What are the long-term implications of this problem and its solutions?",
        "9. How can I break down this problem into smaller, more manageable parts?",
        "10. Critical Thinking: This style involves analyzing the problem from different perspectives, questioning assumptions, and evaluating the evidence or information available. It focuses on logical reasoning, evidence-based decision-making, and identifying potential biases or flaws in thinking.",
        "11. Try creative thinking, generate innovative and out-of-the-box ideas to solve the problem. Explore unconventional solutions, thinking beyond traditional boundaries, and encouraging imagination and originality.",
        "13. Use systems thinking: Consider the problem as part of a larger system and understanding the interconnectedness of various elements. Focuses on identifying the underlying causes, feedback loops, and interdependencies that influence the problem, and developing holistic solutions that address the system as a whole.",
        "14. Use Risk Analysis: Evaluate potential risks, uncertainties, and tradeoffs associated with different solutions or approaches to a problem. Emphasize assessing the potential consequences and likelihood of success or failure, and making informed decisions based on a balanced analysis of risks and benefits.",
        "15. Use Reflective Thinking: Step back from the problem, take the time for introspection and self-reflection. Examine personal biases, assumptions, and mental models that may influence problem-solving, and being open to learning from past experiences to improve future approaches.",
        "16. What is the core issue or problem that needs to be addressed?",
        "17. What are the underlying causes or factors contributing to the problem?",
        "19. What are the potential obstacles or challenges that might arise in solving this problem?",
        "20. Are there any relevant data or information that can provide insights into the problem? If yes, what data sources are available, and how can they be analyzed?",
        "21. Are there any stakeholders or individuals who are directly affected by the problem? What are their perspectives and needs?",
        "23. How can progress or success in solving the problem be measured or evaluated?",
        "24. What indicators or metrics can be used?",
        "25. Is the problem a technical or practical one that requires a specific expertise or skill set? Or is it more of a conceptual or theoretical problem?",
        "27. Is the problem related to human behavior, such as a social, cultural, or psychological issue?",
        "28. Does the problem involve decision-making or planning, where choices need to be made under uncertainty or with competing objectives?",
        "29. Is the problem an analytical one that requires data analysis, modeling, or optimization techniques?",
        "30. Is the problem a design challenge that requires creative solutions and innovation?",
        "31. Does the problem require addressing systemic or structural issues rather than just individual instances?",
        "33. What kinds of solution typically are produced for this kind of problem specification?",
        #"38. Let’s think step by step."
        "39. Let’s make a step by step plan and implement it with good notation and explanation."
        "40. Analyze the linguistic style of the target publication, including elements such as vocabulary, sentence structure, and idiomatic expressions.",
        "41. Identify the target audience of the publication and consider their preferences, background knowledge, and expectations.",
        "42. Determine the purpose and intent behind the original document and ensure that it is preserved during the style transfer process.",
        "43. Assess the formality level of the target publication and adjust the language accordingly, using more casual or formal expressions as needed.",
        "44. Incorporate humor, wit, or satire in a way that aligns with the target publication's style and tone.",
        "45. Use figurative language, such as metaphors, similes, and analogies, to create vivid imagery and engage the reader, as appropriate for the target style.",
        "46. Adapt the narrative perspective and point of view to match the target publication's conventions (e.g., first-person, third-person, or editorial 'we').",
        "47. Manage the flow of information and structure of the document to ensure coherence and cohesion, even as the style is modified.",
        "48. Consider the cultural context and references that the target audience might be familiar with and incorporate them when appropriate.",
        "49. Analyze the emotional tone of the original document and adapt it to evoke the desired emotional response in the target audience.",
        "50. Examine the use of rhetorical devices, such as repetition, rhetorical questions, and hyperbole, in the target publication and employ them judiciously.",
        "51. Identify and maintain the key themes, arguments, and messages of the original document throughout the style transfer process.",
        "52. Assess the need for additional context or background information for the target audience and provide it seamlessly within the adapted document.",
        "53. Use appropriate transitional phrases and devices to maintain the logical flow of the document, even as the style is altered.",
        "54. Consider the visual formatting and presentation of the target publication, such as paragraph length, headings, and pull-quotes, and adapt the document accordingly.",
        "55. Evaluate the effectiveness of the style transfer by comparing the adapted document to exemplars from the target publication and making necessary adjustments.",
    ]

    task_example = '''[INSTRUCTION]: 
Please rewrite the provided source document to match the writing style of the reference document. First, analyze the reference document to identify and understand the key style attributes, such as:

1. Overall document structure and organization
2. Formality and tone of language
3. Vocabulary, terminology, and jargon typical of financial writing
4. Sentence structure, length, and complexity
5. Paragraph structure and length
6. Grammatical and syntactical patterns
7. Use of active vs. passive voice
8. Perspective (e.g. first person, third person)
9. Persuasive techniques and rhetorical devices employed
10. Formatting elements like headings, bullets, emphasis

After gaining a high-level understanding of these style attributes, list them out and apply them to the source document to effectively transfer the writing style. Aim to incorporate these attributes in a manner that maintains the coherence and logical flow of the source document's content.

The rewritten document should convincingly read as though it were written by the same author or publication of the reference document, while preserving the key informational content of the original source document. Please provide the full rewritten document.

[SOURCE DOCUMENT]:
"

"

[REFERENCE DOCUMENT]:
“

”
'''



    with open('responses.md', 'w') as file:
        print("## Stage 1 SELECT: Selected Modules:")
        selected_modules = select_reasoning_modules(task_example, reasoning_modules)
        print(selected_modules + "\n")
        file.write("## Stage 1 SELECT: Selected Modules:\n" + selected_modules + "\n\n")
        
        print("## Stage 1 ADAPT: Adapted Modules:")
        adapted_modules = adapt_reasoning_modules(selected_modules, task_example)
        print(adapted_modules + "\n")
        file.write("## Stage 1 ADAPT: Adapted Modules:\n" + adapted_modules + "\n\n")
        
        print("## Stage 1 IMPLEMENT: Reasoning Structure:")
        reasoning_structure = implement_reasoning_structure(adapted_modules, task_example)
        print(reasoning_structure + "\n")
        file.write("## Stage 1 IMPLEMENT: Reasoning Structure:\n" + reasoning_structure + "\n\n")

        print("# Stage 2: Final Result:")
        result = execute_reasoning_structure(reasoning_structure, task_example)
        print(result + "\n")
        file.write("# Stage 2: Final Result:\n" + result + "\n")
