# pip3 install openai
import openai
import os
import time
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

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
    "1. Analyze the target publication's style, including elements such as tone, voice, vocabulary, sentence structure, and formatting conventions."
    "2. Identify the target audience of the publication and consider their preferences, background knowledge, and expectations when developing the style guideline."
    "3. Use systems thinking to consider how the style guideline fits within the larger context of the publication's brand identity, mission, and values."
    "4. Conduct a risk analysis to evaluate potential challenges or inconsistencies that may arise when applying the style guideline across various types of content."
    "5. Engage in reflective thinking to examine any biases or assumptions that may influence the development of the style guideline, ensuring it remains objective and inclusive."
    "6. Consider how the style guideline can address issues related to human behavior, such as readability, engagement, and accessibility for diverse audiences."
    "7. Develop decision-making frameworks within the style guideline to help writers navigate choices related to language, structure, and presentation."
    "8. Incorporate data analysis and modeling techniques to optimize elements of the style guideline, such as readability scores or word choice, based on audience engagement metrics."
    "9. Approach the style guideline as a design challenge, seeking innovative ways to present information and examples that are both functional and visually appealing."
    "10. Address systemic issues within the style guideline, such as promoting consistent branding, maintaining a cohesive voice, and ensuring alignment with editorial standards."
    "11. Identify the types of content typically produced by the publication and tailor the style guideline to provide relevant guidance and examples for each content type."
    "12. Develop a step-by-step process for applying the style guideline to different types of content, from initial drafting to final editing and formatting."
    "13. Analyze the linguistic style of the target publication, including elements such as regional dialects, industry-specific jargon, and preferred terminology."
    "14. Assess the formality level required for different types of content within the publication and provide guidance on adjusting language and tone accordingly."
    "15. Offer guidance on incorporating humor, wit, or satire in a way that aligns with the publication's brand voice and target audience expectations."
    "16. Provide examples of figurative language, such as metaphors and analogies, that are appropriate for the publication's style and subject matter."
    "17. Specify the preferred narrative perspective and point of view for different types of content, such as news articles, opinion pieces, or feature stories."
    "18. Include recommendations for structuring articles and managing the flow of information to ensure coherence and engagement."
    "19. Consider the cultural context and diversity of the target audience when developing guidelines related to inclusive language, representation, and sensitivity."
    "20. Provide guidance on adapting the emotional tone of content to align with the publication's editorial stance and desired audience response."
    "21. Identify effective rhetorical devices and persuasive techniques commonly used within the publication and offer examples of their appropriate usage."
    "22. Develop guidelines for maintaining the integrity of key messages and arguments when adapting content to fit the publication's style."
    "23. Offer strategies for seamlessly incorporating necessary context or background information for the target audience without disrupting the flow of the content."
    "24. Provide examples of effective transitional phrases and devices that maintain logical coherence while adhering to the publication's style preferences."
    "25. Include guidelines for visual formatting and presentation, such as image selection, caption writing, and pull-quote usage, that align with the publication's design aesthetics."
    "26. Develop a process for evaluating the effectiveness of the style guideline and gathering feedback from writers, editors, and readers to inform future iterations and improvements."
    ]

        # "10. Critical Thinking: This style involves analyzing the problem from different perspectives, questioning assumptions, and evaluating the evidence or information available. It focuses on logical reasoning, evidence-based decision-making, and identifying potential biases or flaws in thinking.",
        # "11. Try creative thinking, generate innovative and out-of-the-box ideas to solve the problem. Explore unconventional solutions, thinking beyond traditional boundaries, and encouraging imagination and originality.",
        # "13. Use systems thinking: Consider the problem as part of a larger system and understanding the interconnectedness of various elements. Focuses on identifying the underlying causes, feedback loops, and interdependencies that influence the problem, and developing holistic solutions that address the system as a whole.",
        # "14. Use Risk Analysis: Evaluate potential risks, uncertainties, and tradeoffs associated with different solutions or approaches to a problem. Emphasize assessing the potential consequences and likelihood of success or failure, and making informed decisions based on a balanced analysis of risks and benefits.",
        # "15. Use Reflective Thinking: Step back from the problem, take the time for introspection and self-reflection. Examine personal biases, assumptions, and mental models that may influence problem-solving, and being open to learning from past experiences to improve future approaches.",
        # "27. Is the problem related to human behavior, such as a social, cultural, or psychological issue?",
        # "28. Does the problem involve decision-making or planning, where choices need to be made under uncertainty or with competing objectives?",
        # "29. Is the problem an analytical one that requires data analysis, modeling, or optimization techniques?",
        # "30. Is the problem a design challenge that requires creative solutions and innovation?",
        # "31. Does the problem require addressing systemic or structural issues rather than just individual instances?",
        # "33. What kinds of solution typically are produced for this kind of problem specification?",
        # #"38. Let’s think step by step."
        # "39. Let’s make a step by step plan and implement it with good notation and explanation."
        # "40. Analyze the linguistic style of the target publication, including elements such as vocabulary, sentence structure, and idiomatic expressions.",
        # "41. Identify the target audience of the publication and consider their preferences, background knowledge, and expectations.",
        # "42. Determine the purpose and intent behind the original document and ensure that it is preserved during the style transfer process.",
        # "43. Assess the formality level of the target publication and adjust the language accordingly, using more casual or formal expressions as needed.",
        # "44. Incorporate humor, wit, or satire in a way that aligns with the target publication's style and tone.",
        # "45. Use figurative language, such as metaphors, similes, and analogies, to create vivid imagery and engage the reader, as appropriate for the target style.",
        # "46. Adapt the narrative perspective and point of view to match the target publication's conventions (e.g., first-person, third-person, or editorial 'we').",
        # "47. Manage the flow of information and structure of the document to ensure coherence and cohesion, even as the style is modified.",
        # "48. Consider the cultural context and references that the target audience might be familiar with and incorporate them when appropriate.",
        # "49. Analyze the emotional tone of the original document and adapt it to evoke the desired emotional response in the target audience.",
        # "50. Examine the use of rhetorical devices, such as repetition, rhetorical questions, and hyperbole, in the target publication and employ them judiciously.",
        # "51. Identify and maintain the key themes, arguments, and messages of the original document throughout the style transfer process.",
        # "52. Assess the need for additional context or background information for the target audience and provide it seamlessly within the adapted document.",
        # "53. Use appropriate transitional phrases and devices to maintain the logical flow of the document, even as the style is altered.",
        # "54. Consider the visual formatting and presentation of the target publication, such as paragraph length, headings, and pull-quotes, and adapt the document accordingly.",
        # "55. Evaluate the effectiveness of the style transfer by comparing the adapted document to exemplars from the target publication and making necessary adjustments.",

#     task_example = '''[INSTRUCTION]: 
# Please rewrite the provided source document to match the writing style of the reference document. First, analyze the reference document to identify and understand the key style attributes, such as:

# 1. Overall document structure and organization
# 2. Formality and tone of language
# 3. Vocabulary, terminology, and jargon typical of financial writing
# 4. Sentence structure, length, and complexity
# 5. Paragraph structure and length
# 6. Grammatical and syntactical patterns
# 7. Use of active vs. passive voice
# 8. Perspective (e.g. first person, third person)
# 9. Persuasive techniques and rhetorical devices employed
# 10. Formatting elements like headings, bullets, emphasis

# After gaining a high-level understanding of these style attributes, list them out and apply them to the source document to effectively transfer the writing style. Aim to incorporate these attributes in a manner that maintains the coherence and logical flow of the source document's content.

# The rewritten document should convincingly read as though it were written by the same author or publication of the reference document, while preserving the key informational content of the original source document. Please provide the full rewritten document.

# [SOURCE DOCUMENT]:
# "

# "

# [REFERENCE DOCUMENT]:
# “

# ”
# '''
#Review the reference documents and identify the following style attributes and add new ones as you deem fit, the below list is not comprehensive and can and should be added on to as needed, also provide specific details for each and how each style is reflected in the referenced article. Do all this while not copying any text/word/phrase from the reference document:
# Review the reference documents and identify the following style attributes, providing clear examples for each:
#    Please analyze the provided reference documents to create a comprehensive JSON-formatted style guide for Nerdwallet. 
    # The style guide should capture the key elements and attributes of Nerdwallet's distinctive writing style, enabling writers
    # to consistently produce content that aligns with their brand voice and tone.
    # Review the reference documents and identify the following style attributes and add new ones as you deem fit,
    # the below list is not comprehensive and can and should be added on to as needed, also provide specific details
    # and generate examples for each and how each style is reflected in the referenced article. Do all this while not
    # copying any text/word/phrase from the reference document:
    publication = 'engadget.com'
    task_example = f'''
Your task is to create   a comprehensive style guide for {publication} based on the provided reference {publication}'s articles.
The style guide should serve as a detailed reference for writers, editors, and content creators to ensure consistency
and adherence to the publication's unique voice and style.

To create the style guide, carefully analyze the reference documents and identify the key attributes that define the
{publication}'s style. Consider the following attributes as a starting point, but feel free to discard, modify, or add new
attributes as necessary to accurately capture the {publication}'s style:


1. "documentStructure": [Document structure and organization]
2. "languageTone": [Language formality and tone]
3. "vocabulary": [Vocabulary, terminology, and jargon]
4. "sentenceStructure": [Sentence structure, length, and complexity and formality]
5. "paragraphStructure": [Paragraph structure and length] 
6. "grammarSyntax": [Grammar and syntax patterns]
7. "voiceUsage": [Active vs. passive voice usage]
8. "perspective": [Perspective (e.g., first person, third person)]
9. "persuasiveTechniques": [Persuasive techniques and rhetorical devices]
10. "formattingElements": [Formatting and visual elements]
11. "targetAudience": [Target audience and purpose of the content]
12. "inclusiveLanguage": [Preferred pronouns and inclusive language guidelines]  
13. "punctuationCapitalization": [Punctuation and capitalization rules]
14. "citationStyle": [Acceptable sources and citation styles]
15. "writingTips": [Any specific do's and don'ts for writers to keep in mind]
..... add new attributes 

For each attribute, provide a brief description and relevant examples in an array.
Add new attributes as required by the reference document, the list isnt comprehensive.

Organize the style guide in JSON into a clear hierarchy, making it easy for LLM's to parse and reference.

The final JSON style guide should be comprehensive, enabling LLM's to internalize {publication}'s unique style
and consistently produce content that resonates with their target audience.

Please provide the complete style guide in valid JSON format.

[SOURCE DOCUMENT]:
{Source_Doc}
[REFERENCE DOCUMENT 2]:
{Reference_Doc}


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
        file.write(f"# Stage 2: Final Result:\n```{result}```\n")
        
