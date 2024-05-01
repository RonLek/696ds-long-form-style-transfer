import pandas as pd
import tkinter as tk
from tkinter import filedialog
from prompting import perform_zero_shot, perform_few_shot, perform_self_discover

class StyleTransfer:
    def __init__(self):
        self.paired_df = None
        self.publications_df = None

    def select_paired_csv(self):
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            self.paired_df = pd.read_csv(file_path)
            print(f"Paired CSV file selected: {file_path}")

    def select_publications_csv(self):
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            self.publications_df = pd.read_csv(file_path)
            print(f"Publications CSV file selected: {file_path}")

    def run_prompting(self, source_doc, publication_name, num_references=None, use_publication_name=False, use_reference_doc=True):
        reference_doc = self.publications_df.loc[self.publications_df['Publication_name'] == publication_name, 'reference_doc'].iloc[0]
        
        # Zero-shot
        zero_shot_result = perform_zero_shot(source_doc, reference_doc, publication_name, use_publication_name, use_reference_doc)
        
        # Few-shot
        few_shot_result = ""
        if num_references:
            reference_docs = self.publications_df.loc[self.publications_df['Publication_name'] == publication_name, 'reference_doc'].iloc[:num_references]
            few_shot_result = perform_few_shot(source_doc, reference_docs, publication_name)

        
        # Self-discover
        reasoning_modules = [
    "1. Analyze the target publication's style, including elements such as tone, voice, vocabulary, sentence structure, and formatting conventions.",
    "2. Identify the target audience of the publication and consider their preferences, background knowledge, and expectations when developing the style guideline.",
    "3. Use systems thinking to consider how the style guideline fits within the larger context of the publication's brand identity, mission, and values.",
    "4. Conduct a risk analysis to evaluate potential challenges or inconsistencies that may arise when applying the style guideline across various types of content.",
    "5. Engage in reflective thinking to examine any biases or assumptions that may influence the development of the style guideline, ensuring it remains objective and inclusive.",
    "6. Consider how the style guideline can address issues related to human behavior, such as readability, engagement, and accessibility for diverse audiences.",
    "7. Develop decision-making frameworks within the style guideline to help writers navigate choices related to language, structure, and presentation.",
    "8. Incorporate data analysis and modeling techniques to optimize elements of the style guideline, such as readability scores or word choice, based on audience engagement metrics.",
    "9. Approach the style guideline as a design challenge, seeking innovative ways to present information and examples that are both functional and visually appealing.",
    "10. Address systemic issues within the style guideline, such as promoting consistent branding, maintaining a cohesive voice, and ensuring alignment with editorial standards.",
    "11. Identify the types of content typically produced by the publication and tailor the style guideline to provide relevant guidance and examples for each content type.",
    "12. Develop a step-by-step process for applying the style guideline to different types of content, from initial drafting to final editing and formatting.",
    "13. Analyze the linguistic style of the target publication, including elements such as regional dialects, industry-specific jargon, and preferred terminology.",
    "14. Assess the formality level required for different types of content within the publication and provide guidance on adjusting language and tone accordingly.",
    "15. Offer guidance on incorporating humor, wit, or satire in a way that aligns with the publication's brand voice and target audience expectations.",
    "16. Provide examples of figurative language, such as metaphors and analogies, that are appropriate for the publication's style and subject matter.",
    "17. Specify the preferred narrative perspective and point of view for different types of content, such as news articles, opinion pieces, or feature stories.",
    "18. Include recommendations for structuring articles and managing the flow of information to ensure coherence and engagement.",
    "19. Consider the cultural context and diversity of the target audience when developing guidelines related to inclusive language, representation, and sensitivity.",
    "20. Provide guidance on adapting the emotional tone of content to align with the publication's editorial stance and desired audience response.",
    "21. Identify effective rhetorical devices and persuasive techniques commonly used within the publication and offer examples of their appropriate usage.",
    "22. Develop guidelines for maintaining the integrity of key messages and arguments when adapting content to fit the publication's style.",
    "23. Offer strategies for seamlessly incorporating necessary context or background information for the target audience without disrupting the flow of the content.",
    "24. Provide examples of effective transitional phrases and devices that maintain logical coherence while adhering to the publication's style preferences.",
    "25. Include guidelines for visual formatting and presentation, such as image selection, caption writing, and pull-quote usage, that align with the publication's design aesthetics.",
    "26. Develop a process for evaluating the effectiveness of the style guideline and gathering feedback from writers, editors, and readers to inform future iterations and improvements."
    
        ]
        reference_docs_self_discover = self.publications_df.loc[self.publications_df['Publication_name'] == publication_name, 'reference_doc'].sample(n=num_references)
        self_discover_result = perform_self_discover(source_doc, reference_docs_self_discover, publication_name, reasoning_modules)

        return zero_shot_result, few_shot_result, self_discover_result

    def process_dataset(self, num_docs=0, num_references=None):
        if num_docs == 0:
            num_docs = len(self.paired_df)

        for index, row in self.paired_df.iloc[:num_docs].iterrows():
            source_doc = row.iloc[1]  # Assumes 'paired_doc1' is the second column (index 1)
            publication_name = row.iloc[4]  # Assumes 'Pub2' is the fourth column (index 3)
            zero_shot_result, few_shot_result, self_discover_result = self.run_prompting(source_doc, publication_name, num_references=None,  use_publication_name=False, use_reference_doc=True)
            
            # Add new columns if they don't exist
            if 'Zeroshot_output' not in self.paired_df.columns:
                self.paired_df['Zeroshot_output'] = ''
            if 'Fewshot_output' not in self.paired_df.columns:
                self.paired_df['Fewshot_output'] = ''
            if 'SelfDiscover_output' not in self.paired_df.columns:
                self.paired_df['SelfDiscover_output'] = ''
            
            self.paired_df.at[index, 'Zeroshot_output'] = zero_shot_result
            self.paired_df.at[index, 'Fewshot_output'] = few_shot_result
            self.paired_df.at[index, 'SelfDiscover_output'] = self_discover_result

    def save_output_csv(self):
        output_file = "output_finance.csv"
        self.paired_df.to_csv(output_file, index=False)
        print(f"Output CSV file saved: {output_file}")

if __name__ == "__main__":
    style_transfer = StyleTransfer()

    style_transfer.select_paired_csv()
    style_transfer.select_publications_csv()

    num_docs = int(input("Enter the number of documents to perform style transfer on (0 for all documents): "))
    num_references = int(input("Enter the number of reference documents for few-shot prompting: "))

    style_transfer.process_dataset(num_docs, num_references)
    style_transfer.save_output_csv()