# Runs in venv "researchgpt" on local machine
# deactivate
# source /Users/johncole/Github/researchgpt-demo/researchgpt/bin/activate

from PyPDF2 import PdfReader
import pandas as pd
from openai.embeddings_utils import get_embedding, cosine_similarity
import openai

path_to_pdf = "/Users/johncole/Desktop/DALL-E/30 Minute Summaries/How to Get Rich-long.pdf"

def open_file(filepath):
    '''
    Open a file and return its contents.
    
    Parameters
    ----------
    filepath : str
        The path to the file to open.
    
    Returns
    -------
    str
        The contents of the file.
    '''
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

debug = True

open_ai_key = open_file("openai_api.key")

class Chatbot():
    
    def parse_paper(self, pdf):
        print("Parsing paper")
        number_of_pages = len(pdf.pages)
        print(f"Total number of pages: {number_of_pages}")
        paper_text = []
        for i in range(number_of_pages):
            print("Parsing page: " + str(i))
            page = pdf.pages[i]
            page_text = []

            def visitor_body(text, cm, tm, fontDict, fontSize):
                # print(f"Text: {text}")
                # print(f'tm: {tm}')
                x = tm[4]
                y = tm[5]
                # ignore header/footer
                # if (y > 50 and y < 720) and (len(text.strip()) > 1):
                if (len(text.strip()) > 1):
                    page_text.append({
                    'fontsize': fontSize,
                    'text': text.strip().replace('\x03', ''),
                    'x': x,
                    'y': y
                    })

            _ = page.extract_text(visitor_text=visitor_body)

            blob_font_size = None
            blob_text = ''
            processed_text = []

            for t in page_text:
                if t['fontsize'] == blob_font_size:
                    blob_text += f" {t['text']}"
                    if len(blob_text) >= 2000:
                        processed_text.append({
                            'fontsize': blob_font_size,
                            'text': blob_text,
                            'page': i
                        })
                        blob_font_size = None
                        blob_text = ''
                else:
                    if blob_font_size is not None and len(blob_text) >= 1:
                        processed_text.append({
                            'fontsize': blob_font_size,
                            'text': blob_text,
                            'page': i
                        })
                    blob_font_size = t['fontsize']
                    blob_text = t['text']
                paper_text += processed_text
                # print(f'processed_text: {processed_text}')
        print("Done parsing paper")
        #if debug:
        #    print(paper_text)
        return paper_text

    def paper_df(self, pdf):
        """
        Creates a dataframe from a paper object.

        Parameters
        ----------
        pdf : list
            A list of dictionaries that contains data about the paper.

        Returns
        -------
        df : pd.DataFrame
            A pandas dataframe containing the paper's data.

        """
        filtered_pdf= []
        for row in pdf:
            if len(row['text']) < 30:
                print(f"Skipping row: {row['text']}")
                print(f"Row length: {len(row['text'])}")
                input()
                continue
            filtered_pdf.append(row)
        df = pd.DataFrame(filtered_pdf)
        print('Data frame shape: ')
        print(df.shape)
        # remove elements with identical df[text] and df[page] values
        df = df.drop_duplicates(subset=['text', 'page'], keep='first')
        df['length'] = df['text'].apply(lambda x: len(x))
        print('Done creating dataframe')
        return df

    def calculate_embeddings(self, df):
        """
        Calculates text embeddings for a dataframe of text.
        Parameters
        ----------
        df : Pandas dataframe
            The dataframe to calculate text embeddings for.
        Returns
        -------
        df : Pandas dataframe
            The dataframe with the embeddings added as a column.
        """
        print('Calculating embeddings')
        print(f'OpenAPI key: {open_ai_key}')
        openai.api_key = open_ai_key
        embedding_model = "text-embedding-ada-002"
        embeddings = df.text.apply([lambda x: get_embedding(x, engine=embedding_model)])
        df["embeddings"] = embeddings
        print('Done calculating embeddings')
        return df

    def search_embeddings(self, df, query, n=3, pprint=True):
        """
        Search the dataframe for the query and return the n most similar documents.
        
        Parameters
        ----------
        df : pd.DataFrame
            Dataframe with the documents to search
        query : str
            The query to search the documents
        n : int, default=3
            The number of results to return
        pprint : bool, default=True
            If True, print the results
        
        Returns
        -------
        results : pd.DataFrame
            A dataframe with the n most similar documents
        """
        query_embedding = get_embedding(
            query,
            engine="text-embedding-ada-002"
        )
        df["similarity"] = df.embeddings.apply(lambda x: cosine_similarity(x, query_embedding))
        
        results = df.sort_values("similarity", ascending=False, ignore_index=True)
        # make a dictionary of the the first three results with the page number as the key and the text as the value. The page number is a column in the dataframe.
        results = results.head(n)
        global sources 
        sources = []
        for i in range(n):
            # append the page number and the text as a dict to the sources list
            sources.append({'Page '+str(results.iloc[i]['page']): results.iloc[i]['text'][:150]+'...'})
        print(sources)
        return results.head(n)
    
    def create_prompt(self, df, user_input):
        result = self.search_embeddings(df, user_input, n=5)
        print(result)
        original_prompt = """You are a large language model whose expertise is reading and summarizing scientific papers. 
        You are given a query and a series of text embeddings from a paper in order of their cosine similarity to the query.
        You must take the given embeddings and return a very detailed summary of the paper that answers the query.
            
            Given the question: """+ user_input + """
            
            and the following embeddings as data: 
            
            1.""" + str(result.iloc[0]['text']) + """
            2.""" + str(result.iloc[1]['text']) + """
            3.""" + str(result.iloc[2]['text']) + """
            4.""" + str(result.iloc[3]['text']) + """
            5.""" + str(result.iloc[4]['text']) + """

            Return a detailed answer based on the paper:"""
        
        prompt = """You are a large language model whose expertise is reading and summarizing literature. 
        You are given a query and a series of text embeddings from a paper in order of their cosine similarity to the query.
        You must take the given embeddings and return a very detailed answer the query.
            
            Given the question: """+ user_input + """
            
            and the following embeddings as data: 
            
            1.""" + str(result.iloc[0]['text']) + """
            2.""" + str(result.iloc[1]['text']) + """
            3.""" + str(result.iloc[2]['text']) + """

            Return a detailed answer based on the literature:"""

        print('Done creating prompt')
        return prompt

    def gpt(self, prompt):
        print('Sending request to GPT-3')
        openai.api_key = open_ai_key
        r = openai.Completion.create(model="text-davinci-003", prompt=prompt, temperature=0.4, max_tokens=1500)
        answer = r.choices[0]['text']
        print('Done sending request to GPT-3')
        response = {'answer': answer, 'sources': sources}
        return response

    '''
    def reply(self, prompt):
        # print(prompt)
        prompt = self.create_prompt(df, prompt)
        return self.gpt(prompt)
    '''

def main():
    open_ai_key = open_file("openai_api.key")
    print(f'OpenApi Key: {open_ai_key}')

    # Load up the pdf file data.
    print("Processing pdf")
    print("Processing pdf: " + path_to_pdf)
    pdf = PdfReader(path_to_pdf)
    
    chatbot = Chatbot()
    paper_text = chatbot.parse_paper(pdf)       # parses the text into some weird data structure.
    global df
    df = chatbot.paper_df(paper_text)           # creates a dataframe from the weird data structure.
    df = chatbot.calculate_embeddings(df)       # calculates the embeddings for each row in the dataframe.
    print("Done processing pdf")

    while True:
            chatbot = Chatbot()
            query = input("Enter your query: ")
            query = str(query)
            prompt = chatbot.create_prompt(df, query)
            response = chatbot.gpt(prompt)
            print(response)
            print(" # # # ")


if __name__ == '__main__':
    main()


