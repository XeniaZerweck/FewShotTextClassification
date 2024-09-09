# =========================== GitHub API Request ===========================

# ==================== Install Packages ====================
import requests as req
import csv
from pprint import pprint
from random import sample
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
import re

# ==================== For Preparation: Check current rate limit ==================== 
url = "https://api.github.com/rate_limit"
response = req.get(url)
pprint(response.json())

# ==================== Get Started ====================
# Requirements
token = "" # required!

headers = {
    "Authorization": f"token {token}",
    "Accept": "application/vnd.github.v3+json" # specifying json-formatted data
}

repositories = ['tensorflow', 'vscode', 'pytorch']

repositories_dict = {
    'tensorflow': {'owner': 'tensorflow', 'repo': 'tensorflow'},
    'vscode': {'owner': 'microsoft', 'repo': 'vscode'},
    'pytorch': {'owner': 'pytorch', 'repo': 'pytorch'}}

# ==================== First Request  ====================
# Get general Information on all issues, keep those that have more than min_comments comments.
def req_issues(min_comments, repository):
    ans = []
    page = 1
    while 1 == 1:
        owner = repositories_dict[repository]['owner']
        repo = repositories_dict[repository]['repo']
        
        response = req.request(
            'get',
            f'https://api.github.com/repos/{owner}/{repo}/issues?sort=comments&direction=desc&per_page=100&page={page}'.format(page),
            headers=headers)
        
        all_results = response.json()
        
         # keep identifier of issue, if it has at least min_comments:
        filtered_results = [x['number'] for x in all_results if x['comments'] >= min_comments]
        ans.extend(filtered_results)
        page += 1
        
        # since sort=comments&direction=desc the loop can stop once results are not added any more
        
        if len(filtered_results) != len(all_results):
            return ans
        
# call function & request
relevant_issues_tensorflow = req_issues(10, 'tensorflow')
relevant_issues_vscode = req_issues(10, 'vscode')
relevant_issues_pytorch = req_issues(10, 'pytorch')

# Sample 20 ramdom issues from each community (so far only ID is stored)
sample_issues_tensorflow = sample(relevant_issues_tensorflow, 20)
sample_issues_vscode = sample(relevant_issues_vscode, 20)
sample_issues_pytorch = sample(relevant_issues_pytorch, 20)

sample_issues_mapping = {
    'tensorflow': sample_issues_tensorflow,
    'vscode': sample_issues_vscode,
    'pytorch': sample_issues_pytorch
}

for repo_name, sample_issues in sample_issues_mapping.items():
    repositories_dict[repo_name]['sample_issues'] = sample_issues


# Get comments from specific issue
df = pd.DataFrame(columns = ['id', 'body', 'issue_number', 'repo', 'created'])
def get_sample_comments(owner, repo, sample_issues):
    for sample_issue in sample_issues:
        
        # get original issue-text
        response = req.get(f"https://api.github.com/repos/{owner}/{repo}/issues/{sample_issue}",
                          headers = headers)
        data = response.json()
        
        df.loc[len(df)] = [data['number'], data['body'], data['number'], repo, data['created_at']]
        
        # get comments
        response = req.get(f"https://api.github.com/repos/{owner}/{repo}/issues/{sample_issue}/comments",
                          headers = headers)
        data = response.json()
        
        for item in data:
            df.loc[len(df)] = [item['id'], item['body'], sample_issue, repo, item['created_at']]

for repository in repositories:
    print(repository)
    owner = repositories_dict[repository]['owner']
    repo = repositories_dict[repository]['repo']
    sample_issues = repositories_dict[repository]['sample_issues']
    
    get_sample_comments(owner, repo, sample_issues)

# ==================== Create (csv-) datasets to work with ====================
    
# Full comments
df.to_csv('/Users/xeniamarlenezerweck/Documents/Verzeichnis/Master/Thesis/Daten/original_dataset.csv', )

for repository in repositories:
    full_entries_count = len(df[df['repo'] == repository])
    print(f'Anzahl voller Beitraege {repository}: {full_entries_count}')

'''
Anzahl voller Beitraege tensorflow: 345
Anzahl voller Beitraege vscode: 375
Anzahl voller Beitraege pytorch: 366
Anzahl Datenpunkte insgesamt (len(df)) = 1086
'''

# Data Cleaning for easier classification
def clean_text(text):
    # Remove everything from the beginning to "### Current behavior?" (GitHub-Form to report issues)
    text = re.sub(r'^.*### Current behavior\?\s*', '', text, flags=re.DOTALL)
    # Remove code snippets enclosed in triple backticks
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    # Remove links
    text = re.sub(r'http\S+|www\.\S+', '', text)
    # Remove usernames (starting with '@')
    text = re.sub(r'@\w+', '', text)
    # Replace multiple line breaks with a single line break (for easier readability)
    text = re.sub(r'\n+', '\n', text)
    return text.strip()

df['body'] = df['body'].apply(clean_text)
df.to_csv('/Users/xeniamarlenezerweck/Documents/Verzeichnis/Master/Thesis/Daten/full_comments_cleaned.csv')

# ==================== Split into Sentence Level ====================
# nltk.download('punkt')

def explode_sentences(df):
    df['sentences'] = df['body'].apply(sent_tokenize)
    exploded_df = df.explode('sentences').reset_index(drop=True)
    return exploded_df

sentence_based_df = explode_sentences(df)

# lenght before dropping duplicates:
for repository in repositories:
    full_entries_count = len(sentence_based_df[sentence_based_df['repo'] == repository])
    print(f'Anzahl satzbasierter Beitraege {repository}: {full_entries_count}')
    
'''
Anzahl satzbasierter Beitraege tensorflow: 1307
Anzahl satzbasierter Beitraege vscode: 1227
Anzahl satzbasierter Beitraege pytorch: 1319

Anzahl Datenpunkte insgesamt (len(sentence_based_df)) = 3853
'''

# drop duplicate sentences
sentence_based_df = sentence_based_df.drop_duplicates(subset='sentences')

for repository in repositories:
    full_entries_count = len(sentence_based_df[sentence_based_df['repo'] == repository])
    print(f'Anzahl satzbasierter Beitraege {repository}: {full_entries_count}')
    
'''
Anzahl satzbasierter Beitraege tensorflow: 1115
Anzahl satzbasierter Beitraege vscode: 1166
Anzahl satzbasierter Beitraege pytorch: 1216

Anzahl Datenpunkte insgesamt (len(sentence_based_df)) = 3497
'''

sentence_based_df.to_csv('/Users/xeniamarlenezerweck/Documents/Verzeichnis/Master/Thesis/Daten/sentences_cleaned.csv', )
