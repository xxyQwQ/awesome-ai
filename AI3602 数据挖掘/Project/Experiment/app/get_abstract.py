import requests
from bs4 import BeautifulSoup

def search_bing(query, key_path='.key'):
    with open(key_path) as k:
        api_key = k.readline().strip()
    endpoint = 'https://api.bing.microsoft.com/v7.0/search'
    headers = {'Ocp-Apim-Subscription-Key': api_key}
    params = {'q': query, 'count': 10}

    response = requests.get(endpoint, headers=headers, params=params)
    response.raise_for_status()
    return response.json()

def find_arxiv_link(search_results):
    for result in search_results.get('webPages', {}).get('value', []):
        url = result.get('url')
        if 'arxiv.org' in url:
            title = result.get('name')
            return url, title
    return None, None

def get_arxiv_abstract(arxiv_url):
    response = requests.get(arxiv_url)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, 'html.parser')
    abstract = soup.find('blockquote', class_='abstract')
    if abstract:
        return abstract.text.replace('Abstract:', '').strip()
    return None

def title_to_abs(title=None):
    if title is None:
        title = input("Enter the article title: ")
    search_results = search_bing(title+' arxiv')
    arxiv_url, paper_title = find_arxiv_link(search_results)
    if arxiv_url:
        abstract = get_arxiv_abstract(arxiv_url)
        if abstract:
            print(f"\nPaper Title: {paper_title}\nURL: {arxiv_url}\nAbstract: \n{abstract}")
        else:
            print("Abstract not found.")
    else:
        print("arXiv link not found.")
    return (paper_title, abstract) if arxiv_url else (None, None)
    

if __name__ == "__main__":
    title_to_abs()