import pandas as pd
from bs4 import BeautifulSoup as bs
import requests
import re
from pynytimes import NYTAPI
from datetime import datetime
from tqdm import tqdm
from nyt_categories import categories
from nyt_apikey import key

nyt = NYTAPI(key)

data = pd.read_csv('data/nyt/nytimes.csv', encoding='unicode_escape')

headers = {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'GET',
    'Access-Control-Allow-Headers': 'Content-Type',
    'Access-Control-Max-Age': '3600',
    'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0'
}

def get_article_url(title, date):
    '''
    Given a title and date, return the url of the article
    '''
    pd_date = pd.to_datetime(date)
    date = datetime(pd_date.year, pd_date.month, pd_date.day)
    articles = nyt.article_search(
        query = title,
        results = 1,
        dates = {
            "begin": date,
            "end": date,
        },
        options = {
            "sort": "relevance",
            'title': title,
            "sources": [
                "New York Times",
            ],
        }
    )
    return articles[0]['web_url']

def get_soup(url):
    '''
    Return a beautiful soup object from a url
    '''
    req = requests.get(url, headers, timeout=5)
    soup = bs(req.content, 'html.parser')
    return soup

def get_title(soup, n=3):
    '''
    Return the title of the article from a beautiful soup object
    '''
    title = soup.find('h1', {'data-testid': 'headline'})
    return title.text

def get_paragraphs(soup, n=3):
    '''
    Return the first n paragraphs from a beautiful soup object for a nyt article
    '''
    paragraph_tag = 'css-axufdj evys1bk0'
    paragraphs = soup.find_all('p', {'class': paragraph_tag})
    paragraphs = paragraphs[:n]
    paragraphs = [p.text for p in paragraphs]
    text = '\n\n'.join(paragraphs)
    return text


def scrape_all():
    '''
    Scrape all the articles in the dataframe
    '''
    i = 0
    bodies = []
    save_every = 100
    for title, date in zip(data.title, data.date):
    # for title, date in tqdm(zip(data.title, data.date)):
        try:
            # url = google_article(title)
            url = get_article_url(title, date)
            soup = get_soup(url)
            if title != get_title(soup):
                raise Exception("Title doesn't match")
            body = get_paragraphs(soup)
            bodies.append((title, body))
            # print(title)
            # print(get_title(soup))
            # print(body)
            # print('\n\n')
        except Exception as e:
            bodies.append((title, ''))
            # print(title)
            # print(url)
            # print(e)
            pass
        i += 1
        if i % save_every == 0:
            df = pd.DataFrame(bodies, columns=['title', 'body'])
            df.to_pickle('data/nyt/bodies.pkl')
            print(i)
            pass

def scrape_n(n=30):
    '''
    Scrape only n random articles for each category
    '''
    i = 0
    bodies = []
    save_every = 10
    categories = data.topic_2digit.unique()
    for category in tqdm(categories):
        category_saved = 0
        # filter to only articles in category
        data_category = data[data.topic_2digit == category]
        # shuffle
        data_category = data_category.sample(frac=1, random_state=42)
        # iterate through articles
        for title, date in zip(data_category.title, data_category.date):
            try:
                # url = google_article(title)
                url = get_article_url(title, date)
                soup = get_soup(url)
                if title != get_title(soup):
                    raise Exception("Title doesn't match")
                body = get_paragraphs(soup)
                bodies.append((title, body))
                category_saved += 1
                i += 1
            except Exception as e:
                pass
            if i % save_every == 0:
                df = pd.DataFrame(bodies, columns=['title', 'body'])
                df.to_pickle('data/nyt/bodies-small.pkl')
                print(i)
                pass
            # break if we've scraped enough in category
            if category_saved == n:
                break
    df = pd.DataFrame(bodies, columns=['title', 'body'])
    df.to_pickle('data/nyt/bodies-small.pkl')
            

if __name__ == '__main__':
    # scrape n articles
    scrape_n()
    # scrape all articles
    # scrape_all()