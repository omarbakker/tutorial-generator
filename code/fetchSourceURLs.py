from bs4 import BeautifulSoup
from bs4 import SoupStrainer
from requests import get

print('Fetching tutorials')
temp = 'https://www.raywenderlich.com/category/'
categories = ['ios','macos','android','swift','apple-game-frameworks','unity']
topURLs = [temp + category for category in categories]
tutorialUrls = []

def representsInt(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def urlIsTutorialUrl(url):
    url = url.replace('https://','').split('/')
    return len(url) == 3 \
            and representsInt(url[1]) \
            and url[0] == 'www.raywenderlich.com'


for url in topURLs:
    result = get(url=url)
    content = BeautifulSoup(result.content,
                            "html.parser",
                            parse_only=SoupStrainer('a'))
    for atag in content:
        if not atag.has_attr('href'): continue
        url = atag['href']
        if not urlIsTutorialUrl(url): continue
        tutorialUrls.append(atag['href'])

with open('data/tutorialURLs.txt', 'w') as urlFile:
    urlFile.write('\n'.join(tutorialUrls))

print('Found ', len(tutorialUrls), ' tutorials')
print('Output in data/tutorialURLs.txt')
