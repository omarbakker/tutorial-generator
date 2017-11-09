from bs4 import BeautifulSoup
from bs4 import SoupStrainer
from requests import get
from random import shuffle

print('Fetching tutorial urls')
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

shuffle(tutorialUrls)
with open('data/tutorialURLs.txt', 'w') as urlFile:
    urlFile.write('\n'.join(tutorialUrls))

print('Found ', len(tutorialUrls), ' tutorials')
print('Output in data/tutorialURLs.txt')

with open('data/tutorialUrls.txt') as tutorialsFile:
    urls = tutorialsFile.readlines()

    with open('data/dataset.txt', 'w') as dataset:
        i = 0
        for url in urls:
            result = get(url)

            if result.status_code != 200:
                print('failed to fetch ', url)

            content = BeautifulSoup(result.content,
                                    "html.parser",
                                    parse_only=SoupStrainer('article'))
            content = content.find_all(['h2','p','pre'])

            for item in content:
                dataset.write(str(item) + '\n')
            i += 1

            if i % 5 == 0 or i == len(urls):
                print('Finished adding ', i, ' tutorials to dataset')
    print("Finished creating dataset, output in data/dataset.txt")
