from bs4 import BeautifulSoup
from bs4 import SoupStrainer
from requests import get

with open('data/tutorialUrls.txt') as tutorialsFile:
    urls = tutorialsFile.readlines()

    with open('dataset.txt', 'w') as dataset:
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
