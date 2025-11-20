import requests
from bs4 import BeautifulSoup
import pandas as pd

def fetch_immoscout_data(search_url):
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(search_url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')

    listings = []
    for item in soup.find_all('div', {'class':'result-list__listing'}):
        try:
            living_space = item.find('span', {'class':'living-space'}).text
            rooms = item.find('span', {'class':'rooms'}).text
            rent = item.find('span', {'class':'total-rent'}).text
            # Extract other features similarly
            listings.append({
                'livingSpace': float(living_space.replace('m²','')),
                'noRooms': float(rooms),
                'totalRent': float(rent.replace('€','').replace('.',''))
            })
        except:
            continue

    df = pd.DataFrame(listings)
    return df

# Example URL
url = "https://www.immobilienscout24.de/Suche/S-2/Wohnung-Miete/Bayern/Muenchen"
df = fetch_immoscout_data(url)
print(df.head())
