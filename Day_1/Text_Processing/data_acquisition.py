import requests
import time
import pandas as pd
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
TMDB_API_KEY = "82b791a2fb1d24a8cd3cedd5bc8a0344"
BASE_URL = "https://api.themoviedb.org/3"
session = requests.Session()

# 2. Add a standard User-Agent header so you don't look like a bot
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept': 'application/json'
})

# 3. Configure automatic retries for dropped connections
retries = Retry(total=5, backoff_factor=1, status_forcelist=[ 500, 502, 503, 504 ])
session.mount('https://', HTTPAdapter(max_retries=retries))

def search_movie(query, page=1):
    endpoint = f"{BASE_URL}/search/movie"
    params = {
        'api_key': TMDB_API_KEY,
        'query': query,
        'page': page,
        'language': 'en-US'
    }
    # Use session.get instead of requests.get
    response = session.get(endpoint, params=params)
    response.raise_for_status() # Raises an error for bad HTTP status codes
    return response.json()

def get_movie_details(movie_id):
    endpoint = f"{BASE_URL}/movie/{movie_id}"
    params = {
        'api_key': TMDB_API_KEY,
        'append_to_response': 'keywords,reviews' 
    }
    response = session.get(endpoint, params=params)
    response.raise_for_status()
    return response.json()

# Fetch data
movies_data = []

try:
    # Loop through pages 1 to 5 (20 movies per page = 100 total)
    for page_num in range(1, 11):
        print(f"Fetching page {page_num}...")
        
        search_results = search_movie("action", page=page_num)
        
        # Stop if we run out of results before page 5
        if not search_results.get('results'):
            break
            
        # Loop through ALL movies on the current page
        for movie in search_results['results']:
            movie_id = movie['id']
            details = get_movie_details(movie_id)
            
            movies_data.append({
                'title': details.get('title'),
                'overview': details.get('overview'),
                'genres': [g['name'] for g in details.get('genres', [])],
                'release_date': details.get('release_date'),
                'vote_average': details.get('vote_average'),
                'popularity': details.get('popularity')
            })
            
            # Keep the sleep here to respect TMDB's rate limit
            time.sleep(0.3)

    # Create DataFrame and save
    df = pd.DataFrame(movies_data)
    csv_filename = "tmdb_action_movies_100.csv"
    df.to_csv(csv_filename, index=False, encoding='utf-8')
    
    print(f"\nSuccess! Saved {len(df)} rows to {csv_filename}")

except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")

# Save the DataFrame to a CSV file
csv_filename = "tmdb_action_movies.csv"

# index=False prevents Pandas from writing the row numbers as the first column
df.to_csv(csv_filename, index=False, encoding='utf-8')

print(f"\nSuccess! Data successfully saved to {csv_filename}")