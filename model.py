import requests
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Load sentence transformer model for semantic similarity
model = SentenceTransformer('all-MiniLM-L6-v2')

# Your YouTube Data API key (remember: keep it secret!)
YOUTUBE_API_KEY = 'AIzaSyDx4EV58jEh_j1TQGHJuA5kTrCRqzlkKUQ'

# Fetch videos from YouTube based on a query
def fetch_youtube_videos(query, max_results=25):
    url = 'https://www.googleapis.com/youtube/v3/search'
    params = {
        'part': 'snippet',
        'q': query,
        'type': 'video',
        'maxResults': max_results,
        'key': YOUTUBE_API_KEY
    }
    res = requests.get(url, params=params)
    data = res.json()

    videos = []
    for item in data.get('items', []):
        video_id = item['id']['videoId']
        title = item['snippet']['title']
        description = item['snippet']['description']
        channel = item['snippet']['channelTitle']
        videos.append({
            'videoId': video_id,
            'title': title,
            'description': description,
            'channel': channel
        })

    return pd.DataFrame(videos)

# Rank videos using sentence transformer similarity
def rank_videos(df, query):
    query_embedding = model.encode(query, convert_to_tensor=True)
    df['combined'] = df['title'] + ' ' + df['description']
    df['embedding'] = df['combined'].apply(lambda x: model.encode(x, convert_to_tensor=True))
    df['similarity'] = df['embedding'].apply(lambda emb: float(util.pytorch_cos_sim(query_embedding, emb)))
    ranked = df.sort_values(by='similarity', ascending=False).head(5)
    return ranked[['title', 'channel', 'videoId', 'similarity']]

# Recommend best video for a chapter/topic
def recommend_video(chapter):
    df = fetch_youtube_videos(chapter)
    
    if df.empty:
        return {
            "title": "No videos found",
            "url": "",
            "channel": ""
        }

    ranked_df = rank_videos(df, chapter)
    top = ranked_df.iloc[0]

    return {
        "title": top['title'],
        "url": f"https://www.youtube.com/watch?v={top['videoId']}",
        "channel": top['channel']
    }

# Run this file directly to test
if __name__ == "__main__":
    user_input = input("Enter your topic/chapter: ")
    video = recommend_video(user_input)
    print(f"\n🎓 Recommended Video:\nTitle: {video['title']}\nChannel: {video['channel']}\nURL: {video['url']}")



