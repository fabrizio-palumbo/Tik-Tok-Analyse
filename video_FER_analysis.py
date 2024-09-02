import glob
import json
import pandas as pd
from fer import Video
from fer import FER
import os
import concurrent.futures
import tensorflow as tf

# Function to list all .mp4 files
def list_mp4_files(root_dir, pattern="**/*.mp4"):
    return [f for f in glob.glob(f"{root_dir}/{pattern}", recursive=True)]

# Function to load video details from JSON
def load_video_details(json_path):
    with open(json_path, 'r') as file:
        all_user_data = json.load(file)
        username = all_user_data[0]["username"]
    video_details = {}
    for user_data in all_user_data:
        for video in user_data['videoIds']:
            video_id = video['video_data']['id']
            video_details[video_id] = video
    return video_details, username

# Function to get video ID from file path
def get_video_id_from_path(path):
    return os.path.basename(path).replace('.mp4', '')

# Function to analyze videos and save emotions
def analyze_video(video_path, video_details, username):
    try:
        video_id = get_video_id_from_path(video_path)
        print(video_id)
        video_obj = Video(video_path)
        detector = FER(mtcnn=True)
        raw_data = video_obj.analyze(detector, display=False, frequency=5, save_frames=False, save_video=False)
        df = video_obj.to_pandas(raw_data)
        video_id = get_video_id_from_path(video_path)
        details = video_details.get(video_id, {})
        created_time = details.get('video_data', {}).get('create_time', 'Unknown')

        df['video_id'] = video_id
        df['created_time'] = created_time
        df['creator_username'] = username

        return df[['video_id', 'creator_username', 'created_time', 'happy', 'sad', 'surprise', 'neutral', 'fear', 'disgust', 'angry']]
    except Exception as e:
        print(f"Error processing video {video_path}. Error: {e}")
        return pd.DataFrame()

# Function to load processed videos
def load_processed_videos(output_dir):
    processed_videos = set()
    for file in glob.glob(f"{output_dir}/extracted_FER_emotions_*.csv"):
        interim_data = pd.read_csv(file)
        processed_videos.update(interim_data['video_id'].unique())
    return processed_videos

import multiprocessing

def analyze_videos_multiprocessed(root_dir, video_details, username, output_dir, max_workers=3):
    mylist = list_mp4_files(root_dir)
    all_emotions = []
    counter = 0
    processed_videos = load_processed_videos(output_dir)

    with multiprocessing.Pool(max_workers) as pool:
        results = [pool.apply_async(analyze_video, (video_path, video_details, username))
                   for video_path in mylist if get_video_id_from_path(video_path) not in processed_videos]

        for result in results:
            try:
                emotions = result.get()
                if not emotions.empty:
                    all_emotions.append(emotions)
                    counter += 1

                    if counter % 100 == 0:
                        interim_csv_path = os.path.join(output_dir, f"extracted_FER_emotions_{counter}.csv")
                        pd.concat(all_emotions, ignore_index=True).to_csv(interim_csv_path, index=False)
                        print(f"Saved interim data at iteration {counter}")
            except Exception as e:
                print(f"Error processing video. Error: {e}")

    # Saving the final emotions data to CSV
    if all_emotions:
        final_csv_path = os.path.join(output_dir, "extracted_FER_emotions.csv")
        pd.concat(all_emotions, ignore_index=True).to_csv(final_csv_path, index=False)
        print("Saved final data")

# Updated function to analyze videos using multithreading and save every 100 iterations
def analyze_videos_multithreaded(root_dir, video_details, username, output_dir, max_workers=2):
    mylist = list_mp4_files(root_dir)
    all_emotions = []
    counter = 0
    processed_videos = load_processed_videos(output_dir)

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        future_to_video = {}
        for video_path in mylist:
            video_id = get_video_id_from_path(video_path)
            if video_id not in processed_videos:
                future = executor.submit(analyze_video, video_path, video_details, username)
                future_to_video[future] = video_path

        for future in concurrent.futures.as_completed(future_to_video):
            video_path = future_to_video[future]
            try:
                emotions = future.result()
                if not emotions.empty:
                    all_emotions.append(emotions)
                    counter += 1

                    if counter % 100 == 0:
                        interim_csv_path = os.path.join(output_dir, f"extracted_FER_emotions_{counter}.csv")
                        pd.concat(all_emotions, ignore_index=True).to_csv(interim_csv_path, index=False)
                        print(f"Saved interim data at iteration {counter}")

            except Exception as e:
                print(f"Error processing video {video_path}. Error: {e}")

    # Saving the final emotions data to CSV
    if all_emotions:
        final_csv_path = os.path.join(output_dir, "extracted_FER_emotions.csv")
        pd.concat(all_emotions, ignore_index=True).to_csv(final_csv_path, index=False)
        print("Saved final data")

# Main execution
if __name__ == "__main__":
    #user_names=["henki_viken","henrikschatvet","jacob_karlsen","jenny.huse"]#["eivindtr","emiliepaus","emmathevampireslayer","fetishawilliams"]#["camillalor","davidmokel16","delshad01"]
#user_names=["jennygehrken","johannemasseyy""barisbrevik3","brilleslangen2","brilleslangen2","ceciliateigen","cristianbrennhovd","danbychoi","davidmokel16","delshad01","amalieolsengjerse"]
    
    
    folder_path = "/Users/fabrizio/Documents/GitHub/video_sentiment_analysis/pipeline/project_root/data"

    # Get a list of all items in the folder that are not the 'done' folder
    folder_contents = [item for item in os.listdir(folder_path) 
                    if os.path.isdir(os.path.join(folder_path, item)) 
                    and item != 'done']

    # Select any folder name from the list, excluding 'done'
    user_names = folder_contents if folder_contents else None
    for user in user_names:# = "oliverbergset"#"muskelbunt1"#"linnealotvedt"#"lassesaelevik2"#danian92"
        json_path = os.path.join(os.path.dirname(__file__), '..', 'data', user, user + '.json')
        root_dir = os.path.join(os.path.dirname(__file__), '..', 'data', user, 'videos')
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'data', user, 'output')

        # Check if the folder exists
        if not os.path.exists(output_dir):
            # Create the folder
            os.makedirs(output_dir)
            print(f"Folder '{output_dir}' created.")
        else:
            print(f"Folder '{output_dir}' already exists.")

        video_details, username = load_video_details(json_path)
        analyze_videos_multiprocessed(root_dir, video_details, username, output_dir)
        #analyze_videos_multithreaded