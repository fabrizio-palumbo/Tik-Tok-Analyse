import os
import json
import pandas as pd
from moviepy.editor import VideoFileClip
import whisper
import torch
from transformers import pipeline
from concurrent.futures import ThreadPoolExecutor, as_completed
import glob
from multiprocessing import Pool




# Define the base project directory
project_root = 'project_root'

# Function to load processed videos
def load_processed_videos(output_dir):
    processed_videos = set()
    for file in glob.glob(f"{output_dir}/audio_transcription_output_*.csv"):
        interim_data = pd.read_csv(file)
        processed_videos.update(interim_data['video_id'].unique())
    return processed_videos
# Load the Whisper model
#model = whisper.load_model("medium").to(torch.float32)

def extract_audio_from_video(video_path, output_dir):
    audio_filename = f"{os.path.splitext(os.path.basename(video_path))[0]}.mp3"
    audio_path = os.path.join(output_dir, audio_filename)
    if os.path.exists(audio_path):
        print(f"Audio file {audio_filename} already exists.")
        return audio_path
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path, codec='mp3')
    return audio_path

def transcribe_audio(audio_path, model_name="large"):
    # Load the model inside the function to ensure thread-safe operation
    model = whisper.load_model(model_name).to(torch.float32)
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio,n_mels=128).to(dtype=torch.float32)
    options = whisper.DecodingOptions(fp16=False, language='Norwegian')#, )#,lower_quantile=0.05, lower_threshold=0.1)
    #options.suppress_tokens = ""  # Equivalent to --suppress_tokens ""
    result = whisper.decode(model, mel, options)
    #result = model.transcribe(mel,suppress_tokens = [],fp16=False,condition_on_previous_text=False)
    return result.text


def load_video_details(json_filepath):
    with open(json_filepath, 'r') as file:
        json_data = json.load(file)
    video_details = {}
    for user_data in json_data:
        creator_username=user_data["username"]
        for video in user_data['videoIds']:
            video_id = video['video_data']['id']
            create_time = video['video_data'].get('create_time', 'Unknown')
            video_details[video_id] = {
                "video_data": video,
                "created_time": create_time,
                "creator_username":creator_username,
            }
    return video_details

def process_video(video, video_dir, audio_output_dir, video_details):
    try:
        print(f"Processing video: {video}")
        video_id = os.path.splitext(video)[0]
        video_path = os.path.join(video_dir, video)
        audio_path = extract_audio_from_video(video_path, audio_output_dir)
        
        # Load model inside this function to ensure it's loaded per thread/process
        #model = whisper.load_model(model_name).to(torch.float32)

        transcription = transcribe_audio(audio_path, model_name="large")
        creator_username=video_details.get(video_id, {}).get("creator_username", 'Unknown')

        create_time = video_details.get(video_id, {}).get('created_time', 'Unknown')
        return {
            "video_id": video_id,
            "created_time": create_time,
            "transcription": transcription,
            "creator_username":creator_username,

        }
    except Exception as e:
        print(f"Error processing video {video}: {str(e)}")
        return None


def analyze_sentiment(output_data, model_name="alexandrainst/scandi-nli-large"):#"facebook/bart-large-mnli"):#
    classifier = pipeline("zero-shot-classification", model=model_name)
    emotion_labels = ['glad','trist','overrasket', 'nÃ¸ytral', 'redd', 'avsky', 'sint']
    #emotion_labels = ['happy', 'sad', 'surprise', 'neutral', 'fear', 'disgust', 'angry']
    #hypothesis_template = "Denne meldingen fÃ¸lelse {}."
    hypothesis_template = "Denne snakken er {}."
    video_ids = []
    create_times = []
    transcriptions = []
    sentiments = []#pd.DataFrame(columns=emotion_labels)
    creators_usernames=[]
    for data in output_data:
        video_id = data['video_id']
        create_time = pd.to_datetime(data['created_time'], unit='s',errors='coerce')#data['create_time']
        transcription = str(data['transcription'])
        creator_username= data['creator_username']
        video_ids.append(video_id)
        create_times.append(create_time)
        transcriptions.append(transcription)
        creators_usernames.append(creator_username)
        result=[]
        result = classifier(transcription, emotion_labels,hypothesis_template=hypothesis_template, multi_label=False)
        sentiment_scores = {label: score for label, score in zip(result['labels'], result['scores'])}
        sentiments.append(sentiment_scores)
    #print(sentiments)
    result_df = pd.DataFrame({
        # 'video_id': video_ids,
        # "creator_username":creator_username,
        # 'created_time': create_times,
        'transcription': transcriptions,
        **pd.DataFrame(sentiments)
    })
    return result_df

# Main execution
if __name__ == "__main__":
    # Set up paths and directories
    #user_names=["eivindtr","emiliepaus","emmathevampireslayer","fetishawilliams"]#["camillalor","davidmokel16","delshad01"]
    #user_names=["henki_viken","henrikschatvet","jacob_karlsen","jenny.huse"]#["eivindtr","emiliepaus","emmathevampireslayer","fetishawilliams"]#["camillalor","davidmokel16","delshad01"]
    #user_names=["jennygehrken","johannemasseyy"]
    
    folder_path = "/Users/fabrizio/Documents/GitHub/video_sentiment_analysis/pipeline/project_root/data"

    # Get a list of all items in the folder that are not the 'done' folder
    folder_contents = [item for item in os.listdir(folder_path) 
                    if os.path.isdir(os.path.join(folder_path, item)) 
                    and item != 'done']

    # Select any folder name from the list, excluding 'done'
    user_names = folder_contents if folder_contents else None
    #user_names =["kontoretpodcast",""]
    for user in user_names:
        json_path = os.path.join(os.path.dirname(__file__), '..', 'data', user,user+'.json')
        video_dir = os.path.join(os.path.dirname(__file__), '..', 'data',user ,'videos')
        output_dir = os.path.join(os.path.dirname(__file__), '..','data',user, 'output')

        # Check if the folder exists
        if not os.path.exists(output_dir):
            # Create the folder
            os.makedirs(output_dir)
            print(f"Folder '{output_dir}' created.")
        else:
            print(f"Folder '{output_dir}' already exists.")
        
        
        # Load video details
        video_details = load_video_details(json_path)
        processed_videos = load_processed_videos(output_dir)

        # Ensure the output directories exist
        audio_output_dir = os.path.join(output_dir, "audio_files")
        os.makedirs(audio_output_dir, exist_ok=True)

        # List video files
        videos = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi', '.mkv'))]

        output_data = []
        counter = 0

        # Use multiprocessing.Pool instead of ThreadPoolExecutor
        with Pool(processes=4) as pool:
            results = []
            for video in videos:
                video_id = os.path.splitext(video)[0]
                if video_id not in processed_videos:
                    result = pool.apply_async(process_video, (video, video_dir, audio_output_dir, video_details))
                    results.append(result)

            for result in results:
                output = result.get()
                if output:
                    output_data.append(output)
                    counter += 1

                    # Save interim data every 100 iterations
                    if counter % 100 == 0:
                        interim_csv_path = os.path.join(output_dir, f"audio_transcription_output_{counter}.csv")
                        pd.DataFrame(output_data).to_csv(interim_csv_path, index=False)
                        print(f"Saved interim data at iteration {counter}")

        # Save transcriptions to CSV
        transcription_output_csv = os.path.join(output_dir, "audio_transcription_output.csv")
        pd.DataFrame(output_data).to_csv(transcription_output_csv, index=False)
        print(f"Transcription completed. Data saved to {transcription_output_csv}.")

        # Step 2: Sentiment analysis
        output_data = pd.read_csv(transcription_output_csv).to_dict('records')
        result_df = analyze_sentiment(output_data)
        sentiment_result_csv = os.path.join(output_dir, "audio_text_emotion_result.csv")
        result_df.to_csv(sentiment_result_csv, index=False)
        print(f"Sentiment analysis completed. Results saved to {sentiment_result_csv}.")