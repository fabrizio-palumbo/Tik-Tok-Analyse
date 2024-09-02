import os
import json
import pandas as pd
from transformers import pipeline
import numpy as np
import emoji
def load_video_details(json_path):
    with open(json_path, 'r') as file:
        all_user_data = json.load(file)
    return all_user_data

def analyze_video_titles(all_user_data, classifier, emotion_labels, hypothesis_template):
    video_texts = []

    for user_data in all_user_data:
        creator_username=user_data["username"]

        for video in user_data['videoIds']:
            video_id = video['video_data']['id']
            created_time = pd.to_datetime(video['video_data']['create_time'], unit='s') 
            #text_title = emoji.demojize(video['video_data']['title'].replace('#', ''))
            if video['video_data']['title'] is not None:
                text_title = emoji.demojize(video['video_data']['title'].replace('#', ''))
                raw_title=video['video_data']['title']
            else:
                text_title = ""  # or any other default value or handling you prefer
                raw_title=""
            video_texts.append({'video_id': video_id,'creator_username':creator_username, 'created_time': created_time, 'title': text_title, "unprocessed_title":raw_title})
    
    # Analyze emotions in the video titles
    for item in video_texts:
        if len(item["title"]) > 0:
            result = classifier(item['title'], emotion_labels, hypothesis_template=hypothesis_template, multi_label=False)
            for score, label in zip(result['scores'], result['labels']):
                item[label] = score
        else:
            for label in emotion_labels:
                item[label] = np.nan
    return video_texts

def save_results(video_texts, output_path):
    df = pd.DataFrame(video_texts)
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    # Paths relative to the script's location
    #json_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'enorwich.json')
    #output_path = os.path.join(os.path.dirname(__file__), '..', 'output', 'title_emotion_results.csv')
    #user_names=["adibwaezz","anny.isabella","barisbrevik3","brilleslangen2","brilleslangen2","ceciliateigen","cristianbrennhovd","danbychoi","davidmokel16","delshad01","amalieolsengjerse"]
    #user_names=["anny.isabella","adibwaezz","brilleslangen2"]
    
    folder_path = "/Users/fabrizio/Documents/GitHub/video_sentiment_analysis/pipeline/project_root/data/done"

    # Get a list of all items in the folder that are not the 'done' folder
    folder_contents = [item for item in os.listdir(folder_path) 
                    if os.path.isdir(os.path.join(folder_path, item)) 
                    and item != 'summary_output']

    # Select any folder name from the list, excluding 'done'
    user_names = folder_contents if folder_contents else None
    
    
    for user in user_names:
        #user="oliverbergset"#"linnealotvedt"#"mankegard"#"danian92"#"oskarwesterlin"
        json_path = os.path.join(os.path.dirname(__file__), '..', 'data/done', user,user+'.json')
        #root_dir = os.path.join(os.path.dirname(__file__), '..', 'data',user ,'videos')
        output_path = os.path.join(os.path.dirname(__file__), '..','data/done',user, 'output', 'title_emotion_results.csv')
        if (os.path.exists(output_path)):
            #print("user already processed")
        #else:

            # Load video details
            all_user_data = load_video_details(json_path)

            # Initialize the zero-shot classification pipeline
            classifier = pipeline("zero-shot-classification", model="alexandrainst/scandi-nli-large")
            emotion_labels = ['glad','trist','overrasket', 'nÃ¸ytral', 'redd', 'avsky', 'sint']
            #emotion_labels = ['happy', 'sad', 'surprise', 'neutral', 'fear', 'disgust', 'angry']
            #hypothesis_template = "Denne meldingen fÃ¸lelse {}."
            hypothesis_template = "Denne snakken er {}."
        
            # Perform zero-shot classification to determine emotions in titles
            video_texts = analyze_video_titles(all_user_data, classifier, emotion_labels, hypothesis_template)

            # Save the results to a CSV file
            save_results(video_texts, output_path)

            print(f"Emotion analysis complete. Results saved to {output_path}.")
