# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 11:38:08 2024

@author: olahuser
"""

import audio_text_analysis as ata
import os
import pandas as pd
from pathlib import Path
from time import time
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

#%%

def transcribe_audio_files(audio_folder, model_name="NbAiLabBeta/nb-whisper-large"):
    """
    Transcribe all audio files in the specified folder.

    Parameters:
    - audio_folder (str): The path to the folder containing audio files.
    - model_name (str): The model name to use for transcription.

    Returns:
    - dict: A dictionary with audio file names as keys and transcriptions as values.
    """
    pathlist = Path(audio_folder).rglob('*.mp3')
    transcription_dict = {}

    for file in tqdm(pathlist):
        t = time()
        transcription = ata.transcribe_audio(file, model_name=model_name)
        transcription_dict[file.name] = transcription
        print(f"Transcribed {file.name} in {time() - t:.2f} seconds")

    return transcription_dict

#%%

def analyze_sentiments(transcriptions, emotion_labels):
    """
    Analyze sentiments of transcriptions using a zero-shot classification model.

    Parameters:
    - transcriptions (list): List of transcriptions to analyze.
    - emotion_labels (list): List of possible emotion labels.

    Returns:
    - list: A list of dictionaries with sentiment scores for each transcription.
    """
    classifier = pipeline("zero-shot-classification", model="alexandrainst/scandi-nli-large")
    hypothesis_template = "Denne snakken er {}."
    sentiments = []

    for transcription in tqdm(transcriptions, desc='Analyzing sentiments'):
        t = time()
        result = classifier(transcription, emotion_labels, hypothesis_template=hypothesis_template, multi_label=False)
        sentiment_scores = {label: score for label, score in zip(result['labels'], result['scores'])}
        sentiments.append(sentiment_scores)
        
    print(f"Sentiment analysis completed in {time() - t:.2f} seconds: {sentiment_scores}")
    return sentiments

#%%

def label_topics(transcriptions, candidate_labels):
    """
    Label topics of transcriptions using a zero-shot classification model.

    Parameters:
    - transcriptions (list): List of transcriptions to label.
    - candidate_labels (list): List of possible topic labels.

    Returns:
    - list: A list of dictionaries with label scores for each transcription.
    """
    classifier = pipeline("zero-shot-classification", model='NbAiLab/nb-bert-base-mnli')
    hypothesis_template = "Dette eksempelet handler om {}."
    labels = []

    for transcription in tqdm(transcriptions, desc='Topic labelling'):
        t = time()
        result = classifier(transcription, candidate_labels, hypothesis_template=hypothesis_template, multi_label=True, model_max_length=None)
        label_scores = {label: score for label, score in zip(result['labels'], result['scores'])}
        labels.append(label_scores)
        
    print(f"Topic labeling completed in {time() - t:.2f} seconds: {label_scores}")
    return labels

#%%

def perform_ner(transcriptions, model_name='NbAiLab/nb-bert-base'):
    """
    Perform Named Entity Recognition (NER) on transcriptions.

    Parameters:
    - transcriptions (list): List of transcriptions to analyze.
    - model_name (str): The model name to use for NER.

    Returns:
    - list: A list of NER results for each transcription.
    """

    
    tokenizer = AutoTokenizer.from_pretrained("NbAiLab/nb-bert-base-ner", model_max_length=None)
    model = AutoModelForTokenClassification.from_pretrained("NbAiLab/nb-bert-base-ner")

    classifier_NER = pipeline("ner", model=model, tokenizer=tokenizer)

    ners = []
    entities_mapping = {}
    for i, transcription in tqdm(enumerate(transcriptions), desc='Estimating NER'):
        t = time()
        result = classifier_NER(transcription)
        ners.append(result)
        names, locations, geo_locations, geo_orgs, orgs,events, dvts, miscelaneous, products = merge_and_extract_entities(result)
        entities_mapping[i] = {'Names': names, 'Locations': locations, 'Geo Locations':geo_locations, 'Geo organizzation':geo_orgs, 'Organizzation':orgs, 'Events':events, 'Dvts':dvts, 'Miscelaneous':miscelaneous, 'Products':products}

    print(f"NER completed in {time() - t:.2f} seconds")
    return entities_mapping

def save_to_csv(data, output_dir, filename):
    """
    Save data to a CSV file.

    Parameters:
    - data (DataFrame): The data to save.
    - output_dir (str): The directory to save the file in.
    - filename (str): The name of the output CSV file.
    """
    output_path = os.path.join(output_dir, filename)
    data.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}.")

#%%

def merge_and_extract_entities(ner_results):
    names = []
    locations = []
    geo_locations = []
    geo_orgs = []
    orgs = []
    events = []
    current_entity = []
    products=[]
    current_entity_type = None
    dvts=[]
    miscelaneous=[]
    for token in ner_results:
        word = token['word']
        entity_type = token['entity'][2:]  # Extract entity type (e.g., 'PER' from 'B-PER')

        # Check if the token is a continuation of a word and merge it with the previous token
        if word.startswith("##"):
            word = word[2:]  # Remove the '##'
            if current_entity:
                current_entity[-1] += word  # Merge with the previous token
            continue

        if token['entity'].startswith('B-'):
            if current_entity:
                # Append the previous entity to the appropriate list
                if current_entity_type == 'PER':
                    names.append(current_entity)
                elif current_entity_type == 'LOC':
                    locations.append(current_entity)
                elif current_entity_type == 'GPE_LOC':
                    geo_locations.append(current_entity)
                elif current_entity_type == 'GPE_ORG':
                    geo_orgs.append(current_entity)
                elif current_entity_type == 'ORG':
                    orgs.append(current_entity)
                elif current_entity_type == 'EVT':
                    events.append(current_entity)
                elif current_entity_type == 'DVT':
                    dvts.append(current_entity)
                elif current_entity_type == 'MISC':
                    miscelaneous.append(current_entity)
                elif current_entity_type == 'PROD':
                    products.append(current_entity)
                # Reset for the new entity
                current_entity = []

            current_entity.append(word)
            current_entity_type = entity_type

        elif token['entity'].startswith('I-') and current_entity_type == entity_type:
            if current_entity:  # Ensure it's a continuation of the same entity type
                current_entity.append(word)

    # Add the last entity if the sentence ends with an entity
    if current_entity:
        if current_entity_type == 'PER':
            names.append(current_entity)
        elif current_entity_type == 'LOC':
            locations.append(current_entity)
        elif current_entity_type == 'GPE_LOC':
            geo_locations.append(current_entity)
        elif current_entity_type == 'GPE_ORG':
            geo_orgs.append(current_entity)
        elif current_entity_type == 'ORG':
            orgs.append(current_entity)
        elif current_entity_type == 'EVT':
            events.append(current_entity)
        elif current_entity_type == 'DVT':
            dvts.append(current_entity)
        elif current_entity_type == 'MISC':
            miscelaneous.append(current_entity)
        elif current_entity_type == 'PROD':
            products.append(current_entity)
    # Combine multi-token entities into single strings
    names = [' '.join(name) for name in names]
    locations = [' '.join(location) for location in locations]
    geo_locations = [' '.join(geo_loc) for geo_loc in geo_locations]
    geo_orgs = [' '.join(geo_org) for geo_org in geo_orgs]
    orgs = [' '.join(org) for org in orgs]
    events = [' '.join(event) for event in events]
    dvts=[' '.join(dvts) for dvt in dvts]
    miscelaneous=[' '.join(miscelaneou) for miscelaneou in miscelaneous]
    products=[' '.join(product) for product in products]

    return names, locations, geo_locations, geo_orgs, orgs, events,dvts, miscelaneous,products


#%% Main loop

if __name__ == "__main__":
       
    
    output_dir = r"output"
    # audio_folder = r"data\audio_files"
    # transcription_dict = transcribe_audio_files(audio_folder, model_name="large-v3")
    # dfScript = pd.DataFrame(transcription_dict.values(), columns=['transcription'])
    # transcription_output_csv = os.path.join(output_dir, "audio_transcription_output.csv")
    # dfScript.to_csv(transcription_output_csv, index=False)
    # print(f"Transcription completed. Data saved to {transcription_output_csv}.")
 
    dfScript = pd.read_csv(r"output\transcription_output_olasvenneby.csv")
    dfScript = dfScript.drop('Unnamed: 0', axis=1)
    dfScript.rename(columns={'0':'transcription'}, inplace=True)

    emotion_labels = ['glad', 'trist', 'overrasket', 'nøytral', 'redd', 'avsky', 'sint']
    sentiments = analyze_sentiments(dfScript['transcription'].tolist(), emotion_labels)

    candidate_labels = ["sykdom", "sport", "helse", "religion", 'miljø', 'klima']
    labels = label_topics(dfScript['transcription'].tolist(), candidate_labels)

    entities_mapping = perform_ner(dfScript['transcription'].tolist())

    # Combine results and save to CSV
    result_df = pd.DataFrame({
        'transcription': dfScript['transcription'],
        **pd.DataFrame(sentiments),
        **pd.DataFrame(labels),
        **pd.DataFrame(entities_mapping)
    })
    save_to_csv(result_df, output_dir, 'audio_analysis_output.csv')



#%% Heatmap

    sentiment_columns = ['glad', 'trist', 'overrasket', 'nøytral', 'redd', 'avsky', 'sint']
    topic_columns = ['sykdom', 'sport', 'helse', 'religion', 'miljø', 'klima']

    topic_counts = result_df[topic_columns].sum().values
    sentiment_counts = result_df[sentiment_columns].sum().values

    association_df = pd.DataFrame(index=topic_columns, columns=sentiment_columns)

    for topic in topic_columns:
        for sentiment in sentiment_columns:
            association_df.loc[topic, sentiment] = (result_df.loc[result_df[topic] > 0, sentiment].mean() if result_df[topic].sum() > 0 else 0)

    association_df = association_df.astype(float)

    plt.figure(figsize=(10, 8))
    sns.heatmap(association_df, annot=True, cmap='coolwarm', cbar=True, fmt=".2f", linewidths=.5)
    plt.title('Sentiment Scores Associated with Topics')
    plt.xlabel('Sentiment')
    plt.ylabel('Topics')
    plt.show()
