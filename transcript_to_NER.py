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
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification, AutoModelForSeq2SeqLM
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

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
    for i, transcription in tqdm(enumerate(transcriptions), desc='Estimating NER', total=len(transcriptions)):
        t = time()
        result = classifier_NER(transcription)
        ners.append(result)
        names, locations, geo_locations, geo_orgs, orgs,events, dvts, miscellaneous, products = merge_and_extract_entities(result)
        entities_mapping[i] = {'Names': names, 'Locations': locations, 'Geo Locations':geo_locations, 'Geo organization':geo_orgs, 'Organization':orgs, 'Events':events, 'Dvts':dvts, 'Miscellaneous':miscellaneous, 'Products':products}

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
    miscellaneous=[]
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
                    miscellaneous.append(current_entity)
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
            miscellaneous.append(current_entity)
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
    miscellaneous=[' '.join(miscellaneous) for miscellaneous in miscellaneous]
    products=[' '.join(product) for product in products]

    return names, locations, geo_locations, geo_orgs, orgs, events, dvts, miscellaneous, products


#%% Function to parse the generated text and extract the triplets (relation extraction)

def extract_triplets_typed(transcriptions):
    
    # Load model and tokenizer
    # tokenizer = AutoTokenizer.from_pretrained("Babelscape/mrebel-large") 
    # model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/mrebel-large")
    tokenizer = AutoTokenizer.from_pretrained("ltg/nort5-base", trust_remote_code=True) 
    model = AutoModelForSeq2SeqLM.from_pretrained("ltg/nort5-base", trust_remote_code=True)
    
    gen_kwargs = {
        "length_penalty": 0,
        "num_beams": 3,
        "num_return_sequences": 1,
        "forced_bos_token_id": None,
    }
    
    relations = []
    
    for i, transcription in tqdm(enumerate(transcriptions), desc='Extracting relations', total=len(transcriptions)):
        t = time()

        # Tokenizer text
        model_inputs = tokenizer(transcription, max_length=256, padding=True, truncation=True, return_tensors = 'pt')
        
        # Generate
        generated_tokens = model.generate(
            model_inputs["input_ids"].to(model.device),
            attention_mask=model_inputs["attention_mask"].to(model.device),
            decoder_start_token_id = tokenizer.convert_tokens_to_ids("tp_XX"),
            **gen_kwargs,
        )
        
        # Extract text
        decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)


        triplets = []
        relation = ''
        text = decoded_preds[0].strip()
        current = 'x'
        subject, relation, object_, object_type, subject_type = '','','','',''
    
        for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").replace("tp_XX", "").replace("__en__", "").split():
            if token == "<triplet>" or token == "<relation>":
                current = 't'
                if relation != '':
                    triplets.append({'head': subject.strip(), 'head_type': subject_type, 'type': relation.strip(),'tail': object_.strip(), 'tail_type': object_type})
                    relation = ''
                subject = ''
            elif token.startswith("<") and token.endswith(">"):
                if current == 't' or current == 'o':
                    current = 's'
                    if relation != '':
                        triplets.append({'head': subject.strip(), 'head_type': subject_type, 'type': relation.strip(),'tail': object_.strip(), 'tail_type': object_type})
                    object_ = ''
                    subject_type = token[1:-1]
                else:
                    current = 'o'
                    object_type = token[1:-1]
                    relation = ''
            else:
                if current == 't':
                    subject += ' ' + token
                elif current == 's':
                    object_ += ' ' + token
                elif current == 'o':
                    relation += ' ' + token
        if subject != '' and relation != '' and object_ != '' and object_type != '' and subject_type != '':
            triplets.append({'head': subject.strip(), 'head_type': subject_type, 'type': relation.strip(),'tail': object_.strip(), 'tail_type': object_type})
        
        relations.append(triplets)
        
    print(f"Relation extraction completed in {time() - t:.2f} seconds")
    return relations
    

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

    sentiment_columns = ['glad', 'trist', 'overrasket', 'nøytral', 'redd', 'avsky', 'sint']
    sentiments = analyze_sentiments(dfScript['transcription'].tolist(), sentiment_columns)

    topic_columns = ["sykdom", "sport", "helse", "religion", 'miljø', 'politikk', 'økonomi']
    labels = label_topics(dfScript['transcription'].tolist(), topic_columns)

    entities_mapping = perform_ner(dfScript['transcription'].tolist())

    extracted_triplets = extract_triplets_typed(dfScript['transcription'].tolist())

    # Combine results and save to CSV
    result_df = pd.DataFrame({
        'transcription': dfScript['transcription'],
        **pd.DataFrame(sentiments),
        **pd.DataFrame(labels),
        **pd.DataFrame.from_dict(entities_mapping, orient='index'),
        **pd.DataFrame(extracted_triplets)
    })
    save_to_csv(result_df, output_dir, 'audio_analysis_output.csv')



#%% Heatmap


    ad_matrix = np.zeros(shape=[len(topic_columns), len(sentiment_columns)])
    for i, col in enumerate(topic_columns):
        query = '%s > 0.6' %col
        ad_matrix[i,:] = result_df.query(query)[sentiment_columns].mean().values

    ad_matrix = np.nan_to_num(ad_matrix)

    # topic_counts = result_df[topic_columns].sum().values
    # sentiment_counts = result_df[sentiment_columns].sum().values

    # association_df = pd.DataFrame(index=topic_columns, columns=sentiment_columns)

    # for topic in topic_columns:
    #     for sentiment in sentiment_columns:
    #         association_df.loc[topic, sentiment] = (result_df.loc[result_df[topic] > 0, sentiment].mean() if result_df[topic].sum() > 0 else 0)

    # association_df = association_df.astype(float)
    
    association_df = pd.DataFrame(data=ad_matrix, index=topic_columns, columns=sentiment_columns)

    plt.figure(figsize=(10, 8))
    sns.heatmap(association_df, annot=True, cmap='coolwarm', cbar=True, fmt=".2f", linewidths=.5)
    plt.title('Sentiment Scores Associated with Topics')
    plt.xlabel('Sentiment')
    plt.ylabel('Topics')
    
    G = nx.from_numpy_array(ad_matrix)
    pos = nx.circular_layout(G)
    
    edges,weights = zip(*nx.get_edge_attributes(G,'weight').items())
    
    node_labels = {}
    for i, topic in enumerate(topic_columns):
        node_labels[i] = topic

    
    pos = nx.spring_layout(G)
    nx.draw(G,  node_color='b', labels=node_labels, edgelist=edges, edge_color=weights, width=10.0, edge_cmap=plt.cm.Blues)


    # nx.draw(G, node_color='#00b4d9', pos=pos, with_labels=True) 



#%% Function to color entities
    # def color_entities(transcript, doc):
    #     colored_text = transcript
    
    #     # Define colors for different entity types
    #     colors = {
    #         "PERSON": "blue",
    #         "ORG": "green",
    #         "GPE": "orange",  # Geographic entities
    #         # Add more entity types and colors as needed
    #     }
    
    #     # Loop through named entities and replace them with colored HTML
    #     for ent in doc.ents:
    #         if ent.label_ in colors:
    #             colored_text = colored_text.replace(ent.text, f'<span style="color: {colors[ent.label_]};">{ent.text}</span>')
    
    #     return colored_text
    
    # # Get the colored transcript
    # colored_transcript = color_entities(transcript)
    
    # # Display the colored transcript
    # display(HTML(colored_transcript))