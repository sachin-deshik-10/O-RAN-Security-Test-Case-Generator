import streamlit as st
from streamlit import components
import logging
import time
from functools import wraps

# Move set_page_config to the very top before any other Streamlit commands
st.set_page_config(
    page_title="O-RAN Security Test Case Generator",
    layout="wide",
    initial_sidebar_state="expanded",
)

from pathlib import Path
import base64
import json
from spacy import displacy
import nltk
from nltk.tokenize import TreebankWordTokenizer as twt
import google.generativeai as genai
from dotenv import load_dotenv
import os
from datetime import datetime

# Import data analysis modules
try:
    from oran_data_analyzer import ORANDataAnalyzer
    from collect_oran_data import ORANDataCollector
    ANALYSIS_MODULES_AVAILABLE = True
except ImportError:
    ANALYSIS_MODULES_AVAILABLE = False
    st.warning("Data analysis modules not found. Some features may be limited.")

# Import additional data analysis libraries
try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.express as px
    import plotly.graph_objects as go
    DATA_ANALYSIS_LIBS_AVAILABLE = True
except ImportError:
    DATA_ANALYSIS_LIBS_AVAILABLE = False
    st.warning("Data analysis libraries not fully available. Install pandas, numpy, matplotlib, seaborn, plotly for full functionality.")

load_dotenv()  # take environment variables from .env.

# Setup logging
logger = logging.getLogger("O-RAN-Security-Test-Case-Generator")
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

# Initialize API key from secrets
default_api_key = st.secrets.get('gemini_api_key', os.getenv('GEMINI_API_KEY', ''))
if default_api_key:
    genai.configure(api_key=default_api_key)

def log_api_call(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.info(f"API call: {func.__name__} with args: {args} kwargs: {kwargs}")
        try:
            result = func(*args, **kwargs)
            logger.info("API call successful")
            return result
        except Exception as e:
            logger.error(f"API call error: {e}")
            raise
    return wrapper

# Add a function to update API key dynamically with logging
def update_api_key(new_key):
    genai.configure(api_key=new_key)
    logger.info(f"Gemini API key updated to: {'[REDACTED]' if new_key else 'None'}")

# Add API key input in sidebar for dynamic change with validation
def api_key_input():
    st.sidebar.subheader("Gemini API Key Input")

    # Input for Gemini API key
    gemini_key = st.sidebar.text_input(
        "Enter your Gemini API key",
        value=default_api_key,
        type="password",
        help="You can change the Gemini API key here to avoid quota issues."
    )

    # Input validation for Gemini key
    if gemini_key and len(gemini_key) < 20:
        st.sidebar.warning("Gemini API key format looks invalid or too short.")

    # Update API key if changed
    if gemini_key != default_api_key:
        update_api_key(gemini_key)
        st.sidebar.success("Using Gemini API key.")

    return gemini_key

# Call the API key input function after set_page_config
api_key_input()

nltk.download("averaged_perceptron_tagger")
nltk.download("universal_tagset")

st.session_state.pos_tags = ["PRON", "VERB", "NOUN", "ADJ", "ADP", "ADV", "CONJ", "DET", "NUM", "PRT"]

st.session_state.NEAR_RT_RIC_ASSETS = ["RMR", "RIC MESSAGE ROUTER", "RMR TRANSMISSION MEDIUM", "TRANSMISSION MEDIUM"]
st.session_state.DATA_ASSETS = ["MESSAGE", "MESSAGES"]
st.session_state.ASVS = []
st.session_state.CAPEC = {}
st.session_state.CWE = []
st.session_state.ORAN_COMPONENTS = []
st.session_state.ORAN_NEAR_RT_RIC = {}
st.session_state.ORAN_SECURITY_ANALYSIS_NEAR_RT_RIC_XAPPS = {}
st.session_state.MCS = []

def gen_ents(text):
    # Tokenize text and pos tag each token
    tokens = twt().tokenize(text)
    tags = nltk.pos_tag(tokens, tagset="universal")

    # Get start and end index (span) for each token
    span_generator = twt().span_tokenize(text)
    spans = [span for span in span_generator]

    # Create dictionary with start index, end index,
    # pos_tag for each token
    ents = []
    for tag, span in zip(tags, spans):
        if tag[1] in st.session_state.pos_tags:
            ents.append({"start": span[0], "end": span[1], "label": tag[1]})

    return ents

def visualize_pos(text):
    ents = gen_ents(text)

    doc = {"text": text, "ents": ents}

    colors = {
        "PRON": "blueviolet",
        "VERB": "lightpink",
        "NOUN": "turquoise",
        "ADJ": "lime",
        "ADP": "khaki",
        "ADV": "orange",
        "CONJ": "cornflowerblue",
        "DET": "forestgreen",
        "NUM": "salmon",
        "PRT": "yellow",
    }

    options = {"ents": st.session_state.pos_tags, "colors": colors}

    return ents, displacy.render(
        doc, style="ent", jupyter=False, options=options, manual=True
    )

def gen_ent_with_word(ents, text):
    for ent in ents:
        start = ent["start"]
        end = ent["end"]
        ent["word"] = text[start:end]

def concat_nouns(ents):
    nouns = []
    left = 0
    word = ""
    while left < len(ents):
        while left < len(ents):
            if ents[left]["label"] != "NOUN":
                break
            word += f"{ents[left]['word']} "
            left += 1

        if word:
            nouns.append(word)
            word = ""

        left += 1

    return nouns

def concat_verbs(ents):
    verbs = []
    left = 0
    word = ""
    while left < len(ents):
        while left < len(ents):
            if ents[left]["label"] != "VERB":
                break
            word += f"{ents[left]['word']} "
            left += 1

        if word:
            verbs.append(word)
            word = ""

        left += 1

    return verbs

def select_outcome(texts):
    index = 0
    for i in range(len(texts)):
        if texts[i].split()[0].lower() == "then":
            index = i
            break

    return index, texts[index:]

def select_sequence(texts, outcome_index):
    index = 0
    for i in range(len(texts)):
        if texts[i].split()[0].lower() == "when":
            index = i
            break

    return texts[index:outcome_index]

def select_data_asset(ents):
    nouns = concat_nouns(ents)
    return [
        noun.strip()
        for noun in nouns
        if noun.strip().upper() in st.session_state.DATA_ASSETS
    ]

def select_near_rt_ric_asset(ents):
    nouns = concat_nouns(ents)
    return [
        noun.strip()
        for noun in nouns
        if noun.strip().upper() in st.session_state.NEAR_RT_RIC_ASSETS
    ]

def ucs_graph(graph):
    components.v1.html(
        f"""
        <pre class="mermaid">
            {graph}
        </pre>

        <script type="module">
            import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
            mermaid.initialize({{ startOnLoad: true }});
        </script>
        """,
        scrolling=True,
        height=450,
    )

def capec_related_attacks_graph(graph):
    components.v1.html(
        f"""
        <pre class="mermaid">
            {graph}
        </pre>

        <script type="module">
            import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
            mermaid.initialize({{ startOnLoad: true }});
        </script>
        """,
        scrolling=True,
        height=150,
    )

import time
import functools
import threading

@log_api_call
def call_gemini_with_retry(system_prompt, user_prompt, max_retries=5, initial_delay=1):
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            # Configure the model
            model = genai.GenerativeModel('gemini-2.0-flash-exp')
            
            # Combine system and user prompts
            combined_prompt = f"{system_prompt}\n\n{user_prompt}"
            
            # Generate response
            response = model.generate_content(combined_prompt)
            
            if response.text:
                return response.text
            else:
                raise Exception("Empty response from Gemini")
                
        except Exception as e:
            logger.warning(f"Gemini API error on attempt {attempt+1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                st.error("Gemini API error. Please check your API key or try again later.")
                raise e

def cache_with_expiration(expiration_seconds=3600):
    def decorator(func):
        cache = {}
        lock = threading.Lock()

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = (args, frozenset(kwargs.items()))
            with lock:
                if key in cache:
                    result, timestamp = cache[key]
                    if time.time() - timestamp < expiration_seconds:
                        st.info("Using cached result.")
                        return result
                    else:
                        del cache[key]
                result = func(*args, **kwargs)
                cache[key] = (result, time.time())
                return result
        return wrapper
    return decorator

@cache_with_expiration(expiration_seconds=3600)
def application_test_case_to_ucs(app_source_code, use_case_scenario_examples):
    system = "You are a cyber security testing expert. You are familiar with writing security test cases and C/C++ programming.\n\n"
    system += f"Given these application source code in C/C++,\n{app_source_code}\n\n"
    system += f"Given these examples of Use Case Scenario in Gherkin language syntax,\n{use_case_scenario_examples}\n\n"
    user = 'Understand the test case in C/C++ and examples of Use Case Scenario in Gherkin language syntax, write only 1 Use Case Scenario based on the given application source code in C/C++.'

    llm_contents = call_gemini_with_retry(system, user)
    return llm_contents

def find_capec_related_attacks(data_assets, near_rt_ric_assets, actions):
    related_attacks = set()
    for capec_key, capec_val in st.session_state.CAPEC.items():
        tags = [tag.lower() for tag in capec_val["tags"]]
        for action in actions:
            if action in tags:
                related_attacks.add(capec_key)
        for data_asset in data_assets:
            if data_asset in tags:
                related_attacks.add(capec_key)
        for near_rt_ric_asset in near_rt_ric_assets:
            if near_rt_ric_asset in tags:
                related_attacks.add(capec_key)

    text = ""
    for related_attack in related_attacks:
        parent_relation = st.session_state.CAPEC[related_attack]['parent_relation']
        if len(parent_relation) > 0:
            for parent in parent_relation:
                if parent in st.session_state.CAPEC.keys():
                    text += f"{related_attack}({related_attack}:{st.session_state.CAPEC[related_attack]['type']}) --> {parent}({parent}: {st.session_state.CAPEC[parent]['type']})\n"

        child_relation = st.session_state.CAPEC[related_attack]['child_relation']
        if len(child_relation) > 0:
            for child in child_relation:
                if child in st.session_state.CAPEC.keys():
                    text += f"{related_attack}({related_attack}:{st.session_state.CAPEC[related_attack]['type']}) --> {child}({child}: {st.session_state.CAPEC[child]['type']})\n"

    if text:
        capec_related_attacks_graph(
            f"""
            graph LR
            {text}
            """
        )

    return related_attacks

def find_capec_related_attacks_llm(use_case_scenario_title, use_case_scenario_description, capec_attack_patterns):
    related_attacks = set()

    system = "You are a cyber security testing expert. You are familiar with writing security test cases. Also, you are familiar with CAPEC.\n\n"
    user = f"Given this Use Case Scenario Title,\n{use_case_scenario_title}\n\n"
    user += f"Given this Use Case Scenario Description,\n{use_case_scenario_description}\n\n"
    user += f"Given these CAPEC attack patterns,\n{capec_attack_patterns}\n\n"
    user += 'From your understanding of the Use Case Scenario Title and Use Case Scenario Description and CAPEC attack patterns, find CAPEC attack pattern(s) that have high relevance and high match with the threat model(s) with confidence of above 95%% only. Also, for the found and matched attack pattern(s), give an explanation and confidence score as to why the attack pattern is found and matched. Do not include any explanations, only provide a RFC8259 compliant JSON response following this format without deviation.\n{"content": [{"capec_id":"", "explanation":"", "confidence":""}]}\nThe JSON response:'

    completion_text = call_gemini_with_retry(system, user)
    llm_contents = json.loads(completion_text)

    for content in llm_contents["content"]:
        id_and_explain_map = {"capec_id": content["capec_id"], "explanation":content["explanation"], "confidence":content["confidence"]}
        related_attacks.add(tuple(id_and_explain_map.items()))

    text = ""
    for related_attack in related_attacks:
        related_capec_id = dict(related_attack)["capec_id"].strip()
        if st.session_state.CAPEC.get(related_capec_id) is None:
            continue
        
        parent_relation = st.session_state.CAPEC[related_capec_id]['parent_relation']
        if len(parent_relation) > 0:
            for parent in parent_relation:
                if parent in st.session_state.CAPEC.keys():
                    text += f"{related_capec_id}({related_capec_id}:{st.session_state.CAPEC[related_capec_id]['type']}) --> {parent}({parent}: {st.session_state.CAPEC[parent]['type']})\n"

        child_relation = st.session_state.CAPEC[related_capec_id]['child_relation']
        if len(child_relation) > 0:
            for child in child_relation:
                if child in st.session_state.CAPEC.keys():
                    text += f"{related_capec_id}({related_capec_id}:{st.session_state.CAPEC[related_capec_id]['type']}) --> {child}({child}: {st.session_state.CAPEC[child]['type']})\n"

    if text:
        capec_related_attacks_graph(
            f"""
            graph LR
            {text}
            """
        )

    return related_attacks

def find_oran_components_related_attacks(data_assets, near_rt_ric_assets, actions):
    related_attacks = set()
    for oran_component in st.session_state.ORAN_COMPONENTS:
        tags = [tag.lower() for tag in oran_component["tags"]]
        for action in actions:
            if action in tags:
                related_attacks.add(oran_component["threat_id"])
        for data_asset in data_assets:
            if data_asset in tags:
                related_attacks.add(oran_component["threat_id"])
        for near_rt_ric_asset in near_rt_ric_assets:
            if near_rt_ric_asset in tags:
                related_attacks.add(oran_component["threat_id"])

    text = ""
    if text:
        ucs_graph(
            f"""
            graph LR
            {text}
            """
        )
    
    return related_attacks

def find_oran_components_related_attacks_llm(use_case_scenario_title, use_case_scenario_description, oran_components_attack_patterns):
    related_attacks = set()

    system = "You are a cyber security testing expert. You are familiar with writing security test cases. Also, you are familiar with SWG O-RAN Security.\n\n"
    user = f"Given this Use Case Scenario Title,\n{use_case_scenario_title}\n\n"
    user += f"Given this Use Case Scenario Description,\n{use_case_scenario_description}\n\n"
    user += f"Given these OpenRAN attack patterns,\n{oran_components_attack_patterns}\n\n"
    user += 'From your understanding of the Use Case Scenario Title and Use Case Scenario Description and OpenRAN attack patterns, find OpenRAN attack pattern(s) that have high relevance and high match with the threat model(s) with confidence of above 95%% only. Also, for the found and matched attack pattern(s), give an explanation and confidence score as to why the attack pattern is found and matched. Do not include any explanations, only provide a RFC8259 compliant JSON response following this format without deviation.\n{"content": [{"threat_id":"", "explanation":"", "confidence":""}]}\nThe JSON response:'

    completion_text = call_gemini_with_retry(system, user)
    llm_contents = json.loads(completion_text)

    for content in llm_contents["content"]:
        id_and_explain_map = {"threat_id": content["threat_id"], "explanation":content["explanation"], "confidence":content["confidence"]}
        related_attacks.add(tuple(id_and_explain_map.items()))

    text = ""
    for related_attack in related_attacks:
        related_threat_id = dict(related_attack)["threat_id"].strip()
        if st.session_state.ORAN_COMPONENTS.get(related_threat_id) is None:
            continue
        
        parent_relation = st.session_state.ORAN_COMPONENTS[related_threat_id]['parent_relation']
        if len(parent_relation) > 0:
            for parent in parent_relation:
                if parent in st.session_state.ORAN_COMPONENTS.keys():
                    text += f"{related_threat_id}({related_threat_id}:{st.session_state.ORAN_COMPONENTS[related_threat_id]['type']}) --> {parent}({parent}: {st.session_state.ORAN_COMPONENTS[parent]['type']})\n"

        child_relation = st.session_state.ORAN_COMPONENTS[related_threat_id]['child_relation']
        if len(child_relation) > 0:
            for child in child_relation:
                if child in st.session_state.ORAN_COMPONENTS.keys():
                    text += f"{related_threat_id}({related_threat_id}:{st.session_state.ORAN_COMPONENTS[related_threat_id]['type']}) --> {child}({child}: {st.session_state.ORAN_COMPONENTS[child]['type']})\n"

    if text:
        capec_related_attacks_graph(
            f"""
            graph LR
            {text}
            """
        )

    return related_attacks

def find_oran_near_rt_ric_related_attacks(data_assets, near_rt_ric_assets, actions):
    related_attacks = set()
    for oran_near_rt_ric_key, oran_near_rt_ric_val in st.session_state.ORAN_NEAR_RT_RIC.items():
        tags = [tag.lower() for tag in oran_near_rt_ric_val["tags"]]
        for action in actions:
            if action in tags:
                related_attacks.add(oran_near_rt_ric_key)
        for data_asset in data_assets:
            if data_asset in tags:
                related_attacks.add(oran_near_rt_ric_key)
        for near_rt_ric_asset in near_rt_ric_assets:
            if near_rt_ric_asset in tags:
                related_attacks.add(oran_near_rt_ric_key)

    text = ""
    if text:
        ucs_graph(
            f"""
            graph LR
            {text}
            """
        )
    
    return related_attacks

def find_oran_near_rt_ric_related_attacks_llm(use_case_scenario_title, use_case_scenario_description, oran_near_rt_ric_attack_patterns):
    related_attacks = set()

    system = "You are a cyber security testing expert. You are familiar with writing security test cases. Also, you are familiar with SWG O-RAN Security.\n\n"
    user = f"Given this Use Case Scenario Title,\n{use_case_scenario_title}\n\n"
    user += f"Given this Use Case Scenario Description,\n{use_case_scenario_description}\n\n"
    user += f"Given these OpenRAN Near-RT RIC attack patterns,\n{oran_near_rt_ric_attack_patterns}\n\n"
    user += 'From your understanding of the Use Case Scenario Title and Use Case Scenario Description and OpenRAN Near-RT RIC attack patterns, find OpenRAN Near-RT RIC attack pattern(s) that have high relevance and high match with the threat model(s) with confidence of above 95%% only. Also, for the found and matched attack pattern(s), give an explanation and confidence score as to why the attack pattern is found and matched. Do not include any explanations, only provide a RFC8259 compliant JSON response following this format without deviation.\n{"content": [{"threat_id":"", "explanation":"", "confidence":""}]}\nThe JSON response:'

    completion_text = call_gemini_with_retry(system, user)
    llm_contents = json.loads(completion_text)

    for content in llm_contents["content"]:
        id_and_explain_map = {"threat_id": content["threat_id"], "explanation":content["explanation"], "confidence":content["confidence"]}
        related_attacks.add(tuple(id_and_explain_map.items()))

    return related_attacks

def find_oran_security_analysis_related_attacks(data_assets, near_rt_ric_assets, actions):
    related_attacks = set()
    for oran_security_analysis_key, oran_security_analysis_val in st.session_state.ORAN_SECURITY_ANALYSIS_NEAR_RT_RIC_XAPPS.items():
        tags = [tag.lower() for tag in oran_security_analysis_val["tags"]]
        for action in actions:
            if action in tags:
                related_attacks.add(oran_security_analysis_key)
        for data_asset in data_assets:
            if data_asset in tags:
                related_attacks.add(oran_security_analysis_key)
        for near_rt_ric_asset in near_rt_ric_assets:
            if near_rt_ric_asset in tags:
                related_attacks.add(oran_security_analysis_key)

    text = ""
    for related_attack in related_attacks:
        key_issue_relations = st.session_state.ORAN_SECURITY_ANALYSIS_NEAR_RT_RIC_XAPPS[related_attack]['key_issue_relation']
        if len(key_issue_relations) > 0:
            for threat_id in key_issue_relations:
                if threat_id in st.session_state.ORAN_NEAR_RT_RIC.keys():
                    text += f"{related_attack}({related_attack}) --> {threat_id}({threat_id}: {st.session_state.ORAN_NEAR_RT_RIC[threat_id]['threat_title']})\n"

    if text:
        ucs_graph(
            f"""
            graph LR
            {text}
            """
        )
    
    return related_attacks

def find_oran_security_analysis_related_attacks_llm(use_case_scenario_title, use_case_scenario_description, oran_security_analysis_attack_patterns):
    related_attacks = set()

    system = "You are a cyber security testing expert. You are familiar with writing security test cases. Also, you are familiar with SWG O-RAN Security.\n\n"
    user = f"Given this Use Case Scenario Title,\n{use_case_scenario_title}\n\n"
    user += f"Given this Use Case Scenario Description,\n{use_case_scenario_description}\n\n"
    user += f"Given these OpenRAN Near-RT RIC and xApps attack patterns,\n{oran_security_analysis_attack_patterns}\n\n"
    user += 'From your understanding of the Use Case Scenario Title and Use Case Scenario Description and OpenRAN Near-RT RIC and xApps attack patterns, find OpenRAN Near-RT RIC and xApps attack pattern(s) that have high relevance and high match with the threat model(s) with confidence of above 95%% only. Also, for the found and matched attack pattern(s), give an explanation and confidence score as to why the attack pattern is found and matched. Do not include any explanations, only provide a RFC8259 compliant JSON response following this format without deviation.\n{"content": [{"threat_id":"", "explanation":"", "confidence":""}]}\nThe JSON response:'

    completion_text = call_gemini_with_retry(system, user)
    llm_contents = json.loads(completion_text)

    for content in llm_contents["content"]:
        id_and_explain_map = {"threat_id": content["threat_id"], "explanation":content["explanation"], "confidence":content["confidence"]}
        related_attacks.add(tuple(id_and_explain_map.items()))

    return related_attacks

def find_weaknesses_and_countermeasures(found_CAPEC_attacks):
    CWEs_matched = set()
    ASVSs_matched = set()
    for found_CAPEC_attack in found_CAPEC_attacks:
        capec_id = dict(found_CAPEC_attack)["capec_id"]
        if capec_id in st.session_state.CAPEC.keys():
            related_weaknesses = st.session_state.CAPEC[capec_id]["related_weaknesses"]
            if related_weaknesses:
                for related_weakness in related_weaknesses:
                    if st.session_state.CWE[related_weakness]:
                        CWEs_matched.add(related_weakness)

    # find related CWEs by matched CWEs
    # parent_child_matched_CWEs = set()
    # for CWE_matched in CWEs_matched:
    #     if CWE[CWE_matched]:
    #         parents_relation_to = CWE[CWE_matched]["parent_relation_to"]
    #         for parent in parents_relation_to:
    #             if CWE[parent]:
    #                 parent_child_matched_CWEs.add(parent)
            
    #         children_relation_to = CWE[CWE_matched]["child_relation_to"]
    #         for child in children_relation_to:
    #             if CWE[child]:
    #                 parent_child_matched_CWEs.add(child)

    # CWEs_matched = CWEs_matched.union(parent_child_matched_CWEs)

    for CWE_matched in CWEs_matched:
        for ASVS_item_key, ASVS_item_val in st.session_state.ASVS.items():
            if CWE_matched in ASVS_item_val["related_cwe_ids"]:
                ASVSs_matched.add(ASVS_item_key)

    return CWEs_matched, ASVSs_matched

def gen_prompt(
    use_case_scenario,
    use_case_scenario_title,
    CAPEC,
    CWE,
    ASVS,
    SWG_O_RAN_Components_Threat_Model,
    SWG_O_RAN_Near_RT_RIC_Components_Threat_Model,
    SWG_Security_Analysis_for_Near_RT_RIC_and_xApps,
    SWG_Security_Analysis_for_Near_RT_RIC_and_xApps_mitigations,
    Examples_Misuse_Case_Scenario,
):
    NONE = "None"
    system = "You are a cyber security testing expert. You are familiar with writing security test cases. Also, you are familiar with CAPEC, CWE and SWG O-RAN Security.\n\n"
    user = f"Use Case Scenario Title,\n{use_case_scenario_title}\n\n"
    user += f"Use Case Scenario in Gherkin language syntax,\n{use_case_scenario}\n\n"
    user += f"CAPEC,\n{CAPEC}\n\n"
    user += f"CWE mitigations or solutions,\n{CWE}\n\n"
    user += f"ASVS mitigations or solutions,\n{ASVS}\n\n"
    user += f"SWG O-RAN Components Threat Model,\n{SWG_O_RAN_Components_Threat_Model if SWG_O_RAN_Components_Threat_Model else NONE}\n\n"
    user += f"SWG O-RAN Near-RT RIC Component Threat Model,\n{SWG_O_RAN_Near_RT_RIC_Components_Threat_Model if SWG_O_RAN_Near_RT_RIC_Components_Threat_Model else NONE}\n\n"
    user += f"SWG Security Analysis for Near-RT RIC and xApps,\n{SWG_Security_Analysis_for_Near_RT_RIC_and_xApps if SWG_Security_Analysis_for_Near_RT_RIC_and_xApps else NONE}\n\n"
    user += f"SWG Security Analysis for Near-RT RIC and xApps mitigations or solutions,\n{SWG_Security_Analysis_for_Near_RT_RIC_and_xApps_mitigations}\n\n"
    user += "Purpose of Misuse Case Scenario?\n- provides additional information about the potential threats and security controls that security engineers or researchers can use to counter those threats. \n\n"
    user += "How to construct a Misuse Case Scenario in Gherkin language?\n- provide additional information about the potential threats and security controls that security engineers or researchers can use to counter those threats. \n- For constructing the When statement, use the threat patterns from CAPEC, SWG O-RAN Components Threat Model, SWG O-RAN Near-RT RIC Component Threat Model and SWG Security Analysis for Near-RT RIC and xApps. \n- For constructing the Then statement, use CWE mitigations or solutions, ASVS mitigations or solutions and SWG Security Analysis for Near-RT RIC and xApps mitigations or solutions.\n\n"
    user += f"Examples of Misuse Case Scenario in Gherkin language syntax,\n{Examples_Misuse_Case_Scenario if Examples_Misuse_Case_Scenario else NONE}\n\n"
    user += 'From your understanding of how to construct a Misuse Case Scenario and the given examples of Misuse Case Scenario, propose best 5 unique Misuse Case Scenarios in Gherkin language syntax from above Use Case Scenario, CAPEC, CWEs, SWG O-RAN Components Threat Model (if not none), SWG O-RAN Near-RT RIC Component Threat Model (if not none) and SWG Security Analysis for Near-RT RIC and xApps (if not none). Output this in a JSON array of objects, the object must follow in this format, {"misuse_case_scenario":""}. The misuse case scenarios proposed should not be exactly the same as the use case scenario.'
    
    prompt = system + "\n\n" + user
    return system, user, prompt




def cs_sidebar():
    project_name = "O-RAN-Security-Test-Case-Generator"
    project_version = "v0.1.0"
    project_url = "https://github.com/leonardyeoxl/O-RAN-Near-RT-RIC-Misuse-Case-Scenario-Generator"
    date = "Jun 2023"
    st.sidebar.header(project_name)
    st.sidebar.markdown(
        f"""
        <small>[{project_name} {project_version}]({project_url})  | {date}</small>
        """,
        unsafe_allow_html=True,
    )
    
    # Add navigation menu for different sections
    st.sidebar.subheader("Navigation")
    page = st.sidebar.selectbox(
        "Select Page", 
        ["Security Test Case Generator", "O-RAN Dataset Analysis", "Data Collection"]
    )
    
    if page == "O-RAN Dataset Analysis":
        st.session_state.current_page = "analysis"
    elif page == "Data Collection":
        st.session_state.current_page = "collection"
    else:
        st.session_state.current_page = "main"
    
    return None

def read_data():
    with open('./data/asvs.json', "r", encoding="utf-8") as asvs_file:
        st.session_state.ASVS = json.load(asvs_file)
    
    with open('./data/capec.json', "r", encoding="utf-8") as capec_file:
        st.session_state.CAPEC = json.load(capec_file)

    with open('./data/cwe.json', "r", encoding="utf-8") as cwe_file:
        st.session_state.CWE = json.load(cwe_file)

    with open('./data/oran-components.json', "r", encoding="utf-8") as oran_components_file:
        st.session_state.ORAN_COMPONENTS = json.load(oran_components_file)
    
    with open('./data/oran-near-rt-ric.json', "r", encoding="utf-8") as oran_near_rt_ric_file:
        st.session_state.ORAN_NEAR_RT_RIC = json.load(oran_near_rt_ric_file)
    
    with open('./data/oran-security-analysis.json', "r", encoding="utf-8") as oran_security_analysis_file:
        st.session_state.ORAN_SECURITY_ANALYSIS_NEAR_RT_RIC_XAPPS = json.load(oran_security_analysis_file)

    with open('./data/misuse-case-scenario-examples.json', "r", encoding="utf-8") as mcs_examples_file:
        st.session_state.MCS = json.load(mcs_examples_file)

def start_section():
    # First section with a form
    st.title("1. Build Use Case Scenario Model")
    
    with st.form(key='start_section'):
        st.header("Step 1. Application Source Code")

        if 'ucs_from_llm' not in st.session_state:
            st.session_state.ucs_from_llm = ""

        if 'section1_triggered' not in st.session_state:
            st.session_state.section1_triggered = False

        st.session_state.app_source_code = st.text_area(
            "Test Case in Application Source Code",
            value="",
            on_change=None,
            height=350,
            placeholder="Test Case in Application Source Code here",
        )
        
        section1_button = st.form_submit_button("Section 1")

        if section1_button:
            # Trigger the second section
            st.session_state.show_section_1 = True
            st.session_state.section1_triggered = True

    # This will only show the second section if 'show_section_2' is set in session_state.
    if st.session_state.get('show_section_1', False):
        first_section()

    # Improvement: Add a button to clear all inputs and reset the app state
    if st.button("Reset All Inputs and State"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.experimental_rerun()

    # Improvement: Add a sidebar info box with instructions and links
    with st.sidebar.expander("About This App"):
        st.markdown(
            """
            **O-RAN Near-RT RIC Misuse Case Scenario Generator**

            This app helps generate security test cases and misuse case scenarios for O-RAN Near-RT RIC components.

            **Usage Instructions:**
            1. Input your application source code in C/C++ in Step 1.
            2. Review and edit the generated Use Case Scenario in Step 2.
            3. Follow the steps to generate related attacks, countermeasures, and misuse case scenarios.
            4. Use the generated scenarios for security testing and analysis.

            **Resources:**
            - [GitHub Repository](https://github.com/leonardyeoxl/O-RAN-Near-RT-RIC-Misuse-Case-Scenario-Generator)
            - [O-RAN Alliance Security Work Group](https://www.o-ran.org/workinggroups/security)
            - [Google Gemini API Documentation](https://ai.google.dev/docs)

            """
        )

def first_section():
    if st.session_state.app_source_code and st.session_state.section1_triggered:
        with st.spinner("Getting LLM generated Use Case Scenario"):
            use_case_scenario_examples = "Example 1:\nGiven a dialer xApp and a listener xApp \nAnd dialer xApp connected to RMR transmission medium successfully \nAnd listener xApp connected to RMR transmission medium successfully \nWhen dialer xApp sends a message to the listener xApp via RMR transmission medium \nThen the listener xApp receive the message\n\n"
            use_case_scenario_examples += "Example 2:\nGiven a new xApp registers with the Near-RT RIC \nAnd the new xApp subscribe to the desired RAN stacks through the E2 termination in the near-RT RICs and the E2 agents on the RAN nodes \nAnd a target xApp is already registered with the Near-RT RIC \nAnd the target xApp subscribed to the desired RAN stacks through the E2 termination in the near-RT RICs and the E2 agents on the RAN nodes \nWhen the new xApp wants to access resources from target xApp \nThen target xApp responds with its resources to the new xApp\n\n"
            st.session_state.ucs_from_llm = application_test_case_to_ucs(st.session_state.app_source_code, use_case_scenario_examples)
            st.session_state.section1_triggered = False
    
    # Navigation button without form wrapper since it contains no form inputs
    section2_button = st.button("Continue to Section 2")
    if section2_button:
        # Trigger the second section
        st.session_state.show_section_2 = True

    if st.session_state.get('show_section_2', False):
        second_section()

def second_section():
    with st.form(key='second_section'):
        st.subheader("Step 2: Review Use Case Scenario")

        st.session_state.ucstitle = st.text_input(
            "Title",
            value="",
            on_change=None,
            placeholder="Use Case Scenario Title here",
        )
        manual_ucs = st.text_area(
            "Use Case Scenario",
            value="" if not st.session_state.ucs_from_llm else st.session_state.ucs_from_llm,
            height=350,
            help="use Gherkin language syntax",
            on_change=None,
            placeholder="Use Case Scenario here",
        )

        if st.session_state.ucs_from_llm == "":
            st.session_state.ucs = st.session_state.ucs_from_llm
        else:
            st.session_state.ucs = manual_ucs

        section3_button = st.form_submit_button("Section 3")

        if section3_button:
            # Trigger the second section
            st.session_state.show_section_3 = True

    # This will only show the second section if 'show_section_2' is set in session_state.
    if st.session_state.get('show_section_3', False):
        third_section()

def third_section():
    st.title("Section 3")

    with st.form(key='third_section'):
        st.subheader("Step 3: Parts-of-Speech Tagging")
        st.session_state.new_ucs = "".join([sentence.strip() + " " for sentence in st.session_state.ucs.split("\n")])
        st.session_state.ents, st.session_state.ent_html = visualize_pos(st.session_state.new_ucs)
        st.markdown(st.session_state.ent_html, unsafe_allow_html=True)

        st.subheader("Step 4: Generate Use Case Scenario Model")
        st.selected_seqs = []
        if st.session_state.ents:
            gen_ent_with_word(st.session_state.ents, st.session_state.new_ucs)
            st.session_state.nouns = concat_nouns(st.session_state.ents)
            st.session_state.subject = st.radio(
                "Step 3.1: Select Subject",
                [noun.strip() for noun in set(st.session_state.nouns)],
            )

            st.session_state.outcome_index, st.session_state.outcomes = select_outcome(
                [sentence.strip() for sentence in st.session_state.ucs.split("\n")]
            )
            st.session_state.outcome = st.radio("Step 3.2: Select Outcome", st.session_state.outcomes)

            st.session_state.sequences = select_sequence(
                [sentence.strip() for sentence in st.session_state.ucs.split("\n")], st.session_state.outcome_index
            )
            st.session_state.selected_seqs = st.multiselect("Step 3.3: Select Sequences", st.session_state.sequences)

            st.session_state.selected_seqs_graph = ""
            selected_seqs_graph_temp = ""
            st.session_state.data_assets = []
            st.session_state.near_rt_ric_assets = []
            st.session_state.actions = []
            for index in range(len(st.session_state.selected_seqs)):
                st.session_state.seq_text = st.session_state.selected_seqs[index]
                st.session_state.seq_ents = gen_ents(st.session_state.seq_text)
                gen_ent_with_word(st.session_state.seq_ents, st.session_state.seq_text)
                st.session_state.actions = concat_verbs(st.session_state.seq_ents)
                st.session_state.data_assets = select_data_asset(st.session_state.seq_ents)
                st.session_state.near_rt_ric_assets = select_near_rt_ric_asset(st.session_state.seq_ents)
                selected_seqs_graph_temp += f"D --> E{index}(Action: {','.join(st.session_state.actions)})\n"
                selected_seqs_graph_temp += (
                    f"E{index} --> F{index}(Data Assets: {','.join(st.session_state.data_assets)})\n"
                )
                selected_seqs_graph_temp += f"F{index} --> G{index}(O-RAN Assets: {','.join(st.session_state.near_rt_ric_assets)})\n"
                st.session_state.selected_seqs_graph = selected_seqs_graph_temp
        
        # Submit button for the form
        section4_button = st.form_submit_button("section 4")
        
        if section4_button:
            # Trigger the second section
            st.session_state.show_section_4 = True

    if st.session_state.get('show_section_4', False):
        fourth_section()

def fourth_section():
    ucs_graph(
        f"""
        graph TD
            A({st.session_state.ucstitle})
            A --> B(Subject: {st.session_state.subject})
            A --> C(Outcome: {st.session_state.outcome})
            A --> D(Sequences)
            {st.session_state.selected_seqs_graph}
        """
    )

    with st.form(key='fourth_section'):
        st.header("2. Find Related Attacks")
        
        # Submit button for the form
        section5_button = st.form_submit_button("section 5")
        
        if section5_button:
            # Trigger the second section
            st.session_state.show_section_5 = True

    if st.session_state.get('show_section_5', False):
        fifth_section()

def fifth_section():
    capec_related_attacks = set()
    oran_components_related_attacks = set()
    oran_near_rt_ric_related_attacks = set()
    oran_security_analysis_related_attacks = set()

    if st.session_state.ucs != "" and st.session_state.ucstitle != "":
        st.session_state.capec_attack_patterns = ""
        capec_attack_patterns_temp = ""
        for CAPEC_atk_pattern_id, CAPEC_atk_pattern in st.session_state.CAPEC.items():
            capec_attack_patterns_temp += f"CAPEC id: {CAPEC_atk_pattern_id}: CAPEC Title: {CAPEC_atk_pattern['type']}. CAPEC description: {CAPEC_atk_pattern['description']}\n"

        st.session_state.capec_related_attacks = find_capec_related_attacks_llm(st.session_state.ucs, st.session_state.ucstitle, capec_attack_patterns_temp)

        st.session_state.oran_components_attack_patterns = ""
        oran_components_attack_patterns_temp = ""
        for oran_components_atk_pattern in st.session_state.ORAN_COMPONENTS:
            oran_components_attack_patterns_temp += f"Threat id: {oran_components_atk_pattern['threat_id']}: Threat Title: {oran_components_atk_pattern['threat_title']}. Threat description: {oran_components_atk_pattern['threat_description']}\n"

        st.session_state.oran_components_related_attacks = find_oran_components_related_attacks_llm(st.session_state.ucs, st.session_state.ucstitle, oran_components_attack_patterns_temp)
        
        st.session_state.oran_near_rt_ric_attack_patterns = ""
        oran_near_rt_ric_attack_patterns_temp = ""
        for oran_near_rt_ric_atk_pattern_id, oran_near_rt_ric_atk_pattern in st.session_state.ORAN_NEAR_RT_RIC.items():
            oran_near_rt_ric_attack_patterns_temp += f"Threat id: {oran_near_rt_ric_atk_pattern_id}: Threat Title: {oran_near_rt_ric_atk_pattern['threat_title']}. Threat description: {oran_near_rt_ric_atk_pattern['threat_description']}\n"

        st.session_state.oran_near_rt_ric_related_attacks = find_oran_near_rt_ric_related_attacks_llm(st.session_state.ucs, st.session_state.ucstitle, oran_near_rt_ric_attack_patterns_temp)

        st.session_state.oran_security_analysis_attack_patterns = ""
        oran_security_analysis_attack_patterns_temp = ""
        for oran_security_analysis_atk_pattern_title, oran_security_analysis_atk_pattern in st.session_state.ORAN_SECURITY_ANALYSIS_NEAR_RT_RIC_XAPPS.items():
            oran_security_analysis_attack_patterns_temp += f"Threat Title: {oran_security_analysis_atk_pattern['key_issue_title']}. Threat description: {oran_security_analysis_atk_pattern['key_issue_detail']}. Security threats: {'.'.join(oran_security_analysis_atk_pattern['security_threats'])}\n"

        st.session_state.oran_security_analysis_related_attacks = find_oran_security_analysis_related_attacks_llm(st.session_state.ucs, st.session_state.ucstitle, oran_security_analysis_attack_patterns_temp)

        st.subheader("CAPEC Related Attacks")
        if st.session_state.capec_related_attacks:
            for capec_related_attack in st.session_state.capec_related_attacks:
                related_capec_id = dict(capec_related_attack)["capec_id"]
                related_capec_explain = dict(capec_related_attack)['explanation']
                related_capec_confidence = dict(capec_related_attack)['confidence']
                
                if st.session_state.CAPEC.get(related_capec_id) is None:
                    continue
                
                CAPEC_ID = st.session_state.CAPEC[related_capec_id]["capec_id"]
                CAPEC_TITLE = st.session_state.CAPEC[related_capec_id]["type"]
                CAPEC_DESCRIPTION = st.session_state.CAPEC[related_capec_id]["description"]
                st.write(f"ID: {CAPEC_ID}")
                st.write(f"Title: {CAPEC_TITLE}")
                st.write(f"Description: {CAPEC_DESCRIPTION}")
                st.write(f"Explanation: {related_capec_explain}")
                st.write(f"Confidence Score: {related_capec_confidence}")
                st.write("")
        else:
            st.write("There are no CAPEC Related Attacks found.")

        st.subheader("O-RAN Components Related Attacks")
        if len(st.session_state.oran_components_related_attacks) > 0:
            for oran_components_atk_pattern in st.session_state.ORAN_COMPONENTS:
                for related_attack in st.session_state.oran_components_related_attacks:
                    related_id = dict(related_attack)["threat_id"]
                    related_explain = dict(related_attack)['explanation']
                    related_confidence = dict(related_attack)['confidence']
                    if oran_components_atk_pattern["threat_id"] == related_id:
                        ORAN_COMPONENS_ID = oran_components_atk_pattern["threat_id"]
                        ORAN_COMPONENT_TITLE = oran_components_atk_pattern["threat_title"]
                        ORAN_COMPONENT_DESCRIPTION = oran_components_atk_pattern["threat_description"]
                        st.write(f"ID: {ORAN_COMPONENS_ID}")
                        st.write(f"Title: {ORAN_COMPONENT_TITLE}")
                        st.write(f"Description: {ORAN_COMPONENT_DESCRIPTION}")
                        st.write(f"Explanation: {related_explain}")
                        st.write(f"Confidence Score: {related_confidence}")
                        st.write("")
        else:
            st.write("There are no O-RAN Components Related Attacks found.")

        st.subheader("O-RAN Near-RT RIC Related Attacks")
        if len(st.session_state.oran_near_rt_ric_related_attacks) > 0:
            for oran_near_rt_ric_related_attack in st.session_state.oran_near_rt_ric_related_attacks:
                related_oran_near_rt_ric_id = dict(oran_near_rt_ric_related_attack)["threat_id"]
                related_oran_near_rt_ric_explain = dict(oran_near_rt_ric_related_attack)['explanation']
                related_oran_near_rt_ric_confidence = dict(oran_near_rt_ric_related_attack)['confidence']
                
                if related_oran_near_rt_ric_id not in st.session_state.ORAN_NEAR_RT_RIC:
                    continue
                
                ID = st.session_state.ORAN_NEAR_RT_RIC[related_oran_near_rt_ric_id]["threat_id"]
                TITLE = st.session_state.ORAN_NEAR_RT_RIC[related_oran_near_rt_ric_id]["threat_title"]
                DESCRIPTION = st.session_state.ORAN_NEAR_RT_RIC[related_oran_near_rt_ric_id]["threat_description"]
                st.write(f"ID: {ID}")
                st.write(f"Title: {TITLE}")
                st.write(f"Description: {DESCRIPTION}")
                st.write(f"Explanation: {related_oran_near_rt_ric_explain}")
                st.write(f"Confidence Score: {related_oran_near_rt_ric_confidence}")
                st.write("")
        else:
            st.write("There are no O-RAN Near-RT RIC Related Attacks found.")

        st.subheader("O-RAN Security Analysis on Near-RT RIC and xApps Related Attacks")
        if len(st.session_state.oran_security_analysis_related_attacks) > 0:
            for oran_security_analysis_related_attack in st.session_state.oran_security_analysis_related_attacks:
                related_oran_security_analysis_id = dict(oran_security_analysis_related_attack)["threat_id"]
                related_oran_security_analysis_explain = dict(oran_security_analysis_related_attack)['explanation']
                related_oran_security_analysis_confidence = dict(oran_security_analysis_related_attack)['confidence']
                
                if related_oran_security_analysis_id not in st.session_state.ORAN_SECURITY_ANALYSIS_NEAR_RT_RIC_XAPPS:
                    continue

                TITLE = st.session_state.ORAN_SECURITY_ANALYSIS_NEAR_RT_RIC_XAPPS[related_oran_security_analysis_id]["key_issue_title"]
                DESCRIPTION = st.session_state.ORAN_SECURITY_ANALYSIS_NEAR_RT_RIC_XAPPS[related_oran_security_analysis_id]["key_issue_detail"]
                SECURITY_THREATS = st.session_state.ORAN_SECURITY_ANALYSIS_NEAR_RT_RIC_XAPPS[related_oran_security_analysis_id]["security_threats"]
                st.write(f"Title: {TITLE}")
                st.write(f"Description: {DESCRIPTION}")
                st.write(f"Security Threats: {SECURITY_THREATS}")
                st.write(f"Explanation: {related_oran_security_analysis_explain}")
                st.write(f"Confidence Score: {related_oran_security_analysis_confidence}")
                st.write("")
        else:
            st.write("There are no O-RAN Security Analysis on Near-RT RIC and xApps Related Attacks found.")

    with st.form(key='fifth_section'):
        # Submit button for the form
        section6_button = st.form_submit_button("Recommend Countermeasures and Construct Misuse Case Scenario")
        
        if section6_button:
            # Trigger the second section
            st.session_state.show_section_6 = True

    if st.session_state.get('show_section_6', False):
        sixth_section()

def sixth_section():
    st.header("3. Construct Misuse Case Scenario")

    st.session_state.CWEs_matched, st.session_state.ASVSs_matched = find_weaknesses_and_countermeasures(
        st.session_state.capec_related_attacks
    )
    st.subheader("CWE")
    if st.session_state.CWEs_matched:
        for CWE_matched in st.session_state.CWEs_matched:
            CWE_id = st.session_state.CWE[CWE_matched]["cwe_id"]
            CWE_type = st.session_state.CWE[CWE_matched]["type"]
            CWE_description = st.session_state.CWE[CWE_matched]["description"]
            st.write(f"ID: {CWE_id}")
            st.write(f"Type: {CWE_type}")
            st.write(f"Description: {CWE_description}\n")
            st.write("")
    else:
        st.write("CWE not found")

    st.subheader("ASVS Countermeasures")
    if st.session_state.ASVSs_matched:
        for ASVS_matched in st.session_state.ASVSs_matched:
            ASVS_id = st.session_state.ASVS[ASVS_matched]["asvs_id"]
            ASVS_type = st.session_state.ASVS[ASVS_matched]["type"]
            ASVS_description = st.session_state.ASVS[ASVS_matched]["description"]
            st.write(f"ID: {ASVS_id}")
            st.write(f"Type: {ASVS_type}")
            st.write(f"Description: {ASVS_description}\n")
            st.write("")
    else:
        st.write("ASVS Countermeasures not found")

    st.subheader("O-RAN Near-RT RIC Countermeasures")
    st.write("There are no O-RAN Near-RT RIC Countermeasures found.")

    st.subheader("O-RAN Near-RT RIC xApp Countermeasures")
    st.write("There are no O-RAN Near-RT RIC xApp Countermeasures found.")

    with st.form(key='sixth_section'):
        # Submit button for the form
        section7_button = st.form_submit_button("Recommend Prompt Design")
        
        if section7_button:
            # Trigger the second section
            st.session_state.show_section_7 = True

    if st.session_state.get('show_section_7', False):
        seventh_section()

def seventh_section():
    st.subheader("Suggested Prompt Design")
    CAPEC_prompt = ""
    for capec_related_attack in st.session_state.capec_related_attacks:
        capec_id = dict(capec_related_attack)['capec_id']
        CAPEC_type = st.session_state.CAPEC[capec_id]["type"]
        CAPEC_description = st.session_state.CAPEC[capec_id]["description"]
        CAPEC_prompt += f"{capec_id}: {CAPEC_type}. {CAPEC_description}\n"

    CWE_prompt = ""
    for CWE_matched in st.session_state.CWEs_matched:
        CWE_id = st.session_state.CWE[CWE_matched]["cwe_id"]
        CWE_type = st.session_state.CWE[CWE_matched]["type"]
        CWE_description = st.session_state.CWE[CWE_matched]["description"]
        CWE_prompt += f"{CWE_id}: {CWE_type}. {CWE_description}\n"

    ASVS_prompt = ""
    for ASVS_matched in st.session_state.ASVSs_matched:
        ASVS_id = st.session_state.ASVS[ASVS_matched]["asvs_id"]
        ASVS_type = st.session_state.ASVS[ASVS_matched]["type"]
        ASVS_description = st.session_state.ASVS[ASVS_matched]["description"]
        ASVS_prompt += f"{ASVS_id}: {ASVS_type}. {ASVS_description}\n"

    ORAN_COMPONENTS_prompt = ""
    for oran_components_atk_pattern in st.session_state.ORAN_COMPONENTS:
        for related_attack in st.session_state.oran_components_related_attacks:
            related_id = dict(related_attack)["threat_id"]
            if oran_components_atk_pattern["threat_id"] == related_id:
                ORAN_COMPONENT_TITLE = oran_components_atk_pattern["threat_title"]
                ORAN_COMPONENT_DESCRIPTION = oran_components_atk_pattern["threat_description"]
                ORAN_COMPONENTS_prompt += f"Title: {ORAN_COMPONENT_TITLE} Description: {ORAN_COMPONENT_DESCRIPTION}\n"

    ORAN_NEARRT_RIC_prompt = ""
    for oran_near_rt_ric_related_attack in st.session_state.oran_near_rt_ric_related_attacks:
        related_oran_near_rt_ric_id = dict(oran_near_rt_ric_related_attack)["threat_id"]
        TITLE = st.session_state.ORAN_NEAR_RT_RIC[related_oran_near_rt_ric_id]["threat_title"]
        DESCRIPTION = st.session_state.ORAN_NEAR_RT_RIC[related_oran_near_rt_ric_id]["threat_description"]
        ORAN_NEARRT_RIC_prompt += f"Title: {TITLE} Description: {DESCRIPTION}\n"

    ORAN_SECURITY_ANALYSIS_prompt = ""
    ORAN_SECURITY_ANALYSIS_SECURITY_REQS_prompt = ""
    for oran_security_analysis_related_attack in st.session_state.oran_security_analysis_related_attacks:
        related_oran_security_analysis_id = dict(oran_security_analysis_related_attack)["threat_id"]
        TITLE = st.session_state.ORAN_SECURITY_ANALYSIS_NEAR_RT_RIC_XAPPS[related_oran_security_analysis_id]["key_issue_title"]
        DESCRIPTION = st.session_state.ORAN_SECURITY_ANALYSIS_NEAR_RT_RIC_XAPPS[related_oran_security_analysis_id]["key_issue_detail"]
        SECURITY_THREATS = ", ".join(st.session_state.ORAN_SECURITY_ANALYSIS_NEAR_RT_RIC_XAPPS[related_oran_security_analysis_id]["security_threats"])
        SECURITY_REQUIREMENTS = ", ".join(st.session_state.ORAN_SECURITY_ANALYSIS_NEAR_RT_RIC_XAPPS[related_oran_security_analysis_id]["potential_security_requirements"])
        ORAN_SECURITY_ANALYSIS_prompt += f"Title: {TITLE} Description: {DESCRIPTION} Security Threats: {SECURITY_THREATS}\n"
        ORAN_SECURITY_ANALYSIS_SECURITY_REQS_prompt += f"Security Mitigations or Solutions: {SECURITY_REQUIREMENTS}\n"

    Examples_Misuse_Case_Scenario = ""
    for index in range(len(st.session_state.MCS)):
        Examples_Misuse_Case_Scenario += f"Misuse Case Scenario #{index+1}: "+st.session_state.MCS[index]+"\n"

    st.session_state.system, st.session_state.user, st.session_state.prompt = gen_prompt(
        st.session_state.ucs,
        st.session_state.ucstitle,
        CAPEC_prompt,
        CWE_prompt,
        ASVS_prompt,
        ORAN_COMPONENTS_prompt,
        ORAN_NEARRT_RIC_prompt,
        ORAN_SECURITY_ANALYSIS_prompt,
        ORAN_SECURITY_ANALYSIS_SECURITY_REQS_prompt,
        Examples_Misuse_Case_Scenario,
    )

    st.text_area(label="prompt_design", height=850, value=st.session_state.prompt, disabled=True)

    with st.form(key='seventh_section'):
        # Submit button for the form
        section8_button = st.form_submit_button("Generate Security Test Cases")
        
        if section8_button:
            # Trigger the second section
            st.session_state.show_section_8 = True

    if st.session_state.get('show_section_8', False):
        eighth_section()

def eighth_section():
    st.session_state.option = st.selectbox(
        'Which Generative AI LLM Model?',
        ('gemini-2.0-flash-exp',)
    )

    if st.session_state.system and st.session_state.user and st.session_state.prompt and st.session_state.option:
        with st.spinner("Getting LLM generated Misuse Case Scenarios"):
            try:
                completion_text = call_gemini_with_retry(st.session_state.system, st.session_state.user)

                gen_llm_contents = json.loads(completion_text)
                for llm_content_index in range(len(gen_llm_contents)):
                    st.text_area(label=f"llm_completion_{llm_content_index+1}", height=150, value=gen_llm_contents[llm_content_index]["misuse_case_scenario"], disabled=True)
            except Exception as e:
                st.error(f"Gemini API error: {e}")
                st.error(f"Unexpected error: {e}")

def data_analysis_page():
    """O-RAN Dataset Analysis Page"""
    st.title("O-RAN Dataset Analysis & Quantitative Assessment")
    st.markdown("Comprehensive analysis of O-RAN datasets with security and performance metrics")
    
    if not ANALYSIS_MODULES_AVAILABLE:
        st.error("Data analysis modules not available. Please ensure oran_data_analyzer.py is in the project directory.")
        return
    
    if not DATA_ANALYSIS_LIBS_AVAILABLE:
        st.error("Data analysis libraries not fully available. Please install required dependencies.")
        return
    
    try:
        # Initialize the analyzer
        analyzer = ORANDataAnalyzer()
        
        # Dataset selection
        st.header("1. Dataset Selection")
        available_datasets = list(analyzer.datasets.keys())
        
        if not available_datasets:
            st.warning("No datasets found. Please ensure data files are in the ./data directory.")
            return
        
        selected_datasets = st.multiselect(
            "Select datasets to analyze:",
            available_datasets,
            default=available_datasets[:3] if len(available_datasets) >= 3 else available_datasets
        )
        
        if not selected_datasets:
            st.info("Please select at least one dataset to analyze.")
            return
        
        # Analysis options
        st.header("2. Analysis Options")
        col1, col2 = st.columns(2)
        
        with col1:
            show_basic_stats = st.checkbox("Basic Statistics", value=True)
            show_security_metrics = st.checkbox("Security Metrics", value=True)
            show_performance_metrics = st.checkbox("Performance Metrics", value=True)
        
        with col2:
            show_visualizations = st.checkbox("Visualizations", value=True)
            show_correlations = st.checkbox("Correlation Analysis", value=True)
            show_anomalies = st.checkbox("Anomaly Detection", value=True)
        
        # Run analysis button
        if st.button("Run Analysis", type="primary"):
            with st.spinner("Analyzing datasets..."):
                results = {}
                
                for dataset_name in selected_datasets:
                    st.subheader(f"Analysis Results for {dataset_name}")
                    
                    # Basic statistics
                    if show_basic_stats:
                        st.write("### Basic Statistics")
                        try:
                            stats = analyzer.get_basic_statistics(dataset_name)
                            if stats:
                                df_stats = pd.DataFrame(stats).T
                                st.dataframe(df_stats)
                        except Exception as e:
                            st.error(f"Error generating basic statistics: {e}")
                    
                    # Security metrics
                    if show_security_metrics:
                        st.write("### Security Metrics")
                        try:
                            security_metrics = analyzer.analyze_security_metrics(dataset_name)
                            if security_metrics:
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Vulnerabilities", security_metrics.get('total_vulnerabilities', 0))
                                with col2:
                                    st.metric("Critical Issues", security_metrics.get('critical_issues', 0))
                                with col3:
                                    st.metric("Security Score", f"{security_metrics.get('security_score', 0):.1f}/10")
                        except Exception as e:
                            st.error(f"Error analyzing security metrics: {e}")
                    
                    # Performance metrics
                    if show_performance_metrics:
                        st.write("### Performance Metrics")
                        try:
                            perf_metrics = analyzer.analyze_performance_metrics(dataset_name)
                            if perf_metrics:
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Avg Latency", f"{perf_metrics.get('avg_latency', 0):.2f}ms")
                                with col2:
                                    st.metric("Throughput", f"{perf_metrics.get('throughput', 0):.1f} ops/s")
                                with col3:
                                    st.metric("Availability", f"{perf_metrics.get('availability', 0):.1f}%")
                        except Exception as e:
                            st.error(f"Error analyzing performance metrics: {e}")
                    
                    # Visualizations
                    if show_visualizations:
                        st.write("### Visualizations")
                        try:
                            # Generate sample visualization
                            fig = analyzer.create_security_dashboard(dataset_name)
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error generating visualizations: {e}")
                    
                    st.divider()
                
                st.success("Analysis completed successfully!")
        
        # Dataset information
        st.header("3. Dataset Information")
        for dataset_name in selected_datasets:
            with st.expander(f"Dataset: {dataset_name}"):
                try:
                    dataset = analyzer.datasets[dataset_name]
                    if isinstance(dataset, dict):
                        st.json(dataset)
                    elif isinstance(dataset, list):
                        st.write(f"Dataset contains {len(dataset)} items")
                        if len(dataset) > 0:
                            st.write("Sample item:")
                            st.json(dataset[0])
                    else:
                        st.write(f"Dataset type: {type(dataset)}")
                        st.write(dataset)
                except Exception as e:
                    st.error(f"Error displaying dataset info: {e}")
    
    except Exception as e:
        st.error(f"Error initializing data analyzer: {e}")
        st.error("Please check that all required files are in the ./data directory.")

def data_collection_page():
    """O-RAN Data Collection Page"""
    st.title("O-RAN Free Source Data Collection")
    st.markdown("Collect O-RAN data from publicly available sources")
    
    try:
        # Initialize collector
        collector = ORANDataCollector()
        
        # Data source selection
        st.header("1. Data Source Selection")
        
        data_sources = st.multiselect(
            "Select data sources:",
            [
                "O-RAN Alliance Specifications",
                "3GPP Standards",
                "NIST Cybersecurity Framework",
                "MITRE ATT&CK for ICS",
                "Common Weakness Enumeration (CWE)",
                "CAPEC Attack Patterns"
            ],
            default=["O-RAN Alliance Specifications"]
        )
        
        # Collection parameters
        st.header("2. Collection Parameters")
        col1, col2 = st.columns(2)
        
        with col1:
            max_items = st.number_input("Maximum items to collect", min_value=1, max_value=1000, value=100)
            include_metadata = st.checkbox("Include metadata", value=True)
        
        with col2:
            format_option = st.selectbox("Output format", ["JSON", "CSV", "Both"])
            save_to_file = st.checkbox("Save to file", value=True)
        
        # Collection button
        if st.button("Start Data Collection", type="primary"):
            with st.spinner("Collecting data from selected sources..."):
                collected_data = {}
                
                for source in data_sources:
                    st.subheader(f"Collecting from: {source}")
                    
                    try:
                        if source == "O-RAN Alliance Specifications":
                            data = collector.collect_oran_alliance_specs()
                            collected_data[source] = data
                            st.success(f"Collected {len(data.get('specifications', []))} specifications")
                        
                        elif source == "3GPP Standards":
                            data = collector.collect_3gpp_standards()
                            collected_data[source] = data
                            st.success(f"Collected {len(data.get('standards', []))} standards")
                        
                        elif source == "NIST Cybersecurity Framework":
                            data = collector.collect_nist_framework()
                            collected_data[source] = data
                            st.success(f"Collected {len(data.get('controls', []))} controls")
                        
                        elif source == "MITRE ATT&CK for ICS":
                            data = collector.collect_mitre_attack_ics()
                            collected_data[source] = data
                            st.success(f"Collected {len(data.get('techniques', []))} techniques")
                        
                        elif source == "Common Weakness Enumeration (CWE)":
                            data = collector.collect_cwe_data()
                            collected_data[source] = data
                            st.success(f"Collected {len(data.get('weaknesses', []))} weaknesses")
                        
                        elif source == "CAPEC Attack Patterns":
                            data = collector.collect_capec_data()
                            collected_data[source] = data
                            st.success(f"Collected {len(data.get('attack_patterns', []))} attack patterns")
                        
                        # Display sample data
                        with st.expander(f"Sample data from {source}"):
                            st.json(data)
                    
                    except Exception as e:
                        st.error(f"Error collecting from {source}: {e}")
                
                if collected_data:
                    # Save collected data
                    if save_to_file:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"oran_collected_data_{timestamp}"
                        
                        try:
                            if format_option in ["JSON", "Both"]:
                                with open(f"./output/{filename}.json", "w") as f:
                                    json.dump(collected_data, f, indent=2)
                                st.success(f"Data saved to ./output/{filename}.json")
                            
                            if format_option in ["CSV", "Both"]:
                                # Convert to CSV format (flattened)
                                csv_data = []
                                for source, data in collected_data.items():
                                    csv_data.append({
                                        'source': source,
                                        'data': json.dumps(data),
                                        'collected_at': datetime.now().isoformat()
                                    })
                                
                                df = pd.DataFrame(csv_data)
                                df.to_csv(f"./output/{filename}.csv", index=False)
                                st.success(f"Data saved to ./output/{filename}.csv")
                        
                        except Exception as e:
                            st.error(f"Error saving data: {e}")
                    
                    # Display collection summary
                    st.header("3. Collection Summary")
                    total_items = sum(len(data.get(list(data.keys())[1], [])) for data in collected_data.values() if isinstance(data, dict) and len(data) > 1)
                    st.metric("Total Items Collected", total_items)
                    
                    # Show collected data structure
                    st.subheader("Collected Data Structure")
                    for source, data in collected_data.items():
                        with st.expander(f"{source} Structure"):
                            if isinstance(data, dict):
                                for key, value in data.items():
                                    if isinstance(value, list):
                                        st.write(f"**{key}**: {len(value)} items")
                                    else:
                                        st.write(f"**{key}**: {type(value).__name__}")
                            else:
                                st.write(f"Data type: {type(data).__name__}")
                else:
                    st.warning("No data collected. Please check your selections and try again.")
    
    except Exception as e:
        st.error(f"Error initializing data collector: {e}")
        st.error("Please ensure the collect_oran_data.py module is available.")

# Update main function to handle different pages
def main():
    read_data()
    cs_sidebar()
    
    # Initialize current page if not set
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "main"
    
    # Route to appropriate page
    if st.session_state.current_page == "analysis":
        data_analysis_page()
    elif st.session_state.current_page == "collection":
        data_collection_page()
    else:
        start_section()

if __name__ == "__main__":
    main()