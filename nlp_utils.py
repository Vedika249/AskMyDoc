def extract_entities(text: str) -> dict:
    try:
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            return {"INFO": ["Run: python -m spacy download en_core_web_sm"]}
        doc = nlp(text[:5000])
        entities = {}
        for ent in doc.ents:
            if ent.label_ not in entities:
                entities[ent.label_] = []
            if ent.text not in entities[ent.label_]:
                entities[ent.label_].append(ent.text)
        return entities
    except ImportError:
        return {"INFO": ["spaCy not installed"]}
    except Exception as e:
        return {"ERROR": [str(e)]}


def extract_keywords(text: str, top_n: int = 10) -> list:
    try:
        from keybert import KeyBERT
        kw_model = KeyBERT()
        keywords = kw_model.extract_keywords(
            text[:5000],
            keyphrase_ngram_range=(1, 2),
            stop_words="english",
            top_n=top_n
        )
        return keywords
    except ImportError:
        return [("keybert not installed", 0.0)]
    except Exception as e:
        return [(f"Error: {str(e)}", 0.0)]


def extract_topics(texts: list) -> list:
    try:
        from bertopic import BERTopic
        if len(texts) < 5:
            return ["Not enough text for topic modeling"]
        topic_model = BERTopic(
            language="english",
            calculate_probabilities=False,
            verbose=False,
            min_topic_size=2
        )
        topics, _ = topic_model.fit_transform(texts)
        topic_info = topic_model.get_topic_info()
        top_topics = topic_info[topic_info["Topic"] != -1].head(5)
        return top_topics["Name"].tolist()
    except ImportError:
        return ["BERTopic not installed"]
    except Exception as e:
        return [f"Topic modeling error: {str(e)}"]