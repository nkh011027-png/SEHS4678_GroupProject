import os
import time
import nltk
import numpy
import random
import json
import re
from nltk.stem.lancaster import LancasterStemmer
from tensorflow.keras import layers, models
from pathlib import Path
from nltk.tokenize import word_tokenize
from collections import Counter, defaultdict


class IntentModel:
    model:models
    intent_file_path:str
    stemmer:LancasterStemmer
    words:list
    labels:list
    data:dict

    def __init__(self, intent_file_path, force_retrain=False, epochs_list=[5, 10, 15, 20], batch_size_list=[4, 8, 16, 32, 64]) -> None:
        self.intent_file_path = intent_file_path
        self.stemmer = LancasterStemmer()
        self.labels, self.words, docs_x, docs_y, self.data = self.read_intent()

        model_path = "{}/rsc/models/{}.keras".format(Path(__file__).parent, Path(intent_file_path).stem)

        if os.path.exists(model_path) and not force_retrain:
            self.model = models.load_model(model_path)
        else:   
            training, output = self.make_BOW(self.labels, self.words, docs_x, docs_y)

            best_epochs = None
            best_batch_size = None
            best_accuracy = None

            for epochs in epochs_list:
                for batch_size in batch_size_list:
                    # Define the model
                    tmp_model = models.Sequential()
                    tmp_model.add(layers.Input(shape=(len(training[0]),)))             # Input layer
                    tmp_model.add(layers.Dense(8, activation='relu'))                  # First hidden layer
                    tmp_model.add(layers.Dense(8, activation='relu'))                  # Second hidden layer
                    tmp_model.add(layers.Dense(len(output[0]), activation='softmax'))  # Output layer

                    # Compile the model
                    tmp_model.compile(optimizer='adam',
                                    loss='categorical_crossentropy',
                                    metrics=['accuracy'])

                    tmp_model.summary()

                    history = tmp_model.fit(training, output, epochs=epochs, batch_size=batch_size, verbose=0)
                    accuracy = max(history.history['accuracy'])

                    if best_accuracy is None or accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_epochs = epochs
                        best_batch_size = batch_size
                        self.model = tmp_model

            print("===== {} =====".format(Path(intent_file_path).stem))
            print(" Best Epochs: ", best_epochs)
            print(" Best Batch Size: ", best_batch_size)
            print(" Best Accuracy: ", best_accuracy)
            self.model.save(model_path)
        pass

    def read_intent(self):
        with open(self.intent_file_path) as file:
            data = json.load(file)
        words = []
        labels = []
        docs_x = []
        docs_y = []
        
        for intent in data["intents"]:
            for pattern in intent["patterns"]:
                wrds = word_tokenize(pattern)
                words.extend(wrds)
                docs_x.append(wrds)
                docs_y.append(intent["tag"])

            if intent["tag"] not in labels:
                labels.append(intent["tag"])

        words = [self.stemmer.stem(w.lower()) for w in words if w != "?"]
        words = sorted(list(set(words)))
        labels = sorted(labels)
        return labels, words, docs_x, docs_y, data

    def make_BOW(self, labels, words, docs_x, docs_y):
        training = []
        output = []
        out_empty = [0 for _ in range(len(labels))]

        for x, doc in enumerate(docs_x):
            bag = []
            wrds = [self.stemmer.stem(w.lower()) for w in doc]

            for w in words:
                if w in wrds:
                    bag.append(1)
                else:
                    bag.append(0)

            output_row = out_empty[:]
            output_row[labels.index(docs_y[x])] = 1
            training.append(bag)
            output.append(output_row)

        return numpy.array(training), numpy.array(output)
    
    def bag_of_words(self, s, words):
        bag = [0 for _ in range(len(words))]

        s_words = word_tokenize(s)
        s_words = [self.stemmer.stem(word.lower()) for word in s_words]

        for se in s_words:
            for i, w in enumerate(words):
                if w == se:
                    bag[i] = 1

        return numpy.array(bag)
    
    def predict_intent(self, input_str, result_tag=['responses']):
        ip = self.bag_of_words(input_str, self.words)
        ip = numpy.array(ip).reshape(1, -1)

        results = self.model.predict(ip, verbose=0)
        results_index = numpy.argmax(results)
        tag = self.labels[results_index]

        result = list()
        for tg in self.data["intents"]:
            if tg['tag'] == tag:
                for r_tg in result_tag:
                    if isinstance(tg[r_tg], list):
                        result.append(random.choice(tg[r_tg]))
                    else:
                        result.append(tg[r_tg])
                
        return result


class PythonIntentModel(IntentModel):
    # mapping the topic to aliases
    topic_aliases = {
        "getstarted": [
            "getting started", "intro", "introduction", "overview", 
            "beginner guide", "first steps", "how to start", "python basics", 
            "setup python", "hello world", "starting with python"
        ],
        "intro": [
            "introduction", "overview", "what is python", "python basics", 
            "python for beginners", "why learn python", "python fundamentals"
        ],
        "syntax": [
            "syntax", "rules", "structure", "python syntax", "indentation", 
            "code structure", "writing python code", "basic syntax"
        ],
        "comments": [
            "comments", "comment", "notes", "hash comment", "# comment", 
            "multiline comments", "docstring", "code documentation"
        ],
        "variables": [
            "variables", "variable", "names", "variable names", "assigning variables", 
            "storing data", "naming variables", "python variables"
        ],
        "datatypes": [
            "datatypes", "data types", "types", "data type", "python data types", 
            "type checking", "built-in types"
        ],
        "casting": [
            "casting", "type conversion", "convert types", "type casting", 
            "converting data types", "str to int", "float conversion"
        ],
        "numbers": [
            "numbers", "numeric", "integer", "float", "int", "floating point", 
            "number operations", "numeric types"
        ],
        "output": [
            "output", "print", "console", "printing output", "display output", 
            "print function", "output to screen"
        ],
        "booleans": [
            "booleans", "boolean", "true false", "True", "False", "bool", 
            "boolean values", "logical values"
        ],
        "operators": [
            "operators", "operation", "expressions", "arithmetic operators", 
            "comparison operators", "logical operators", "operator precedence"
        ],
        "strings": [
            "strings", "string", "text", "string manipulation", "string methods", 
            "text processing", "f-strings", "string formatting"
        ],
        "lists": [
            "lists", "list", "array", "python lists", "list operations", 
            "mutable sequences", "list methods", "list comprehension"
        ],
        "tuples": [
            "tuples", "tuple", "immutable list", "tuple unpacking", 
            "read-only list"
        ],
        "sets": [
            "sets", "set", "unique values", "unique collection", "set operations", 
            "unordered collection"
        ],
        "dictionaries": [
            "dictionaries", "dictionary", "map", "key value", "key-value pair", 
            "dict", "hashmap", "associative array"
        ],
        "conditions": [
            "conditions", "if", "else", "elif", "conditional statements", 
            "if else", "decision making", "control flow"
        ],
        "loops": [
            "loops", "for", "while", "iteration", "looping", "for loop", 
            "while loop", "iterating", "repeating code"
        ],
        "functions": [
            "functions", "def", "parameters", "function definition", "reusable code", 
            "defining functions", "return value"
        ],
        "arguments": [
            "arguments", "args", "kwargs", "parameters", "function arguments", 
            "*args", "**kwargs", "variable arguments"
        ],
        "classes": [
            "classes", "object oriented", "oop", "class", "objects", 
            "object-oriented programming", "python oop"
        ],
        "inheritance": [
            "inheritance", "parent class", "child class", "superclass", "subclass", 
            "extending classes", "base class"
        ],
        "encapsulation": [
            "encapsulation", "private", "public", "data hiding", "_private", 
            "access modifiers"
        ],
        "polymorphism": [
            "polymorphism", "override", "duck typing", "method overriding", 
            "operator overloading"
        ],
        "modules": [
            "modules", "import", "package", "python modules", "importing modules", 
            "modular code", "__init__.py"
        ],
        "json": [
            "json", "serialization", "data exchange", "json module", "parsing json", 
            "json dump", "json load"
        ],
        "datetime": [
            "datetime", "date", "time", "date and time", "timestamp", 
            "datetime module", "handling dates"
        ],
        "regex": [
            "regex", "regular expressions", "pattern matching", "re module", 
            "regexp", "text pattern"
        ],
        "math": [
            "math", "maths", "calculations", "math module", "mathematical operations", 
            "numeric computations"
        ],
        "compiler": [
            "compiler", "compile", "execution", "interpreter", "how python runs", 
            "python execution model"
        ],
        "arrays": [
            "arrays", "array", "numpy array", "fixed size array"
        ],
    }

    # using to filter out the common words aim to find more distinctive keywords for intent patterns and subtopic aliases.
    stop_words = {
        "the", "and", "for", "with", "that", "this", "from", "are", "was", "were", "you", "your",
        "have", "has", "had", "can", "will", "shall", "about", "into", "more", "use", "used", "using",
        "what", "when", "where", "why", "how", "which", "their", "there", "then", "than", "also",
        "only", "each", "some", "many", "most", "very", "such", "like", "just", "over", "under",
        "html", "python", "code", "example", "examples", "page", "pages", "w3schools", "learn"
    }

    subtopic_models:dict

    def __init__(self, kb_dir, intent_file_path, force_retrain=False, epochs_list=[20], batch_size_list=[8]) -> None:
        self.kb_dir = Path(__file__).parent / kb_dir
        self.intent_file_path = Path(__file__).parent / intent_file_path
        self.intent_file_path.parent.mkdir(parents=True, exist_ok=True)
        self.subtopic_models = {}
        self.subtopic_json_files = self.build_intent_file()
        
        super().__init__(
           str(self.intent_file_path),
            force_retrain=force_retrain,
            epochs_list=epochs_list,
            batch_size_list=batch_size_list
        )

        self.subtopic_models = dict()
        for tag in self.subtopic_json_files:
            sub_model = IntentModel(
                intent_file_path=str(self.subtopic_json_files[tag]),
                force_retrain=force_retrain,
                epochs_list=epochs_list,
                batch_size_list=batch_size_list
            )
            self.subtopic_models[tag] = sub_model

        pass

    def build_intent_file(self) :
        intents = defaultdict(lambda: {"patterns": [], "responses": [], "kb_files": [], "subtopics": []})

        for kb_file in sorted(self.kb_dir.glob("*.txt")):
            topic = self.map_file_to_topic(kb_file.name)
            if topic is None:
                continue
            
            # read first 2500 characters to extract keywords for intent patterns and subtopic aliases
            content = None
            with open(kb_file, "r", encoding="utf-8") as file:
                content = file.read(2500)

            keywords = self.extract_keywords(content, top_n=10)
            subtopic_name = kb_file.name.split(".")[0].split("_")[-1]

            entry = intents[topic]
            entry["kb_files"].append(kb_file.name)
            entry["patterns"].extend(self.build_patterns(topic, keywords))
            
            if len(entry["responses"]) == 0:
                entry["responses"] = f"PythonIntent_{topic}.json"

            filter_words = []
            if subtopic_name != topic:
                filter_words = [topic, self.normalize_token(topic)]

            subtopic_keywords = self.extract_keywords(content, top_n=5, filter_words=filter_words)
            subtopic_pattern = []
            for k in subtopic_keywords:
                subtopic_pattern.extend(self.gen_pattern(k))

            entry["subtopics"].append(
                {
                    "tag": subtopic_name,
                    "patterns": self.dedupe(subtopic_pattern),
                    "responses": [
                        f"I found information about ({subtopic_name}) in ({topic}).",
                        f"I have located documentation on ({subtopic_name}) within ({topic}).",
                        f"I have retrieved material pertaining to ({subtopic_name}) in the context of ({topic}).",
                        f"I found material that aligns with ({subtopic_name}) under the topic of ({topic}).",
                    ],
                    "knowledge_base_files": kb_file.name,
                }
            )
        for tag, entry in sorted(intents.items()):
            print(tag)

        json_data = {"intents": []}
        subt_json_files = dict()
        for tag, entry in sorted(intents.items()):
            json_data["intents"].append(
                {
                    "tag": tag,
                    "patterns": self.dedupe(entry["patterns"]),
                    "responses": entry["responses"]
                }
            )

            subtopic_json_data = {"intents": []}
            for subtopic in entry["subtopics"]:
                subtopic_json_data["intents"].append(subtopic)
            
            sub_file = self.intent_file_path.parent / f"{entry['responses']}"
            with open(sub_file, "w", encoding="utf-8") as file:
                json.dump(subtopic_json_data, file, indent=2)
            subt_json_files[tag] = sub_file

        with open(self.intent_file_path, "w", encoding="utf-8") as file:
            json.dump(json_data, file, indent=2)

        return subt_json_files
    
    def predict_intent(self, input_str, result_tag=['responses']):
        ip = self.bag_of_words(input_str, self.words)
        ip = numpy.array(ip).reshape(1, -1)

        results = self.model.predict(ip, verbose=0)
        results_index = numpy.argmax(results)
        tag = self.labels[results_index]

        responses = ""
        if tag in self.subtopic_models:
            sub_result = self.subtopic_models[tag].predict_intent(input_str, result_tag=["responses", "knowledge_base_files"])

            responses = f"\n####################################################################\n\n{sub_result[0]}"
            with open(f"{self.kb_dir}/{sub_result[1]}", "r", encoding="utf-8") as file:
                responses += f"\n\n{file.read()}"
                
        return [responses]

    def map_file_to_topic(self, filename:str):
        stem = filename.replace(".txt", "")
        stem = stem.replace("python_", "", 1)
        stem = stem.replace("python", "", 1)
        stem = stem.replace(".asp", "")

        for topic in self.topic_aliases:
            if stem == topic or stem.startswith(f"{topic}"):
                return topic

        base_topic = stem.split("_")[0]
        return base_topic if base_topic else None

    # append extracted text with same meaning 
    def topic_variants(self, topic:str):
        aliases = [topic]
        aliases.extend(self.topic_aliases.get(topic, []))
        return self.dedupe(aliases)

    def extract_subtopic_name(self, filename:str):
        stem = filename.replace(".txt", "")
        stem = stem.replace("python_", "", 1)
        stem = stem.replace("python", "", 1)
        stem = stem.replace(".asp", "")
        return stem

    def build_subtopic_aliases(self, subtopic_name:str, filename:str):
        aliases = [subtopic_name, subtopic_name.replace("_", " "), filename.split(".")[0].replace("_", " ")]
        expanded_aliases = []
        for alias in aliases:
            expanded_aliases.extend(self.plural_variants(alias))
        return self.dedupe(expanded_aliases)

    def normalize_token(self, token:str):
        token = token.lower()
        if len(token) <= 3:
            return token

        if token.endswith("ies") and len(token) > 4:
            return token[:-3] + "y"

        if token.endswith("es") and len(token) > 4:
            return token[:-2]

        if token.endswith("s") and not token.endswith("ss"):
            return token[:-1]

        return token

    def plural_variants(self, token:str):
        variants = {token}
        normalized = self.normalize_token(token)
        variants.add(normalized)
        return [value for value in variants if value]

    def build_patterns(self, topic, keywords):
        patterns = list()

        patterns.extend(self.gen_pattern(topic))

        normalized_topic = self.normalize_token(topic)
        if normalized_topic != topic:
            patterns.extend(self.gen_pattern(normalized_topic))

        for keyword in keywords:
            patterns.extend(self.gen_pattern(keyword + " in " + normalized_topic))

        return patterns

    def extract_keywords(self, text:str, top_n:int = 3, filter_words=[]):
        # Find all words that are at least 3 characters long, start with a letter, 
        # and can include letters, numbers, underscores, pluses, or hyphens.
        words = re.findall(r"[a-zA-Z][a-zA-Z0-9_+-]{2,}", text.lower())

        # Count the frequency of each word, excluding stop words and digits.
        # Aim to find the most common keywords that are likely relevant to the topic.
        counts = Counter()
        for word in words:
            if word in self.stop_words or word.isdigit() or word in filter_words:
                continue
            # Keep the original token and also include a singularized variant.
            counts[word] += 1
        return [word for word, _ in counts.most_common(max(1, top_n))]


    def tokenize_text(self, text:str):
        # Find all words that are at least 2 characters long, start with a letter,
        tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_+-]{1,}", text.lower())
        expanded_tokens = []

        # For each token, generate plural variants and add them to the expanded list.
        for token in tokens:
            expanded_tokens.extend(self.plural_variants(token))

        return expanded_tokens

    def dedupe(self, values):
        return sorted(list(set(values)))# Remove exact duplicates
    
    def gen_pattern(self, topic:str):
        patterns = [
            f"{topic}",
            f"what is {topic}",
            f"tell me about {topic}",
            f"how do i use {topic}",
            f"explain {topic}",
            f"help with {topic}",
        ]
        return patterns