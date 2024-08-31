### **1. Importing Modules**

```bash
import nltk from nltk.stem.lancaster 
import LancasterStemmer from nltk 
import word_tokenize 
import numpy as np 
import tensorflow as tf 
import tflearn 
import random 
import json 
import time 
import pickle
```

- **`nltk`**: Natural Language Toolkit, used for working with human language data.
- **`LancasterStemmer`**: A stemmer from NLTK for reducing words to their root forms.
- **`word_tokenize`**: Tokenizes sentences into words.
- **`numpy`**: Provides support for arrays and mathematical operations.
- **`tensorflow`**: A library for machine learning and neural networks.
- **`tflearn`**: A high-level API for TensorFlow that simplifies building and training neural networks.
- **`random`**: Used for selecting random responses.
- **`json`**: For handling JSON data.
- **`time`**: Provides time-related functions.
- **`pickle`**: For serializing and deserializing Python objects.

### **2. Loading and Preprocessing Data**


```bash
with open("responses.json") as file:     data = json.load(file)
```

- **Load JSON Data**: Reads the `responses.json` file, which contains the intents and responses for the chatbot.

```bash
stemmer = LancasterStemmer()  
words = [] 
labels = [] 
docs_x = [] 
docs_y = []  
for intent in data["intents"]:     
    for pattern in intent["patterns"]:         
        word_list = word_tokenize(pattern)         
        words.extend(word_list)         
        docs_x.append(word_list)         
        docs_y.append(intent["tag"])         
        if intent["tag"] not in labels:             
            labels.append(intent["tag"])
```

- **Initialize Stemmer**: `LancasterStemmer` is used to normalize words.
- **Lists**:
    - `words`: Stores all unique words from the patterns.
    - `labels`: Stores the unique tags from the intents.
    - `docs_x`: Stores the tokenized words from each pattern.
    - `docs_y`: Stores the tags corresponding to each pattern.
- **Tokenization and Collection**:
    - Tokenizes each pattern into words and appends them to `words`.
    - Appends tokenized patterns to `docs_x` and their tags to `docs_y`.
    - Ensures each tag is added only once to `labels`.

```bash
words = [stemmer.stem(w.lower()) for w in words] 
words = sorted(set(words)) labels = sorted(labels)
```

- **Stemming**: Reduces words to their root forms to standardize them.
- **Deduplication**: Removes duplicate words and sorts the list.

```bash
training = [] 
output = [] 
out_empty = [0 for _ in range(len(labels))]  
for x, doc in enumerate(docs_x):
    bag = [0 for _ in range(len(words))]
    wrds = [stemmer.stem(w) for w in doc if w != "?"]
    for word in words:
        if word in wrds:
            bag[words.index(word)] = 1
    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)

training = np.array(training)
output = np.array(output)

with open("saved_model.pickle", "wb") as file:
    pickle.dump((words, labels, training, output), file)

```

- **Creating "Bag of Words"**:
    - For each document, creates a binary vector (`bag`) where each position represents whether a word is present or not.
    - **`out_empty`**: A vector of zeros with a length equal to the number of labels. Used to create the output vectors.
    - **`output_row`**: A one-hot encoded vector where the position of the label is set to 1.
    - **Appending**: Adds the `bag` and `output_row` to `training` and `output` lists respectively.
- **Saving Data**: Stores the processed data into `saved_model.pickle` using `pickle`.

### **3. Building and Training the Model**

```bash
model = Sequential()
model.add(Dense(8, input_shape=(len(training[0]),), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(len(output[0]), activation='softmax'))

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

try:
    model.load_weights("model.h5")
    print("model loaded")
except:
    model.fit(training, output, epochs=1000, batch_size=8, verbose=1)
    model.save_weights("model.h5")
    print("model trained")

```
- **Model Construction**:
    - **`Sequential()`**: Creates a sequential model.
    - **`Dense(8, input_shape=(len(training[0]),), activation='relu')`**: Input layer with 8 neurons and ReLU activation.
    - **`Dense(8, activation='relu')`**: Hidden layer with 8 neurons and ReLU activation.
    - **`Dense(len(output[0]), activation='softmax')`**: Output layer with neurons equal to the number of classes, using softmax activation to produce probabilities.
    - 
- **Model Compilation**:
    - **`optimizer=Adam()`**: Uses Adam optimizer.
    - **`loss='categorical_crossentropy'`**: Measures the error for classification.
    - **`metrics=['accuracy']`**: Tracks accuracy during training.
    - 
- **Model Loading or Training**:
    - **`model.load_weights("model.h5")`**: Tries to load pre-saved weights.
    - If loading fails, it trains the model with the data and saves the weights.

### **4. Defining Helper Functions**

```bash
def bag_of_words(S, words):
    bag = [0 for _ in range(len(words))]
    S_words = nltk.word_tokenize(S)
    S_words = [stemmer.stem(word.lower()) for word in S_words]
    for i in S_words:
        for x, y in enumerate(words):
            if y == i:
                bag[x] = 1
    return np.array(bag)  

def chat():
    print("What do you want to know?")
    run = True
    while run:
        inpt = input("You: ")
        if inpt.lower() == "quit":
            run = False        
        input_data = bag_of_words(inpt, words)        
        prediction = model.predict(np.array([input_data]))  
        result = np.argmax(prediction)
        tag = labels[result]
        
        for i in data["intents"]:
            if i["tag"] == tag:
                responses = i["responses"]
                try:                    
                    ttime = time.localtime()
                    ttime = time.strftime("%H:%M", ttime)
                    print(random.choice(responses).replace("'time'", ttime))
                    if (tag == "goodbye"):                        
                        run = False
                except:
                    print(random.choice(responses))
                    if (tag == "goodbye"):                        
                        run = False

```

- **`bag_of_words` Function**:
    - Converts a sentence into the same "bag of words" format used for training.
    - Tokenizes and stems the input sentence, then creates a binary vector representing the presence of each word.
- **`chat` Function**:
    - Prompts the user for input and processes it using the model.
    - If the input is "quit", exits the loop.
    - Uses the trained model to predict the tag for the input, finds the corresponding responses, and prints one of them.
    - Replaces `"'time'"` with the current time if present in the response.