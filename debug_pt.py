import json
import logging
import pickle

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.models import load_model

# logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# max_len = 200
# label_mapping = {0: "Neutral", 1: "Negative", 2: "Positive"}

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

# def predict_sentiment(list_text, model, tokenizer, max_len=max_len, mapping=label_mapping):
#     sequence = tokenizer.texts_to_sequences(list_text)
#     sequence = pad_sequences(sequence, maxlen=max_len)
#     preds = np.round(model.predict(sequence), decimals=0) \
#             .squeeze() \
#             .astype("int") \
#             .tolist()
#     labels = [mapping[pred] for pred in preds]
#     return preds, labels

def input_fn(request_body, content_type="application/json"):
#     logger.info("Request received.")
    assert content_type == "application/json"
    request = json.loads(request_body)
    try:
        model_type = request["model_type"]
        data_input = request["inputs"]
    except TypeError:
        request = json.loads(request)
        model_type = request["model_type"]
        data_input = request["inputs"]

    if model_type == "image":
        data_input = torch.tensor(data_input).to(device)

    return data_input

def model_fn(model_dir):
#     logger.info("Loading image model ...")
    model_image = Net().to(device)
    model_image.load_state_dict(torch.load(f"{model_dir}/image_model/mnist_cnn.pt"))

#     logger.info("Loading text model ...")
#     with open(f"{model_dir}/text_model/tokenizer.pkl", "rb") as file_in:
#         tokenizer = pickle.load(file_in)
#     model_text = load_model(f"{model_dir}/text_model")

#     logger.info("Model loaded.")

#     return model_image, (tokenizer, model_text)
    return model_image
    
def predict_fn(data_input, model):
    # split models
#     model_image, (tokenizer, model_text) = model

#     logger.info("Predicting ...")
    if isinstance(data_input, torch.Tensor):
        prediction = model(data_input)
        prediction = prediction.argmax(dim=1, keepdim=True)
#     elif isinstance(data_input, list):
#         prediction = predict(data_input, model_text, tokenizer)

    return prediction

def output_fn(prediction, accept="application/json"):
#     logger.info("Sending result ...")
    assert accept == "application/json"

    if isinstance(prediction, torch.Tensor):
        output = prediction.cpu().detach().numpy().tolist()[0][0]
#     elif isinstance(prediction, tuple):
#         output

    response = json.dumps({"pred": output})
    return response
