import pickle
import os

BASE_DIR = "./history_saves/"

def save_history(hist, filename):
    dictionary_data = hist.history
    file = open(os.path.join(BASE_DIR, filename), 'wb')
    pickle.dump(dictionary_data, file)
    file.close()

def load_history(filename):
    file = open(filename, "rb")
    history = pickle.load(file)
    file.close()
    return history