# python3 -m venv sklearn-env
import pickle

def saveWeight(W1, b1, W2, b2):
  with open('model_params.pkl', 'wb') as f:
    pickle.dump((W1, b1, W2, b2), f)
    
    
def OpenWeights():   
  with open('model_params.pkl', 'rb') as f:
    W1, b1, W2, b2 = pickle.load(f)
  return W1, b1, W2, b2