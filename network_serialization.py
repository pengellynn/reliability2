import pickle


# Pickle the network model
def pickle_network(wn, file_name):
    f = open(file_name, 'wb')
    pickle.dump(wn, f)
    f.close()


# Reload the water network model
def reload_network(file_name):
    f = open(file_name, 'rb')
    wn = pickle.load(f)
    f.close()
    return wn
