import os
import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
# from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pickle

def get_features(file):
    data=json.load(file)
    struct={}
    # feat.append(data["user"]["verified"])
    # feat.append(data["user"]["friends_count"])
    # feat.append(data["user"]["followers_count"])
    # feat.append(data["retweet_count"])
    # feat.append(data["favorite_count"])
    struct["verified"]=data["user"]["verified"]
    struct["following"]=data["user"]["friends_count"]
    struct["followers"]=data["user"]["followers_count"]
    struct["retweets"]=data["retweet_count"]
    struct["favourites"]=data["favorite_count"]

    return(struct)
    # return(tuple(feat))

def create_tree(old_dict,rum,key,event,Bool_rum):
    new_dict = { }
    for key in old_dict.keys():
        # print(key)
        try:
            if(Bool_rum==True):
                file=open("D:/Docs/PES/Capstone/all-rnr-annotated-threads_or_pheme/"+event+"/rumours/"+str(rum)+"/reactions/"+str(key)+".json")
            else:
                file=open("D:/Docs/PES/Capstone/all-rnr-annotated-threads_or_pheme/"+event+"/non-rumours/"+str(rum)+"/reactions/"+str(key)+".json")
        except:
            if(Bool_rum==True):
                file=open("D:/Docs/PES/Capstone/all-rnr-annotated-threads_or_pheme/"+event+"/rumours/"+str(rum)+"/source-tweets/"+str(rum)+".json")
            else:
                file=open("D:/Docs/PES/Capstone/all-rnr-annotated-threads_or_pheme/"+event+"/non-rumours/"+str(rum)+"/source-tweets/"+str(rum)+".json")

        new_key = get_features(file)
        if isinstance(old_dict[key], dict):
            new_dict[new_key] = create_tree(old_dict[key],rum,key,event,Bool_rum)
        else:
            new_dict[new_key] = old_dict[key]
    return new_dict

def get_dataset():
    path="D:/Docs/PES/Capstone/all-rnr-annotated-threads_or_pheme"
    raw_data={"id":[],"struct":[],"label":[]}
    files=os.listdir(path)
    events=[]
    for file in files:
        if os.path.isdir(os.path.join(os.path.abspath(path), file)):
            events.append(file)
    #print(events)
    for event in events:
        event_path_rumours="D:/Docs/PES/Capstone/all-rnr-annotated-threads_or_pheme/"+event+"/rumours"
        rumour_files=os.listdir(event_path_rumours)
        rumour_dirs=[]
        for f in rumour_files:
            if os.path.isdir(os.path.join(os.path.abspath(event_path_rumours), f)):
                rumour_dirs.append(f)
        for rum in rumour_dirs:
            file=open("D:/Docs/PES/Capstone/all-rnr-annotated-threads_or_pheme/"+event+"/rumours/"+rum+"/source-tweets/"+rum+".json")
            # structure=open("D:/Courses/VII Sem/Capstone/PHEME_veracity/PHEME_veracity/all-rnr-annotated-threads/"+event+"/rumours/"+rum+"/structure.json",encoding="utf8")
            # data=json.load(file) 
            struct=get_features(file)
            # structure=json.load(structure)
            # struct=create_tree(structure,rum,rum,event,True)
            raw_data["id"].append(rum)
            # struct["verified"]=data["user"]["verified"]
            # struct["following"]=data["user"]["friends_count"]
            # struct["followers"]=data["user"]["followers_count"]
            # struct["retweets"]=data["retweet_count"]
            # struct["favourites"]=data["favorite_count"]
            # raw_data["struct"].append(struct)
            raw_data["struct"].append(struct)
            raw_data["label"].append(1)
        #print(raw_data)

        event_path_nonrumours="D:/Docs/PES/Capstone/all-rnr-annotated-threads_or_pheme/"+event+"/non-rumours"
        non_rumour_files=os.listdir(event_path_nonrumours)
        non_rumour_dirs=[]
        for f in non_rumour_files:
            if os.path.isdir(os.path.join(os.path.abspath(event_path_nonrumours), f)):
                non_rumour_dirs.append(f)
        for nrum in non_rumour_dirs:
            file=open("D:/Docs/PES/Capstone/all-rnr-annotated-threads_or_pheme/"+event+"/non-rumours/"+nrum+"/source-tweets/"+nrum+".json")
            # structure=open("D:/Courses/VII Sem/Capstone/PHEME_veracity/PHEME_veracity/all-rnr-annotated-threads/"+event+"/non-rumours/"+nrum+"/structure.json")
            # structure=json.load(structure)
            # struct=create_tree(structure,nrum,nrum,event,False)
            struct=get_features(file)
            raw_data["id"].append(nrum)
            # struct["verified"]=data["user"]["verified"]
            # struct["following"]=data["user"]["friends_count"]
            # struct["followers"]=data["user"]["followers_count"]
            # struct["retweets"]=data["retweet_count"]
            # struct["favourites"]=data["favorite_count"]
            raw_data["struct"].append(struct)
            raw_data["label"].append(0)
    file1 = open('D:/Docs/PES/Capstone/demom/demo1/Tweetstructs/trial1.json')
    data = json.load(file1)
    print(data,'***********************************\n')
    raw_data["id"].append(1388390214347882499)
    raw_data["struct"].append(data)
    raw_data["label"].append(1)
    print(type(data))
        
    
    X_train, X_test, y_train, y_test = train_test_split(raw_data['struct'], raw_data['label'], test_size=0.3, random_state=2018,stratify=raw_data['label'])
        
    return X_train, X_test, y_train, y_test        



# structure=open("D:/Courses/VII Sem/Capstone/PHEME_veracity/PHEME_veracity/all-rnr-annotated-threads/ferguson-all-rnr-threads/rumours/500278858156085248/structure.json")
# structure=json.load(structure)
# print(structure)
# structure=create_tree(structure,500278858156085248,500278858156085248,"ferguson-all-rnr-threads",True)
# print(structure)
train_text,temp_text, train_labels, temp_labels = get_dataset()
print(temp_text,temp_labels)
v = DictVectorizer(sparse=False)
train_x=v.fit_transform(train_text)
test_x=v.fit_transform(temp_text)
print(train_x)
# svclassifier = SVC(kernel='rbf')
# svclassifier.fit(train_x, train_labels)
# y_pred = svclassifier.predict(test_x)

# clf=DecisionTreeClassifier()
clf=RandomForestClassifier(n_estimators=100)
clf=clf.fit(train_x,train_labels)
y_pred=clf.predict(test_x)
filename = 'rf_tree.pkl'
pickle.dump(clf, open(filename, 'wb'))
# hidden_units=100
# learning_rate=0.01
# hidden_layer_act='tanh'
# output_layer_act='sigmoid'
# no_epochs=100
# model = Sequential()
# model.add(Dense(hidden_units, input_dim=5, activation=hidden_layer_act))
# model.add(Dense(hidden_units, activation=hidden_layer_act))
# model.add(Dense(1, activation=output_layer_act))
# sgd=optimizers.SGD(lr=learning_rate)
# model.compile(loss='binary_crossentropy',optimizer=sgd, metrics=['acc'])
# model.fit(train_x, train_labels, epochs=no_epochs, batch_size=len(train_x),  verbose=2)
# predictions = model.predict(test_x)
# y_pred = [int(round(x[0])) for x in predictions]

print(confusion_matrix(temp_labels,y_pred))
print(classification_report(temp_labels,y_pred))

loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(test_x, temp_labels)
print(result)