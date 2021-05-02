# Capstone-2021--Rumour-Detection-and-Veracity-Verification
Sharing content on social media platforms such as Twitter is one of the most effective ways to target a large audience and spread awareness. However, this privilege can be misused when the shared content includes unverified and mistrusted information in the form of rumours. 
Hence, early detection of these rumours and assessing their veracity is a key problem to be addressed on such platforms. In this project, we have proposed a learning mechanism which incorporates analysis of socio-linguistic data and social graph perspectives for rumour detection and veracity verification.

Demo execution: 
python3 /demo codes/demoData.ipynb
python3 /demo codes/generateDataset.ipynb
python3 /demo codes/final_demo.ipynb

bert_saved_weights.pt can be downloaded from https://drive.google.com/file/d/1AajakD8iOvJigw29DOS374zEafq2yG4e/view?usp=sharing
GoogleNews-vectors-negative300.bin can be downloaded from https://code.google.com/archive/p/word2vec/

BERT model training: 
python3 BERT-Rumour Detection.ipynb

Random Forest Classifier Training:
python3 rum_det_prop.py

Emoji Prediction code:
python3 stance_emoji.ipynb

Stance and Veracity using bilstm:
python3 stance_veracity.ipynb
