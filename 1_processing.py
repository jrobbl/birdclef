### This script takes everything from folder "datos"
### which is expected to have structure of one folder per class
### and .ogg files per audio
### It then splits into test and train datasets and processes each audio
### the result of this processing is the spectrograms of each audio
### which are stored in train_test_spectrograms

from procesing_imports import *

# variables
RANDOM_STATE = 42
FOLDER = "datos"

# 
print("##### Grabbing and shuffling elements from data/ #####")

metadata = get_metadata(FOLDER)
species_list = os.listdir(FOLDER)

# train_test_split
X_train, X_test = train_test_split(metadata,test_size=0.3,random_state=RANDOM_STATE)
print(f"train shape: {X_train.shape}")
print(f"test shape: {X_test.shape}")

## loading metadata and missing datasets
if not "metadata.json" in os.listdir("train_test_spectrograms"):
    print("##### Filling a metadata dictionary #####")
    missing_species = get_missing_dictionary(FOLDER)

    with open("train_test_spectrograms/metadata.json","w") as outfile:
        json.dump(missing_species, outfile)
else:
    print("##### metadata.json file already exists ##### \n##### Loading... #####")
    metadata = js_r("train_test_spectrograms/metadata.json")

if not "missing.json" in os.listdir("train_test_spectrograms"):
    print("##### Filling a missing dictionary #####")
    missing_species = get_missing_dictionary(FOLDER)

    with open("train_test_spectrograms/missing.json","w") as outfile:
        json.dump(missing_species, outfile)
else:
    print("##### missing.json file already exists ##### \n##### Loading... #####")
    missing_species = js_r("train_test_spectrograms/missing.json")


# If folders don't exist, create them
print("Checking train/ and test/ folders")
try:
    os.mkdir("train_test_spectrograms/train")
except:
    pass
for species_folder in os.listdir(os.path.join(FOLDER)):
    try: 
        os.mkdir(os.path.join("train_test_spectrograms/train",species_folder))
        os.mkdir(os.path.join("train_test_spectrograms/test",species_folder))
    except:
        pass

## Filling folders
#### Filling train data
print("Filling train data folder")
for i in tqdm(X_train):
    if i[1] in missing_species[i[0]]:
        file_name = i[1].split(".")[0]
        FILE_PATH = os.path.join(FOLDER, i[0], i[1])
        audio, sr = librosa.load(FILE_PATH,sr=32000)
        time_vector = [i/sr for i in range(len(audio))]

        SNR = SNR_filter(BASE_PATH=FOLDER, 
                species=i[0],
                item=i[1])
        
        SNR.filter()

        item_number = 1
        for segmento in SNR.segments:
            array = SNR.spectrogram_db[:,segmento[0]:segmento[1]]
            np.savetxt(f"train_test_spectrograms/train/{i[0]}/{file_name}_{item_number}.txt",array)
            item_number += 1

        with open("train_test_spectrograms/missing.json", "w") as outfile: 
            json.dump(missing_species, outfile)

        missing_species[i[0]].pop(missing_species[i[0]].index(i[1]))

#### Filling test data
print("Filling test data folder")
for i in tqdm(X_test):
    if i[1] in missing_species[i[0]]:
        file_name = i[1].split(".")[0]
        FILE_PATH = os.path.join(FOLDER, i[0], i[1])
        audio, sr = librosa.load(FILE_PATH,sr=32000)
        time_vector = [i/sr for i in range(len(audio))]

        SNR = SNR_filter(BASE_PATH=FOLDER, 
                species=i[0],
                item=i[1])
        
        SNR.filter()

        item_number = 1
        for segmento in SNR.segments:
            array = SNR.spectrogram_db[:,segmento[0]:segmento[1]]
            np.savetxt(f"train_test_spectrograms/test/{i[0]}/{file_name}_{item_number}.txt",array)
            item_number += 1

        with open("train_test_spectrograms/missing.json", "w") as outfile: 
            json.dump(missing_species, outfile)

        missing_species[i[0]].pop(missing_species[i[0]].index(i[1]))