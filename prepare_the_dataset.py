import librosa
import os
import json

#specify some constants (when we pass information to our prepare dataset)

DATAET_PATH= "data_speech_commands_v0.02"
JSON_PATH="data.json"  #WHERE we want to store these data using for training
SAMPLE_TO_CONSIDER=22050 #1 SEC worth of sound given the default setting that librosa are uses for loading


def prepare_dataset(dataset_path,json_path, n_mfcc=13, hop_length=512, n_fft=2048): #go through all the audio files extract the MFCC and store in json file (benefits is offline fast retrieve at training time.) (hop_length is how big the segements should be.)
    # creating data dictionary
    data={
        "mappings": [],
        "lables": [],
        "MFCCs": [],
        "files": []
    }
    #loop through all the sub_dir
    for i, (dirpath,dirname, filenames) in enumerate (os.walk(dataset_path)):
        #ensure that we are not in the root level
        if dirpath is not dataset_path:
            #update the mapping
            category=dirpath.split("/")[-1] #daset/up ->
            data["mappings"].append(category)
            print(f"Processing {category}")
            #loop through all the file
            for f in filenames:
                #get the file path
                file_path=os.path.join(dirpath,f)
                #load audio
                signal, sr=librosa.load(file_path)       #sr single rate
                #ensure the audio is at least 1 secs
                if len(signal)>= SAMPLE_TO_CONSIDER:
                    #Ensure it is 1 sec long
                    signal=signal[: SAMPLE_TO_CONSIDER]
                    # extract the MFCCs
                    MFCCs = librosa.feature.mfcc(signal, sr, n_mfcc=n_mfcc, n_fft=n_fft,
                                                 hop_length=hop_length)
                    data["lables"].append(i - 1)
                    data["MFCCs"].append(MFCCs.T.tolist())
                    data["files"].append(file_path)
                    print(f"{file_path}, {i-1}")



    # save data in json file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)
if __name__ == "__main__":
    prepare_dataset(DATAET_PATH, JSON_PATH)
