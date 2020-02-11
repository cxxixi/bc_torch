"""
 Dataset preparation code for ESC-50 and ESC-10 [Piczak, 2015].
 Usage: python esc_gen.py [path]
 FFmpeg should be installed.

"""

import sys
import os
import subprocess
import pandas as pd
import glob
import numpy as np
import wavio


def main():
    fsd_path = os.path.join(sys.argv[1], 'fsd')
#     os.mkdir(fsd_path)
    fs_list = [16000, 44100] 

    # Download freesound from kaggle
#     subprocess.call('kaggle competitions download -c freesound-audio-tagging -p {}'.format(
#         fsd_path), shell=True)
#     subprocess.call('unzip -d {} {}'.format(
#         fsd_path, os.path.join(fsd_path, 'freesound-audio-tagging.zip')), shell=True)
#     os.remove(os.path.join(fsd_path, 'freesound-audio-tagging.zip'))

#     # Convert sampling rate
#     for fs in fs_list:
#         if fs == 44100:
#             continue
#         else:
#             convert_fs(os.path.join(fsd_path, 'audio_train'),
#                        os.path.join(fsd_path, 'wav{}_train'.format(fs // 1000)),
#                        fs)
#             convert_fs(os.path.join(fsd_path, 'audio_test'),
#                        os.path.join(fsd_path, 'wav{}_test'.format(fs // 1000)),
#                        fs)
    
    dic = pd.read_csv('../datasets/fsd/train.csv')
    label_map = list(dic["label"].unique())
    label_map = {label_map[i]:i for i in range(len(label_map))}

    class_map_train = dic[["fname","label"]]
    class_map_train.index = class_map_train["fname"]
    class_map_train = class_map_train.drop(["fname"],axis=1)
    class_map_train["label"] = class_map_train["label"].map(label_map)
    class_map_train = class_map_train.to_dict()["label"]
#     class_map_train
    dic = pd.read_csv('../datasets/fsd/test_post_competition.csv')
    dic = dic[dic["label"]!="None"]
    class_map_test = dic[["fname","label"]]
    class_map_test.index = class_map_test["fname"]
    class_map_test = class_map_test.drop(["fname"],axis=1)
    class_map_test["label"] = class_map_test["label"].map(label_map)
    class_map_test = class_map_test.to_dict()["label"]
    
    
    # class_map_test
    # Create npz files
    for fs in fs_list:
        if fs == 44100:
            src_path_train = os.path.join(fsd_path, 'audio_train')
            src_path_test = os.path.join(fsd_path, 'audio_test')
        else:
            src_path_train = os.path.join(fsd_path, 'wav{}_train'.format(fs // 1000))
            src_path_test = os.path.join(fsd_path, 'wav{}_test'.format(fs // 1000))
            
        
        create_dataset(src_path_train, os.path.join(fsd_path, 'wav{}_train.npz'.format(fs // 1000)), class_map_train)
        create_dataset(src_path_test, os.path.join(fsd_path, 'wav{}_test.npz'.format(fs // 1000)), class_map_test)
                  


def convert_fs(src_path, dst_path, fs):
    print('* {} -> {}'.format(src_path, dst_path))
    os.mkdir(dst_path)
    for src_file in sorted(glob.glob(os.path.join(src_path, '*.wav'))):
        dst_file = src_file.replace(src_path, dst_path)
        print('ffmpeg -i {} -ac 1 -ar {} -loglevel error -y {}'.format(
            src_file, fs, dst_file))
        
        subprocess.call('ffmpeg -i {} -ac 1 -ar {} -loglevel error -y {}'.format(
            src_file, fs, dst_file), shell=True)


def create_dataset(src_path, dst_path, mapping):
    print('* {} -> {}'.format(src_path, dst_path))
    fsd_dataset = {}

    fold_list = [("1","2","3"),("4","5","6"),("7","8","9"),("0","a","b"),("c","d","e","f")]
    count = 1
    for fold in fold_list:
        fsd_dataset['fold{}'.format(count)] = {}
        fsd_sounds = []
        fsd_labels = []
        
        print(os.path.join(src_path, '{}-*.wav'.format(fold)))
        for sub_fold in fold:
            for wav_file in sorted(glob.glob(os.path.join(src_path, '{}*.wav'.format(sub_fold)))):
                sound = wavio.read(wav_file).data.T[0]
                try:
                    start = sound.nonzero()[0].min()
                    end = sound.nonzero()[0].max()
                    sound = sound[start: end + 1]  # Remove silent sections
                    fname = wav_file.split("/")[-1]
                
                    label = mapping[fname]#int(os.path.splitext(wav_file)[0].split('-')[-1])
                    fsd_sounds.append(sound)
                    fsd_labels.append(label)
                except:
                    pass
                
            

        fsd_dataset['fold{}'.format(count)]['sounds'] = fsd_sounds
        fsd_dataset['fold{}'.format(count)]['labels'] = fsd_labels
        count += 1

    np.savez(dst_path, **fsd_dataset)


if __name__ == '__main__':
    main()
