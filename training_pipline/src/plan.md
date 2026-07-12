I want you to write me a training code and put it at here
/home/alicekenway/Dev/project/WUW/openWakeWord/training_pipline/src
you can call the program here if necessary
/home/alicekenway/Dev/project/WUW/openWakeWord/openwakeword
you can also modify them if which is convenient.

I want you to write these thing, the train.py has these functions but I want to separate from it, 
you can call the functions in the openwakeword, dont need to write everything from scratch.


1. audio augmentor
I can use it to mix the audio with background noise, 
I will provide the input dir and the noise dir, noise dir can be multiple
will also provide how many times it will augment, also it may support the artifical augment.
and also I provide output dir 

2. Model downloader
it will call the model download of the program, it can choose download all, or download some specific one, so i provide the output dir to save it.

3. feature generator
I provide the audio dir, model path, and batch size if gpu model, ncpu if cpu model, and I will provide the output file path, you write the output npy file there.

write the code (you can write python or bash or both depend on which one use think is better)  to here
/home/alicekenway/Dev/project/WUW/openWakeWord/training_pipline/src
as long as make it good and clear tool. 


After that, please run a training and test it, to verify the pipline and the trained model can truly work. 
for training setting
you can see the default setting and example here
/home/alicekenway/Dev/project/WUW/openWakeWord/examples/custom_model.yml
/home/alicekenway/Dev/project/WUW/openWakeWord/notebooks/training_models.ipynb

for data, 
positive data is here
/mnt/d/wuw_data/eng/turn_on_the_office_lights
dont use all, get some for val and test, you decide val. test, 300 is good. 

negative speech data is here
/mnt/d/wuw_data/eng/cv-corpus-26.0-2026-06-12
but dont use all of them, I think use about 200K utterances are okay, the files are mp3, probably you need to transfer it to wav, just transfer the needed, not all
it has the train.tsv, use it, for test.tsv, we use later for testing.

background data is here
/mnt/d/wuw_data/background
VISC_Dataset_SON  fma_sample  fsd50k_sample
car background, noise and music.
you can use them as negative case, also for data augment mixing. 


firstly, you need to augment the data, augment the positive case and the negative case
randomly choose snr in the range and use the background data for mixing. 
let just do one augment for each utterance. so in the train data, we have 1 original and 1 augment for each utt. Save the augmented data in the current dir.

for percentage of each part of the training data, you can 

secondly, for the augmented data, do the feature extraction, and save them to negative npy and positive npy, train and dev.

thirdly, run the training, for hyperparameters, you can decide it, use my cuda gpu to train it.

for each step you do, about data processing or training or testing, please keep them as code or bash or tools, so I can easily use the pipline in the future.
/home/alicekenway/Dev/project/WUW/training/expts1
here is the training dir.

after trained the model, test it with test data for positive case, (this part Im new, maybe wrong, judge it by yourself.) we want fr, how many times the waking failed, so we test with the positive test set. for fa, we test, in a time range, how many times it is wrongly awaked. So we may get 2 negative set. background and negative, you design and save the testsets, then test if , lets just do 1 hours of data, you test the fa. 

I give you full permission, but dont delete or modify anything outside this dir 
~/Dev/project/WUW/openWakeWord. 
in this dir, you can modify.

you can install lib to this env, this is okay
/home/alicekenway/miniconda3/envs/openwake/bin/python

you can modify the code here, this is also okay
/home/alicekenway/Dev/project/WUW/openWakeWord

for code writing, give me readme to explain them, and explain how to use
for training, record what you did and what your conclusion from the experiment and test.



