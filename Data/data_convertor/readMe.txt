step one Download raw dataset to filter and preprocessing to create a dataset needed for our task:
we used dataset from roboflow named Football Jersey Tracker that we forked.
URL: https://universe.roboflow.com/football-tracking/football-jersey-tracker/browse?queryText=&pageSize=50&startingIndex=50&browseQuery=true

### First step 
start by running filter_and_merge_coco.py 
### description
The first stage of the code let the user pick images that are needed for our task.
aprocheded by 3 keys:
KEEP_KEY => "k"  # keep the image if useful for future preprocessing
DELETE_KEY => "d"  # delete if no need for it in the dataset
QUIT_KEY = "q"  # quit early because the loop passes through 3 phases train valid and test you can quit any one of you do not want to procees with picking in one of those three phases

The second stage is merging the picked images and anotations of the 3 phases in same image folder and aontation file for better and more ballanced dataset split.

### Second step 
Running classify_image_roles.py
### description
the code allow user to classify the team with the atack possition and diffence possition and give the formate of offnsive team, as well blur ref and unknown player in image and can be unblured.
aproched:
pick the bbox of catigory 1 if atack or defense and the catigory 3 will be the oppsite automaticly:
ATTACK_FIRST_KEY => "a"
DEFENSE_FIRST_KEY => "d"

picking formation by key number:
    "1": "shotgun"
    "2": "i-formation"
    "3": "singleback"
    "4": "trips-right"
    "5": "trips-left"
    "6": "empty"
    "7": "pistol"

after that confirm information by 
CONFIRM_KEYS => Enter or Space

other options:
UNBLUR_KEY => "u" # to save unblurred
SKIP_KEY => "s"
QUIT_KEY => "q"

### Third step
Run auto_segment_sam.py
### description
It uses SAM and make use of bbox to segment the player.

### Fourth step
Run review_and_resegment.py
### description
It can allow user to edit segmentation of images if found any unproper segmentation:
KEEP_KEY=> "k" # keep
RESEG_KEY=> "r" # resegment
QUIT_KEY=> "q" # quit