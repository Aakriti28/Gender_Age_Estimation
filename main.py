from email.policy import default
import string
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import copy
import ffmpeg
from person_extraction import track
from face_detection import retina_face
from face_sr import face_sr
from age_gender import check
from make_csv import make_csv
import argparse

# Initialize the Parser
parser = argparse.ArgumentParser(description ='Inputs')
  
# Adding Arguments
parser.add_argument('--video', type = str, help ='give path to input video')  
parser.add_argument('--visual', type = int, help = 'show visualization', default = 1 )
args = parser.parse_args()

source = '0'
source = args.video
results = "./Yolov5_DeepSort/results"

track(source=source, save_vid=True, save_txt=True, save_dir=results)

global_frame_arr = []
txt_res = results +"/" + source.split("/")[-1].split(".")[0] + ".txt"
#print(txt_res)

df = pd.read_csv(txt_res, delimiter=' ', header=None)
df = df[[0, 1, 2, 3, 4, 5, 6]]

video = cv2.VideoCapture(source)
fps = video.get(cv2.CAP_PROP_FPS)
frame_number = 0
while(video.isOpened()):
    ret, frame = video.read()
    if ret == False:
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    global_frame_arr.append(frame)
    frame_number += 1

detection_list = df.to_numpy()

class People_wrt_frame:
    def __init__(self):
        self.frame_number = 0
        self.id = -1
        self.person_x_min = -1
        self.person_y_min = -1
        self.person_w = -1
        self.person_h = -1
        self.frame_person = None
        self.face_x_min = -1
        self.face_y_min = -1
        self.face_w = -1
        self.face_h = -1
        self.frame_face = None
        self.person_score = -1
        self.sr_img = None
        self.age = -1
        self.gender = "N"
        self.is_person = False
        self.is_face = False
        self.is_SR = False
        self.is_top = False

global_people_info = []

for i in range(len(detection_list)):
    temp_instance = People_wrt_frame()
    global_people_info.append(temp_instance)

for i in range(len(detection_list)):
    frame = global_frame_arr[int(detection_list[i][0] - 1)]

    frame_cut = frame[int(detection_list[i][3]): int(detection_list[i][3] + detection_list[i][5]), int(detection_list[i][2]): int(detection_list[i][2] + detection_list[i][4])]

    global_people_info[i].frame_number = int(detection_list[i][0])
    global_people_info[i].id = int(detection_list[i][1])
    global_people_info[i].person_x_min = int(detection_list[i][3])
    global_people_info[i].person_y_min = int(detection_list[i][2])
    global_people_info[i].person_w = int(detection_list[i][5])
    global_people_info[i].person_h = int(detection_list[i][4])
    global_people_info[i].frame_person = copy.deepcopy(frame_cut)
    global_people_info[i].is_person = True

#print(len(global_people_info), len(global_frame_arr))

global_people_info = retina_face(global_people_info)
#print(len(global_people_info))
global_people_info = face_sr(global_people_info)

for i in range(len(global_people_info)):
    if global_people_info[i].is_face and global_people_info[i].frame_number % 5 == 0:
        (age, gender) = check(global_people_info[i].sr_img)
        global_people_info[i].age = age
        global_people_info[i].gender = gender
        # #print(age)
        # #print(gender)
        # #print("--")
        # plt.figure()
        # plt.imshow(global_people_info[i].frame_face)
        # plt.figure()
        # plt.imshow(global_people_info[i].sr_img)

is_first = True
last_frame_num = -1
for i in range(len(global_people_info)):
    if global_people_info[i].is_face and global_people_info[i].frame_number % 5 == 0:
        last_frame_num = global_people_info[i].frame_number
        break

dict_age = dict()
dict_gender = dict()

for i in range(len(global_people_info)):
    if global_people_info[i].is_face and global_people_info[i].frame_number % 5 == 0:
        # #print("**************")
        # #print(global_people_info[i].frame_number, i, global_people_info[i].id)
        if last_frame_num != global_people_info[i].frame_number:
            # #print("I am saving the dictionaries")
            is_first = True
            dict_age[last_frame_num] = temp_dict_age
            dict_gender[last_frame_num] = temp_dict_gender
            last_frame_num = global_people_info[i].frame_number
        
        if is_first:
            # #print("This is the first time using a new dictionary")
            temp_dict_age = dict()
            temp_dict_gender = dict()
            temp_dict_age[global_people_info[i].id] = global_people_info[i].age
            temp_dict_gender[global_people_info[i].id] = global_people_info[i].gender
            is_first = False
        else:
            # #print("Filling up the dictionary")
            temp_dict_age[global_people_info[i].id] = global_people_info[i].age
            temp_dict_gender[global_people_info[i].id] = global_people_info[i].gender

# #print("I am saving the dictionaries")
dict_age[last_frame_num] = temp_dict_age
dict_gender[last_frame_num] = temp_dict_gender

# #print("Assigning part")
for i in range(len(global_people_info)):
    temp = global_people_info[i].frame_number / 5
    temp = round(temp)    
    nearest_5 = int(5 * temp)

    try:
        if global_people_info[i].id in dict_age[nearest_5].keys():
            global_people_info[i].age = dict_age[nearest_5][global_people_info[i].id]
            global_people_info[i].gender = dict_gender[nearest_5][global_people_info[i].id]
    except:
        continue

age_label_total = dict()
age_label_count = dict()
age_label_avg = dict()

gender_label_count = dict()

for i in range(len(global_people_info)):
    if global_people_info[i].age != -1:
        if global_people_info[i].id not in age_label_total.keys():
            age_label_total[global_people_info[i].id] = global_people_info[i].age
            age_label_count[global_people_info[i].id] = 1
            if global_people_info[i].gender == "M":
                gender_label_count[global_people_info[i].id] = 1
            elif global_people_info[i].gender == "F":
                gender_label_count[global_people_info[i].id] = -1
        else:
            age_label_total[global_people_info[i].id] += global_people_info[i].age
            age_label_count[global_people_info[i].id] += 1
            if global_people_info[i].gender == "M":
                gender_label_count[global_people_info[i].id] += 1
            elif global_people_info[i].gender == "F":
                gender_label_count[global_people_info[i].id] += -1
            else:
                pass

for key in age_label_total.keys():
    age_label_avg[key] = age_label_total[key] / age_label_count[key]

for i in range(len(global_people_info)):
    if global_people_info[i].id in age_label_avg.keys():
        global_people_info[i].age = int(round(age_label_avg[global_people_info[i].id]))
    
    if global_people_info[i].id in gender_label_count.keys():
        if gender_label_count[global_people_info[i].id] <= 0:
            global_people_info[i].gender = "F"
        else:
            global_people_info[i].gender = "M"

df = make_csv(global_people_info, source)

if args.visual:

    last_frame_num = global_people_info[0].frame_number
    temp_frame = copy.deepcopy(global_frame_arr[global_people_info[0].frame_number])
    global_final_arr = []

    
    for i in range(len(global_people_info)):
        if last_frame_num != global_people_info[i].frame_number - 1:
            last_frame_num = global_people_info[i].frame_number - 1
            global_final_arr.append(temp_frame)
            temp_frame = copy.deepcopy(global_frame_arr[last_frame_num])

        x1 = int(df["bb_xmin"][i])
        y1 = int(df["bb_ymin"][i])
        x2 = int(int(df["bb_xmin"][i]) + int(df["bb_width"][i]))
        y2 = int(int(df["bb_ymin"][i]) + int(df["bb_height"][i]))
        #print("x2", x2, "y2", y2)

        temp_frame = cv2.rectangle(temp_frame, (x1, y1), (x2, y2), (255,0,0), 2)
        font = cv2.FONT_HERSHEY_TRIPLEX
        if global_people_info[i].age != -1:
            temp_frame = cv2.putText(temp_frame, 'L{} {}{}'.format(global_people_info[i].id, global_people_info[i].age, global_people_info[i].gender), (x1,y1+15), font, 1.5, (255, 0, 0), 1, cv2.LINE_AA)
    
    height, width, layers = global_final_arr[0].shape
    size = (width,height)
    #print('h', height)
    #print('w', width)
    
    video_path = 'Results/video/{}.mp4'.format(source.split("/")[-1].split(".")[0]+'_infer')
    out = cv2.VideoWriter(video_path,cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    #out = cv2.VideoWriter('{}.avi'.format(source + "_infer"),cv2.VideoWriter_fourcc(*'XVID'), fps, size)
    for i in range(len(global_final_arr)):
        out.write(cv2.cvtColor(global_final_arr[i],cv2.COLOR_BGR2RGB))
    #print(video_path)
    #print("Video Released")
    out.release()
    cv2.destroyAllWindows()



