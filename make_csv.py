import pandas as pd
import os, sys

def make_csv(global_people_info, input_vid):
    dict = {'frame num':[],
        'person id':[],
        'bb_xmin':[],
        'bb_ymin':[],
        'bb_height': [],
        'bb_width': [],
        'age_min': [],
        'age_max': [],
        'age_actual': [],
        'gender': []
       }
  
    df = pd.DataFrame(dict)
    for i in range(len(global_people_info)):
        if global_people_info[i].age != -1:
            age = str(global_people_info[i].age)
            age_min = str(5*int(int(age)/5))
            age_max = str(5*int(int(age)/5 + 1))

            df.loc[len(df.index)] = [str(global_people_info[i].frame_number), 
                                    str(global_people_info[i].id),
                                    str(global_people_info[i].person_y_min), 
                                    str(global_people_info[i].person_x_min), 
                                    str(global_people_info[i].person_w), 
                                    str(global_people_info[i].person_h), 
                                    age_min, age_max, age, 
                                    global_people_info[i].gender]
        else:
            df.loc[len(df.index)] = [str(global_people_info[i].frame_number), 
                                    str(global_people_info[i].id),
                                    str(global_people_info[i].person_y_min), 
                                    str(global_people_info[i].person_x_min), 
                                    str(global_people_info[i].person_w), 
                                    str(global_people_info[i].person_h), 
                                    -1, -1, -1, 
                                    global_people_info[i].gender]
    res_file = './Results/csv/' + input_vid.split("/")[-1].split(".")[0] +'.csv'
    ##print(res_file)
    if os.path.exists(res_file):
        os.remove(res_file)
    df.to_csv(res_file, index=False)

    return df