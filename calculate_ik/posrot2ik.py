from multiprocessing.sharedctypes import Value
import numpy as np
import csv
import nxtik
from nxt import NxtRobot

def rotation_matrix(theta1, theta2, theta3):
    """
    入力
        theta1, theta2, theta3 = 回転角度 回転順にtheta 1, 2, 3
        oreder = 回転順　たとえば X, Z, Y順なら'xzy'
    出力
        3x3回転行列
    """
    cr = np.cos(theta1 * np.pi / 180)
    sr = np.sin(theta1 * np.pi / 180)
    cp = np.cos(theta2 * np.pi / 180)
    sp = np.sin(theta2 * np.pi / 180)
    cy = np.cos(theta3 * np.pi / 180)
    sy = np.sin(theta3 * np.pi / 180)

    matrix=np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                        [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
                        [-sp, cp*sr, cp*cr]])

    return matrix

def first_motion_seq():
    return [5, 2, 0, 0, 0, 0, -25.7, -127.5, 0, 0, 0, 8, -25.7, -133.7, -7, 0, 0, 2.86487, -2.86487, 2.86487, -2.86487]

def make_motionik(motion_seq, time, option, armid, armjntsgoal6):
    """
    armid:0(right), 1(left)
    option:0(close), 1(open), 2(stay), 3(pause)
    """

    motion_seq[0] = time
    motion_seq[1] = option

    if option == 0 and armid == 0:
        motion_seq[17:19] = [0, 0]
    elif option == 0 and armid == 1:
        motion_seq[19:21] = [0, 0]
    elif option == 1 and armid == 0:
        motion_seq[17:19] = [2.86487, -2.86487]
    elif option == 1 and armid == 1:
         motion_seq[19:21] = [2.86487, -2.86487]

    if not armjntsgoal6 == []:
        if armid == 0:
            motion_seq[5:11] = armjntsgoal6
        elif armid == 1:
            motion_seq[11:17] = armjntsgoal6

    return motion_seq

def calculate_ik(line):
    x = float(line[3])*1000
    y = float(line[4])*1000
    z = float(line[5])*1000
    roll = float(line[8])
    pitch = -90
    yaw = float(line[6])
    
    pos = [x,y,z]
    rot = rotation_matrix(roll, pitch, yaw)
    
    nxtrobot = NxtRobot()
    armjntsgoal6 = nxtik.numik(nxtrobot, pos, rot, armid="lft")
    print(pos, rot, armjntsgoal6)

    if not isinstance(armjntsgoal6, np.ndarray):
        raise ValueError("Arm joints is None!")

    return armjntsgoal6

def read_file(filepath):
    armjntsgoal6, motion_seq_list = [], []
    motion_seq = first_motion_seq()
    motion_seq_list.append(list(motion_seq))

    with open(filepath, "r") as f:
        reader = csv.reader(f)
        for line in reader:
            time_start = int(line[0])
            time_end = int(line[1])
            time = time_end - time_start
            option = line[2]
            if option == "LARM_XYZ_ABS":
                option_num = 2
                armid = 1
                armjntsgoal6 = calculate_ik(line)
                motion_seq = make_motionik(motion_seq, time, option_num, armid, armjntsgoal6)
            elif option == "LHAND_JNT_CLOSE":
                option_num = 0
                armid = 1
                motion_seq = make_motionik(motion_seq, time, option_num, armid, armjntsgoal6)
            elif option == "LHAND_JNT_OPEN":
                option_num = 1
                armid = 1
                motion_seq = make_motionik(motion_seq, time, option_num, armid, armjntsgoal6)
            elif option == "RARM_XYZ_ABS":
                option_num = 2
                armid = 0
                armjntsgoal6 = calculate_ik(line)
                motion_seq = make_motionik(motion_seq, time, option_num, armid, armjntsgoal6)
            elif option == "RHAND_JNT_CLOSE":
                option_num = 0
                armid = 0
                motion_seq = make_motionik(motion_seq, time, option_num, armid, armjntsgoal6)
            elif option == "RHAND_JNT_OPEN":
                option_num = 1
                armid = 0
                motion_seq = make_motionik(motion_seq, time, option_num, armid, armjntsgoal6)
            else:
                raise ValueError("Motion Option Error!")
            motion_seq_list.append(list(motion_seq))

    motion_seq = first_motion_seq()
    motion_seq_list.append(list(motion_seq))

    f.close()

    for motion_seq in motion_seq_list:
        print(motion_seq)

    print("IK is solved ans motion sequence is constructed!")

    return motion_seq_list

def write_file(motion_seq_list):
    with open("/home/takasu/ダウンロード/u_cylinder/motionfile/motionfile_ik.dat", "w") as f:
        for motion in motion_seq_list:
            length = len(motion)
            for i in range(length):
                f.write(str(motion[i]))
                if not i == length - 1:
                    f.write(" ")
                else:
                    f.write("\n")
    f.close()

    print("Making IK Motionfile is succeeded! : file name is motionfile_ik.dat")

if __name__ == "__main__":
    filepath = "/home/takasu/ダウンロード/u_cylinder/motionfile/motionfile_csv.csv"
    motion_seq_list = read_file(filepath)
    write_file(motion_seq_list)