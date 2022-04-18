import cv2
import glob


dir = "./ad_data/"

for filename in glob.glob(dir+'images/*/*', recursive=True):
    img = cv2.imread(filename)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    h1 = cv2.inRange(h, 173, 175)
    h2 = cv2.inRange(h, 176, 179)
    h3 = cv2.inRange(h, 1, 5)
    h4 = cv2.inRange(h, 6, 12)
    h5 = cv2.inRange(h, 12, 20)
    h6 = cv2.inRange(h, 20, 24)
    h7 = cv2.inRange(h, 24, 28)
    h8 = cv2.inRange(h, 29, 34)
    h9 = cv2.inRange(h, 35, 59)
    h10 = cv2.inRange(h, 60, 65)
    h11 = cv2.inRange(h, 66, 75)
    h12 = cv2.inRange(h, 76, 81)
    h13 = cv2.inRange(h, 82, 86)
    h14 = cv2.inRange(h, 87, 90)
    h15 = cv2.inRange(h, 91, 93)
    h16 = cv2.inRange(h, 94, 97)
    h17 = cv2.inRange(h, 98, 100)
    h18 = cv2.inRange(h, 101, 106)
    h19 = cv2.inRange(h, 107, 111)
    h20 = cv2.inRange(h, 112, 116)
    h21 = cv2.inRange(h, 117, 120)
    h22 = cv2.inRange(h, 121, 133)
    h23 = cv2.inRange(h, 134, 140)
    h24 = cv2.inRange(h, 140, 145)
    h25 = cv2.inRange(h, 146, 154)
    h26 = cv2.inRange(h, 155, 162)
    h27 = cv2.inRange(h, 163, 172)

    total_color = {'pink_red': h1, 'red1': h2, 'red2': h3, 'orange1': h4,
                   'orange2': h5, 'orange3': h6, 'yellow1': h7, 'yellow2': h8,
                   'l_green': h9, 'green1': h10,'green2': h11, 'b_green1': h12,
                   'b_green2': h13, 'b_green3': h14, 'b_green4': h15, 'b_green5': h16,
                   'blue1': h17, 'blue2': h18, 'dark_blue1': h19, 'dark_blue2': h20,
                   'blue_purple1': h21, 'blue_purple2': h22, 'purple1': h23, 'purple2': h24,
                   'purple3': h25, 'light_pink': h26, 'pink': h27}
    for t, val in total_color.items():

        # mask 부분 빼고 검출
        mask_inv = cv2.bitwise_not(val)
        result = cv2.bitwise_and(hsv, hsv, mask=mask_inv)
        orange = cv2.bitwise_not(hsv, hsv, mask=val)
        color_img = cv2.cvtColor(orange, cv2.COLOR_HSV2BGR)
        save_path = dir + 'excluding_one_color27/' + t + filename[17:]
        cv2.imwrite(save_path, color_img)