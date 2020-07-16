
import numpy as np
import cv2
import os

counter=1
SHOW_RESULTS=False


def get_backpaired_patches(img, thresh):
    patch_size=70
    i_x, i_y = 0, 0
    x_margin, y_margin = 5,5
    
    while True:
        p1_pos = [np.random.randint(2 * x_margin + (i_x + 1) * patch_size,
                                 img.shape[1] - (i_x + 2) * patch_size - 2 * x_margin),
               np.random.randint((i_y + 1) * patch_size + 2 * y_margin,
                                 img.shape[0] - (i_y + 2) * patch_size - 2 * y_margin)]
        p2_pos = [np.random.randint(2 * x_margin + (i_x + 1) * patch_size,
                                    img.shape[1] - (i_x + 2) * patch_size - 2 * x_margin),
                  np.random.randint((i_y + 1) * patch_size + 2 * y_margin,
                                    img.shape[0] - (i_y + 2) * patch_size - 2 * y_margin)]
        p1 = img[p1_pos[1]:p1_pos[1] + patch_size, p1_pos[0]:p1_pos[0] + patch_size]
        p1_thresh = thresh[p1_pos[1]:p1_pos[1] + patch_size, p1_pos[0]:p1_pos[0] + patch_size]
        p2 = img[p2_pos[1]:p2_pos[1] + patch_size, p2_pos[0]:p2_pos[0] + patch_size]
        p2_thresh = thresh[p2_pos[1]:p2_pos[1] + patch_size, p2_pos[0]:p2_pos[0] + patch_size]
        
        p1_nonzero=cv2.countNonZero(p1_thresh)
        p2_nonzero=cv2.countNonZero(p2_thresh)
        
        if p1_nonzero>10 and p1_nonzero<300 and p2_nonzero>1100:
            label = 1
            break
        elif p2_nonzero>10 and p2_nonzero<300 and p1_nonzero>1100:
            label = 1
            break        
        else:
            continue


    if SHOW_RESULTS:
        global counter
        disp_img = cv2.cvtColor(thresh.copy(), cv2.COLOR_GRAY2BGR)
        cv2.rectangle(disp_img,  (int(p1_pos[0]), int(p1_pos[1])), (int(p1_pos[0] + patch_size), int(p1_pos[1] + patch_size)), (255, 0, 0), 5, lineType=cv2.LINE_AA)
        cv2.rectangle(disp_img, (int(p2_pos[0]), int(p2_pos[1])), (int(p2_pos[0] + patch_size), int(p2_pos[1] + patch_size)), (0, 0, 255), 5, lineType=cv2.LINE_AA)
        
        cv2.imwrite('sample_pairs/'+str(counter)+'page'+str(label)+'.png',disp_img)
        cv2.imwrite('sample_pairs/'+str(counter)+'firstpair'+str(label)+'.png',p1)
        cv2.imwrite('sample_pairs/'+str(counter)+'secondpair'+str(label)+'.png',p2)
        counter=counter+1        

    return p1, p2, label

    

    
def calculate_cropped_component_average(img):
    ccs = cv2.connectedComponentsWithStatsWithAlgorithm(img, 4, cv2.CV_32S,cv2.CCL_GRANA)
    cc_number=ccs[0]-1
    cc_stats=ccs[2]
    
    a_sum, w_sum, h_sum = 0, 0, 0    
    for i in range(len(cc_stats)):
        if i==0:
            continue
        a_sum=a_sum+cc_stats[i][cv2.CC_STAT_AREA]
        w_sum=w_sum+cc_stats[i][cv2.CC_STAT_WIDTH]
        h_sum=h_sum+cc_stats[i][cv2.CC_STAT_HEIGHT]
    a_avg=a_sum/cc_number
    h_avg=h_sum/cc_number
    w_avg=w_sum/cc_number
    return a_avg,h_avg, w_avg
     
def get_different_patches(img, thresh):
    patch_size=70
    epsilon = 0.0001
    ccs = cv2.connectedComponentsWithStatsWithAlgorithm(thresh, 4, cv2.CV_32S, cv2.CCL_GRANA)
    cc_number=ccs[0]
    cc_labels=ccs[1]
    cc_stats=ccs[2]

    i_x, i_y = 0, 0
    x_margin, y_margin = 5,5
    
    while True:
        p1_pos = [np.random.randint(2 * x_margin + (i_x + 1) * patch_size,
                                 img.shape[1] - (i_x + 2) * patch_size - 2 * x_margin),
               np.random.randint((i_y + 1) * patch_size + 2 * y_margin,
                                 img.shape[0] - (i_y + 2) * patch_size - 2 * y_margin)]
        p2_pos = [np.random.randint(2 * x_margin + (i_x + 1) * patch_size,
                                    img.shape[1] - (i_x + 2) * patch_size - 2 * x_margin),
                  np.random.randint((i_y + 1) * patch_size + 2 * y_margin,
                                    img.shape[0] - (i_y + 2) * patch_size - 2 * y_margin)]
        
        p1 = img[p1_pos[1]:p1_pos[1] + patch_size, p1_pos[0]:p1_pos[0] + patch_size]
        p1_thresh = thresh[p1_pos[1]:p1_pos[1] + patch_size, p1_pos[0]:p1_pos[0] + patch_size]


        p2 = img[p2_pos[1]:p2_pos[1] + patch_size, p2_pos[0]:p2_pos[0] + patch_size]
        p2_thresh = thresh[p2_pos[1]:p2_pos[1] + patch_size, p2_pos[0]:p2_pos[0] + patch_size]

        
        if cv2.countNonZero(p1_thresh)<10 or cv2.countNonZero(p2_thresh)<10 :
            continue

        a1,h1, w1 = calculate_cropped_component_average(p1_thresh)
        a2,h2, w2 = calculate_cropped_component_average(p2_thresh)

        a1,a2,w1, w2, h1, h2 =a1 + epsilon, a2 + epsilon, w1 + epsilon, w2 + epsilon, h1 + epsilon, h2 + epsilon
        if min(a1, a2)/max(a1, a2) < 0.5 :  # todo: improve on this condition
            label = 1
            break
        else:
            continue

    if SHOW_RESULTS:
        global counter
        disp_img = cv2.cvtColor(thresh.copy(), cv2.COLOR_GRAY2BGR)
        for cc_i in range(1,cc_number):
            x, y, w, h = cc_stats[cc_i][cv2.CC_STAT_LEFT], cc_stats[cc_i][cv2.CC_STAT_TOP], cc_stats[cc_i][cv2.CC_STAT_WIDTH], cc_stats[cc_i][cv2.CC_STAT_HEIGHT]
            cv2.rectangle(disp_img, (int(x), int(y)), (int(x + w), int(y + h)), (60, 200, 200), 1, lineType=cv2.LINE_AA)

        cv2.rectangle(disp_img,  (int(p1_pos[0]), int(p1_pos[1])), (int(p1_pos[0] + patch_size), int(p1_pos[1] + patch_size)), (255, 0, 0),5, lineType=cv2.LINE_AA)
        cv2.rectangle(disp_img, (int(p2_pos[0]), int(p2_pos[1])), (int(p2_pos[0] + patch_size), int(p2_pos[1] + patch_size)), (0, 0, 255), 5, lineType=cv2.LINE_AA)
        
        cv2.imwrite('sample_pairs/'+str(counter)+'page'+str(label)+'.png',disp_img)
        cv2.imwrite('sample_pairs/'+str(counter)+'firstpair'+str(label)+'.png',p1)
        cv2.imwrite('sample_pairs/'+str(counter)+'secondpair'+str(label)+'.png',p2)
        counter=counter+1        

    return p1, p2, label


def get_same_patches(img, thresh):
    patch_size=70
    epsilon = 0.0001
    ccs = cv2.connectedComponentsWithStatsWithAlgorithm(thresh, 4, cv2.CV_32S, cv2.CCL_GRANA)
    cc_number=ccs[0]
    cc_labels=ccs[1]
    cc_stats=ccs[2]

    i_x, i_y = 0, 0
    x_margin, y_margin = 5,5
    
    while True:
        p1_pos = [np.random.randint(2 * x_margin + (i_x + 1) * patch_size,
                                 img.shape[1] - (i_x + 2) * patch_size - 2 * x_margin),
               np.random.randint((i_y + 1) * patch_size + 2 * y_margin,
                                 img.shape[0] - (i_y + 2) * patch_size - 2 * y_margin)]
        p2_pos = [np.random.randint(2 * x_margin + (i_x + 1) * patch_size,
                                    img.shape[1] - (i_x + 2) * patch_size - 2 * x_margin),
                  np.random.randint((i_y + 1) * patch_size + 2 * y_margin,
                                    img.shape[0] - (i_y + 2) * patch_size - 2 * y_margin)]
        
        p1 = img[p1_pos[1]:p1_pos[1] + patch_size, p1_pos[0]:p1_pos[0] + patch_size]
        p1_thresh = thresh[p1_pos[1]:p1_pos[1] + patch_size, p1_pos[0]:p1_pos[0] + patch_size]


        p2 = img[p2_pos[1]:p2_pos[1] + patch_size, p2_pos[0]:p2_pos[0] + patch_size]
        p2_thresh = thresh[p2_pos[1]:p2_pos[1] + patch_size, p2_pos[0]:p2_pos[0] + patch_size]

        
        if cv2.countNonZero(p1_thresh)<10 or cv2.countNonZero(p2_thresh)<10 :
            continue

        a1,h1, w1 = calculate_cropped_component_average(p1_thresh)
        a2,h2, w2 = calculate_cropped_component_average(p2_thresh)

        a1,a2,w1, w2, h1, h2 =a1 + epsilon, a2 + epsilon, w1 + epsilon, w2 + epsilon, h1 + epsilon, h2 + epsilon

        if min(a1, a2)/max(a1, a2) > 0.7 :  # todo: improve on this condition
            label = 0
            break
        else:
            continue

    if SHOW_RESULTS:
        global counter
        disp_img = cv2.cvtColor(thresh.copy(), cv2.COLOR_GRAY2BGR)
        for cc_i in range(1,cc_number):
            x, y, w, h = cc_stats[cc_i][cv2.CC_STAT_LEFT], cc_stats[cc_i][cv2.CC_STAT_TOP], cc_stats[cc_i][cv2.CC_STAT_WIDTH], cc_stats[cc_i][cv2.CC_STAT_HEIGHT]
            cv2.rectangle(disp_img, (int(x), int(y)), (int(x + w), int(y + h)), (60, 200, 200), 1, lineType=cv2.LINE_AA)

        cv2.rectangle(disp_img,  (int(p1_pos[0]), int(p1_pos[1])), (int(p1_pos[0] + patch_size), int(p1_pos[1] + patch_size)), (255, 0, 0),5, lineType=cv2.LINE_AA)
        cv2.rectangle(disp_img, (int(p2_pos[0]), int(p2_pos[1])), (int(p2_pos[0] + patch_size), int(p2_pos[1] + patch_size)), (0, 0, 255), 5, lineType=cv2.LINE_AA)
        
        cv2.imwrite('sample_pairs/'+str(counter)+'page'+str(label)+'.png',disp_img)
        cv2.imwrite('sample_pairs/'+str(counter)+'firstpair'+str(label)+'.png',p1)
        cv2.imwrite('sample_pairs/'+str(counter)+'secondpair'+str(label)+'.png',p2)
        counter=counter+1        

    return p1, p2, label
    
def get_random_pair(images_path):
    images = os.listdir(images_path)
    image_name = np.random.choice(images)
    img = cv2.imread(os.path.join(images_path, image_name), 0)
    ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    gen_func = np.random.choice([get_same_patches,get_backpaired_patches,get_same_patches, get_different_patches])
    p1, p2, label = gen_func(img,thresh)
    return p1, p2, label

#images_path='ahte_test'
#for i in range (10):
#  get_random_pair(images_path)
