import cv2
import numpy as np
import os
import imutils

def select_cluster(org_folder,vis_folder,page_name,line_folder,overlap_folder):
    vis_img=cv2.imread(os.path.join(vis_folder,page_name))
    org_img=cv2.imread(os.path.join(org_folder,page_name),0)
    ret, thresh = cv2.threshold(org_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    rows,cols,ch=vis_img.shape 
    red=vis_img[:,:,2]
    green=vis_img[:,:,1]
    blue=vis_img[:,:,0]
    #select cluster
    lines=np.zeros((rows,cols),dtype=np.uint8)
    s=(green>219)&(blue>225)
    lines[s]=255
    #find contours
    cnts = cv2.findContours(lines.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    mask = np.ones(lines.shape[:2], dtype="uint8") * 255
    # loop over the contours
    for c in cnts:
	# if the contour is small, draw it on the mask
        if cv2.contourArea(c)<7000:
            cv2.drawContours(mask, [c], -1, 0, -1)
    lines = cv2.bitwise_and(lines, lines, mask=mask)

    kernel=np.ones((15,75),np.uint8)
    dilation=cv2.dilate(lines,kernel,1)

    cv2.imwrite(os.path.join(line_folder,page_name),dilation)
    cv2.imwrite(os.path.join(overlap_folder,page_name),dilation&thresh)

vis_folder='ahte_line_7_10/cv2_vis2'
org_folder='ahte_test'
line_folder='ahte_line_5'
overlap_folder='ahte_line_overlap_5'

os.makedirs(line_folder, exist_ok=True)
os.makedirs(overlap_folder, exist_ok=True)

#page_name='book1_page11.png'
#page_name='moc_test_10.png'

for page_name in os.listdir(vis_folder):
    select_cluster(org_folder,vis_folder,page_name,line_folder,overlap_folder)



