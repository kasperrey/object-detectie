import urllib.request
import cv2
import requests
from ultralytics import YOLO

model = YOLO()

r = requests.get("https://api.geekdo.com/api/images?ajax=1&gallery=all&nosession=1&objectid=136063&objecttype=thing&pageid=1&showcount=36&size=thumb&sort=hot")
json = r.json()

for image in json["images"]:
    person = False
    src = image["imageurl"]
    urllib.request.urlretrieve(src, "img.jpg")
    img = cv2.imread("img.jpg")
    results = model(img, verbose=False)
    for r in results:
        for c in r.boxes.cls:
            if model.names[int(c)] == "person":
                person = True
    if not person:
        cv2.imwrite("dataset2/train/images/img_"+src[len(src)-10:], img)
        f = open("dataset2/train/labels/img_"+src[len(src)-10:len(src)-4]+".txt", "w")
        f.write("2 0.5 0.5 1 1")
        f.close()

