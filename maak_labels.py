import cv2
import os
from pybboxes import BoundingBox
import random
import yaml


class Start:
    def __init__(self):
        self.lijst_fotos, self.lijst_labels = self.maak_lijst()
        self.labels = {}
        self.read_yaml()
        Scherm(self)

    def maak_lijst(self):
        lijst_images = []
        lijst_labels = []
        for file in os.listdir("add_to_dataset"):
            if not file[len(file)-3:] == "txt":
                lijst_images.append((self.resize(cv2.imread("add_to_dataset/"+file)), "add_to_dataset/"+file))
                print()
                if os.path.exists("add_to_dataset/"+file[:len(file)-4]+".txt"):
                    lijst_labels.append((self.verander_yolo_in_co(self.resize(cv2.imread("add_to_dataset/"+file)), "add_to_dataset/"+file[:len(file)-4]+".txt"),
                                         "add_to_dataset/"+file[:len(file)-4]+".txt"))
        return lijst_images, lijst_labels

    def verander_yolo_in_co(self, img, txt_file):
        lijst = []
        dh, dw, _ = img.shape
        fl = open(txt_file, 'r')
        data = fl.readlines()
        fl.close()
        for dt in data:
            k, x, y, w, h = map(float, dt.split(' '))
            lijst.append((BoundingBox.from_yolo(*[x, y, w, h], image_size=(img.shape[1], img.shape[0])).to_voc(return_values=True), int(k)))
        return lijst

    def verander_co_in_yolo(self, img, x1, y1, x2, y2):
        return BoundingBox.from_voc(*[x1, y1, x2, y2], image_size=(img.shape[1], img.shape[0])).to_yolo(return_values=True)

    def resize(self, img):
        delen_door = max(img.shape[1], img.shape[0])/1000
        img = cv2.resize(img, (int(img.shape[1]/delen_door), int(img.shape[0]/delen_door)))
        return img

    def read_yaml(self):
        f = open('spellen2.yaml', 'r')
        data = yaml.load(f, Loader=yaml.SafeLoader)
        for name in data["names"]:
            self.labels[str(name)] = (data["names"][name], (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
        f.close()



class Scherm:
    def __init__(self, start):
        self.plus = 0
        self.labels = []
        self.start_pos = None
        self.end_pos = None
        self.start = start
        self.image = self.start.lijst_fotos[self.plus][0]
        self.aangepast_image = self.image.copy()
        cv2.namedWindow("Display")
        self.add_loaded_labels()
        cv2.setMouseCallback('Display', self.mouse_klick)
        cv2.imshow("Display", self.image)
        self.loop()
        cv2.destroyAllWindows()

    def mouse_klick(self, event, x, y, flags, param):
        if event == 1:
            self.start_pos = (x, y)
        elif event == 4:
            self.end_pos = (x, y)
        elif event == 0:
            if not self.end_pos:
                self.aangepast_image = self.image.copy()
                if self.start_pos:
                    cv2.rectangle(self.aangepast_image, self.start_pos, (x, y), (0, 0, 255))
                cv2.line(self.aangepast_image, (0, y), (self.aangepast_image.shape[1], y), (255, 255, 255))
                cv2.line(self.aangepast_image, (x, 0), (x, self.aangepast_image.shape[0]), (255, 255, 255))
                cv2.imshow("Display", self.aangepast_image)

    def loop(self):
        while True:
            gewacht = cv2.waitKey(1)
            if self.end_pos:
                if gewacht == ord("d"):
                    self.aangepast_image = self.image.copy()
                    cv2.imshow("Display", self.aangepast_image)
                    self.start_pos = None
                    self.end_pos = None
                for key in self.start.labels.keys():
                    if gewacht == ord(key):
                        self.aangepast_image = self.image.copy()
                        cv2.rectangle(self.aangepast_image, self.start_pos, self.end_pos, self.start.labels[key][1])
                        cv2.putText(self.aangepast_image, self.start.labels[key][0],
                                    (self.start_pos[0], self.start_pos[1]-5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.start.labels[key][1])
                        cv2.imshow("Display", self.aangepast_image)
                        self.labels.append((self.start_pos, self.end_pos, key, self.start.lijst_fotos[self.plus][1]))
                        self.image = self.aangepast_image.copy()
                        self.start_pos = None
                        self.end_pos = None
            if gewacht == ord("q"):
                break
            elif gewacht == ord("s"):
                lijst = []
                files = []
                for label in self.labels:
                    for image in self.start.lijst_fotos:
                        if image[1] == label[3]:
                            yolo = self.start.verander_co_in_yolo(image[0], label[0][0], label[0][1], label[1][0], label[1][1])
                            lijst.append((f"{label[2]} {yolo[0]} {yolo[1]} {yolo[2]} {yolo[3]}", label[3]))
                for foto in self.start.lijst_fotos:
                    files.append(open(foto[1][:len(foto[1])-4]+".txt", "w+"))
                for file in files:
                    for text in lijst:
                        if text[1][:len(text[1])-4]+".txt" == file.name:
                            file.write(text[0]+"\n")
                    file.close()
                break
            elif gewacht == ord("v"):
                self.plus += 1
                if len(self.start.lijst_fotos) <= self.plus:
                    self.plus = 0
                self.image = self.start.lijst_fotos[self.plus][0].copy()
                for label in self.labels:
                    if label[3] == self.start.lijst_fotos[self.plus][1]:
                        cv2.rectangle(self.image, label[0], label[1], self.start.labels[label[2]][1])
                        cv2.putText(self.image, self.start.labels[str(label[2])][0],
                                        (label[0][0], label[0][1] - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.start.labels[str(label[2])][1])
                self.aangepast_image = self.image.copy()
                cv2.imshow("Display", self.image)

    def add_loaded_labels(self):
        for label in self.start.lijst_labels:
            for image in self.start.lijst_fotos:
                if label[1][:len(label[1])-4] == image[1][:len(image[1])-4]:
                    for coord in label[0]:
                        self.labels.append(((coord[0][0], coord[0][1]), (coord[0][2], coord[0][3]), str(coord[1]), image[1]))
                        if image[1] == self.start.lijst_fotos[self.plus][1]:
                            cv2.rectangle(self.image, (coord[0][0], coord[0][1]), (coord[0][2], coord[0][3]), self.start.labels[str(coord[1])][1])
                            cv2.putText(self.image, self.start.labels[str(coord[1])][0],
                                        (coord[0][0], coord[0][1] - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.start.labels[str(coord[1])][1])


Start()
