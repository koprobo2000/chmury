import cv2
from flask import Flask
from flask_restful import Resource, Api

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


app = Flask(__name__)
api = Api(app)


class PeopleCounter(Resource):
    def get(self):

        cv2.waitKey(0)
        image = cv2.imread('ludzie.jpg')
        image = cv2.resize(image, (1000, 900))
# detect people in the image
        (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.05)
        return {'peopleCount': len(rects)}


api.add_resource(PeopleCounter, '/')


if __name__ == '__main__':
    app.run(debug=True)
