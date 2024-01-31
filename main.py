import cv2
import requests
from flask import Flask, request
from flask_restful import Resource, Api
import numpy as np


# metoda przyjmuje obraz i zwraca ilość osób
def process_image(image) -> int:
    # detekcja osób algorytmem hog
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    (rects, weights) = hog.detectMultiScale(image, winStride=(3, 3), padding=(10, 10), scale=1.01)
    # należy odkomentować w celu podejrzenia obrazu
    # for (x, y, w, h) in rects:
    #    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #
    # cv2.imshow('HOGdetector', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return len(rects)


app = Flask(__name__)
api = Api(app)


class PeopleCounter(Resource):
    def get(self):

        image = cv2.imread('ludzie.jpg')
        # skalowanie za dużego obrazu
        scaled_w = 1000
        scale_factor = scaled_w / image.shape[1]
        scaled_h = int(image.shape[0] * scale_factor)

        image = cv2.resize(image, (scaled_w, scaled_h))

        people_count = process_image(image)
        return {'peopleCount': people_count}


# klasa do pobierania url
class GetImageUrl(Resource):
    def get(self):
        try:
            # pobieranie wartości url
            url = request.args.get('url')
            if not url:
                return {"error": "Brak poprawnego URL"}, 400
            # zapytanie get do url i przechowywanie odpowiedzi w zmiennej response
            response = requests.get(url)
            image_array = np.frombuffer(response.content, dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            if image is None:
                return {"error": "Nie można załadować obrazu z URL"}, 404

            people_count = process_image(image)
            return {'peopleCount': people_count}

        except Exception as e:
            print(f'Error: {str(e)}')
            return {'error': str(e)}


# klasa do przesyłania obrazu
class UploadImage(Resource):
    def post(self):
        try:
            file = request.files['file']

            if not file:
                return {"error": "Brak przesłanego pliku"}, 400

            image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

            if image is None:
                return {"error": "Nie można załadować przesłanego obrazu"}, 400

            people_count = process_image(image)
            return {'peopleCount': people_count}

        except Exception as e:
            return {'error': str(e)}


# trzy endpointy (2x get, 1 post)
api.add_resource(PeopleCounter, '/')
api.add_resource(GetImageUrl, '/provideurl')
api.add_resource(UploadImage, '/img_upload')

if __name__ == '__main__':
    app.run(debug=True)
