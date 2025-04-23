import cv2
from paddleclas import PaddleClas

def main():
    model = PaddleClas(
        model_name="vehicle_attribute"
    )
    img = cv2.imread("/home/erwin/Documents/AutoNeura/resources/images/cars/car1.jpg")    

    # # Perform inference
    # attributes = model.predict(
    #     img,
    #     predict_type="cls"
    # )

    print(attributes)
if __name__ == "__main__":
    main()