import src.angle as angle
import src.cutter as cutter
import yaml
import os


if __name__ == "__main__":

    try:
        with open ("paths.yaml", encoding="utf8") as y:
            paths = yaml.safe_load(y)
    except FileNotFoundError:
        print("paths.yaml not found! Please put paths.yaml in the same directory as the .exe file")
        input("Press any key to exit...")
        exit()
    
    try:
        model_path = paths["model_path"]
        vertebra_path = paths["vertebra_path"]
        results_path = paths["results_path"]
    except KeyError:
        print("paths header not found!")
    
    if model_path == None or vertebra_path == None or results_path == None:
        print("paths.yaml error, please ensure you have fill the paths")
        input("Press any key to exit...")
        exit()
    elif not os.path.exists(model_path) or not model_path.endswith(".pt"):
        print("Model not found!")
        input("Press any key to exit...")
        exit()
    elif not os.path.exists(vertebra_path):
        print("Images not found!")
        input("Press any key to exit...")
        exit()

    def mode_selector(mode):
        if(mode == "1"):
            cutter_object = cutter.Cutter(model_path, vertebra_path, results_path)
            cutter_object.run()
            print("Predict and croods save done!")
        elif(mode == "2"):
            cutter_object = cutter.Cutter(model_path, vertebra_path, results_path)
            cutter_object.run()
            print("Predict and croods save done!")
            angle_calculator = angle.AngleSVACalculator(result_save_path=results_path, original_vertebra_path=vertebra_path)
            angle_calculator.run()
            print("Angle and SVA calculate done!")
        else:
            print("Invalid mode!")

    print(f"Model Path: {model_path}\nDataset Path: {vertebra_path}\nResults Path: {results_path}")

    mode = input("Enter Mode (1: C2-C7 predict only, 2: C2-C7 predict and calculate Cobb Angle and SVA): ")

    mode_selector(mode)

    input("Press any key to exit...")