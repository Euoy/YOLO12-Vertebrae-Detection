import angle
import cutter
import yaml


if __name__ == "__main__":

    try:
        with open ("paths.yaml", encoding="utf8") as y:
            paths = yaml.safe_load(y)
    except FileNotFoundError:
        print("paths.yaml not found! Please put paths.yaml in the same directory as the .exe file")
        input("Press any key to exit...")
        exit()
    
    model_path = paths["model_path"]
    vertebra_path = paths["vertebra_path"]
    results_path = paths["results_path"]

    def mode(mode):
        if(mode == "1"):
            cutter = cutter.Cutter(model_path, vertebra_path, results_path)
            cutter.run()
            print("Predict and croods save done!")
        elif(mode == "2"):
            cutter = cutter.Cutter(model_path, vertebra_path, results_path)
            cutter.run()
            print("Predict and croods save done!")
            angle_calculator = angle.AngleSVACalculator(result_save_path=results_path, original_vertebra_path=vertebra_path)
            angle_calculator.run()
            print("Angle and SVA calculate done!")

    print(f"Model Path: {model_path}\nDataset Path: {vertebra_path}\nResults Path: {results_path}")

    mode = input("Enter Mode (1: C2-C7 predict only, 2: C2-C7 predict and calculate Cobb Angle and SVA): ")

    mode(mode)

    input("Press any key to exit...")