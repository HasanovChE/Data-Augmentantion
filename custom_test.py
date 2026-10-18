import os
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms
from model import CNN
from datasets import get_datasets

def transform_custom_image(
    image,
    rotation=0,
    shift_x=0.0,
    shift_y=0.0
):
    transform = transforms.Compose([
        transforms.RandomAffine(
            degrees=(rotation, rotation),
            translate=(shift_x, shift_y)
        )
    ])
    return transform(image)

def evaluate_custom_scenarios():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    models_config = {
        "Baseline": "results/models/baseline.pt",
        "Rotation": "results/models/rotation.pt",
        "Rotation+Shift": "results/models/rotation_shift.pt",
        "Rotation+Shift+Noise": "results/models/rotation_shift_noise.pt"
    }
    
    loaded_models = {}
    for name, path in models_config.items():
        if os.path.exists(path):
            model = CNN().to(device)
            model.load_state_dict(torch.load(path, map_location=device))
            model.eval()
            loaded_models[name] = model
        else:
            print(f"Error: {path} model not found!")

    scenarios = {
        "Original": {"rotation": 0, "shift_x": 0.0, "shift_y": 0.0},
        "+10° Rotation": {"rotation": 10, "shift_x": 0.0, "shift_y": 0.0},
        "-10° Rotation": {"rotation": -10, "shift_x": 0.0, "shift_y": 0.0},
        "Shift Left": {"rotation": 0, "shift_x": 0.1, "shift_y": 0.0},
        "Shift Right": {"rotation": 0, "shift_x": -0.1, "shift_y": 0.0},
        "Rotation + Shift": {"rotation": 10, "shift_x": 0.1, "shift_y": 0.1}
    }

    _, test_dataset = get_datasets(augmentation=None)
    
    results = []

    for scenario_name, params in scenarios.items():
        row = {"Test Scenario": scenario_name}
        
        for model_name, model in loaded_models.items():
            correct = 0
            total = 0
            
            with torch.no_grad():
                for idx in range(min(1000, len(test_dataset))):
                    img_tensor, label = test_dataset[idx]
                    
                    pil_img = transforms.ToPILImage()(img_tensor)
                    trans_img = transform_custom_image(
                        pil_img,
                        rotation=params["rotation"],
                        shift_x=params["shift_x"],
                        shift_y=params["shift_y"]
                    )

                    input_tensor = transforms.Compose([
                        transforms.ToTensor()
                    ])(trans_img).unsqueeze(0).to(device)

                    output = model(input_tensor)
                    pred = output.argmax(dim=1).item()
                    
                    if pred == label:
                        correct += 1
                    total += 1
            
            acc = correct / total if total > 0 else 0
            row[model_name] = f"{acc * 100:.2f}%"
            
        results.append(row)

    df_results = pd.DataFrame(results)
    os.makedirs("results/custom_tests", exist_ok=True)
    df_results.to_csv("results/custom_tests/custom_test_results.csv", index=False)
    
    print("\n--- Custom Test Scenarios Comparison Table ---")
    print(df_results.to_string(index=False))

if __name__ == "__main__":
    evaluate_custom_scenarios()